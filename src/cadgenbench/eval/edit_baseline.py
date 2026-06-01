# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Editing-task no-op baseline + shape-axis renormalization.

For an editing fixture the unedited ``input.step`` is a valid,
near-GT solid, so the global shape-similarity metric scores it high.
The "no-op" strategy (return the input unchanged) can therefore beat a
real edit attempt that perturbs the unchanged bulk. To stop rewarding
that, the **shape-similarity axis is renormalized against the no-op
baseline**::

    b_shape   = shape_similarity(input.step, GT)        # the no-op's score
    s_renorm  = max(0, (s_raw - b_shape) / (1 - b_shape))

The no-op maps to ``0``, a perfect candidate stays at ``1``, and
anything at or below the no-op floors at ``0``. Topology and interface
match stay **raw** (most edits leave them unchanged, and where they do
not they already discriminate). For editing fixtures the per-fixture
``cad_score`` reweights the three axes :data:`EDITING_AXIS_WEIGHTS`
(shape ``0.5``, topology ``0.25``, interface ``0.25``) instead of the
equal mean used for generation. The validity gate still hard-zeros.

``b_shape`` depends only on ``input.step`` + ``ground_truth.step`` +
the shape / alignment implementation, so it is a **fixture constant**.
It is precomputed once at authoring time (``sanity_check_gt.py
--write-baselines``) and committed to the GT dataset as
``<fixture>/edit_baseline.json``. The grader reads it back at eval
time (the GT-side files are exactly what HF Jobs downloads); it never
recomputes it per submission. The **presence** of ``edit_baseline.json``
in a fixture's GT dir is also the single source of truth that tells the
grader "this is an editing fixture, renormalize + reweight" — the
scorer never has to read the inputs-side ``description.yaml``.

Staleness: the baseline is stamped with ``cadgenbench_version``. If the
shape-similarity or alignment math changes (bump ``cadgenbench``'s
version), the committed baseline must be regenerated; the grader
hard-errors on a version mismatch rather than scoring against a stale
number.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from cadgenbench import __version__
from cadgenbench.eval.shape_similarity import compare_step_files

logger = logging.getLogger(__name__)

# Committed per-fixture GT file holding the no-op baseline. Its presence
# is also the editing-fixture flag the grader keys off.
EDIT_BASELINE_NAME = "edit_baseline.json"

# Per-axis weights applied to ``cad_score`` for editing fixtures. Shape
# is the axis that actually resolves most edits, so it dominates 2:1
# over topology and interface (which are frequently non-discriminating
# on a given edit). Generation fixtures keep the equal mean.
EDITING_AXIS_WEIGHTS: dict[str, float] = {
    "shape": 0.5,
    "topology": 0.25,
    "interface": 0.25,
}

# Authoring gate: the edit must leave the shape metric at least this
# much headroom (``1 - b_shape``). Below this the no-op already scores
# essentially perfectly, so the renormalized axis cannot resolve the
# edit — the *fixture* is rejected at authoring time, not special-cased
# in the scorer. The shape metric is very stable (tessellation residue
# saturates IoU near 0.99), so the floor is small; it sits ~10x above
# the run-to-run alignment jitter on ``b_shape`` (~1e-3) so the gate is
# a stable pass/fail rather than noise.
EDIT_HEADROOM_FLOOR = 1e-2


def compute_edit_baseline(input_step: str | Path, gt_step: str | Path) -> dict:
    """Score the no-op (``input.step`` against the GT) and return the baseline dict.

    The input is aligned to the GT through the **exact same path** a
    candidate goes through in :func:`cadgenbench.eval.evaluate.evaluate_result`
    (``align_step`` with ``refine=True, pca_top_k=12``, then
    :func:`compare_step_files` with ``align=False``), so ``b_shape`` is
    apples-to-apples with the score any submission's ``output.step``
    receives.

    Returns a JSON-serializable dict with the headline
    ``shape_similarity_score`` (``b_shape``), its three sub-metrics, the
    alignment ``rmse``, the derived ``headroom`` (``1 - b_shape``), and a
    ``cadgenbench_version`` stamp for staleness detection.
    """
    input_step = Path(input_step)
    gt_step = Path(gt_step)

    from cadgenbench.eval.alignment import align_step

    with tempfile.TemporaryDirectory() as td:
        aligned = Path(td) / "input_aligned.step"
        ar = align_step(
            input_step, gt_step, output=aligned, refine=True, pca_top_k=12,
        )
        comparison = compare_step_files(
            aligned, gt_step, align=False, alignment_rmse=ar.rmse,
        )

    scores = comparison.scores
    b_shape = scores.get("shape_similarity_score")
    if b_shape is None:
        raise RuntimeError(
            f"could not compute shape_similarity for the no-op baseline "
            f"(input={input_step.name} vs gt={gt_step.name}); shape "
            f"sub-metrics were all unavailable",
        )
    b_shape = float(b_shape)
    return {
        "shape_similarity_score": b_shape,
        "shape_point_cloud_f1": scores.get("shape_point_cloud_f1"),
        "shape_volume_iou": scores.get("shape_volume_iou"),
        "shape_feature_edge_f1": scores.get("shape_feature_edge_f1"),
        "alignment_rmse": round(float(ar.rmse), 4),
        "headroom": round(1.0 - b_shape, 6),
        "cadgenbench_version": __version__,
    }


def write_edit_baseline(gt_dir: str | Path, baseline: dict) -> Path:
    """Write *baseline* to ``<gt_dir>/edit_baseline.json`` and return the path."""
    gt_dir = Path(gt_dir)
    out = gt_dir / EDIT_BASELINE_NAME
    out.write_text(json.dumps(baseline, indent=2) + "\n")
    return out


def read_edit_baseline(gt_dir: str | Path) -> dict | None:
    """Return the committed no-op baseline for a fixture, or ``None``.

    ``None`` means the fixture has no ``edit_baseline.json`` — i.e. it is
    not an editing fixture and is scored with the generation rules.
    """
    path = Path(gt_dir) / EDIT_BASELINE_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_baseline_fresh(baseline: dict, fixture_name: str) -> None:
    """Raise if the committed baseline was produced by a different cadgenbench.

    Called on the grader's hot path for editing fixtures. A version
    mismatch means the shape / alignment math may have moved under the
    cached ``b_shape``; rather than silently score against a stale
    number we fail loud and point the operator at the regeneration step.
    """
    base_ver = baseline.get("cadgenbench_version")
    if base_ver != __version__:
        raise RuntimeError(
            f"{EDIT_BASELINE_NAME} for fixture {fixture_name!r} was computed "
            f"with cadgenbench {base_ver!r} but the grader is running "
            f"{__version__!r}. The no-op baseline is stale; regenerate it "
            f"with `sanity_check_gt.py --all --write-baselines` and re-commit "
            f"the GT dataset.",
        )


def renormalize_shape(raw_shape: float, b_shape: float) -> float:
    """Map the raw shape-similarity score onto the no-op-anchored ``[0, 1]`` scale.

    ``b_shape`` (the no-op) maps to ``0``; a perfect candidate maps to
    ``1``; anything at or below the no-op floors at ``0``. The degenerate
    zero-headroom case (``b_shape >= 1``) should never reach scoring
    because the authoring gate (:data:`EDIT_HEADROOM_FLOOR`) rejects such
    fixtures, but it is handled defensively.
    """
    headroom = 1.0 - b_shape
    if headroom <= 0.0:
        return 1.0 if raw_shape > b_shape else 0.0
    return max(0.0, min(1.0, (raw_shape - b_shape) / headroom))
