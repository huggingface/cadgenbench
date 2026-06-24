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

"""Editing-task no-op baseline + per-axis renormalization.

For an editing fixture the unedited ``input.step`` is a valid, near-GT
solid, so every global similarity metric scores it high. The "no-op"
strategy (return the input unchanged) must therefore score **0**, not
partial credit. To get that, each axis is anchored on its own no-op
baseline ``b_axis = metric(input.step, GT)`` and assigned a *role*:

- ``target``  -- the edit moves this axis. The raw score is renormalized,
  ``renorm = max(0, (raw - b)/(1 - b))``, so the no-op maps to ``0`` and a
  perfect candidate to ``1``. Targets combine as a weighted mean
  (:data:`EDITING_AXIS_WEIGHTS`, renormalized over the targets present).
- ``preserve`` -- the edit leaves this axis unchanged (``b`` ~ 1). It earns
  no positive credit (that would re-float the no-op); it is a one-sided
  *gate* ``min(1, raw/b)`` that only pulls the score down when the candidate
  breaks what should have stayed put.
- ``absent`` -- metric not applicable (e.g. no jig sub-volumes); dropped.

``cad_score = mean_target(renorm) * prod_preserve(gate)``. Shape is the
anchor and is always ``target``; topology / interface are ``target`` only
when the edit moved them past :data:`PRESERVE_HEADROOM_FLOOR`. The validity
gate still hard-zeros. Generation fixtures are unchanged
(``GENERATION_AXIS_WEIGHTS``, raw axes).

Each ``b_axis`` depends only on ``input.step`` + the GT + the metric /
alignment implementation, so the baseline is a **fixture constant**. It is
precomputed once at authoring time (``sanity_check_gt.py --write-baselines``)
and committed to the GT dataset as ``<fixture>/edit_baseline.json``. The
grader reads it back at eval time (the GT-side files are exactly what HF Jobs
downloads); it never recomputes it per submission. The **presence** of
``edit_baseline.json`` in a fixture's GT dir is also the single source of
truth that tells the grader "this is an editing fixture" — the scorer never
reads the inputs-side ``description.yaml``.

Staleness: the baseline is stamped with ``cadgenbench_version``. If the
metric or alignment math changes (bump ``cadgenbench``'s version), the
committed baseline must be regenerated; the grader hard-errors on a version
mismatch rather than scoring against a stale number.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from cadgenbench import __version__

logger = logging.getLogger(__name__)

# Committed per-fixture GT file holding the no-op baseline. Its presence
# is also the editing-fixture flag the grader keys off.
EDIT_BASELINE_NAME = "edit_baseline.json"

# Per-axis roles, decided once at authoring time from the no-op headroom and
# committed in ``edit_baseline.json``. The grader reads them back.
ROLE_TARGET = "target"      # the edit moves this axis -> renormalize to the no-op
ROLE_PRESERVE = "preserve"  # edit leaves it unchanged -> one-sided break gate
ROLE_ABSENT = "absent"      # metric N/A for this fixture (e.g. no jig) -> dropped

# Relative weights among the **target** axes of an editing fixture (the no-op
# is now structurally 0 regardless of weights, so these express importance
# only, not a no-op cap). Shape is the axis that resolves most edits, so it
# dominates; weights renormalize over whichever axes are actually targets on a
# given fixture, and preserve / absent axes never enter the mean. See
# ``GENERATION_AXIS_WEIGHTS`` for the generation-task counterpart.
EDITING_AXIS_WEIGHTS: dict[str, float] = {
    "shape": 0.6,
    "interface": 0.3,
    "topology": 0.1,
}

# Authoring gate: the edit must leave the **shape** metric at least this
# much headroom (``1 - b_shape``). Below this the no-op already scores
# essentially perfectly, so the renormalized axis cannot resolve the
# edit — the fixture is rejected at authoring time, not special-cased in
# the scorer. Alignment is deterministic, so the only noise in ``b_shape``
# is F1 point-sampling (~2e-4); the floor sits ~10x above it so the
# renormalized shape axis carries signal rather than noise. Shape is the
# anchor axis and is always a ``target``.
EDIT_HEADROOM_FLOOR = 2e-3

# Target-vs-preserve threshold for topology and interface. A headroom at or
# above this means the edit genuinely moves the axis, so it is renormalized
# against the no-op (``target``); below it the edit didn't touch the axis and
# it becomes a one-sided break gate (``preserve``). A scan of the v1 editing
# fixtures showed both axes are cleanly bimodal — headroom is either ~0 or
# well above 1e-2 — so this split is robust and needs no per-fixture tuning.
PRESERVE_HEADROOM_FLOOR = 1e-2


def classify_axis(axis: str, baseline: float | None) -> str:
    """Assign an axis its grading role from the no-op baseline.

    Shape is always a ``target`` (the authoring gate guarantees it has
    headroom). Topology / interface are ``target`` when the edit moved them
    past :data:`PRESERVE_HEADROOM_FLOOR`, else ``preserve``. A ``None``
    baseline (metric not applicable) is ``absent``.
    """
    if baseline is None:
        return ROLE_ABSENT
    if axis == "shape":
        return ROLE_TARGET
    return ROLE_TARGET if (1.0 - baseline) >= PRESERVE_HEADROOM_FLOOR else ROLE_PRESERVE


def compute_edit_baseline(input_step: str | Path, gt_dir: str | Path) -> dict:
    """Score the no-op (``input.step`` vs GT) on all axes and return the baseline.

    The input is aligned and scored through the **exact grader path**
    (:func:`cadgenbench.eval.evaluate.compute_noop_axes`): the production
    aligner on the trusted mesh sidecar, then shape / topology / interface.
    This keeps each per-axis baseline apples-to-apples with the score any
    submission's ``output.step`` receives.

    Returns a JSON-serializable dict whose ``axes`` maps each of ``shape``,
    ``topology``, ``interface`` to ``{baseline, headroom, role}`` (role per
    :func:`classify_axis`; ``interface``/``topology`` baseline is ``None``
    when the metric does not apply). Also carries the alignment ``rmse`` and a
    ``cadgenbench_version`` stamp for staleness detection.
    """
    from cadgenbench.eval.evaluate import compute_noop_axes  # noqa: PLC0415

    input_step = Path(input_step)
    gt_dir = Path(gt_dir)

    raw = compute_noop_axes(input_step, gt_dir)
    if raw.get("shape") is None:
        raise RuntimeError(
            f"could not compute shape_similarity for the no-op baseline "
            f"(input={input_step.name} vs gt_dir={gt_dir.name}); the anchor "
            f"shape axis is unavailable",
        )

    axes: dict[str, dict] = {}
    for axis in ("shape", "topology", "interface"):
        b = raw.get(axis)
        axes[axis] = {
            "baseline": None if b is None else round(float(b), 6),
            "headroom": None if b is None else round(1.0 - float(b), 6),
            "role": classify_axis(axis, b),
        }
    return {
        "axes": axes,
        "alignment_rmse": raw.get("alignment_rmse"),
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


def renormalize(raw: float, baseline: float) -> float:
    """Map a raw axis score onto the no-op-anchored ``[0, 1]`` scale (target axes).

    ``baseline`` (the no-op) maps to ``0``; a perfect candidate maps to
    ``1``; anything at or below the no-op floors at ``0``. The degenerate
    zero-headroom case (``baseline >= 1``) should not reach a target axis —
    such an axis is classified ``preserve`` — but it is handled defensively.
    """
    headroom = 1.0 - baseline
    if headroom <= 0.0:
        return 1.0 if raw > baseline else 0.0
    return max(0.0, min(1.0, (raw - baseline) / headroom))


def preserve_gate(raw: float, baseline: float) -> float:
    """One-sided break gate for a ``preserve`` axis, in ``[0, 1]``.

    The edit left this axis unchanged (``baseline`` is ~1), so the axis earns
    no positive credit — that would re-float the no-op. Instead it only pulls
    the score down when the candidate *breaks* what should have stayed put:
    at or above the baseline the gate is ``1`` (no penalty); below it the
    gate falls off as ``raw / baseline``.
    """
    if baseline <= 0.0:
        return 1.0
    return max(0.0, min(1.0, raw / baseline))
