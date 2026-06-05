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

"""End-to-end evaluation of a result directory against a benchmark GT.

:func:`evaluate_result` is the single entry point used by both the
baseline pipeline and the standalone CLI (``cadgenbench evaluate``).
It is idempotent: given the same inputs it reuses any previously
computed alignment + renders, so it is safe to re-run after a metric
set changes.

The function writes a fully agent-agnostic ``<result_dir>/result.json``
with these keys:

- ``status``            , ``"valid" | "invalid" | "missing"``. Single
  source of truth for "did this fixture produce a scorable STEP?".
- ``validation``        , validity + measurements for the raw candidate.
  Absent (with ``status == "missing"``) when no candidate STEP exists.
- ``gt_metrics``        , normalized shape-similarity metrics.
- ``shape_diagnostics`` , raw shape distances/volumes (non-metric).
- ``alignment``         , ``{"rmse": <float>}`` (cached for re-runs).
- ``interface_metrics`` , interface-match score + per-context breakdown
  when fixture jig sub-volumes are present.
- ``topology_metrics``  , Betti-number agreement (b0, b1, b2) between
  candidate and GT, plus the per-axis fuzzy log-ratio scores
  (``per_axis_scores``) and the aggregate ``topo_match`` score. See
  ``docs/metrics/topo_match.md``.
- ``cad_score``         , the **CAD Score**: single ``[0, 1]``
  headline number per fixture. For generation fixtures it is the
  weighted mean of every available component score (shape similarity,
  interface match, topology match) with weights
  ``GENERATION_AXIS_WEIGHTS`` (shape 0.4 / interface 0.4 / topology
  0.2); only the axes actually computed for a fixture enter the mean and
  the weights renormalize over those present. For editing fixtures
  (those with a committed ``edit_baseline.json`` in the GT dir) the
  shape axis is first renormalized against the no-op baseline and the
  axes are reweighted ``EDITING_AXIS_WEIGHTS`` (shape 0.5 / interface
  0.3 / topology 0.2; see ``cadgenbench.eval.edit_baseline``). Zero when
  the candidate is not a valid solid or no STEP was produced.
- ``edit_metrics``      , editing fixtures only. The no-op baseline
  (``baseline_shape_similarity``), the raw and renormalized shape-axis
  values, the ``headroom`` (``1 - baseline``), and the per-axis
  ``axis_weights`` used. Absent for generation fixtures.

``result.json`` carries no information about *how* the STEP was
produced; that's the point. The baseline agent stashes its run-only
state (``stopped_reason``, ``total_duration_s``) in a sibling
``baseline_debug.json`` which the report tools never read.

The metric *categories* (CAD validity, shape similarity, interface
match, topology match) live in dedicated sibling modules; this file
is the orchestrator that calls them and persists the combined output.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from cadgenbench.common.artifacts import (
    StepArtifacts,
    sidecar_path_for,
    write_mesh_sidecar,
)
from cadgenbench.common.profiling import note, phase
from cadgenbench.common.validity import analyze_step
from cadgenbench.eval.edit_baseline import (
    EDITING_AXIS_WEIGHTS,
    check_baseline_fresh,
    read_edit_baseline,
    renormalize_shape,
)

# Per-axis weights applied to ``cad_score`` for generation fixtures.
# Topology is toned down (it is comparatively easy to score well here, so
# it should not carry a full third of the headline); shape and interface
# split the rest. Weights renormalize over whichever axes are present.
GENERATION_AXIS_WEIGHTS: dict[str, float] = {
    "shape": 0.4,
    "interface": 0.4,
    "topology": 0.2,
}
from cadgenbench.eval.interface_match import (
    InterfaceMatchArtifacts,
    SubVolume,
    best_iou_in_context,
    discover_sub_volumes,
)
from cadgenbench.eval.shape_similarity import compare_step_files
from cadgenbench.eval.topo_match import topo_match
from cadgenbench.common.mesh import MeshSanityError

logger = logging.getLogger(__name__)

ALIGNED_STEP = "aligned/output_aligned.step"
RENDERS_DIR = "renders"
GT_STEP_NAME = "ground_truth.step"
EDIT_DIFF_WEBP = "edit_diff.webp"
# Static frame-0 still beside the turntable. Used as the grid thumbnail for
# editing samples (an animated WebP can't be frozen to one angle in HTML, and
# 35 looping clips on one page is wasteful); the full turntable still plays in
# the detail card. Rides to the render bucket like any other ``renders/*.png``.
EDIT_DIFF_PNG = "edit_diff.png"

# Per-fixture status enum surfaced as ``result.json["status"]``.  Agnostic
# to the generator: same three states whether the STEP was produced by the
# baseline agent, a script, or a human.
STATUS_VALID = "valid"      # output.step exists and passes the validity gate
STATUS_INVALID = "invalid"  # output.step exists but failed the gate
STATUS_MISSING = "missing"  # no output.step in the work dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_result(
    result_dir: Path,
    gt_dir: Path,
    *,
    candidate_step: Path | None = None,
    force_align: bool = False,
) -> dict:
    """Compute all applicable metrics for *result_dir* and persist them.

    Args:
        result_dir: Directory produced by a single fixture run.  Must
            contain ``result.json``.  Will also contain (or have created)
            ``aligned/output_aligned.step`` and ``renders/*.png``.
        gt_dir: Ground-truth source directory (``data/gt/<fixture>``).
            Must contain ``ground_truth.step``.
        candidate_step: Raw candidate STEP (pre-alignment).  Only needed if
            the candidate is not already discoverable inside *result_dir*
            (e.g. live pipeline that points at a workdir file).
        force_align: If True, realign even when a cached aligned STEP is
            fresher than the raw candidate.  Discards cached renders.

    Returns:
        The shape-similarity scores dict written to
        ``result.json["gt_metrics"]``. Empty when no candidate STEP could
        be located.
    """
    result_dir = Path(result_dir)
    gt_dir = Path(gt_dir)
    result_json = result_dir / "result.json"
    name = result_dir.name  # profiling tag (see cadgenbench.common.profiling)

    gt_step = gt_dir / GT_STEP_NAME
    if not gt_step.exists():
        raise FileNotFoundError(f"GT step missing: {gt_step}")

    raw_candidate = candidate_step or _find_candidate_step(result_dir)

    # --- Missing-candidate fast path ---------------------------------------
    if raw_candidate is None:
        logger.warning("No candidate STEP in %s; recording status=missing", result_dir)
        # Preserve only cached alignment if present; everything else gets
        # rewritten so result.json stays a pure evaluator artefact.
        prior = _read_json(result_json)
        data: dict = {
            "status": STATUS_MISSING,
            "cad_score": 0.0,
        }
        if "alignment" in prior:
            data["alignment"] = prior["alignment"]
        result_json.write_text(json.dumps(data, indent=2))
        return {}

    # --- CAD validity (raw candidate; runs BEFORE alignment / shape_similarity
    # / interface / topology). An invalid candidate (non-watertight, mesh
    # non-manifold, BRepCheck errors, etc.) is a *score signal*, not an
    # eval-pipeline failure: it lands as status=invalid + cad_score=0 and
    # the run continues. The downstream metric modules tessellate the
    # candidate and would crash on the same non-manifold geometry, so we
    # short-circuit here both to avoid the crash and to keep
    # ``evaluate_result`` exit-clean: per-fixture validity failures
    # never bubble out to the caller.
    # One deflection per fixture, derived from the GT bbox, drives every
    # tessellation on both sides (validity, shape, topology, interface,
    # render, and the ICP point clouds). Computing it up front lets the
    # candidate be meshed exactly once (here, at the validity gate) and
    # reused everywhere — a rigid alignment only moves that one mesh.
    from cadgenbench.common.mesh import deflection_for_bbox  # noqa: PLC0415

    gt_artifacts = StepArtifacts(gt_step, is_ground_truth=True)
    shared_deflection = deflection_for_bbox(
        gt_artifacts.analysis.measurements.bounding_box.diagonal,
    )

    raw_artifacts = StepArtifacts(
        raw_candidate, deflection_override=shared_deflection,
    )
    with phase("validity", tag=name):
        validation_dict = _validation_dict(raw_candidate, artifacts=raw_artifacts)
    if not validation_dict.get("is_valid"):
        prior = _read_json(result_json)
        data = {
            "status": STATUS_INVALID,
            "cad_score": 0.0,
            "validation": validation_dict,
        }
        if "alignment" in prior:
            data["alignment"] = prior["alignment"]
        result_json.write_text(json.dumps(data, indent=2))
        return {}

    prior = _read_json(result_json)
    # Carry over only the alignment cache; drop any stale agent metadata or
    # previously-computed metrics so we never publish stale evaluator output.
    data = {}
    if "alignment" in prior:
        data["alignment"] = prior["alignment"]

    aligned_step = result_dir / ALIGNED_STEP
    renders_dir = result_dir / RENDERS_DIR

    # Mesh sizes drive the align cost (selector/sampling scale with vertices);
    # log them so slow fixtures can be correlated with geometry. Both meshes are
    # already cached (candidate from the validity gate, GT from its sidecar).
    cand_mesh = raw_artifacts.mesh()
    gt_mesh = gt_artifacts.mesh()
    note(
        f"meshsize cand_v={len(cand_mesh.vertices)} cand_t={len(cand_mesh.triangles)} "
        f"gt_v={len(gt_mesh.vertices)} gt_t={len(gt_mesh.triangles)}",
        tag=name,
    )

    with phase("align", tag=name):
        rmse = _align_or_reuse(
            raw_candidate, aligned_step, renders_dir,
            raw_artifacts=raw_artifacts,
            gt_artifacts=gt_artifacts,
            data=data, force=force_align,
        )
    aligned_artifacts = StepArtifacts(aligned_step)

    # --- Shape similarity (also fills the candidate renders) ----------------
    with phase("shape", tag=name):
        comparison = compare_step_files(
            aligned_step, gt_step,
            align=False,
            alignment_rmse=rmse,
            candidate_renders_dir=renders_dir,
            candidate_artifacts=aligned_artifacts,
            gt_artifacts=gt_artifacts,
        )
    scores = comparison.scores

    # --- Interface match (aligned candidate; only when jig files exist) -----
    with phase("interface", tag=name):
        interface_metrics = _interface_metrics_dict(
            aligned_step,
            gt_dir,
            gt_step,
            candidate_artifacts=aligned_artifacts,
            gt_artifacts=gt_artifacts,
        )
    if interface_metrics:
        _maybe_render_interface_overlay(
            aligned_step,
            gt_dir,
            result_dir / "interface_overlay.png",
        )

    # --- Topology match (Betti b0/b1/b2 on the tessellated boundary) --------
    with phase("topo", tag=name):
        topology_metrics = _topology_metrics_dict(
            raw_candidate,
            gt_step,
            validation_dict,
            candidate_artifacts=raw_artifacts,
            gt_artifacts=gt_artifacts,
        )

    data["status"] = STATUS_VALID
    data["validation"] = validation_dict
    data["gt_metrics"] = scores
    data["shape_diagnostics"] = comparison.diagnostics
    if comparison.metric_errors:
        data["metric_errors"] = comparison.metric_errors
    data["alignment"] = {"rmse": round(rmse, 4)}
    if interface_metrics:
        data["interface_metrics"] = interface_metrics
    if topology_metrics:
        data["topology_metrics"] = topology_metrics

    # --- Editing-task renormalization -------------------------------------
    # A committed ``edit_baseline.json`` in the GT dir marks this as an
    # editing fixture: the shape axis is renormalized against the no-op
    # baseline and the axes are reweighted with ``EDITING_AXIS_WEIGHTS``
    # (see ``cadgenbench.eval.edit_baseline``). Its absence ⇒ generation
    # rules (raw shape, ``GENERATION_AXIS_WEIGHTS``). The scorer never
    # reads the inputs-side ``description.yaml`` for this.
    edit_baseline = read_edit_baseline(gt_dir)
    if edit_baseline is None:
        data["cad_score"] = _cad_score(
            scores=scores,
            interface_metrics=interface_metrics,
            topology_metrics=topology_metrics,
            validation=validation_dict,
            weights=GENERATION_AXIS_WEIGHTS,
        )
    else:
        check_baseline_fresh(edit_baseline, gt_dir.name)
        b_shape = edit_baseline.get("shape_similarity_score")
        raw_shape = scores.get("shape_similarity_score")
        shape_renorm = (
            renormalize_shape(float(raw_shape), float(b_shape))
            if raw_shape is not None and b_shape is not None
            else None
        )
        data["edit_metrics"] = {
            "baseline_shape_similarity": (
                round(float(b_shape), 4) if b_shape is not None else None
            ),
            "shape_similarity_raw": (
                round(float(raw_shape), 4) if raw_shape is not None else None
            ),
            "shape_similarity_renormalized": (
                round(shape_renorm, 4) if shape_renorm is not None else None
            ),
            "headroom": (
                round(1.0 - float(b_shape), 4) if b_shape is not None else None
            ),
            "axis_weights": EDITING_AXIS_WEIGHTS,
        }
        data["cad_score"] = _cad_score(
            scores=scores,
            interface_metrics=interface_metrics,
            topology_metrics=topology_metrics,
            validation=validation_dict,
            shape_score=shape_renorm,
            weights=EDITING_AXIS_WEIGHTS,
        )
        # Editing fixtures get the ghost/diff turntable alongside the
        # candidate's own renders so the (often tiny or internal) edit is
        # visible in the gallery.
        with phase("edit_diff", tag=name):
            _maybe_render_edit_diff(
                gt_artifacts, aligned_artifacts, aligned_step, renders_dir,
            )

    result_json.write_text(json.dumps(data, indent=2))

    return scores


def _cad_score(
    *,
    scores: dict,
    interface_metrics: dict,
    topology_metrics: dict,
    validation: dict,
    shape_score: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Return the CAD Score (``[0, 1]``) for one fixture.

    The score is a weighted mean of every component score that was
    successfully computed for this fixture: shape similarity, interface
    match, topology match. Missing axes (e.g. fixtures without jig
    sub-volumes) drop out and the remaining weights renormalize, rather
    than diluting the mean. Zeroes out when the candidate failed CAD
    validation so an invalid geometry never wins comparisons.

    Args:
        shape_score: Overrides the shape-similarity axis value. Editing
            fixtures pass the no-op-renormalized shape score here (see
            :mod:`cadgenbench.eval.edit_baseline`); generation fixtures
            leave it ``None`` to use the raw ``shape_similarity_score``.
        weights: Per-axis weights keyed ``"shape" | "interface" |
            "topology"``. ``None`` means equal weighting (a plain
            arithmetic mean over present axes); callers pass
            ``GENERATION_AXIS_WEIGHTS`` / ``EDITING_AXIS_WEIGHTS``. In all
            cases the weights of the axes actually present renormalize, so
            a missing axis (e.g. no jig sub-volumes) redistributes its
            weight over the rest rather than diluting the mean.
    """
    if not validation.get("is_valid"):
        return 0.0
    if shape_score is None:
        shape_score = scores.get("shape_similarity_score")
    axes: dict[str, float | None] = {
        "shape": None if shape_score is None else float(shape_score),
        "interface": (interface_metrics or {}).get("score"),
        "topology": (topology_metrics or {}).get("score"),
    }
    num = 0.0
    den = 0.0
    for axis, value in axes.items():
        if value is None:
            continue
        w = 1.0 if weights is None else float(weights.get(axis, 0.0))
        num += w * float(value)
        den += w
    if den == 0:
        return 0.0
    return num / den


def evaluate_candidate_only(candidate_step: Path, result_dir: Path) -> None:
    """Render + validate a candidate when no GT is available.

    Writes the same agent-agnostic ``result.json`` schema as
    :func:`evaluate_result`, minus the GT-derived metrics. Useful as a
    quick local sanity check; the full grader still requires a GT.
    """
    result_dir = Path(result_dir)
    renders_dir = result_dir / RENDERS_DIR
    renders_dir.mkdir(parents=True, exist_ok=True)
    try:
        from cadgenbench.common.viewer import render_step, render_step_turntable_webp

        for img in render_step(str(candidate_step)):
            (renders_dir / f"{img.name}.png").write_bytes(img.data)
        (renders_dir / "rotating.webp").write_bytes(
            render_step_turntable_webp(str(candidate_step))
        )
    except Exception:
        logger.warning("Render of %s failed", candidate_step, exc_info=True)

    validation_dict = _validation_dict(candidate_step)
    data: dict = {
        "status": (
            STATUS_VALID if validation_dict.get("is_valid") else STATUS_INVALID
        ),
        "validation": validation_dict,
    }
    (result_dir / "result.json").write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Alignment (with caching)
# ---------------------------------------------------------------------------


def _align_or_reuse(
    raw_candidate: Path,
    aligned_step: Path,
    renders_dir: Path,
    *,
    raw_artifacts: StepArtifacts,
    gt_artifacts: StepArtifacts,
    data: dict,
    force: bool,
) -> float:
    """Rigidly align the candidate to the GT, reusing a cached result when fresh.

    Operates on the candidate's *already-computed* mesh (and the GT's) via
    :func:`align_cached_mesh` — no re-tessellation. The recovered transform
    is applied to the BREP to persist ``output_aligned.step`` (a cheap
    geometric export), and the transformed mesh is written as that STEP's
    trusted sidecar so every downstream consumer reuses the one mesh
    instead of meshing the aligned geometry again.
    """
    cached_rmse = (data.get("alignment") or {}).get("rmse")
    sidecar = sidecar_path_for(aligned_step)

    fresh = (
        aligned_step.exists()
        and sidecar.exists()
        and aligned_step.stat().st_mtime >= raw_candidate.stat().st_mtime
        and cached_rmse is not None
    )
    if fresh and not force:
        return float(cached_rmse)

    from cadgenbench.eval.alignment import align_cached_mesh, export_aligned_shape

    car = align_cached_mesh(raw_artifacts, gt_artifacts, pca_top_k=12)
    aligned_step.parent.mkdir(parents=True, exist_ok=True)
    export_aligned_shape(
        raw_artifacts.wrapped, car.rotation, car.translation, aligned_step,
    )
    write_mesh_sidecar(aligned_step, car.mesh)

    # The cached candidate renders are stale once the aligned geometry moves.
    shutil.rmtree(renders_dir, ignore_errors=True)
    return car.rmse


# ---------------------------------------------------------------------------
# Candidate discovery + validation
# ---------------------------------------------------------------------------


def _find_candidate_step(result_dir: Path) -> Path | None:
    """Return the candidate STEP at the fixture root, per the submission contract.

    The evaluator is generator-agnostic: it has no concept of turns,
    baselines, or any other internal layout the generator may use. The
    canonical candidate file is ``<result_dir>/output.step`` (or
    ``.stp``) and nothing else (see
    ``docs/benchmark/submission.md``). Generators that keep their own
    debug artefacts in sibling directories are free to do so; the
    evaluator ignores them.
    """
    for name in ("output.step", "output.stp"):
        p = result_dir / name
        if p.exists():
            return p
    return None


def _validation_dict(
    candidate_step: Path,
    *,
    artifacts: StepArtifacts | None = None,
) -> dict:
    """Flatten validity + measurements into the ``result.json["validation"]`` schema.

    The JSON key is still ``validation`` for backward compatibility with
    downstream reports, even though it now carries measurement fields too.
    """
    try:
        a = artifacts.analysis if artifacts is not None else analyze_step(candidate_step)
        v, m = a.validation, a.measurements
        bb = m.bounding_box
        return {
            "is_valid": v.is_valid,
            "is_watertight": v.is_watertight,
            "solid_count": m.solid_count,
            "shell_count": m.shell_count,
            "face_count": m.face_count,
            "volume": round(m.volume, 2),
            "bbox": {
                "x": round(bb.size_x, 2),
                "y": round(bb.size_y, 2),
                "z": round(bb.size_z, 2),
            },
            "topology_errors": list(v.topology_errors[:10]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _maybe_render_interface_overlay(
    aligned_candidate_step: Path,
    fixture_dir: Path,
    output_png: Path,
) -> None:
    """Render an overlay PNG showing candidate vs sub-volumes (yellow = disagreement).

    Idempotent: only renders when *output_png* is older than the aligned
    candidate or missing. Renderer failures are logged but never abort
    the surrounding metric run.
    """
    try:
        if (
            output_png.exists()
            and output_png.stat().st_mtime >= aligned_candidate_step.stat().st_mtime
        ):
            return
        from cadgenbench.eval.interface_match_viz import (
            composite_grid,
            render_part_with_subvolumes,
        )

        sub_volumes = discover_sub_volumes(fixture_dir)
        if not sub_volumes:
            return
        images = render_part_with_subvolumes(
            aligned_candidate_step,
            sub_volumes,
            views=("iso", "top", "front", "right"),
            width=512,
            height=384,
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_png.write_bytes(composite_grid(images, cols=2))
    except Exception:
        logger.warning(
            "Interface overlay render failed for %s",
            aligned_candidate_step,
            exc_info=True,
        )


def _maybe_render_edit_diff(
    gt_artifacts: StepArtifacts,
    candidate_artifacts: StepArtifacts,
    aligned_candidate_step: Path,
    renders_dir: Path,
) -> None:
    """Render the edit-diff turntable WebP (``renders/edit_diff.webp``).

    Editing fixtures only. Ghosts the aligned candidate translucent and lights
    up only the material that differs from GT, classified by signed distance to
    the other solid (blue = added by the candidate, red = present in GT but
    missing from the candidate). This makes a small or internal edit legible
    where a plain shaded render of a near-no-op output is indistinguishable from
    a correct one. Reuses the welded meshes both artifacts already cache, so no
    re-tessellation and no re-alignment happen here.

    Idempotent: only renders when the output is missing or older than the
    aligned candidate. Renderer failures are logged but never abort the run.
    """
    try:
        output_webp = renders_dir / EDIT_DIFF_WEBP
        output_png = renders_dir / EDIT_DIFF_PNG
        webp_fresh = (
            output_webp.exists()
            and output_webp.stat().st_mtime >= aligned_candidate_step.stat().st_mtime
        )
        # Already current and the still exists -> nothing to do.
        if webp_fresh and output_png.exists():
            return
        from cadgenbench.common.imaging import first_frame_png
        from cadgenbench.common.viewer import render_mesh_diff_turntable_webp

        renders_dir.mkdir(parents=True, exist_ok=True)
        if webp_fresh:
            # Clip is current but the still is missing (e.g. evaluated
            # before the still existed) -> derive it from the clip, no
            # re-render.
            webp_bytes = output_webp.read_bytes()
        else:
            webp_bytes = render_mesh_diff_turntable_webp(
                gt_artifacts.mesh(), candidate_artifacts.mesh(),
            )
            output_webp.write_bytes(webp_bytes)
        # Frame-0 still beside the clip, picked up by the bucket uploader.
        output_png.write_bytes(first_frame_png(webp_bytes))
    except Exception:
        logger.warning(
            "Edit-diff render failed for %s",
            aligned_candidate_step,
            exc_info=True,
        )


def _topology_metrics_dict(
    candidate_step: Path,
    gt_step: Path,
    validation: dict,
    *,
    candidate_artifacts: StepArtifacts | None = None,
    gt_artifacts: StepArtifacts | None = None,
) -> dict:
    """Compute Betti agreement between candidate and GT, return persistable dict.

    Returns an empty dict if the candidate isn't a valid solid (the
    mesh-gate has already failed at the validity layer, so re-running
    here would just raise). For valid candidates whose meshing happens
    to fail at this step (should be rare given the validity gate is
    strict), the exception is propagated rather than silently swallowed
   , the metric must never paper over a deterministic failure.
    """
    if not validation.get("is_valid"):
        return {}
    try:
        result = topo_match(
            candidate_step,
            gt_step,
            candidate_artifacts=candidate_artifacts,
            gt_artifacts=gt_artifacts,
        )
    except MeshSanityError as exc:
        # If the candidate (or GT!) survives is_valid but its mesh
        # pipeline still trips a gate here, surface the discrepancy
        # rather than recording it as a topo_match=0. This means the
        # validity-layer gate and the topo-match gate disagree, which
        # is a bug we want to investigate, not bury.
        raise RuntimeError(
            f"topo_match mesh-gate disagreed with is_valid=True for "
            f"candidate={candidate_step.name} vs gt={gt_step.name}: {exc}",
        ) from exc
    return result.to_dict()


def _interface_metrics_dict(
    aligned_candidate_step: Path,
    fixture_dir: Path,
    gt_step: Path,
    *,
    n_samples: int = 32,
    workers: int = 1,
    candidate_artifacts: StepArtifacts | None = None,
    gt_artifacts: StepArtifacts | None = None,
    interface_artifacts: InterfaceMatchArtifacts | None = None,
) -> dict:
    """Return interface-match metrics for one aligned candidate.

    Returns an empty dict when the fixture has no
    ``jig_<context_id>__<index>__<fit_type>.step`` files.
    """
    sub_volumes = discover_sub_volumes(fixture_dir)
    if not sub_volumes:
        return {}
    interface_artifacts = interface_artifacts or InterfaceMatchArtifacts(
        gt_step=gt_step,
        sub_volumes=sub_volumes,
        gt_artifacts=gt_artifacts,
    )
    candidate_artifacts = candidate_artifacts or StepArtifacts(aligned_candidate_step)

    by_context: dict[int, list[SubVolume]] = {}
    for sv in sub_volumes:
        by_context.setdefault(sv.context_id, []).append(sv)

    contexts: dict[str, dict] = {}
    context_scores: list[float] = []
    for context_id in sorted(by_context):
        ctx_svs = by_context[context_id]
        per_sv = best_iou_in_context(
            aligned_candidate_step,
            ctx_svs,
            gt_step,
            n_samples=n_samples,
            workers=workers,
            candidate_artifacts=candidate_artifacts,
            interface_artifacts=interface_artifacts,
        )
        ctx_score = min(per_sv.values())
        context_scores.append(ctx_score)
        contexts[str(context_id)] = {
            "score": round(ctx_score, 4),
            "sub_volumes": {
                sv.name: round(per_sv[sv.name], 4)
                for sv in sorted(ctx_svs, key=lambda s: s.index)
            },
        }

    overall = sum(context_scores) / len(context_scores)
    return {
        "score": round(overall, 4),
        "contexts": contexts,
    }


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


