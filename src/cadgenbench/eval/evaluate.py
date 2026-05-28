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
  headline number per fixture. Equals the arithmetic mean of every
  available component score (shape similarity, interface match,
  topology match), so each axis contributes equally and only the ones
  actually computed for a fixture enter the mean. Zero when the
  candidate is not a valid solid or no STEP was produced.

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

from cadgenbench.common.validity import analyze_step
from cadgenbench.eval.interface_match import (
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
    validation_dict = _validation_dict(raw_candidate)
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

    rmse = _align_or_reuse(
        raw_candidate, gt_step, aligned_step, renders_dir,
        data=data, force=force_align,
    )

    # --- Shape similarity ---------------------------------------------------
    comparison = compare_step_files(
        aligned_step, gt_step,
        align=False,
        alignment_rmse=rmse,
        candidate_renders_dir=renders_dir,
    )
    scores = comparison.scores

    # --- Interface match (aligned candidate; only when jig files exist) -----
    interface_metrics = _interface_metrics_dict(
        aligned_step,
        gt_dir,
        gt_step,
    )
    if interface_metrics:
        _maybe_render_interface_overlay(
            aligned_step,
            gt_dir,
            result_dir / "interface_overlay.png",
        )

    # --- Topology match (Betti b0/b1/b2 on the tessellated boundary) --------
    topology_metrics = _topology_metrics_dict(
        raw_candidate, gt_step, validation_dict,
    )

    data["status"] = STATUS_VALID
    data["validation"] = validation_dict
    data["gt_metrics"] = scores
    data["shape_diagnostics"] = comparison.diagnostics
    data["alignment"] = {"rmse": round(rmse, 4)}
    if interface_metrics:
        data["interface_metrics"] = interface_metrics
    if topology_metrics:
        data["topology_metrics"] = topology_metrics

    data["cad_score"] = _cad_score(
        scores=scores,
        interface_metrics=interface_metrics,
        topology_metrics=topology_metrics,
        validation=validation_dict,
    )

    result_json.write_text(json.dumps(data, indent=2))

    return scores


def _cad_score(
    *,
    scores: dict,
    interface_metrics: dict,
    topology_metrics: dict,
    validation: dict,
) -> float:
    """Return the CAD Score (``[0, 1]``) for one fixture.

    The score is the arithmetic mean of every component score that was
    successfully computed for this fixture: shape similarity, interface
    match, topology match. Each axis contributes equally; missing axes
    (e.g. fixtures without jig sub-volumes) simply drop out of the mean
    rather than diluting it. Zeroes out when the candidate failed CAD
    validation so an invalid geometry never wins comparisons.
    """
    if not validation.get("is_valid"):
        return 0.0
    terms: list[float] = []
    shape = scores.get("shape_similarity_score")
    if shape is not None:
        terms.append(float(shape))
    iface = (interface_metrics or {}).get("score")
    if iface is not None:
        terms.append(float(iface))
    topo = (topology_metrics or {}).get("score")
    if topo is not None:
        terms.append(float(topo))
    if not terms:
        return 0.0
    return sum(terms) / len(terms)


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
        from cadgenbench.common.viewer import render_step

        for img in render_step(str(candidate_step)):
            (renders_dir / f"{img.name}.png").write_bytes(img.data)
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
    gt_step: Path,
    aligned_step: Path,
    renders_dir: Path,
    *,
    data: dict,
    force: bool,
) -> float:
    """Align *raw_candidate* to *gt_step*, reusing a cached result when fresh."""
    cached_rmse = (data.get("alignment") or {}).get("rmse")

    fresh = (
        aligned_step.exists()
        and aligned_step.stat().st_mtime >= raw_candidate.stat().st_mtime
        and cached_rmse is not None
    )
    if fresh and not force:
        return float(cached_rmse)

    from cadgenbench.eval.alignment import align_step

    aligned_step.parent.mkdir(parents=True, exist_ok=True)
    ar = align_step(raw_candidate, gt_step, output=aligned_step, refine=True, pca_top_k=12)

    # The cached candidate renders are stale once the aligned geometry moves.
    shutil.rmtree(renders_dir, ignore_errors=True)
    return ar.rmse


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


def _validation_dict(candidate_step: Path) -> dict:
    """Flatten validity + measurements into the ``result.json["validation"]`` schema.

    The JSON key is still ``validation`` for backward compatibility with
    downstream reports, even though it now carries measurement fields too.
    """
    try:
        a = analyze_step(candidate_step)
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


def _topology_metrics_dict(
    candidate_step: Path,
    gt_step: Path,
    validation: dict,
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
        result = topo_match(candidate_step, gt_step)
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
) -> dict:
    """Return interface-match metrics for one aligned candidate.

    Returns an empty dict when the fixture has no
    ``jig_<context_id>__<index>__<fit_type>.step`` files.
    """
    sub_volumes = discover_sub_volumes(fixture_dir)
    if not sub_volumes:
        return {}

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


