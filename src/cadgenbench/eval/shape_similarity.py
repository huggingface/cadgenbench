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

"""Shape-similarity metrics, one of the three v1 metric categories.

This module computes global geometry similarity scores in ``[0, 1]`` for
already-aligned candidate/GT STEP files.

Reported shape metrics:

- ``shape_point_cloud_f1``:
  Symmetric F1 of surface point clouds with a normal-agreement gate.
  A point counts as a hit when its nearest neighbor on the other
  cloud is within ``1%`` of the GT bounding-box diagonal **and** the
  two outward unit normals dot above ``0.9`` (≈25° tolerance).
- ``shape_volume_iou``:
  Volumetric IoU of candidate and GT solids.
- ``shape_similarity_score``:
  arithmetic mean of available component scores.

Raw distances/errors are kept in diagnostics and must not be mixed into
the metric aggregate.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cadgenbench.common.artifacts import StepArtifacts
from cadgenbench.common.validity import ValidationResult
from cadgenbench.common.measurements import Measurements

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MetricContext
# ---------------------------------------------------------------------------

_DEFAULT_N_POINTS = 50_000


@dataclass
class MetricContext:
    """Bundle of data available for one shape (candidate or ground truth).

    Populate whichever fields are available; metrics that need missing
    fields will be silently skipped by :func:`compute_metrics`.
    """

    step_path: Path | None = None
    validation: ValidationResult | None = None
    measurements: Measurements | None = None
    renders_dir: Path | None = None
    # Tessellation deflection (mm). When set on both candidate and GT
    # this guarantees their meshes (and the derived manifolds) are
    # produced at the same chord error, so Manifold IoU is comparable.
    # Callers (e.g. :func:`compare_step_files`) typically derive this
    # from the GT bbox via :func:`cadgenbench.common.mesh.deflection_for_bbox`.
    linear_deflection_mm: float | None = None

    _pc_points: np.ndarray | None = field(default=None, repr=False)
    _pc_normals: np.ndarray | None = field(default=None, repr=False)
    _pc_n_points: int = field(default=_DEFAULT_N_POINTS, repr=False)
    _pc_seed: int = field(default=0, repr=False)
    _mesh: object | None = field(default=None, repr=False)
    _manifold: object | None = field(default=None, repr=False)
    artifacts: StepArtifacts | None = field(default=None, repr=False)

    def _ensure_point_cloud(self) -> None:
        """Sample points + outward-unit normals once, then cache them."""
        if self._pc_points is not None or self.step_path is None:
            return
        from cadgenbench.eval.sampling import sample_points_and_normals_from_mesh

        mesh = self.get_mesh()
        if mesh is None:
            return
        pts, nrm = sample_points_and_normals_from_mesh(
            mesh.vertices,
            mesh.triangles,
            n_points=self._pc_n_points,
            seed=self._pc_seed,
            smooth_normals=True,
        )
        self._pc_points = pts
        self._pc_normals = nrm

    @property
    def point_cloud(self) -> np.ndarray | None:
        """Lazily sampled surface point cloud (computed once, then cached)."""
        self._ensure_point_cloud()
        return self._pc_points

    @property
    def point_cloud_normals(self) -> np.ndarray | None:
        """Outward unit normals matched 1:1 with :attr:`point_cloud`."""
        self._ensure_point_cloud()
        return self._pc_normals

    def get_mesh(self):
        """Lazily tessellate the STEP into a welded :class:`Mesh` (cached).

        Uses :attr:`linear_deflection_mm` when set; otherwise derives a
        deflection from this context's own bounding box. For *metric*
        use, callers should set the same ``linear_deflection_mm`` on
        candidate and GT (via the GT bbox) so Manifold IoU is computed
        at a single scale.
        """
        if self._mesh is not None:
            return self._mesh
        if self.step_path is None:
            return None
        from cadgenbench.common.mesh import deflection_for_bbox

        if self.artifacts is not None:
            self._mesh = self.artifacts.mesh()
            return self._mesh
        from cadgenbench.common.mesh import tessellate_and_validate

        if self.linear_deflection_mm is not None:
            defl = float(self.linear_deflection_mm)
        elif self.measurements is not None:
            defl = deflection_for_bbox(self.measurements.bounding_box.diagonal)
        else:
            from cadgenbench.common.measurements import measure_step

            self.measurements = measure_step(self.step_path)
            defl = deflection_for_bbox(self.measurements.bounding_box.diagonal)
        self._mesh = tessellate_and_validate(self.step_path, defl)
        return self._mesh

    def get_manifold(self):
        """Lazily convert the cached mesh to a ``manifold3d.Manifold``."""
        if self._manifold is not None:
            return self._manifold
        mesh = self.get_mesh()
        if mesh is None:
            return None
        from cadgenbench.eval.booleans import mesh_to_manifold

        self._manifold = mesh_to_manifold(mesh)
        return self._manifold


# ---------------------------------------------------------------------------
# Metric type
# ---------------------------------------------------------------------------

MetricFn = Callable[[MetricContext, MetricContext], float | None]


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


POINT_CLOUD_F1_THRESHOLD_FRACTION = 0.01
# Normal-agreement gate on point-cloud F1 hits. A match requires the
# candidate and GT surface normals at the matched pair to agree in
# direction within this cosine, applied to the **absolute** dot product
# (|dot|): we accept a point on the right surface regardless of which way
# the facet is wound, which removes the whole orientation-sensitivity class
# (winding-convention differences and flipped patches) that flat-facet,
# signed-dot matching was fragile to. cos(30°) ≈ 0.866 keeps a generous
# angular tolerance for residual smooth-normal-across-crease wobble.
# (Combined with smooth_normals=True sampling, this makes the metric
# continuous in the triangulation.) "Right place, wrong side" is still
# excluded upstream by the watertight + manifold + winding gate.
POINT_CLOUD_F1_NORMAL_DOT_THRESHOLD = 0.8660254037844387  # cos(30°)


def shape_point_cloud_f1(candidate: MetricContext, gt: MetricContext) -> float | None:
    """Symmetric surface-point-cloud F1 in ``[0, 1]`` (threshold = 1% of GT bbox diag)."""
    stats = _point_cloud_f1_stats(candidate, gt)
    if stats is None:
        return None
    return _clamp01(stats["f1"])


def shape_volume_iou(candidate: MetricContext, gt: MetricContext) -> float | None:
    """Volumetric IoU of candidate and GT solids in ``[0, 1]``."""
    stats = _volume_overlap_stats(candidate, gt)
    if stats is None:
        return None
    inter, union, _sym_diff = stats
    if union <= 0:
        return None
    return _clamp01(inter / union)


# ---------------------------------------------------------------------------
# Registry and display metadata
# ---------------------------------------------------------------------------

DEFAULT_METRICS: dict[str, MetricFn] = {
    "shape_point_cloud_f1": shape_point_cloud_f1,
    "shape_volume_iou": shape_volume_iou,
}


@dataclass(frozen=True)
class MetricMeta:
    """How to display a metric in reports."""

    label: str
    fmt: str
    suffix: str = ""


METRIC_DISPLAY: dict[str, MetricMeta] = {
    "cad_score":              MetricMeta("CAD Score", ".3f"),
    "shape_similarity_score": MetricMeta("Shape Similarity", ".3f"),
    "shape_point_cloud_f1":   MetricMeta("Point Cloud F1", ".3f"),
    "shape_volume_iou":       MetricMeta("Volume IoU", ".3f"),
    # interface_match (jig sub-volumes)
    "interface_match_score":  MetricMeta("Interface Match", ".3f"),
    # topo_match (Betti agreement)
    "topo_match_score":       MetricMeta("Topo Match", ".3f"),
}


def compute_metrics(
    candidate: MetricContext,
    gt: MetricContext,
    metrics: dict[str, MetricFn] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Run all applicable metrics and return ``(scores, errors)``.

    A metric that returns ``None`` (the required data is missing) is
    omitted from ``scores``. A metric that raises is scored ``0.0`` and
    recorded in ``errors`` as ``{name: "ExcType: message"}``: on valid
    CAD these metrics are deterministic and should not fail, so an
    exception is treated as a candidate-side failure (the sub-metric
    counts as 0, which also means a deliberate crash can never raise
    the score) and surfaced loudly instead of silently dropped.
    ``errors`` is empty on a clean run.
    """
    fns = metrics if metrics is not None else DEFAULT_METRICS
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name, fn in fns.items():
        try:
            value = fn(candidate, gt)
        except Exception as exc:
            logger.error("Metric %s raised; scoring it 0", name, exc_info=True)
            scores[name] = 0.0
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if value is not None:
            scores[name] = value
    return scores, errors


# ---------------------------------------------------------------------------
# High-level comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Output of :func:`compare_step_files`."""

    scores: dict[str, float | None]
    diagnostics: dict[str, float]
    alignment_rmse: float | None = None
    aligned_step: Path | None = None
    metric_errors: dict[str, str] = field(default_factory=dict)


def compare_step_files(
    candidate_step: str | Path,
    gt_step: str | Path,
    *,
    align: bool = True,
    aligned_output: str | Path | None = None,
    alignment_rmse: float | None = None,
    candidate_renders_dir: str | Path | None = None,
    gt_renders_dir: str | Path | None = None,
    metrics: dict[str, MetricFn] | None = None,
    candidate_artifacts: StepArtifacts | None = None,
    gt_artifacts: StepArtifacts | None = None,
) -> ComparisonResult:
    """Align two STEP files, render them, and compute all applicable metrics.

    Flow: (optional) align → render → metrics.

    Args:
        candidate_step: Path to the candidate STEP file.
        gt_step: Path to the ground-truth STEP file.
        align: If True, rigidly align the candidate to the GT before
            rendering and computing metrics.  Set to False when the caller
            has already aligned and is passing the aligned STEP directly.
        aligned_output: Where to write the aligned STEP.  Defaults to
            ``<candidate_stem>_aligned.step`` next to *candidate_step*.
        alignment_rmse: Precomputed RMSE from alignment. Ignored when
            ``align=True`` (fresh RMSE is used).
        candidate_renders_dir: If provided, renders of the (post-alignment)
            candidate STEP are written here using the viewer's default views.
            Existing PNGs are left alone; missing views are rendered.
        gt_renders_dir: If provided, GT renders are written here using the
            viewer's default views.  Existing PNGs are left alone.
        metrics: Optional custom metric registry.  Defaults to
            :data:`DEFAULT_METRICS`.
    Returns:
        :class:`ComparisonResult` with scores, alignment RMSE (if known), and
        the path to the aligned STEP file (if produced).
    """
    candidate_step = Path(candidate_step)
    gt_step = Path(gt_step)
    gt_artifacts = gt_artifacts or StepArtifacts(gt_step)

    aligned_path: Path | None = None
    rmse: float | None = alignment_rmse
    step_for_metrics = candidate_step
    aligned_mesh = None  # set on the trusted-candidate path (no STEP round-trip)

    if align:
        cand_art = candidate_artifacts or StepArtifacts(candidate_step)
        if cand_art.has_sidecar:
            # Trusted candidate: align its cached mesh and score from that.
            # Never write an aligned STEP or re-tessellate — the supplied mesh
            # is the reference (a rigid transform keeps it valid).
            from cadgenbench.eval.alignment import align_cached_mesh

            car = align_cached_mesh(cand_art, gt_artifacts)
            aligned_mesh = car.mesh
            rmse = car.rmse
            candidate_artifacts = cand_art
            step_for_metrics = candidate_step
        else:
            # Untrusted candidate (e.g. a submission): align via STEP export,
            # then re-tessellate the aligned geometry. Unchanged behavior.
            from cadgenbench.eval.alignment import align_step

            ar = align_step(
                candidate_step, gt_step,
                output=aligned_output,
            )
            step_for_metrics = ar.output_path
            aligned_path = ar.output_path
            rmse = ar.rmse
            candidate_artifacts = None

    cand_renders = Path(candidate_renders_dir) if candidate_renders_dir else None
    gt_renders = Path(gt_renders_dir) if gt_renders_dir else None

    candidate_artifacts = candidate_artifacts or StepArtifacts(step_for_metrics)
    cand_analysis = candidate_artifacts.analysis
    gt_analysis = gt_artifacts.analysis

    # One deflection drives the mesh-Boolean Volume IoU on both sides,
    # derived from the GT's bbox so candidate and GT are tessellated
    # at the same chord error. Mirrors what topo_match does.
    from cadgenbench.common.mesh import deflection_for_bbox

    shared_deflection = deflection_for_bbox(
        gt_analysis.measurements.bounding_box.diagonal,
    )

    ctx_candidate = MetricContext(
        step_path=step_for_metrics,
        validation=cand_analysis.validation,
        measurements=cand_analysis.measurements,
        renders_dir=cand_renders,
        linear_deflection_mm=shared_deflection,
        # On the trusted path the mesh is injected below; passing artifacts
        # would make get_mesh() return the *un-aligned* cached mesh instead.
        artifacts=None if aligned_mesh is not None else candidate_artifacts,
    )
    if aligned_mesh is not None:
        ctx_candidate._mesh = aligned_mesh
    ctx_gt = MetricContext(
        step_path=gt_step,
        validation=gt_analysis.validation,
        measurements=gt_analysis.measurements,
        renders_dir=gt_renders,
        linear_deflection_mm=shared_deflection,
        artifacts=gt_artifacts,
    )

    # Render after building the contexts so we reuse the welded mesh
    # the metric path is about to compute anyway, halving the BREP
    # tessellation work on the eval hot path.
    if cand_renders is not None:
        _ensure_renders(ctx_candidate, cand_renders)
    if gt_renders is not None:
        _ensure_renders(ctx_gt, gt_renders)

    if metrics is None:
        scores, diagnostics, metric_errors = _compute_default_scores_and_diagnostics(
            ctx_candidate,
            ctx_gt,
        )
    else:
        scores, metric_errors = compute_metrics(ctx_candidate, ctx_gt, metrics=metrics)
        diagnostics = _compute_diagnostics(ctx_candidate, ctx_gt)

    return ComparisonResult(
        scores=scores,
        diagnostics=diagnostics,
        alignment_rmse=rmse,
        aligned_step=aligned_path,
        metric_errors=metric_errors,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bbox_diagonal(ctx: MetricContext) -> float | None:
    """Euclidean diagonal of a context's bounding box, or None."""
    if ctx.measurements is None:
        return None
    return float(ctx.measurements.bounding_box.diagonal)


def _clamp01(value: float) -> float:
    """Clamp a scalar to the unit interval."""
    return max(0.0, min(1.0, value))


def _compute_default_scores_and_diagnostics(
    candidate: MetricContext,
    gt: MetricContext,
) -> tuple[dict[str, float | None], dict[str, float], dict[str, str]]:
    """Compute default shape metrics and diagnostics without duplicate work."""
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    diagnostics: dict[str, float] = {}

    diag = _bbox_diagonal(gt)
    if diag is not None:
        diagnostics["part_diagonal"] = diag

    try:
        pc_stats = _point_cloud_f1_stats(candidate, gt)
    except Exception as exc:
        logger.error("Metric shape_point_cloud_f1 raised; scoring it 0", exc_info=True)
        scores["shape_point_cloud_f1"] = 0.0
        errors["shape_point_cloud_f1"] = f"{type(exc).__name__}: {exc}"
        pc_stats = None
    if pc_stats is not None:
        scores["shape_point_cloud_f1"] = _clamp01(pc_stats["f1"])
        _add_point_cloud_diagnostics(diagnostics, pc_stats)

    try:
        volume_stats = _volume_overlap_stats(candidate, gt)
    except Exception as exc:
        logger.error("Metric shape_volume_iou raised; scoring it 0", exc_info=True)
        scores["shape_volume_iou"] = 0.0
        errors["shape_volume_iou"] = f"{type(exc).__name__}: {exc}"
        volume_stats = None
    if volume_stats is not None:
        inter, union, _sym_diff = volume_stats
        if union > 0:
            scores["shape_volume_iou"] = _clamp01(inter / union)
        _add_volume_diagnostics(diagnostics, volume_stats)

    component_keys = (
        "shape_point_cloud_f1",
        "shape_volume_iou",
    )
    component_values = [scores[k] for k in component_keys if scores.get(k) is not None]
    if component_values:
        scores["shape_similarity_score"] = float(sum(component_values) / len(component_values))
    return scores, diagnostics, errors


def _compute_diagnostics(candidate: MetricContext, gt: MetricContext) -> dict[str, float]:
    """Compute diagnostics for custom metric runs."""
    diagnostics: dict[str, float] = {}
    diag = _bbox_diagonal(gt)
    if diag is not None:
        diagnostics["part_diagonal"] = diag
    pc_stats = _point_cloud_f1_stats(candidate, gt)
    if pc_stats is not None:
        _add_point_cloud_diagnostics(diagnostics, pc_stats)
    stats = _volume_overlap_stats(candidate, gt)
    if stats is not None:
        _add_volume_diagnostics(diagnostics, stats)
    return diagnostics


def _add_point_cloud_diagnostics(
    diagnostics: dict[str, float],
    pc_stats: dict[str, float],
) -> None:
    diagnostics["point_cloud_f1"] = pc_stats["f1"]
    diagnostics["point_cloud_precision"] = pc_stats["precision"]
    diagnostics["point_cloud_recall"] = pc_stats["recall"]
    diagnostics["point_cloud_threshold"] = pc_stats["threshold"]
    diagnostics["point_cloud_mean_chamfer"] = pc_stats["mean_chamfer"]
    diagnostics["point_cloud_mean_normal_dot"] = pc_stats["mean_normal_dot"]


def _add_volume_diagnostics(
    diagnostics: dict[str, float],
    stats: tuple[float, float, float],
) -> None:
    diagnostics["volume_intersection"] = stats[0]
    diagnostics["volume_union"] = stats[1]
    diagnostics["volume_symmetric_difference"] = stats[2]


def _point_cloud_f1_stats(
    candidate: MetricContext, gt: MetricContext,
) -> dict[str, float] | None:
    """Symmetric point-cloud F1 + diagnostics (mean chamfer, threshold).

    A point counts as a hit only when both gates pass:

    1. Distance to its nearest neighbour on the other cloud is within
       :data:`POINT_CLOUD_F1_THRESHOLD_FRACTION` of the GT bbox diagonal.
    2. The outward normal at the source point and at the matched point
       dot above :data:`POINT_CLOUD_F1_NORMAL_DOT_THRESHOLD` (i.e. they
       face roughly the same direction; ≈25° tolerance at 0.9).

    The second gate rejects "right place, wrong side" matches (back-face
    of a thin wall, or a flipped-orientation copy of the part) that
    would otherwise pass on distance alone.
    """
    pts_a = candidate.point_cloud
    pts_b = gt.point_cloud
    nrm_a = candidate.point_cloud_normals
    nrm_b = gt.point_cloud_normals
    if pts_a is None or pts_b is None or nrm_a is None or nrm_b is None:
        return None
    diag = _bbox_diagonal(gt)
    if diag is None or diag <= 0:
        return None

    threshold = max(1e-6, POINT_CLOUD_F1_THRESHOLD_FRACTION * diag)

    from scipy.spatial import cKDTree

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    a_to_b_dist, a_to_b_idx = tree_b.query(pts_a)
    b_to_a_dist, b_to_a_idx = tree_a.query(pts_b)

    a_to_b_dot = np.einsum("ij,ij->i", nrm_a, nrm_b[a_to_b_idx])
    b_to_a_dot = np.einsum("ij,ij->i", nrm_b, nrm_a[b_to_a_idx])

    # |dot|: orientation-insensitive (accept the right surface regardless of
    # winding direction); see POINT_CLOUD_F1_NORMAL_DOT_THRESHOLD.
    a_hit = (a_to_b_dist <= threshold) & (
        np.abs(a_to_b_dot) > POINT_CLOUD_F1_NORMAL_DOT_THRESHOLD
    )
    b_hit = (b_to_a_dist <= threshold) & (
        np.abs(b_to_a_dot) > POINT_CLOUD_F1_NORMAL_DOT_THRESHOLD
    )

    precision = float(a_hit.mean())
    recall = float(b_hit.mean())
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float((2.0 * precision * recall) / (precision + recall))
    mean_chamfer = float((a_to_b_dist.mean() + b_to_a_dist.mean()) / 2.0)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "mean_chamfer": mean_chamfer,
        "mean_normal_dot": float((a_to_b_dot.mean() + b_to_a_dot.mean()) / 2.0),
    }


def _volume_overlap_stats(
    candidate: MetricContext, gt: MetricContext,
) -> tuple[float, float, float] | None:
    """Return ``(intersection, union, symmetric_difference)`` volumes.

    Backed by :mod:`cadgenbench.eval.booleans` (the ``manifold3d``
    mesh-Boolean kernel). Candidate and GT are tessellated at the same
    deflection (derived from the GT bbox) so the IoU is computed at one
    consistent scale.
    """
    from cadgenbench.eval.booleans import (
        intersect,
        manifold_volume,
    )

    cand_manifold = candidate.get_manifold()
    gt_manifold = gt.get_manifold()
    if cand_manifold is None or gt_manifold is None:
        return None
    vol_a = manifold_volume(cand_manifold)
    vol_b = manifold_volume(gt_manifold)
    if vol_a <= 0 or vol_b <= 0:
        return None
    vol_inter = manifold_volume(intersect(cand_manifold, gt_manifold))
    vol_union = max(0.0, vol_a + vol_b - vol_inter)
    if vol_union <= 0:
        return None
    vol_sym_diff = max(0.0, vol_union - vol_inter)
    return vol_inter, vol_union, vol_sym_diff


def _ensure_renders(ctx: MetricContext, renders_dir: Path) -> None:
    """Write canonical-view PNGs and the turntable WebP for *ctx*.

    Reuses the welded mesh already cached on the context (computed at
    the shared GT-derived deflection) so we never tessellate twice for
    the same fixture. The rotating WebP backs the leaderboard gallery; the
    static PNG views back the per-fixture report.
    """
    from cadgenbench.common.viewer import (
        DEFAULT_VIEWS as RENDER_VIEWS,
        render_mesh,
        render_mesh_turntable_webp,
    )

    renders_dir.mkdir(parents=True, exist_ok=True)
    missing = [v for v in RENDER_VIEWS if not (renders_dir / f"{v}.png").exists()]
    webp_path = renders_dir / "rotating.webp"
    if not missing and webp_path.exists():
        return
    mesh = ctx.get_mesh()
    if mesh is None:
        raise RuntimeError(
            f"Cannot render {ctx.step_path}: tessellation produced no mesh.",
        )
    if missing:
        for img in render_mesh(mesh, views=missing):
            (renders_dir / f"{img.name}.png").write_bytes(img.data)
    if not webp_path.exists():
        webp_path.write_bytes(render_mesh_turntable_webp(mesh))
