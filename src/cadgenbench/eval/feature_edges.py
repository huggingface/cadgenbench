"""Mesh-based feature-edge extraction for shape-similarity scoring.

Operates directly on the tessellated candidate / GT meshes used for
volume IoU and topology, so the kept edges are genuine 3D features
(creases, hole rims, slot openings) rather than view-dependent
silhouette pixels.

Algorithm (per mesh):

1. Build the undirected edge map. For each interior edge incident to
   exactly two triangles, compute the dihedral angle from the two
   outward face normals,
   ``angle = arccos(clip(dot(n1, n2), -1, 1))``.
2. Bin the edge:
   - ``angle >= tau_sharp`` (default 30°): **kept**.
   - ``angle <= tau_smooth`` (default 5°): **dropped**.
   - in-between: **ambiguous**, excluded from both candidate and GT so
     uncertain dihedrals (fillet tessellation, sliver-triangle normal
     jitter) don't score on either side.
3. Sample points uniformly along each kept edge at spacing
   ``s = spacing_frac * gt_bbox_diagonal`` (default 0.2% of the GT
   diagonal, ~500 samples across a 100 mm part). Edges shorter than
   ``s`` contribute their midpoint.
4. Return an ``(N, 3)`` array of feature-edge sample points.

Non-manifold edges (≠2 incident faces) are skipped. The closed-manifold
validity gate rules them out anyway, but the local check keeps the
module usable on raw welded meshes during debug.

:func:`extract_feature_edges_debug` returns the kept edges as
line-segment endpoints alongside the sample points, for tuning
``tau_sharp`` / ``tau_smooth`` against a real mesh.
:func:`write_feature_edge_overlay_png` renders a hit / miss overlay
between a candidate and a GT for documentation and debugging.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults (mirrors the spec)
# ---------------------------------------------------------------------------

DEFAULT_TAU_SHARP_DEG = 30.0
DEFAULT_TAU_SMOOTH_DEG = 5.0
# Sample spacing as a fraction of the GT bbox diagonal. 0.002 yields
# ~500 samples for a 1 m of total feature-edge length on a 100 mm part.
DEFAULT_SPACING_FRAC = 0.002


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureEdgeDebug:
    """Diagnostic artifacts from one feature-edge extraction.

    Attributes:
        points: ``(N, 3)`` sampled feature-edge points (the scoring input).
        segments: ``(M, 2, 3)`` line-segment endpoints of every kept
            edge, suitable for ``trimesh.load_path`` or
            ``pyvista.lines_from_points``.
        n_total_edges: Total number of undirected edges in the mesh.
        n_kept: Edges with dihedral >= ``tau_sharp``.
        n_smooth: Edges with dihedral <= ``tau_smooth`` (dropped).
        n_ambiguous: Edges in ``(tau_smooth, tau_sharp)`` (excluded).
        n_non_manifold: Edges incident to ≠2 triangles (skipped).
        tau_sharp_deg: Effective ``tau_sharp`` used.
        tau_smooth_deg: Effective ``tau_smooth`` used.
        spacing: Effective edge-sampling spacing in mm.
    """

    points: np.ndarray
    segments: np.ndarray
    n_total_edges: int
    n_kept: int
    n_smooth: int
    n_ambiguous: int
    n_non_manifold: int
    tau_sharp_deg: float
    tau_smooth_deg: float
    spacing: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_feature_edge_points(
    verts: np.ndarray,
    tris: np.ndarray,
    *,
    bbox_diagonal: float,
    tau_sharp_deg: float = DEFAULT_TAU_SHARP_DEG,
    tau_smooth_deg: float = DEFAULT_TAU_SMOOTH_DEG,
    spacing_frac: float = DEFAULT_SPACING_FRAC,
) -> np.ndarray:
    """Sample points along sharp mesh edges (dihedral above ``tau_sharp``).

    Args:
        verts: ``(V, 3)`` vertex positions.
        tris: ``(F, 3)`` triangle vertex indices with consistent outward
            winding (the gate enforced by
            :mod:`cadgenbench.common.mesh`). Inconsistent winding gives
            sign-flipped dihedrals.
        bbox_diagonal: GT bounding-box diagonal in mm. Drives the edge
            sampling spacing.
        tau_sharp_deg: Lower bound for "this edge is a sharp feature",
            in degrees.
        tau_smooth_deg: Upper bound for "this edge is on a smooth
            patch", in degrees. The ``(tau_smooth, tau_sharp)`` band
            is excluded from both sides of the eventual F1.
        spacing_frac: Edge sampling spacing as a fraction of
            ``bbox_diagonal``.

    Returns:
        ``(N, 3)`` float64 array of sampled feature-edge points. Empty
        when the mesh has no edges that satisfy ``tau_sharp``.
    """
    return _extract(
        verts, tris,
        bbox_diagonal=bbox_diagonal,
        tau_sharp_deg=tau_sharp_deg,
        tau_smooth_deg=tau_smooth_deg,
        spacing_frac=spacing_frac,
        with_debug=False,
    )


def extract_feature_edges_debug(
    verts: np.ndarray,
    tris: np.ndarray,
    *,
    bbox_diagonal: float,
    tau_sharp_deg: float = DEFAULT_TAU_SHARP_DEG,
    tau_smooth_deg: float = DEFAULT_TAU_SMOOTH_DEG,
    spacing_frac: float = DEFAULT_SPACING_FRAC,
) -> FeatureEdgeDebug:
    """Like :func:`extract_feature_edge_points` but also returns the kept
    edges as line segments plus counts per bin, for tau-sweep debugging
    and overlay rendering. Inputs and behaviour are otherwise identical.
    """
    return _extract(
        verts, tris,
        bbox_diagonal=bbox_diagonal,
        tau_sharp_deg=tau_sharp_deg,
        tau_smooth_deg=tau_smooth_deg,
        spacing_frac=spacing_frac,
        with_debug=True,
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _extract(
    verts: np.ndarray,
    tris: np.ndarray,
    *,
    bbox_diagonal: float,
    tau_sharp_deg: float,
    tau_smooth_deg: float,
    spacing_frac: float,
    with_debug: bool,
):
    if tau_smooth_deg < 0 or tau_sharp_deg <= tau_smooth_deg:
        raise ValueError(
            f"need 0 <= tau_smooth_deg < tau_sharp_deg; got "
            f"tau_smooth={tau_smooth_deg}, tau_sharp={tau_sharp_deg}",
        )
    if spacing_frac <= 0:
        raise ValueError(f"spacing_frac must be > 0, got {spacing_frac}")
    if bbox_diagonal <= 0:
        raise ValueError(f"bbox_diagonal must be > 0, got {bbox_diagonal}")

    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (V, 3); got {verts.shape}")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError(f"tris must be (F, 3); got {tris.shape}")

    spacing = float(spacing_frac * bbox_diagonal)
    cos_sharp = math.cos(math.radians(tau_sharp_deg))
    cos_smooth = math.cos(math.radians(tau_smooth_deg))

    n_tris = tris.shape[0]
    if n_tris == 0:
        return _empty_result(
            with_debug=with_debug,
            tau_sharp_deg=tau_sharp_deg,
            tau_smooth_deg=tau_smooth_deg,
            spacing=spacing,
        )

    # ---- per-triangle unit normals ----------------------------------------
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    tri_normals_raw = np.cross(v1 - v0, v2 - v0)
    tri_lengths = np.linalg.norm(tri_normals_raw, axis=1)
    # Truly-degenerate triangles get a placeholder normal; their edges
    # will fall into the ambiguous band whatever the partner triangle.
    safe_lengths = np.where(tri_lengths > 0, tri_lengths, 1.0)
    tri_normals = tri_normals_raw / safe_lengths[:, None]

    # ---- undirected edge -> incident-triangles map ------------------------
    edges = np.concatenate(
        [
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ],
        axis=0,
    )
    # The three slices are stacked in blocks of n_tris rows each, so the
    # parallel triangle index is ``tile``, not ``repeat`` (which would
    # group all three edges of triangle 0 first, etc.).
    tri_of_edge = np.tile(np.arange(n_tris, dtype=np.int64), 3)
    # Sort each undirected edge so (a, b) and (b, a) hash the same.
    sorted_edges = np.sort(edges, axis=1)

    # Lexicographic sort by edge key, then walk runs of equal keys to
    # collect the incident triangles per undirected edge.
    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    sorted_keys = sorted_edges[order]
    sorted_tri_of_edge = tri_of_edge[order]

    # Run-length boundaries.
    diffs = np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)
    run_starts = np.concatenate(([0], np.flatnonzero(diffs) + 1))
    run_ends = np.concatenate((run_starts[1:], [sorted_keys.shape[0]]))
    run_lengths = run_ends - run_starts

    # ---- bin edges --------------------------------------------------------
    n_total = int(run_starts.size)
    n_non_manifold = 0
    n_kept = 0
    n_smooth = 0
    n_ambiguous = 0
    kept_endpoints_a: list[np.ndarray] = []
    kept_endpoints_b: list[np.ndarray] = []

    for rs, rl in zip(run_starts, run_lengths):
        if rl != 2:
            n_non_manifold += 1
            continue
        t1 = sorted_tri_of_edge[rs]
        t2 = sorted_tri_of_edge[rs + 1]
        dot = float(tri_normals[t1] @ tri_normals[t2])
        # dot >= cos_smooth  ->  angle <= tau_smooth (smooth, drop)
        # dot <= cos_sharp   ->  angle >= tau_sharp  (sharp, keep)
        # otherwise           ->  ambiguous (exclude)
        if dot >= cos_smooth:
            n_smooth += 1
        elif dot <= cos_sharp:
            n_kept += 1
            ea, eb = sorted_keys[rs]
            kept_endpoints_a.append(verts[ea])
            kept_endpoints_b.append(verts[eb])
        else:
            n_ambiguous += 1

    if n_kept == 0:
        return _empty_result(
            with_debug=with_debug,
            n_total_edges=n_total,
            n_smooth=n_smooth,
            n_ambiguous=n_ambiguous,
            n_non_manifold=n_non_manifold,
            tau_sharp_deg=tau_sharp_deg,
            tau_smooth_deg=tau_smooth_deg,
            spacing=spacing,
        )

    endpoints_a = np.asarray(kept_endpoints_a, dtype=np.float64)
    endpoints_b = np.asarray(kept_endpoints_b, dtype=np.float64)

    # ---- sample along each kept edge -------------------------------------
    points = _sample_segments(endpoints_a, endpoints_b, spacing)

    if not with_debug:
        return points

    segments = np.stack([endpoints_a, endpoints_b], axis=1)
    return FeatureEdgeDebug(
        points=points,
        segments=segments,
        n_total_edges=n_total,
        n_kept=n_kept,
        n_smooth=n_smooth,
        n_ambiguous=n_ambiguous,
        n_non_manifold=n_non_manifold,
        tau_sharp_deg=tau_sharp_deg,
        tau_smooth_deg=tau_smooth_deg,
        spacing=spacing,
    )


def _sample_segments(
    a: np.ndarray, b: np.ndarray, spacing: float,
) -> np.ndarray:
    """Sample uniform points along each segment at ``spacing`` mm.

    Per edge:
    - length >= spacing -> ceil(length / spacing) + 1 samples between
      endpoints inclusive (so the corner points anchor the edge).
    - length < spacing  -> single midpoint sample.
    """
    diffs = b - a
    lengths = np.linalg.norm(diffs, axis=1)

    out_chunks: list[np.ndarray] = []
    short_mask = lengths < spacing
    if short_mask.any():
        short_a = a[short_mask]
        short_b = b[short_mask]
        out_chunks.append(0.5 * (short_a + short_b))

    long_mask = ~short_mask
    if long_mask.any():
        long_a = a[long_mask]
        long_b = b[long_mask]
        long_len = lengths[long_mask]
        # Equal spacing; np.linspace per edge would be Python-slow.
        # Build a flat list of (t, edge_id) pairs and gather.
        n_samples = np.ceil(long_len / spacing).astype(np.int64) + 1
        n_samples = np.maximum(n_samples, 2)
        starts = np.concatenate(([0], np.cumsum(n_samples)))
        edge_ids = np.repeat(
            np.arange(long_a.shape[0], dtype=np.int64), n_samples,
        )
        t = np.empty(int(starts[-1]), dtype=np.float64)
        for i, n in enumerate(n_samples):
            t[starts[i] : starts[i] + n] = np.linspace(0.0, 1.0, int(n))
        samples = (
            long_a[edge_ids]
            + t[:, None] * (long_b[edge_ids] - long_a[edge_ids])
        )
        out_chunks.append(samples)

    if not out_chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(out_chunks, axis=0)


# ---------------------------------------------------------------------------
# Documentation / debug visualisation
# ---------------------------------------------------------------------------


# Canonical iso view (azimuth, elevation) in degrees; the convention
# matches mpl_toolkits.mplot3d's view_init.
_DEFAULT_ISO_VIEW: tuple[float, float] = (-60.0, 30.0)


def write_feature_edge_overlay_png(
    candidate_step: str | Path,
    gt_step: str | Path,
    output_path: str | Path,
    *,
    view: tuple[float, float] = _DEFAULT_ISO_VIEW,
    tau_sharp_deg: float = DEFAULT_TAU_SHARP_DEG,
    tau_smooth_deg: float = DEFAULT_TAU_SMOOTH_DEG,
    spacing_frac: float = DEFAULT_SPACING_FRAC,
    threshold_frac: float = 0.01,
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: int = 160,
    title: str | None = None,
) -> dict[str, float]:
    """Render the F1 hit / miss overlay for the feature-edge metric.

    Tessellates both STEPs at the GT-derived deflection, extracts feature
    edges, samples them, and classifies each candidate / GT sample by
    Chamfer-F1 hit (distance to nearest neighbour on the other side within
    ``threshold_frac * gt_bbox_diag``). The GT mesh is drawn as a
    translucent grey surface from one canonical view; sample
    classification is overlaid in three colours:

    - green: matched within threshold on both sides.
    - blue: GT-only feature-edge sample (a real feature the candidate
      is missing).
    - red: candidate-only feature-edge sample (a feature the candidate
      invented).

    Returns a dict with the F1 / precision / recall / threshold / counts
    the production metric emits in its diagnostics, suitable for figure
    captions or downstream HTML.

    The renderer is matplotlib's Agg backend, so this function has no
    GPU / OpenGL dependency.

    Example: regenerate the documentation illustration
    (``docs/metrics/illustrations/example_1_shape/feature_edge_overlay_iso.png``)
    on top of any fixture's ``ground_truth.step``::

        import math
        import numpy as np
        from pathlib import Path
        from cadgenbench.common.measurements import measure_step
        from cadgenbench.eval.alignment import _apply_and_export
        from cadgenbench.eval.sampling import _load_occ_shape
        from cadgenbench.eval.feature_edges import write_feature_edge_overlay_png

        from cadgenbench.common.paths import data_gt_dir
        gt = data_gt_dir() / "jig-01-single-hole-plate" / "ground_truth.step"
        # 3° rotation around Z (centred on the part centroid) + 4 mm
        # along X. Produces a balanced overlay with green at the centre
        # and red/blue along the perimeter.
        b = measure_step(gt).bounding_box
        c = np.array([
            0.5 * (b.x_min + b.x_max),
            0.5 * (b.y_min + b.y_max),
            0.5 * (b.z_min + b.z_max),
        ])
        theta = math.radians(3.0)
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta),  math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t = c - R @ c + np.array([4.0, 0.0, 0.0])
        synthetic = Path("/tmp/doc_candidate.step")
        _apply_and_export(_load_occ_shape(gt), R, t, synthetic)

        write_feature_edge_overlay_png(
            candidate_step=synthetic,
            gt_step=gt,
            output_path=Path(
                "docs/metrics/illustrations/example_1_shape/"
                "feature_edge_overlay_iso.png",
            ),
        )
    """
    from cadgenbench.common.measurements import measure_step
    from cadgenbench.common.mesh import (
        deflection_for_bbox,
        tessellate_and_validate,
    )
    from scipy.spatial import cKDTree

    candidate_step = Path(candidate_step)
    gt_step = Path(gt_step)
    output_path = Path(output_path)

    gt_measurements = measure_step(gt_step)
    gt_diag = float(gt_measurements.bounding_box.diagonal)
    if gt_diag <= 0:
        raise ValueError(f"GT bbox diagonal must be > 0; got {gt_diag}")
    deflection = deflection_for_bbox(gt_diag)
    threshold = max(1e-6, threshold_frac * gt_diag)

    gt_mesh = tessellate_and_validate(gt_step, deflection)
    cand_mesh = tessellate_and_validate(candidate_step, deflection)

    pts_gt = extract_feature_edge_points(
        gt_mesh.vertices, gt_mesh.triangles,
        bbox_diagonal=gt_diag,
        tau_sharp_deg=tau_sharp_deg,
        tau_smooth_deg=tau_smooth_deg,
        spacing_frac=spacing_frac,
    )
    pts_cand = extract_feature_edge_points(
        cand_mesh.vertices, cand_mesh.triangles,
        bbox_diagonal=gt_diag,
        tau_sharp_deg=tau_sharp_deg,
        tau_smooth_deg=tau_smooth_deg,
        spacing_frac=spacing_frac,
    )

    if pts_cand.shape[0] and pts_gt.shape[0]:
        tree_gt = cKDTree(pts_gt)
        tree_cand = cKDTree(pts_cand)
        cand_to_gt = tree_gt.query(pts_cand)[0]
        gt_to_cand = tree_cand.query(pts_gt)[0]
        cand_hit = cand_to_gt <= threshold
        gt_hit = gt_to_cand <= threshold
    else:
        cand_hit = np.zeros(pts_cand.shape[0], dtype=bool)
        gt_hit = np.zeros(pts_gt.shape[0], dtype=bool)

    precision = float(cand_hit.mean()) if pts_cand.shape[0] else 0.0
    recall = float(gt_hit.mean()) if pts_gt.shape[0] else 0.0
    if pts_cand.shape[0] == 0 and pts_gt.shape[0] == 0:
        f1 = 1.0
        precision = 1.0
        recall = 1.0
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float((2.0 * precision * recall) / (precision + recall))

    _render_overlay(
        gt_verts=gt_mesh.vertices,
        gt_tris=gt_mesh.triangles,
        pts_gt=pts_gt,
        pts_cand=pts_cand,
        gt_hit=gt_hit,
        cand_hit=cand_hit,
        view=view,
        figsize=figsize,
        dpi=dpi,
        output_path=output_path,
        title=title or _default_title(
            candidate_step=candidate_step,
            gt_step=gt_step,
            f1=f1,
            precision=precision,
            recall=recall,
            threshold=threshold,
            gt_diag=gt_diag,
            n_gt=pts_gt.shape[0],
            n_cand=pts_cand.shape[0],
        ),
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "gt_bbox_diagonal": gt_diag,
        "n_points_candidate": int(pts_cand.shape[0]),
        "n_points_gt": int(pts_gt.shape[0]),
        "n_hits_candidate": int(cand_hit.sum()),
        "n_hits_gt": int(gt_hit.sum()),
    }


def _default_title(
    *,
    candidate_step: Path,
    gt_step: Path,
    f1: float,
    precision: float,
    recall: float,
    threshold: float,
    gt_diag: float,
    n_gt: int,
    n_cand: int,
) -> str:
    gt_name = gt_step.parent.name if gt_step.stem == "ground_truth" else gt_step.stem
    cand_name = (
        candidate_step.parent.name
        if candidate_step.stem == "ground_truth"
        else candidate_step.stem
    )
    return (
        f"{gt_name} vs {cand_name}   "
        f"F1={f1:.3f}  precision={precision:.3f}  recall={recall:.3f}\n"
        f"threshold={threshold:.2f} mm  gt_bbox_diag={gt_diag:.2f} mm  "
        f"samples: gt={n_gt}  candidate={n_cand}\n"
        f"green = within threshold; blue = GT miss; red = candidate miss"
    )


def _render_overlay(
    *,
    gt_verts: np.ndarray,
    gt_tris: np.ndarray,
    pts_gt: np.ndarray,
    pts_cand: np.ndarray,
    gt_hit: np.ndarray,
    cand_hit: np.ndarray,
    view: tuple[float, float],
    figsize: tuple[float, float],
    dpi: int,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    azim, elev = view

    lo = gt_verts.min(axis=0)
    hi = gt_verts.max(axis=0)
    centre = 0.5 * (lo + hi)
    extent = float((hi - lo).max())
    half = 0.55 * extent

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=9)

    tri_polys = gt_verts[gt_tris]
    mesh_collection = Poly3DCollection(
        tri_polys,
        facecolor=(0.85, 0.86, 0.90, 0.45),
        edgecolor="none",
        linewidth=0.0,
    )
    ax.add_collection3d(mesh_collection)

    # green = matched, blue = GT-only miss, red = candidate-only miss.
    color_hit = (0.16, 0.63, 0.31, 1.0)
    color_miss_gt = (0.14, 0.41, 0.90, 1.0)
    color_miss_cand = (0.90, 0.27, 0.14, 1.0)

    if pts_gt.shape[0]:
        gt_miss = pts_gt[~gt_hit]
        gt_match = pts_gt[gt_hit]
        if gt_miss.shape[0]:
            ax.scatter(
                gt_miss[:, 0], gt_miss[:, 1], gt_miss[:, 2],
                color=color_miss_gt, s=1.6, depthshade=False,
            )
        if gt_match.shape[0]:
            ax.scatter(
                gt_match[:, 0], gt_match[:, 1], gt_match[:, 2],
                color=color_hit, s=1.6, depthshade=False,
            )
    if pts_cand.shape[0]:
        cand_miss = pts_cand[~cand_hit]
        cand_match = pts_cand[cand_hit]
        if cand_miss.shape[0]:
            ax.scatter(
                cand_miss[:, 0], cand_miss[:, 1], cand_miss[:, 2],
                color=color_miss_cand, s=1.6, depthshade=False,
            )
        if cand_match.shape[0]:
            ax.scatter(
                cand_match[:, 0], cand_match[:, 1], cand_match[:, 2],
                color=color_hit, s=1.6, depthshade=False,
            )

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _empty_result(
    *,
    with_debug: bool,
    tau_sharp_deg: float,
    tau_smooth_deg: float,
    spacing: float,
    n_total_edges: int = 0,
    n_smooth: int = 0,
    n_ambiguous: int = 0,
    n_non_manifold: int = 0,
):
    empty_pts = np.empty((0, 3), dtype=np.float64)
    if not with_debug:
        return empty_pts
    return FeatureEdgeDebug(
        points=empty_pts,
        segments=np.empty((0, 2, 3), dtype=np.float64),
        n_total_edges=n_total_edges,
        n_kept=0,
        n_smooth=n_smooth,
        n_ambiguous=n_ambiguous,
        n_non_manifold=n_non_manifold,
        tau_sharp_deg=tau_sharp_deg,
        tau_smooth_deg=tau_smooth_deg,
        spacing=spacing,
    )
