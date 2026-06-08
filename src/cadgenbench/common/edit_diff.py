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

"""Edit-diff visualization for editing-task reports.

A plain shaded render of a near-no-op edit is indistinguishable from a correct
one, so the report shows the candidate as a translucent ghost with a per-vertex
**surface-deviation field** painted on top:

- **red**   = candidate material outside the ground truth (extra / too much),
- **amber** = ground-truth material outside the candidate (missing / too little).

The field mirrors the :mod:`cadgenbench.eval.shape_similarity` surface-distance
F1 gates so the picture shows exactly what the score penalises: a point deviates
when its distance to the *other mesh surface* exceeds the F1 distance gate, or
(on the candidate side, among right-place points) the smooth surface normals
disagree by more than the F1 normal gate. Severity is continuous, so genuine
edits saturate while sub-gate tessellation wobble fades out.

Two implementation details make it faithful:

- **Subdivision.** The renderer colours vertices, not arbitrary points, so each
  mesh is linearly subdivided (no geometry change, no T-junctions) until the
  samples are dense enough to resolve thin features (a removed fillet, a small
  boss). Severity is evaluated per subdivided vertex.
- **Consistent normals.** Both normals at a comparison are interpolated from the
  *original* meshes' area-weighted vertex normals at the same surface point, so a
  subdivided edge vertex never averages across a fold while its match takes a
  single face -- that mismatch would otherwise paint a perfect candidate red.

The field is built once per fixture (:func:`build_edit_diff_shapes`) and reused
for the full turntable and the zoomed turntable (:func:`render_edit_diff_turntables`).
"""
from __future__ import annotations

import numpy as np

from cadgenbench.common.mesh import Mesh

# --- tuning -----------------------------------------------------------------

#: Linear subdivisions of each mesh before evaluating severity (each level
#: splits every triangle into 4). Two levels resolves the thin features in the
#: current fixtures without an unreasonable vertex count.
SUBDIVISIONS: int = 2

#: Laplacian smoothing of the per-vertex severity field (1-ring neighbour blend).
SMOOTH_ITERS: int = 2
SMOOTH_ALPHA: float = 0.5

#: Distance severity ramps from the F1 gate to saturation over this fraction of
#: the gate, so a genuine deviation reads punchy rather than faint.
DIST_SPAN_FRACTION: float = 0.5

#: Normal severity (candidate side) ramps from the F1 normal gate to saturation
#: at this dot product (cos 45deg), i.e. a ~45deg disagreement reads full.
NORMAL_SATURATION_DOT: float = 0.7071067811865476

#: Highlight opacity at full severity (slightly < 1 keeps the look soft).
HIGHLIGHT_ALPHA: float = 0.95

#: Edit-region detection for the *zoom* framing (GT vs input). The region is
#: where the two trusted solids disagree by more than ``edit_tol``, found by
#: barycentric face sampling (a face counts once >= REGION_MIN_HOT of its samples
#: exceed). ``edit_tol = max(floor, factor * deflection)``.
ZOOM_EDIT_TOL_FLOOR_MM: float = 0.30
ZOOM_EDIT_TOL_DEFLECTION_FACTOR: float = 0.8
REGION_MIN_HOT: int = 3

#: Zoom box = the largest connected change cluster (points linked within
#: ``max(LINK_FRACTION * GT_diagonal, LINK_FLOOR_MM)``) plus a margin of
#: ``max(MARGIN_FRACTION * cluster_diagonal, MARGIN_FLOOR_MM)``.
ZOOM_CLUSTER_LINK_FRACTION: float = 0.03
ZOOM_CLUSTER_LINK_FLOOR_MM: float = 5.0
ZOOM_MARGIN_FRACTION: float = 0.35
ZOOM_MARGIN_FLOOR_MM: float = 8.0


# --- per-vertex severity field ---------------------------------------------


def _subdivide(mesh: Mesh, levels: int = SUBDIVISIONS) -> Mesh:
    """Linearly subdivide *mesh* so severity samples land on real vertices.

    Linear subdivision only inserts edge-midpoint vertices on existing faces --
    it does not move geometry (no fold rounding) and stays watertight (shared
    edge vertices, no T-junction cracks).
    """
    from cadgenbench.common.viewer import _mesh_to_polydata  # noqa: PLC0415

    poly = _mesh_to_polydata(mesh).subdivide(levels, "linear")
    tris = poly.faces.reshape(-1, 4)[:, 1:]
    return Mesh(
        vertices=np.ascontiguousarray(poly.points, dtype=np.float64),
        triangles=np.ascontiguousarray(tris, dtype=np.int64),
        linear_deflection_mm=mesh.linear_deflection_mm,
    )


def _smooth_scalar(
    mesh: Mesh, scalar: np.ndarray, *, iters: int = SMOOTH_ITERS, alpha: float = SMOOTH_ALPHA,
) -> np.ndarray:
    """Blend a per-vertex scalar toward its 1-ring neighbour mean a few times.

    An isolated trip collapses toward its (zero) neighbours while a coherent
    band survives -- de-speckling via smoothing rather than a component filter.
    """
    from scipy.sparse import coo_matrix  # noqa: PLC0415

    tris = mesh.triangles
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    n = mesh.n_vertices
    adj = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n),
    ).tocsr()
    deg = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
    out = scalar.astype(np.float64, copy=True)
    for _ in range(iters):
        out = (1.0 - alpha) * out + alpha * ((adj @ out) / deg)
    return out


def _severity(
    src: Mesh, ref: Mesh, src_original: Mesh, *, f1_tol: float, normal: bool,
) -> np.ndarray:
    """Per-vertex deviation severity of *src* against the *ref* surface, ``[0, 1]``.

    Directional: only *src* material lying OUTSIDE *ref* contributes, so the
    candidate carries "extra" and the GT carries "missing" -- colours never flip.
    Distance ramps from ``f1_tol``; on the candidate side an additional normal
    term flags orientation edits among right-place points (``|distance| <=
    f1_tol``). Normals come from *src_original* / *ref* originals at the foot
    point for fold-stable, perfect-match-exact comparison.
    """
    from cadgenbench.common.viewer import _signed_distance  # noqa: PLC0415

    signed = _signed_distance(src.vertices, ref)
    outside = np.clip(signed, 0.0, None)
    span = max(DIST_SPAN_FRACTION * f1_tol, 1e-6)
    sev = np.clip((outside - f1_tol) / span, 0.0, 1.0)
    if not normal:
        return sev

    from cadgenbench.eval.sampling import closest_point_distances_and_normals  # noqa: PLC0415
    from cadgenbench.eval.shape_similarity import (  # noqa: PLC0415
        SURFACE_DISTANCE_F1_NORMAL_DOT_THRESHOLD as NDOT,
    )

    _, ref_normal = closest_point_distances_and_normals(
        src.vertices, ref.vertices, ref.triangles)
    _, src_normal = closest_point_distances_and_normals(
        src.vertices, src_original.vertices, src_original.triangles)
    dot = np.abs(np.einsum("ij,ij->i", src_normal, ref_normal))
    right_place = np.abs(signed) <= f1_tol
    nspan = max(NDOT - NORMAL_SATURATION_DOT, 1e-6)
    nsev = np.clip((NDOT - dot) / nspan, 0.0, 1.0) * right_place
    return np.maximum(sev, nsev)


def _field_polydata(mesh: Mesh, severity: np.ndarray, color, *, ghost: bool):
    """PolyData carrying a per-vertex ``"rgba"`` field for :mod:`viewer`'s renderer.

    ``ghost=True`` (candidate): translucent grey body blending to *color* with
    severity. ``ghost=False`` (GT): transparent base, only the misses fade in as
    *color*, so the recall half adds no second opaque body.
    """
    from cadgenbench.common.viewer import (  # noqa: PLC0415
        DIFF_GHOST_ALPHA,
        DIFF_GHOST_RGB,
        _mesh_to_polydata,
    )

    poly = _mesh_to_polydata(mesh)
    t = np.clip(severity, 0.0, 1.0)[:, None]
    color = np.asarray(color, dtype=np.float64)
    if ghost:
        rgb = (1.0 - t) * np.array(DIFF_GHOST_RGB) + t * color
        alpha = (1.0 - t) * DIFF_GHOST_ALPHA + t * HIGHLIGHT_ALPHA
    else:
        rgb = np.tile(color, (len(severity), 1))
        alpha = t[:, 0] * HIGHLIGHT_ALPHA
    rgba = np.concatenate([rgb, alpha.reshape(-1, 1)], axis=1)
    poly.point_data["rgba"] = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    return poly


def build_edit_diff_shapes(gt_mesh: Mesh, candidate_mesh: Mesh) -> list:
    """Build the two per-vertex rgba field shapes (computed once per fixture).

    Returns a ``shapes`` list in the ``(polydata, rgb, alpha)`` form
    :func:`cadgenbench.common.viewer._render_views` /
    ``_render_turntable_frames`` consume; the per-vertex ``"rgba"`` arrays drive
    the colour, the tuple rgb/alpha are placeholders. Reuse the same list for the
    full and the zoomed render so subdivision + severity happen only once.
    """
    from cadgenbench.common.viewer import (  # noqa: PLC0415
        DIFF_EXTRA_RGB,
        DIFF_GHOST_RGB,
        DIFF_MISSING_RGB,
        DIFF_TOL_FLOOR_MM,
        DIFF_TOL_FRACTION,
    )

    lo = gt_mesh.vertices.min(axis=0)
    hi = gt_mesh.vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    f1_tol = max(DIFF_TOL_FLOOR_MM, DIFF_TOL_FRACTION * diag)

    cand_sub = _subdivide(candidate_mesh)
    gt_sub = _subdivide(gt_mesh)
    red = _smooth_scalar(cand_sub, _severity(
        cand_sub, gt_mesh, candidate_mesh, f1_tol=f1_tol, normal=True))
    amber = _smooth_scalar(gt_sub, _severity(
        gt_sub, candidate_mesh, gt_mesh, f1_tol=f1_tol, normal=False))
    return [
        (_field_polydata(cand_sub, red, DIFF_EXTRA_RGB, ghost=True), DIFF_GHOST_RGB, 1.0),
        (_field_polydata(gt_sub, amber, DIFF_MISSING_RGB, ghost=False), DIFF_GHOST_RGB, 1.0),
    ]


# --- zoom framing (candidate-independent: GT vs input) ----------------------

# Barycentric face-sample weights (centroid, edge midpoints, two-thirds points):
# sampling face interiors -- not just shared seam vertices -- is what catches a
# sub-tessellation edit when locating the intended-edit region.
_BARY_WEIGHTS = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
    [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3],
])


def _changed_face_points(
    mesh: Mesh, target: Mesh, tol: float, *, min_hot: int = REGION_MIN_HOT,
) -> np.ndarray:
    """Barycentric sample points of *mesh* faces that differ from *target*.

    A face contributes its exceeding samples when at least ``min_hot`` of its
    barycentric samples sit more than ``tol`` from *target* (either side).
    """
    from cadgenbench.common.viewer import _signed_distance  # noqa: PLC0415

    verts = mesh.vertices[mesh.triangles]                       # (T, 3, 3)
    pts = np.einsum("kj,tjc->tkc", _BARY_WEIGHTS, verts)        # (T, k, 3)
    signed = _signed_distance(pts.reshape(-1, 3), target).reshape(len(mesh.triangles), -1)
    hot = np.abs(signed) > tol                                  # (T, k)
    keep = hot.sum(axis=1) >= min_hot                           # (T,)
    return pts[hot & keep[:, None]]                             # (N, 3)


def _largest_cluster(points: np.ndarray, link: float) -> np.ndarray:
    """Points of the largest spatially-connected cluster (within *link*)."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n = len(points)
    if n <= 1:
        return points
    pairs = cKDTree(points).query_pairs(link, output_type="ndarray")
    if len(pairs) == 0:
        return points
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n),
    )
    _, labels = connected_components(graph, directed=False)
    return points[labels == np.bincount(labels).argmax()]


def edit_region_zoom_bbox(
    gt_mesh: Mesh, input_mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounding box framing the intended edit (GT vs input) for the zoom clip.

    Candidate-independent and stable: the edit is exactly where GT and the
    original input disagree. Frames the largest connected change cluster with a
    margin. Returns ``None`` when no change is found, so the caller can omit the
    zoom clip rather than render a meaningless one.
    """
    if gt_mesh.n_triangles == 0 or input_mesh.n_triangles == 0:
        return None
    edit_tol = max(
        ZOOM_EDIT_TOL_FLOOR_MM,
        ZOOM_EDIT_TOL_DEFLECTION_FACTOR * float(gt_mesh.linear_deflection_mm),
    )
    changed = [
        p for p in (
            _changed_face_points(gt_mesh, input_mesh, edit_tol),
            _changed_face_points(input_mesh, gt_mesh, edit_tol),
        ) if p.shape[0]
    ]
    if not changed:
        return None
    pts = np.vstack(changed)
    diag = float(np.linalg.norm(gt_mesh.vertices.max(0) - gt_mesh.vertices.min(0)))
    link = max(ZOOM_CLUSTER_LINK_FRACTION * diag, ZOOM_CLUSTER_LINK_FLOOR_MM)
    cluster = _largest_cluster(pts, link)
    lo, hi = cluster.min(axis=0), cluster.max(axis=0)
    margin = max(float(np.linalg.norm(hi - lo)) * ZOOM_MARGIN_FRACTION, ZOOM_MARGIN_FLOOR_MM)
    return lo - margin, hi + margin


# --- rendering --------------------------------------------------------------


def render_edit_diff_turntables(
    gt_mesh: Mesh,
    candidate_mesh: Mesh,
    *,
    input_mesh: Mesh | None = None,
    frames: int = 120,
    width: int = 512,
    height: int = 384,
    duration_ms: int = 150,
    quality: int = 68,
) -> tuple[bytes, bytes | None]:
    """Render the edit-diff severity field as a full and a zoomed turntable WebP.

    Builds the field once and orbits it twice. The zoom is framed on the intended
    edit region (GT vs *input_mesh*) when an input is supplied; without it, or
    when no change is found, only the full clip is returned (zoom ``None``).

    Returns ``(full_webp, zoom_webp_or_None)``.
    """
    if candidate_mesh.n_triangles == 0:
        raise ValueError("render_edit_diff_turntables: candidate mesh has zero triangles")
    if frames < 2:
        raise ValueError("render_edit_diff_turntables: frames must be >= 2")

    from cadgenbench.common.viewer import (  # noqa: PLC0415
        _diff_bbox,
        _encode_webp,
        _render_turntable_frames,
    )

    shapes = build_edit_diff_shapes(gt_mesh, candidate_mesh)

    def _webp(bbox_min: np.ndarray, bbox_max: np.ndarray) -> bytes:
        frames_png = _render_turntable_frames(
            shapes=shapes, bbox_min=bbox_min, bbox_max=bbox_max,
            frames=frames, width=width, height=height,
        )
        return _encode_webp(frames_png, duration_ms=duration_ms, quality=quality)

    full_min, full_max = _diff_bbox(gt_mesh, candidate_mesh)
    full_webp = _webp(full_min, full_max)

    zoom_webp = None
    if input_mesh is not None:
        zoom_bbox = edit_region_zoom_bbox(gt_mesh, input_mesh)
        if zoom_bbox is not None:
            zoom_webp = _webp(zoom_bbox[0], zoom_bbox[1])
    return full_webp, zoom_webp
