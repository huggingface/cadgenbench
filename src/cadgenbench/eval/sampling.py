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

"""Sample point clouds from STEP/BREP surfaces via OCC tessellation.

Shared utility used by alignment, metrics (Chamfer/Hausdorff), and
any future code that needs a discrete representation of a BREP shape.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def sample_surface_points(
    step_path: str | Path,
    n_points: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample approximately *n_points* from a STEP shape's tessellated surface.

    Points are drawn uniformly w.r.t. surface area (area-weighted triangle
    sampling followed by uniform barycentric coordinates within each chosen
    triangle).

    Args:
        step_path: Path to a ``.step`` / ``.stp`` file.
        n_points: Desired number of sample points.
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(N, 3)`` float64 array of surface points.

    Raises:
        FileNotFoundError: If *step_path* does not exist.
        RuntimeError: If the file cannot be loaded or tessellated.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    wrapped = _load_occ_shape(step_path)
    verts, tris = _tessellate(wrapped)
    return _area_weighted_sample(verts, tris, n_points, seed)


def sample_surface_points_from_shape(
    shape,  # OCC TopoDS_Shape
    n_points: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Like :func:`sample_surface_points` but accepts a pre-loaded OCC shape."""
    verts, tris = _tessellate(shape)
    return _area_weighted_sample(verts, tris, n_points, seed)


def sample_surface_points_with_normals(
    step_path: str | Path,
    n_points: int = 10_000,
    seed: int | None = None,
    *,
    linear_deflection_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface points + their outward unit normals from a STEP file.

    Goes through :func:`cadgenbench.common.mesh.tessellate_and_validate`
    so each triangle's winding is consistent with the outward normal of
    its originating BREP face (REVERSED faces are flipped during
    tessellation). Each sample inherits the unit normal of the triangle
    it was drawn from, so per-point normals are reliable for downstream
    normal-weighted matching.

    Args:
        step_path: Path to a ``.step`` / ``.stp`` file.
        n_points: Desired number of sample points.
        seed: Optional RNG seed for reproducibility.
        linear_deflection_mm: Tessellation chord error. When ``None``,
            derived from the part's own bounding-box diagonal via
            :func:`cadgenbench.common.mesh.deflection_for_bbox`.

    Returns:
        ``(points, normals)``: both ``(N, 3)`` float64 arrays. Normals
        are unit length (degenerate triangles with zero area never enter
        the area-weighted draw).
    """
    from cadgenbench.common.measurements import measure_step
    from cadgenbench.common.mesh import (
        deflection_for_bbox,
        tessellate_and_validate,
    )

    step_path = Path(step_path)
    if linear_deflection_mm is None:
        linear_deflection_mm = deflection_for_bbox(
            measure_step(step_path).bounding_box.diagonal,
        )
    mesh = tessellate_and_validate(step_path, linear_deflection_mm)
    return sample_points_and_normals_from_mesh(
        mesh.vertices, mesh.triangles, n_points, seed,
    )


def sample_points_and_normals_from_mesh(
    verts: np.ndarray,
    tris: np.ndarray,
    n_points: int = 10_000,
    seed: int | None = None,
    *,
    smooth_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted point sample of a pre-tessellated mesh with per-point normals.

    The mesh is assumed to be orientation-consistent (the gate enforced
    by :mod:`cadgenbench.common.mesh`), so the cross product of triangle
    edges points outward.

    ``smooth_normals``: when True each point's normal is the barycentric blend
    of **area-weighted vertex normals** instead of the flat facet normal. Flat
    facet normals are discontinuous in the triangulation (two valid meshings of
    a curved surface tilt their facets differently at the same point), so they
    make normal-gated metrics tessellation-sensitive; the smooth normal tracks
    the true surface and is largely mesh-independent.

    Returns:
        ``(points, normals)``: both ``(N, 3)`` float64 arrays.
    """
    return _area_weighted_sample(
        verts, tris, n_points, seed, with_normals=True, smooth_normals=smooth_normals,
    )


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals for an orientation-consistent mesh.

    Identical estimator to the smooth-normal path of :func:`_area_weighted_sample`
    (accumulate each triangle's ``cross = 2*area * unit_facet_normal`` onto its
    vertices, then normalise), so a sample's smooth normal and the interpolated
    vertex normal at that same surface point agree exactly. ``(V, 3)`` float64.
    """
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros((verts.shape[0], 3), dtype=np.float64)
    np.add.at(vn, tris[:, 0], cross)
    np.add.at(vn, tris[:, 1], cross)
    np.add.at(vn, tris[:, 2], cross)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def closest_point_distances_and_normals(
    query_points: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-SURFACE distance + smooth normal at the foot point, per query.

    For each query point, find the closest point on the *mesh surface* (not on a
    discrete sample of it) via Open3D's ``RaycastingScene``, and the smooth
    (area-weighted, barycentric-interpolated) unit normal of the surface there.

    This is the correct primitive for shape comparison: sampling one mesh and
    matching to the *nearest sample* of the other injects two errors that vanish
    here -- the discretisation gap (nearest sample != closest surface point) and,
    at sharp edges, the matched sample landing on a different facet. On a perfect
    match the foot point IS the query and its normal equals the query's own
    normal, so distance is 0 and ``|dot|`` is 1 -- exact, artifact-free.

    Returns ``(dist, normals)``: ``(N,)`` distances and ``(N, 3)`` unit normals.
    """
    import open3d as o3d  # noqa: PLC0415

    verts = np.ascontiguousarray(verts, dtype=np.float64)
    tris = np.ascontiguousarray(tris)
    q = np.ascontiguousarray(query_points, dtype=np.float64)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(verts.astype(np.float32)),
            o3d.core.Tensor(tris.astype(np.int32)),
        )
    )
    res = scene.compute_closest_points(
        o3d.core.Tensor(q.astype(np.float32)),
    )
    foot = res["points"].numpy().astype(np.float64)
    prim = res["primitive_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(q - foot, axis=1)

    # Barycentric weights of the foot point in its triangle, recomputed from
    # geometry (independent of Open3D's uv ordering) for robustness.
    vn = vertex_normals(verts, tris)
    tri_v = tris[prim]
    a = verts[tri_v[:, 0]]
    b = verts[tri_v[:, 1]]
    c = verts[tri_v[:, 2]]
    e0 = b - a
    e1 = c - a
    e2 = foot - a
    d00 = np.einsum("ij,ij->i", e0, e0)
    d01 = np.einsum("ij,ij->i", e0, e1)
    d11 = np.einsum("ij,ij->i", e1, e1)
    d20 = np.einsum("ij,ij->i", e2, e0)
    d21 = np.einsum("ij,ij->i", e2, e1)
    denom = d00 * d11 - d01 * d01
    safe = np.abs(denom) > 1e-20
    wv = np.where(safe, (d11 * d20 - d01 * d21) / np.where(safe, denom, 1.0), 0.0)
    ww = np.where(safe, (d00 * d21 - d01 * d20) / np.where(safe, denom, 1.0), 0.0)
    wu = 1.0 - wv - ww
    normals = (
        wu[:, None] * vn[tri_v[:, 0]]
        + wv[:, None] * vn[tri_v[:, 1]]
        + ww[:, None] * vn[tri_v[:, 2]]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return dist, normals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def solids_only(wrapped):
    """Return an OCC compound containing only TopAbs_SOLID subshapes.

    Filters out non-solid topology (PMI annotations, wireframes, etc.) so
    that bounding boxes and point-cloud samples reflect only part geometry.
    Falls back to *wrapped* unchanged when no solids are found.
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    found = False
    explorer = TopExp_Explorer(wrapped, TopAbs_SOLID)
    while explorer.More():
        builder.Add(compound, explorer.Value())
        found = True
        explorer.Next()

    return compound if found else wrapped


def _load_occ_shape(step_path: Path):
    """Load a STEP file and return the raw OCC ``TopoDS_Shape``.

    Returns only solid subshapes so that PMI annotation geometry does not
    pollute point clouds used for alignment and metrics.
    """
    from build123d import import_step

    try:
        shape = import_step(str(step_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load STEP file: {step_path}") from exc

    if shape is None or not shape.wrapped:
        raise RuntimeError(f"STEP file produced no geometry: {step_path}")

    return solids_only(shape.wrapped)


def _tessellate(
    shape,
    linear_deflection: float = 0.1,
    angular_deflection: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate an OCC shape and return (vertices, triangles).

    Returns:
        verts: ``(V, 3)`` float64, unique vertex positions.
        tris:  ``(T, 3)`` int32 , triangle indices into *verts*.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    from cadgenbench.common.profiling import phase  # noqa: PLC0415

    # Separate meshing timer for the alignment point-cloud path (distinct OCC
    # tessellation from cadgenbench.common.mesh.tessellate_shape).
    with phase(f"mesh.align d={linear_deflection:.4g}"):
        BRepMesh_IncrementalMesh(
            shape, linear_deflection, False, angular_deflection, True,
        )

    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Value())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            explorer.Next()
            continue

        trsf = loc.Transformation()
        n_nodes = tri.NbNodes()
        n_triangles = tri.NbTriangles()

        face_verts = np.empty((n_nodes, 3), dtype=np.float64)
        for i in range(1, n_nodes + 1):
            p = tri.Node(i).Transformed(trsf)
            face_verts[i - 1] = (p.X(), p.Y(), p.Z())

        face_tris = np.empty((n_triangles, 3), dtype=np.int32)
        for i in range(1, n_triangles + 1):
            t = tri.Triangle(i)
            i1, i2, i3 = t.Get()
            face_tris[i - 1] = (i1 - 1 + offset, i2 - 1 + offset, i3 - 1 + offset)

        all_verts.append(face_verts)
        all_tris.append(face_tris)
        offset += n_nodes
        explorer.Next()

    if not all_verts:
        raise RuntimeError("Tessellation produced no triangles")

    return np.vstack(all_verts), np.vstack(all_tris)


def _area_weighted_sample(
    verts: np.ndarray,
    tris: np.ndarray,
    n_points: int,
    seed: int | None,
    *,
    with_normals: bool = False,
    smooth_normals: bool = False,
):
    """Uniformly sample points on a triangle mesh by area.

    With ``with_normals=False`` returns ``points`` only (``(N, 3)``).
    With ``with_normals=True`` returns ``(points, normals)``. By default each
    normal is the unit triangle (flat) normal of the source triangle; with
    ``smooth_normals=True`` it is the barycentric blend of area-weighted vertex
    normals (continuous in the triangulation). The caller is responsible for
    feeding in an orientation-consistent mesh if outward-pointing normals matter.
    """
    rng = np.random.default_rng(seed)

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    twice_areas = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_areas
    total_area = areas.sum()
    if total_area <= 0:
        raise RuntimeError("Mesh has zero total surface area")

    probs = areas / total_area
    chosen = rng.choice(len(tris), size=n_points, p=probs)

    # Random barycentric coordinates (uniform in triangle)
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    points = (
        u[:, None] * v0[chosen]
        + v[:, None] * v1[chosen]
        + w[:, None] * v2[chosen]
    )
    if not with_normals:
        return points

    if smooth_normals:
        # Area-weighted vertex normals (cross = 2*area * unit facet normal, so
        # accumulating it per vertex weights by area), then barycentric-blend
        # them at each sample. Continuous in the triangulation.
        vertex_normals = np.zeros_like(verts)
        np.add.at(vertex_normals, tris[:, 0], cross)
        np.add.at(vertex_normals, tris[:, 1], cross)
        np.add.at(vertex_normals, tris[:, 2], cross)
        vn = vertex_normals / np.maximum(
            np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12,
        )
        t = tris[chosen]
        normals = u[:, None] * vn[t[:, 0]] + v[:, None] * vn[t[:, 1]] + w[:, None] * vn[t[:, 2]]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        return points, normals

    # Unit triangle normals; the area filter above already excluded any
    # zero-area triangles from the draw, so the denominator is positive.
    tri_normals = cross / twice_areas[:, None]
    normals = tri_normals[chosen]
    return points, normals
