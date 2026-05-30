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

"""Tessellate STEP files into clean closed manifold triangle meshes.

This module is the foundation under
:mod:`cadgenbench.eval.topo_match`: it produces a triangle mesh
whose Euler characteristic is the genuine topological :math:`\\chi` of the
boundary surface, which the topology metric relies on.

Pipeline:

1. **Tessellate the BREP**: OCC's :class:`BRepMesh_IncrementalMesh`
   produces a per-face triangulation. The deflection that controls
   chord error is chosen by the *caller* (typically via
   :func:`deflection_for_bbox` applied to the GT) and used uniformly
   for GT and candidate within a comparison.

2. **Collect a raw triangle soup** by walking every face. Each face's
   nodes are emitted as fresh global vertices, so vertices on shared
   edges appear N times (once per adjacent face). Triangle winding is
   flipped on REVERSED faces so the soup is wound consistently
   outward.

3. **Weld coincident vertices** using
   :func:`trimesh.Trimesh(..., merge_norm=tol)`. The tolerance is
   anchored to the OCC tessellation deflection so we never weld two
   genuinely-distinct vertices on a feature smaller than the mesh can
   resolve.

4. **Validate** via :func:`validate_mesh`, which raises
   :class:`MeshSanityError` if the welded mesh is not a closed
   orientation-consistent manifold:

   - ``is_watertight``: every edge incident to exactly 2 triangles.
   - ``is_winding_consistent``: shared edges traversed in opposite
     orientations by their two incident triangles.

Any failure surfaces to ``validation.topology_errors`` and cascades
``is_valid = False`` → ``cad_score = 0``. The module never silently
downgrades a check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class MeshSanityError(RuntimeError):
    """Raised when a tessellated mesh fails one of the topology gates."""


@dataclass(frozen=True)
class Mesh:
    """A welded, validated triangle mesh suitable for topological work.

    Attributes:
        vertices: ``(Nv, 3)`` float64 array of vertex positions in the
            shape's coordinate frame (units: mm).
        triangles: ``(Nt, 3)`` int64 array of vertex indices. Each row
            ``(a, b, c)`` is wound consistently with the outward normal
            of the originating face (REVERSED faces have winding
            flipped during tessellation).
        linear_deflection_mm: chord-error deflection used by the
            tessellator. Same value applied to GT and candidate within
            a comparison.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    linear_deflection_mm: float

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])


# ---------------------------------------------------------------------------
# Deflection policy
# ---------------------------------------------------------------------------

DEFLECTION_RELATIVE = 0.001
DEFLECTION_MIN_MM = 0.005
DEFLECTION_MAX_MM = 0.5


def deflection_for_bbox(bbox_diagonal_mm: float) -> float:
    """Choose a tessellation deflection from a bounding-box diagonal.

    Relative to part scale (``DEFLECTION_RELATIVE × diagonal``), clamped
    to ``[DEFLECTION_MIN_MM, DEFLECTION_MAX_MM]``. The caller is
    expected to use the **GT** bounding-box diagonal for both GT and
    candidate within a comparison so the two meshes are produced at the
    same scale.
    """
    return float(
        min(
            DEFLECTION_MAX_MM,
            max(DEFLECTION_MIN_MM, DEFLECTION_RELATIVE * bbox_diagonal_mm),
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tessellate_step(
    step_path: str | Path,
    linear_deflection_mm: float,
    *,
    angular_deflection_rad: float = 0.5,
) -> Mesh:
    """Tessellate a STEP file into a welded :class:`Mesh`.

    Welding tolerance is anchored to *linear_deflection_mm* so vertices
    on a feature smaller than the chord error never get merged across
    distinct topology.

    Raises:
        FileNotFoundError: STEP file missing.
        RuntimeError: STEP file unreadable, mesher failure, or any
            face missing a triangulation after meshing.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    from build123d import import_step

    shape = import_step(str(step_path))
    if shape is None or not shape.wrapped:
        raise RuntimeError(f"STEP file produced no geometry: {step_path}")
    return tessellate_shape(
        shape.wrapped,
        linear_deflection_mm,
        angular_deflection_rad=angular_deflection_rad,
    )


def tessellate_shape(
    wrapped,  # type: ignore[no-untyped-def]
    linear_deflection_mm: float,
    *,
    angular_deflection_rad: float = 0.5,
) -> Mesh:
    """Tessellate a pre-loaded OCC ``TopoDS_Shape``."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    if linear_deflection_mm <= 0:
        raise ValueError(
            f"linear_deflection_mm must be > 0, got {linear_deflection_mm}",
        )

    BRepMesh_IncrementalMesh(
        wrapped,
        linear_deflection_mm,
        False,
        angular_deflection_rad,
        False,
    )

    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_FACE, face_map)
    if face_map.Size() == 0:
        raise MeshSanityError("shape has no faces to tessellate")

    all_vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    for fi in range(1, face_map.Size() + 1):
        face = TopoDS.Face_s(face_map.FindKey(fi))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            raise MeshSanityError(
                f"face #{fi} missing triangulation after meshing "
                f"(deflection={linear_deflection_mm} mm)",
            )
        trsf = loc.Transformation()
        reversed_face = face.Orientation() == TopAbs_REVERSED

        base = len(all_vertices)
        n_nodes = tri.NbNodes()
        for i in range(1, n_nodes + 1):
            p = tri.Node(i).Transformed(trsf)
            all_vertices.append((p.X(), p.Y(), p.Z()))

        n_tris = tri.NbTriangles()
        for i in range(1, n_tris + 1):
            t = tri.Triangle(i)
            n1, n2, n3 = t.Get()
            a = base + (n1 - 1)
            b = base + (n2 - 1)
            c = base + (n3 - 1)
            if reversed_face:
                a, b = b, a
            triangles.append((a, b, c))

    if not triangles:
        raise MeshSanityError("tessellation produced no triangles")

    # Weld coincident vertices. Tolerance is small (0.1 % of deflection)
    # so we only weld nodes that are numerically the same point, e.g.
    # vertices on a shared edge as sampled by two adjacent faces. We
    # never collapse genuine short edges that are below the chord-error
    # threshold.
    raw_vertices = np.asarray(all_vertices, dtype=np.float64)
    raw_faces = np.asarray(triangles, dtype=np.int64)
    merge_tol = max(1e-9, 1e-3 * linear_deflection_mm)
    snapped = np.round(raw_vertices / merge_tol).astype(np.int64)
    _, inv = np.unique(snapped, axis=0, return_inverse=True)
    welded_faces = inv[raw_faces]
    # Average positions across welded duplicates (numerical noise only).
    n_unique = int(_.shape[0])
    centroids = np.zeros((n_unique, 3), dtype=np.float64)
    counts = np.zeros(n_unique, dtype=np.int64)
    np.add.at(centroids, inv, raw_vertices)
    np.add.at(counts, inv, 1)
    centroids /= np.maximum(counts, 1)[:, None]

    # Drop degenerate triangles. These arise on **periodic surface
    # seams** (sphere poles, cylinder/torus seams crossing a topological
    # vertex), where OCC's parametric tessellation lays down a sliver
    # triangle whose two corners weld to the same global point. They
    # carry no surface area and their non-degenerate neighbours still
    # cover the local region, dropping is the standard fix and is
    # safe iff the surviving mesh passes :func:`validate_mesh`.
    keep_mask = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    welded_faces = welded_faces[keep_mask]
    if welded_faces.shape[0] == 0:
        raise MeshSanityError("welding collapsed all triangles")

    # Compact vertex list to only referenced indices.
    used = np.unique(welded_faces.reshape(-1))
    remap = np.full(n_unique, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    compact_vertices = centroids[used]
    compact_faces = remap[welded_faces]

    return Mesh(
        vertices=compact_vertices,
        triangles=compact_faces.astype(np.int64),
        linear_deflection_mm=float(linear_deflection_mm),
    )


def validate_mesh(mesh: Mesh) -> None:
    """Verify *mesh* is a closed orientable manifold.

    Three checks, applied in order; the first to fail raises:

    1. **Manifold**, every undirected edge appears in ≤ 2 triangles.
    2. **Closed**, every undirected edge appears in *exactly* 2
       triangles (equivalently :math:`3F = 2E`).
    3. **Orientation-consistent**, for every shared edge ``(a, b)``,
       the two incident triangles list it in opposite orders.

    Implemented on top of :class:`trimesh.Trimesh` checks so we benefit
    from trimesh's robust edge-counting.

    Raises:
        MeshSanityError: With a human-readable description that callers
            forward into ``validation.topology_errors``.
    """
    import trimesh

    if mesh.n_triangles == 0:
        raise MeshSanityError("empty mesh (0 triangles)")

    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        process=False,
        validate=False,
    )

    edges = tm.edges_sorted
    # Count occurrences per undirected edge using stable hashing.
    n_v = int(mesh.n_vertices)
    keys = edges[:, 0].astype(np.int64) * (n_v + 1) + edges[:, 1]
    _u, counts = np.unique(keys, return_counts=True)

    max_count = int(counts.max())
    if max_count > 2:
        bad_idx = int(np.argmax(counts))
        bad_key = int(_u[bad_idx])
        a = bad_key // (n_v + 1)
        b = bad_key % (n_v + 1)
        raise MeshSanityError(
            f"mesh non-manifold: edge ({a}, {b}) shared by "
            f"{max_count} triangles (expected ≤ 2)",
        )

    expected_2e = 3 * mesh.n_triangles
    n_e = int(len(_u))
    if 2 * n_e != expected_2e:
        # Either edges with count=1 (open mesh) or some other gap.
        n_open = int((counts == 1).sum())
        raise MeshSanityError(
            f"mesh not closed: {n_open} open edges, "
            f"3F − 2E = {expected_2e - 2 * n_e} (expected 0; "
            f"F={mesh.n_triangles}, E={n_e})",
        )

    if not tm.is_winding_consistent:
        raise MeshSanityError(
            "mesh orientation inconsistent: at least one shared edge "
            "is traversed in the same direction by both incident "
            "triangles",
        )


def tessellate_and_validate(
    step_path: str | Path,
    linear_deflection_mm: float,
    *,
    angular_deflection_rad: float = 0.5,
) -> Mesh:
    """One-shot: tessellate + validate. Raises :class:`MeshSanityError`."""
    mesh = tessellate_step(
        step_path,
        linear_deflection_mm,
        angular_deflection_rad=angular_deflection_rad,
    )
    validate_mesh(mesh)
    return mesh
