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

import os
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
    parallel: bool | None = None,
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
        parallel=parallel,
    )


def tessellate_shape(
    wrapped,  # type: ignore[no-untyped-def]
    linear_deflection_mm: float,
    *,
    angular_deflection_rad: float = 0.5,
    parallel: bool | None = None,
) -> Mesh:
    """Tessellate a pre-loaded OCC ``TopoDS_Shape`` into a welded mesh.

    Faces are stitched into one mesh by *topology*, not coordinates.
    OCC discretises every ``TopoDS_Edge`` once, so each face carrying
    that edge stores a polygon of node indices into its own
    triangulation running along the edge. We read those polygons and
    merge corresponding nodes by index (union-find). A periodic seam
    lies on a single face but is stored as a closed representation
    holding two polygons, one per side of the seam; merging those two by
    index stitches the seam the same way. Finally, all nodes that resolve
    to the same BREP vertex are merged, which closes apices where a face
    boundary passes through a point via a degenerate edge (fillet, cone,
    or pole tips) that carries no polygon to stitch along. The result is
    conformal at any deflection whenever the BREP is topologically
    closed, with no coordinate weld tolerance anywhere.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import (
        TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape,
    )

    if linear_deflection_mm <= 0:
        raise ValueError(
            f"linear_deflection_mm must be > 0, got {linear_deflection_mm}",
        )

    if parallel is None:
        parallel = os.environ.get("CADGENBENCH_OCC_PARALLEL_MESH", "1") != "0"

    mesher = BRepMesh_IncrementalMesh()
    mesher.SetShape(wrapped)
    params = mesher.ChangeParameters()
    params.Deflection = float(linear_deflection_mm)
    params.Angle = float(angular_deflection_rad)
    params.Relative = False
    params.InParallel = bool(parallel)
    mesher.Perform()

    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_FACE, faces)
    if faces.Size() == 0:
        raise MeshSanityError("shape has no faces to tessellate")

    # 1. Lay every face's triangulation into one global node/triangle list.
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    face_base: dict[int, int] = {}
    face_tri: dict[int, tuple] = {}
    for fi in range(1, faces.Size() + 1):
        face = TopoDS.Face_s(faces.FindKey(fi))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            raise MeshSanityError(
                f"face #{fi} missing triangulation after meshing "
                f"(deflection={linear_deflection_mm} mm)",
            )
        trsf = loc.Transformation()
        base = len(vertices)
        face_base[fi] = base
        face_tri[fi] = (tri, loc)
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i).Transformed(trsf)
            vertices.append((p.X(), p.Y(), p.Z()))
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for i in range(1, tri.NbTriangles() + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            a, b, c = base + n1 - 1, base + n2 - 1, base + n3 - 1
            if reversed_face:
                a, b = b, a
            triangles.append((a, b, c))

    if not triangles:
        raise MeshSanityError("tessellation produced no triangles")

    # 2. Merge shared-edge nodes by index (union-find). Each face
    # carrying an edge stores a polygon of node indices into its own
    # triangulation running along that edge; the same edge in two faces
    # yields two equal-length lists of corresponding nodes. A periodic
    # seam lies on a single face but is stored as a closed representation
    # carrying two polygons (``PolygonOnTriangulation`` and
    # ``PolygonOnTriangulation2``), one per side of the seam, which we
    # merge the same way.
    parent = list(range(len(vertices)))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    verts = np.asarray(vertices, dtype=np.float64)
    # Each edge runs between two BREP vertices; collect, per vertex, the
    # triangulation nodes that land on it (the endpoints of every edge
    # polygon) so they can be merged in step 3.
    vmap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_VERTEX, vmap)
    vertex_nodes: dict[int, list[int]] = {}
    edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(wrapped, TopAbs_EDGE, TopAbs_FACE, edge_faces)
    for ei in range(1, edge_faces.Extent() + 1):
        edge = TopoDS.Edge_s(edge_faces.FindKey(ei))
        edge_loc = edge.Location()
        edge_curves = tuple(edge.TShape().Curves())
        node_lists: list[list[int]] = []
        seen_tri: set[int] = set()
        for adj_face in edge_faces.FindFromIndex(ei):
            fi = faces.FindIndex(adj_face)
            if fi == 0:
                continue
            tri, loc = face_tri[fi]
            # A seam lists its face twice; its single closed
            # representation already carries both sides, so process each
            # triangulation once.
            if id(tri) in seen_tri:
                continue
            seen_tri.add(id(tri))
            base = face_base[fi]
            pred = loc.Predivided(edge_loc)
            for cr in edge_curves:
                if not cr.IsPolygonOnTriangulation(tri, pred):
                    continue
                p1 = cr.PolygonOnTriangulation()
                if p1 is not None:
                    p1_nodes = np.fromiter(
                        p1.Nodes(),
                        dtype=np.int64,
                        count=p1.NbNodes(),
                    )
                    node_lists.append((p1_nodes + (base - 1)).tolist())
                if cr.IsPolygonOnClosedTriangulation():
                    p2 = cr.PolygonOnTriangulation2()
                    if p2 is not None:
                        p2_nodes = np.fromiter(
                            p2.Nodes(),
                            dtype=np.int64,
                            count=p2.NbNodes(),
                        )
                        node_lists.append((p2_nodes + (base - 1)).tolist())
        # Every array for one edge is indexed along that edge's single
        # curve parametrization, so node k of any array is the same point
        # as node k of every other. Merge by index, never by coordinates.
        # The coordinate checks below only *guard* that assumption: they
        # raise on a length mismatch or opposite storage order rather
        # than choosing a merge, so a silent "closed but twisted" mesh
        # becomes a loud, diagnosable failure.
        if not node_lists:
            continue
        ref = node_lists[0]
        ref_pts = verts[ref]
        for other in node_lists[1:]:
            if len(other) != len(ref):
                raise MeshSanityError(
                    f"edge #{ei}: shared-edge node arrays differ in "
                    f"length ({len(ref)} vs {len(other)}); cannot stitch "
                    "by index",
                )
            other_pts = verts[other]
            fwd = float(np.abs(ref_pts - other_pts).max())
            rev = float(np.abs(ref_pts - other_pts[::-1]).max())
            if rev < fwd:
                raise MeshSanityError(
                    f"edge #{ei}: shared-edge node arrays are stored in "
                    f"opposite order (forward error {fwd:.3e} mm > reverse "
                    f"{rev:.3e} mm); an index stitch would twist the mesh",
                )
            for u, v in zip(ref, other):
                union(u, v)

        # Record this edge's endpoint nodes against its two BREP vertices.
        # A polygon's nodes run along the edge parametrisation, so node 0
        # sits at the edge's first vertex and node -1 at its last.
        v_first = vmap.FindIndex(TopExp.FirstVertex_s(edge))
        v_last = vmap.FindIndex(TopExp.LastVertex_s(edge))
        for arr in node_lists:
            if v_first:
                vertex_nodes.setdefault(v_first, []).append(arr[0])
            if v_last:
                vertex_nodes.setdefault(v_last, []).append(arr[-1])

    # Merge nodes that resolve to the same BREP vertex. Where a face
    # boundary passes through a point via a degenerate edge (a fillet,
    # cone, or pole apex), the face's two boundary nodes there are
    # distinct indices denoting one point, and the edge stitch above
    # never links them because the connecting edge is degenerate and
    # carries no polygon. Unioning by shared vertex closes such apices by
    # topology, with no coordinate tolerance. The coordinate check only
    # *guards* the node-to-vertex assignment, raising rather than choosing.
    for vi, nodes in vertex_nodes.items():
        vp = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vmap.FindKey(vi)))
        vertex_xyz = np.array([vp.X(), vp.Y(), vp.Z()], dtype=np.float64)
        far = np.abs(verts[nodes] - vertex_xyz).max(axis=1)
        if far.size and float(far.max()) > linear_deflection_mm:
            raise MeshSanityError(
                f"vertex #{vi}: an edge endpoint node is "
                f"{float(far.max()):.3e} mm from the vertex (> deflection "
                f"{linear_deflection_mm} mm); edge polygon orientation is "
                "inconsistent with FirstVertex/LastVertex",
            )
        base_node = nodes[0]
        for n in nodes[1:]:
            union(base_node, n)

    roots = np.array([find(i) for i in range(len(vertices))], dtype=np.int64)

    # 3. Relabel to representatives, average merged positions, compact.
    uniq, inv = np.unique(roots, return_inverse=True)
    n_unique = int(uniq.shape[0])
    centroids = np.zeros((n_unique, 3), dtype=np.float64)
    counts = np.zeros(n_unique, dtype=np.int64)
    np.add.at(centroids, inv, verts)
    np.add.at(counts, inv, 1)
    centroids /= np.maximum(counts, 1)[:, None]

    merged_faces = inv[np.asarray(triangles, dtype=np.int64)]
    # Drop triangles that became zero-area after the merge. A closed
    # edge whose closure node one face repeats and the adjacent face
    # splits into two coincident nodes forces those two nodes to unify,
    # which collapses the thin sliver triangle spanning them. The sliver
    # carries no area and its neighbours still cover the region, so
    # dropping it is exact (no tolerance) and closure is reconfirmed by
    # :func:`validate_mesh`.
    keep = (
        (merged_faces[:, 0] != merged_faces[:, 1])
        & (merged_faces[:, 1] != merged_faces[:, 2])
        & (merged_faces[:, 0] != merged_faces[:, 2])
    )
    merged_faces = merged_faces[keep]
    if merged_faces.shape[0] == 0:
        raise MeshSanityError("stitching collapsed all triangles")

    used = np.unique(merged_faces.reshape(-1))
    remap = np.full(n_unique, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    return Mesh(
        vertices=centroids[used],
        triangles=remap[merged_faces].astype(np.int64),
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
