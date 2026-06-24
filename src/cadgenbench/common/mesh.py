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
   - vertex-manifold: no vertex where two surface sheets meet at a
     single point (a "pinch"), which the edge-only checks above cannot
     see yet silently corrupts the Euler characteristic.

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
    _edge_debug: dict | None = None,
    _skip_vertex_merge: bool = False,
    _skip_seam_merge: bool = False,
    _cancel_flaps: bool = True,
    _relative: bool = False,
    _trace: dict | None = None,
) -> Mesh:
    """Tessellate a pre-loaded OCC ``TopoDS_Shape`` into a welded mesh.

    ``_edge_debug`` is a diagnostics-only escape hatch (default ``None`` =
    normal behaviour, unchanged). When a dict is passed, the shared-edge
    stitching does **not** raise on a length / order mismatch; instead it
    tallies every shared edge into ``missing`` / ``difflen`` /
    ``opposite_order`` / ``samelen_far`` / ``samelen_ok`` so a failing part can
    be surveyed in one pass. The returned mesh is meaningless in this mode.

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

    Finally, opposite-winding duplicate triangles (a degenerate face region
    folded onto a triangle and its mirror by the merge) are cancelled in pairs
    (``_cancel_flaps``, on by default): the pair carries no net surface, so
    removing both is topology-preserving and closes the otherwise-non-manifold
    fold. Pass ``_cancel_flaps=False`` to disable for diagnostics.
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
    params.Relative = bool(_relative)
    params.InParallel = bool(parallel)
    # The OCC tessellation is the native call that the killable mesh
    # subprocess + MESH_TIMEOUT_S exist to bound, so it is the prime
    # "meshing by itself" measurement. Timed separately from the Python
    # welding/validation that follows.
    from cadgenbench.common.profiling import phase  # noqa: PLC0415

    with phase(f"mesh.perform d={linear_deflection_mm:.4g}"):
        mesher.Perform()

    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_FACE, faces)
    if faces.Size() == 0:
        raise MeshSanityError("shape has no faces to tessellate")

    # 1. Lay every face's triangulation into one global node/triangle list.
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    tri_face: list[int] = []  # diagnostics: source face index per soup triangle
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
        if _trace is not None:
            _trace.setdefault("face_nnodes", {})[fi] = int(tri.NbNodes())
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
            if _trace is not None:
                tri_face.append(fi)

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
        node_src: list[int] = []  # diagnostics: face index per node_lists entry
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
                    if _trace is not None:
                        node_src.append(fi)
                if cr.IsPolygonOnClosedTriangulation() and not _skip_seam_merge:
                    p2 = cr.PolygonOnTriangulation2()
                    if p2 is not None:
                        if _trace is not None:
                            _trace.setdefault("seam_p2_faces", set()).add(fi)
                        p2_nodes = np.fromiter(
                            p2.Nodes(),
                            dtype=np.int64,
                            count=p2.NbNodes(),
                        )
                        node_lists.append((p2_nodes + (base - 1)).tolist())
                        if _trace is not None:
                            node_src.append(fi)
        # Every array for one edge is indexed along that edge's single
        # curve parametrization, so node k of any array is the same point
        # as node k of every other. Merge by index, never by coordinates.
        # The coordinate checks below only *guard* that assumption: they
        # raise on a length mismatch or opposite storage order rather
        # than choosing a merge, so a silent "closed but twisted" mesh
        # becomes a loud, diagnosable failure.
        if _edge_debug is not None and edge_faces.FindFromIndex(ei).Extent() >= 2 \
                and len(node_lists) < 2:
            _edge_debug["missing"] = _edge_debug.get("missing", 0) + 1
        if not node_lists:
            continue
        ref = node_lists[0]
        ref_pts = verts[ref]
        for oi, other in enumerate(node_lists[1:], start=1):
            if len(other) != len(ref):
                if _edge_debug is not None:
                    _edge_debug["difflen"] = _edge_debug.get("difflen", 0) + 1
                    continue
                raise MeshSanityError(
                    f"edge #{ei}: shared-edge node arrays differ in "
                    f"length ({len(ref)} vs {len(other)}); cannot stitch "
                    "by index",
                )
            other_pts = verts[other]
            fwd = float(np.abs(ref_pts - other_pts).max())
            rev = float(np.abs(ref_pts - other_pts[::-1]).max())
            if rev < fwd:
                if _edge_debug is not None:
                    _edge_debug["opposite_order"] = _edge_debug.get("opposite_order", 0) + 1
                    continue
                raise MeshSanityError(
                    f"edge #{ei}: shared-edge node arrays are stored in "
                    f"opposite order (forward error {fwd:.3e} mm > reverse "
                    f"{rev:.3e} mm); an index stitch would twist the mesh",
                )
            if _edge_debug is not None:
                key = "samelen_far" if min(fwd, rev) > linear_deflection_mm else "samelen_ok"
                _edge_debug[key] = _edge_debug.get(key, 0) + 1
            if _trace is not None:
                tag = ("edge", int(ei), "seam" if node_src and node_src[oi] == node_src[0] else "cross")
                ulog = _trace.setdefault("unions", [])
                for u, v in zip(ref, other):
                    ulog.append((int(u), int(v), tag))
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
    #
    # ``_skip_vertex_merge`` (diagnostics only) bypasses this step to isolate
    # whether a meshing failure originates here. The resulting mesh is only
    # meaningful for that A/B test, never for scoring.
    for vi, nodes in ({} if _skip_vertex_merge else vertex_nodes).items():
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
            if _trace is not None:
                _trace.setdefault("unions", []).append(
                    (int(base_node), int(n), ("vertex", int(vi))),
                )
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
    if _trace is not None:
        _trace["merged_faces_predrop"] = merged_faces.copy()
        _trace["keep"] = keep.copy()
        _trace["tri_face"] = np.asarray(tri_face, dtype=np.int64)
        _trace["n_unique"] = n_unique
        _trace["inv"] = inv.copy()
        _trace["triangles"] = np.asarray(triangles, dtype=np.int64)
        _trace["face_base"] = dict(face_base)
        _trace["verts"] = verts.copy()
    merged_faces = merged_faces[keep]
    if merged_faces.shape[0] == 0:
        raise MeshSanityError("stitching collapsed all triangles")

    if _cancel_flaps:
        # Cancel opposite-winding duplicate pairs (on by default; pass
        # _cancel_flaps=False to A/B it). A degenerate face region can fold into
        # a triangle and its mirror (same 3 merged vertices, opposite
        # orientation), which the zero-area drop misses and which reads as
        # non-manifold. Such a pair carries no net surface, so cancelling both
        # is topology-preserving where the flap is redundant. Verified faithful
        # (every removed pair is a true opposite-winding mirror, 0 same-winding
        # drops) across the dataset, with 0 regressions on the parts that
        # already meshed.
        from collections import defaultdict as _dd

        def _even(t) -> bool:
            x, y, z = sorted(t)
            return (int(t[0]), int(t[1]), int(t[2])) in {(x, y, z), (y, z, x), (z, x, y)}

        groups: dict = _dd(list)
        for i, t in enumerate(merged_faces.tolist()):
            groups[tuple(sorted(t))].append(i)
        remove: list[int] = []
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            even = [i for i in idxs if _even(merged_faces[i])]
            odd = [i for i in idxs if not _even(merged_faces[i])]
            ncancel = min(len(even), len(odd))
            remove.extend(even[:ncancel])
            remove.extend(odd[:ncancel])
        if remove:
            mask = np.ones(merged_faces.shape[0], dtype=bool)
            mask[np.asarray(remove, dtype=np.int64)] = False
            merged_faces = merged_faces[mask]
        if merged_faces.shape[0] == 0:
            raise MeshSanityError("flap cancellation removed all triangles")

    used = np.unique(merged_faces.reshape(-1))
    remap = np.full(n_unique, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    return Mesh(
        vertices=centroids[used],
        triangles=remap[merged_faces].astype(np.int64),
        linear_deflection_mm=float(linear_deflection_mm),
    )


def validate_mesh(mesh: Mesh) -> None:
    """Verify *mesh* is a closed orientable 2-manifold.

    Four checks, applied in order; the first to fail raises:

    1. **Manifold edges**, every undirected edge appears in ≤ 2 triangles.
    2. **Closed**, every undirected edge appears in *exactly* 2
       triangles (equivalently :math:`3F = 2E`).
    3. **Orientation-consistent**, for every shared edge ``(a, b)``,
       the two incident triangles list it in opposite orders.
    4. **Manifold vertices**, no vertex where ≥ 2 surface sheets meet
       at a single point. Edge checks 1-3 all pass for two cones glued
       at their apex, yet that pinch is not a 2-manifold and flips the
       Euler characteristic's parity (the cause of negative Betti
       numbers downstream). Run last because it assumes 1-2 hold.

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

    pinched = _nonmanifold_vertices(mesh)
    if pinched.size:
        raise MeshSanityError(
            f"mesh non-manifold vertex: {pinched.size} vertex/vertices where "
            f"≥2 surface sheets meet at a single point (e.g. vertex "
            f"{int(pinched[0])}); edge-manifold and closed but not a "
            "2-manifold surface",
        )


def _nonmanifold_vertices(mesh: Mesh) -> np.ndarray:
    """Return the indices of pinch vertices (≥2 sheets meeting at a point).

    Precondition: *mesh* is already edge-manifold and closed (checks 1-2
    of :func:`validate_mesh`), so every vertex's *link* - the neighbours
    joined by the edge opposite each incident triangle - is a disjoint
    union of cycles. The vertex is a 2-manifold point iff that link is a
    *single* cycle; ≥2 cycles is a pinch the edge-only checks miss.

    One global graph classifies every vertex at once. Its nodes are the
    directed half-edges ``v→u``; at each triangle corner ``v`` with
    opposite edge ``(p, q)`` the two half-edges ``v→p`` and ``v→q`` are
    joined. Joins only ever connect half-edges sharing a source ``v``, so
    each connected component belongs to one vertex and counts one link
    cycle; a vertex owning >1 component is non-manifold. Fully vectorised
    (no per-vertex Python loop): O(F) work plus one sparse
    connected-components call.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tris = mesh.triangles
    n_v = mesh.n_vertices
    # Per corner v, the opposite edge's two endpoints (p, q): the two
    # half-edges v→p and v→q to be joined.
    src = np.concatenate([tris[:, 0], tris[:, 1], tris[:, 2]])
    p = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    q = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])

    # Stable id per directed half-edge (source, dest); the v→p list alone
    # already covers every directed edge (each is some corner's "first").
    half = np.concatenate([np.stack([src, p], 1), np.stack([src, q], 1)])
    keys = half[:, 0] * (n_v + 1) + half[:, 1]
    node = np.unique(keys, return_inverse=True)[1]
    n_nodes = int(node.max()) + 1
    m = src.size

    graph = coo_matrix(
        (np.ones(m, dtype=np.int8), (node[:m], node[m:])),
        shape=(n_nodes, n_nodes),
    )
    n_comp, label = connected_components(graph, directed=False)

    # Distinct component labels per source vertex; >1 ⇒ pinch.
    node_src = np.empty(n_nodes, dtype=np.int64)
    node_src[node[:m]] = src
    vtx_comp = np.unique(node_src * n_comp + label)  # one entry per (vertex, cycle)
    owners, cycles = np.unique(vtx_comp // n_comp, return_counts=True)
    return owners[cycles > 1]


# ---------------------------------------------------------------------------
# Adaptive (escalating) tessellation
# ---------------------------------------------------------------------------

# Divisors applied to the *requested* deflection, coarsest first. Rung 1 is
# the requested deflection itself (so a shape that already meshes is meshed
# exactly as before); finer rungs are only reached when a coarser one fails
# the manifold/closed checks. Stops at /32 — the finest that proved necessary
# to clear false non-manifolds on real parts — so the ladder is bounded and
# cannot run away to pathological triangle counts. The triangle-count ceiling
# and the process-kill timeout are deliberately *not* here; they are a
# separate safeguard layer.
DEFLECTION_LADDER: tuple[int, ...] = (1, 4, 16, 32)


def robust_tessellate_shape(
    wrapped,  # type: ignore[no-untyped-def]
    linear_deflection_mm: float,
    *,
    angular_deflection_rad: float = 0.5,
    parallel: bool | None = None,
    ladder: tuple[int, ...] = DEFLECTION_LADDER,
) -> Mesh:
    """Tessellate + validate, escalating to finer deflection on failure.

    The single chokepoint both the validity gate and the metric mesh
    accessor route through, so neither tessellates "raw" at a coarse
    deflection. Tries the requested deflection first (ladder rung 1),
    then ``requested / divisor`` for each finer rung, returning the first
    :class:`Mesh` that passes :func:`validate_mesh`. The returned mesh's
    ``linear_deflection_mm`` records the rung that actually succeeded
    (for reproducibility), which may be finer than requested.

    Backwards-compatible: a shape that already meshes at the requested
    deflection is meshed identically (rung 1) and returns immediately.

    Raises:
        MeshSanityError: if no rung in *ladder* produces a closed
            orientable manifold (the failure from the finest rung tried).
    """
    if not ladder:
        raise ValueError("ladder must list at least one divisor")

    last_exc: MeshSanityError | None = None
    for divisor in ladder:
        deflection = linear_deflection_mm / divisor
        try:
            mesh = tessellate_shape(
                wrapped,
                deflection,
                angular_deflection_rad=angular_deflection_rad,
                parallel=parallel,
            )
            validate_mesh(mesh)
            return mesh
        except MeshSanityError as exc:
            last_exc = exc
            continue

    assert last_exc is not None  # ladder is non-empty, so we tried >=1 rung
    raise last_exc


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
