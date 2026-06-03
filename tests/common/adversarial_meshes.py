"""Hand-built triangle meshes that exercise each mesh-gate failure mode.

Each builder returns a :class:`cadgenbench.common.mesh.Mesh` whose
``validate_mesh`` outcome is documented in the module docstring of
:mod:`cadgenbench.common.mesh`. They are used both by
:mod:`tests.geometry.test_mesh` and by a tiny matplotlib render helper
(``python -m tests.geometry.adversarial_meshes``) for visual inspection.

Fixtures:

- ``cube_mesh``              , closed orientable manifold; passes all
                                three gates (happy path).
- ``nonmanifold_t_mesh``     , three triangles sharing one edge; trips
                                the manifold check.
- ``open_tetrahedron_mesh``  , tetrahedron with the bottom face removed;
                                trips the closed check.
- ``flipped_winding_mesh``   , closed cube with one triangle's winding
                                flipped; trips the orientation check.
- ``pinch_vertex_mesh``      , two tetrahedra sharing a single apex
                                vertex; passes the manifold/closed/
                                orientation checks but trips the
                                vertex-manifold check (two sheets meet
                                at one point).

The meshes are deliberately tiny (≤ 12 triangles) so the failure modes
are obvious by eye.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cadgenbench.common.mesh import Mesh


def cube_mesh() -> Mesh:
    """Closed manifold unit cube with outward-consistent winding.

    8 vertices, 12 triangles. Each undirected edge appears in exactly
    2 triangles; each directed edge appears in exactly 1.
    """
    v = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom (z=0)
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top (z=1)
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            # bottom (z=0), normals point -z
            [0, 2, 1], [0, 3, 2],
            # top (z=1), normals point +z
            [4, 5, 6], [4, 6, 7],
            # front (y=0), normals point -y
            [0, 1, 5], [0, 5, 4],
            # back (y=1), normals point +y
            [2, 3, 7], [2, 7, 6],
            # right (x=1), normals point +x
            [1, 2, 6], [1, 6, 5],
            # left (x=0), normals point -x
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=v, triangles=f, linear_deflection_mm=0.01)


def nonmanifold_t_mesh() -> Mesh:
    """Three triangles sharing one common edge (0, 1).

    Trips the manifold gate: the edge (0, 1) is incident to three
    triangles, but a 2-manifold edge must be incident to at most two.
    """
    v = np.array(
        [
            [0.0, 0.0, 0.0],  # 0, shared edge start
            [1.0, 0.0, 0.0],  # 1, shared edge end
            [0.5, 1.0, 0.0],  # 2, wing 1 tip (+y, z=0)
            [0.5, -1.0, 0.0],  # 3, wing 2 tip (-y, z=0)
            [0.5, 0.0, 1.0],  # 4, stem tip (+z)
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 1, 2],  # wing in +y plane
            [0, 1, 3],  # wing in -y plane
            [0, 1, 4],  # stem in +z plane (this is the "extra" leaf)
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=v, triangles=f, linear_deflection_mm=0.01)


def open_tetrahedron_mesh() -> Mesh:
    """Tetrahedron with the bottom triangle removed (3 of 4 faces).

    Trips the closed gate: three undirected edges are now incident to
    only one triangle each (the perimeter of the missing bottom).
    """
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    # NOTE: the bottom face [0, 1, 2] is intentionally omitted.
    f = np.array(
        [
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=v, triangles=f, linear_deflection_mm=0.01)


def flipped_winding_mesh() -> Mesh:
    """Closed manifold cube with one triangle's winding reversed.

    Trips the orientation gate: the flipped triangle traverses two of
    its shared edges in the *same* direction as its neighbour
    triangles, instead of the opposite direction expected of a
    consistently-oriented closed manifold.
    """
    m = cube_mesh()
    f = m.triangles.copy()
    # Reverse triangle 0 (one of the two bottom-face triangles).
    f[0] = f[0][::-1]
    return Mesh(vertices=m.vertices, triangles=f, linear_deflection_mm=0.01)


def pinch_vertex_mesh() -> Mesh:
    """Two closed tetrahedra meeting at one shared apex vertex (index 3).

    Each tetrahedron is on its own a closed orientable manifold, and the
    two share no edge - only the single apex point. So every undirected
    edge is still in exactly 2 triangles (manifold + closed) and winding
    is consistent, yet the surface is *not* a 2-manifold: the apex is a
    pinch where two sheets touch. Its link is two disjoint triangles
    ({0,1,2} and {4,5,6}) rather than one cycle, so it trips only the
    vertex-manifold check. Welding two such vertices into one is exactly
    how a coarse tessellation drives the Euler characteristic odd.
    """
    v = np.array(
        [
            [0.0, 0.0, 0.0],  # 0  tetra A base
            [1.0, 0.0, 0.0],  # 1  tetra A base
            [0.5, 1.0, 0.0],  # 2  tetra A base
            [0.5, 0.5, 1.0],  # 3  shared apex
            [0.0, 0.0, 2.0],  # 4  tetra B base
            [1.0, 0.0, 2.0],  # 5  tetra B base
            [0.5, 1.0, 2.0],  # 6  tetra B base
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            # tetra A (base 0,1,2 below the apex), outward-wound
            [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
            # tetra B (base 4,5,6 above the apex), outward-wound
            [4, 5, 6], [4, 3, 5], [5, 3, 6], [6, 3, 4],
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=v, triangles=f, linear_deflection_mm=0.01)


# ---------------------------------------------------------------------------
# Visualisation helper (matplotlib; safe to run from any environment)
# ---------------------------------------------------------------------------

ALL_FIXTURES: dict[str, tuple[Mesh, str]] = {
    "cube_happy": (
        cube_mesh(),
        "Happy path: closed manifold cube (passes all three gates)",
    ),
    "nonmanifold_T": (
        nonmanifold_t_mesh(),
        "Non-manifold: 3 triangles meeting on edge (0, 1)",
    ),
    "open_tetrahedron": (
        open_tetrahedron_mesh(),
        "Open: tetrahedron with the bottom face removed",
    ),
    "flipped_winding": (
        flipped_winding_mesh(),
        "Orientation: cube with one triangle's winding reversed",
    ),
    "pinch_vertex": (
        pinch_vertex_mesh(),
        "Vertex-manifold: two tetrahedra meeting at a single apex (vertex 3)",
    ),
}


def render_to_pngs(out_dir: str | Path) -> None:
    """Render each fixture to ``<out_dir>/<name>.png`` via matplotlib.

    Uses the Agg backend so it works headless. Triangle edges are drawn
    in black; the flipped triangle (for ``flipped_winding``) is
    highlighted in red so the failure mode is obvious to the eye.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, (mesh, title) in ALL_FIXTURES.items():
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        tris = mesh.vertices[mesh.triangles]
        colors = ["lightblue"] * len(tris)
        edge_colors = ["black"] * len(tris)
        if name == "flipped_winding":
            colors[0] = "salmon"
            edge_colors[0] = "red"

        for i, tri in enumerate(tris):
            poly = Poly3DCollection(
                [tri], alpha=0.55, linewidth=1.2,
            )
            poly.set_facecolor(colors[i])
            poly.set_edgecolor(edge_colors[i])
            ax.add_collection3d(poly)

        v = mesh.vertices
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], color="black", s=15)
        for idx, (x, y, z) in enumerate(v):
            ax.text(x, y, z, str(idx), fontsize=8, color="darkred")

        margin = 0.2
        ax.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
        ax.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
        ax.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=130)
        plt.close(fig)
        print(f"wrote {out / f'{name}.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render the adversarial mesh fixtures to PNGs.",
    )
    parser.add_argument(
        "--out", default="/tmp/mesh_adversarial",
        help="Output directory for the PNGs (default: %(default)s)",
    )
    args = parser.parse_args()
    render_to_pngs(args.out)
