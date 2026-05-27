"""Unit tests for the mesh-based feature-edge extractor.

Pure-geometry tests on hand-built meshes (no STEP I/O, no rendering),
so they're fast and don't depend on any tessellator's idea of "sharp".
The point is to pin the algorithm:

- A 90° dihedral is a kept edge.
- A near-flat dihedral is dropped (smooth band).
- An intermediate dihedral lands in the ambiguous band and is excluded.
- Non-manifold (1- or 3-incident) edges are skipped, not crashed on.
- Sample-spacing controls how many points come out of one long edge.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cadgenbench.eval.feature_edges import (
    DEFAULT_SPACING_FRAC,
    extract_feature_edge_points,
    extract_feature_edges_debug,
)


# ---------------------------------------------------------------------------
# Synthetic meshes
# ---------------------------------------------------------------------------


def _two_tri_dihedral(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Two triangles sharing edge (0, 1) along the x-axis.

    Triangle 0 lies in the xy-plane with outward normal +z.
    Triangle 1 is rotated about the x-axis by (180° - angle_deg) so the
    dihedral *between the outward normals* is exactly angle_deg.

    Wound consistent: both triangles list edge (0, 1) once forward and
    once backward, so the shared edge appears in exactly 2 triangles
    with opposite traversal.
    """
    a = math.radians(180.0 - angle_deg)
    verts = np.array(
        [
            [0.0, 0.0, 0.0],     # 0, shared
            [1.0, 0.0, 0.0],     # 1, shared
            [0.5, 1.0, 0.0],     # 2, in xy-plane
            [0.5, math.cos(a), math.sin(a)],  # 3, rotated about x
        ],
        dtype=np.float64,
    )
    tris = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
        ],
        dtype=np.int64,
    )
    return verts, tris


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    """Regular tetrahedron centred at the origin (all dihedrals ≈ 70.53°)."""
    # Vertices of a regular tetrahedron.
    v = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Outward-wound faces (right-hand rule, normal away from origin).
    t = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    return v, t


# ---------------------------------------------------------------------------
# Single-edge dihedral binning
# ---------------------------------------------------------------------------


class TestTwoTriangleDihedral:

    def test_ninety_degree_kept(self) -> None:
        verts, tris = _two_tri_dihedral(90.0)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0,
        )
        assert debug.n_kept == 1
        assert debug.n_smooth == 0
        assert debug.n_ambiguous == 0
        # Two non-shared edges per triangle, all of those are
        # boundary edges (incident to 1 triangle): 4 non-manifold edges.
        assert debug.n_non_manifold == 4

    def test_two_degrees_smooth(self) -> None:
        verts, tris = _two_tri_dihedral(2.0)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0,
        )
        assert debug.n_kept == 0
        assert debug.n_smooth == 1
        assert debug.n_ambiguous == 0

    @pytest.mark.parametrize("angle_deg", [10.0, 20.0, 29.0])
    def test_ambiguous_band(self, angle_deg: float) -> None:
        verts, tris = _two_tri_dihedral(angle_deg)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0,
        )
        assert debug.n_kept == 0
        assert debug.n_smooth == 0
        assert debug.n_ambiguous == 1

    def test_threshold_boundary_inclusive_at_tau_sharp(self) -> None:
        """Exactly 30° lands on the boundary; spec is ``angle >= tau_sharp``."""
        verts, tris = _two_tri_dihedral(30.0)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0,
        )
        assert debug.n_kept == 1
        assert debug.n_ambiguous == 0

    def test_kept_segment_geometry(self) -> None:
        """Kept edge endpoints match the shared (0, 1) edge."""
        verts, tris = _two_tri_dihedral(60.0)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0,
        )
        assert debug.segments.shape == (1, 2, 3)
        seg = debug.segments[0]
        expected = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Could be in either order; match either.
        same = np.allclose(seg, expected)
        flipped = np.allclose(seg, expected[::-1])
        assert same or flipped


# ---------------------------------------------------------------------------
# Edge sampling
# ---------------------------------------------------------------------------


class TestEdgeSampling:

    def test_short_edge_emits_midpoint(self) -> None:
        verts, tris = _two_tri_dihedral(90.0)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0, spacing_frac=1.0,
        )
        assert debug.points.shape == (1, 3)
        np.testing.assert_allclose(
            debug.points[0], [0.5, 0.0, 0.0],
        )

    def test_long_edge_samples_density(self) -> None:
        """Sampling density scales 1/spacing along a fixed-length edge."""
        verts, tris = _two_tri_dihedral(90.0)
        debug_loose = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0, spacing_frac=0.25,
        )
        debug_tight = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0, spacing_frac=0.05,
        )
        assert debug_tight.points.shape[0] > debug_loose.points.shape[0]
        # Loose: spacing = 0.5 mm on a 1 mm edge -> ceil(2)+1 = 3 samples
        assert debug_loose.points.shape[0] == 3
        # Tight: spacing = 0.1 mm on a 1 mm edge -> ceil(10)+1 = 11 samples
        assert debug_tight.points.shape[0] == 11

    def test_sampled_points_lie_on_edge(self) -> None:
        verts, tris = _two_tri_dihedral(90.0)
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=2.0, spacing_frac=0.1,
        )
        # Points must satisfy y=0, z=0 and x in [0, 1].
        np.testing.assert_allclose(debug.points[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(debug.points[:, 2], 0.0, atol=1e-12)
        assert (debug.points[:, 0] >= 0.0).all()
        assert (debug.points[:, 0] <= 1.0).all()
        # Endpoints must be present.
        assert np.isclose(debug.points[:, 0].min(), 0.0)
        assert np.isclose(debug.points[:, 0].max(), 1.0)


# ---------------------------------------------------------------------------
# Whole-mesh sanity
# ---------------------------------------------------------------------------


class TestTetrahedron:

    def test_tetrahedron_keeps_six_edges(self) -> None:
        """Regular tetrahedron has 6 edges, all dihedral ≈ 70.53° -> all kept."""
        verts, tris = _tetrahedron()
        debug = extract_feature_edges_debug(
            verts, tris, bbox_diagonal=4.0,
        )
        assert debug.n_total_edges == 6
        assert debug.n_kept == 6
        assert debug.n_smooth == 0
        assert debug.n_ambiguous == 0
        assert debug.n_non_manifold == 0
        assert debug.points.shape[0] >= 6
        assert debug.segments.shape == (6, 2, 3)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:

    def test_rejects_negative_tau_smooth(self) -> None:
        verts, tris = _two_tri_dihedral(90.0)
        with pytest.raises(ValueError):
            extract_feature_edge_points(
                verts, tris,
                bbox_diagonal=2.0, tau_smooth_deg=-1.0,
            )

    def test_rejects_tau_smooth_ge_tau_sharp(self) -> None:
        verts, tris = _two_tri_dihedral(90.0)
        with pytest.raises(ValueError):
            extract_feature_edge_points(
                verts, tris,
                bbox_diagonal=2.0, tau_smooth_deg=30.0, tau_sharp_deg=30.0,
            )

    def test_rejects_zero_bbox(self) -> None:
        verts, tris = _two_tri_dihedral(90.0)
        with pytest.raises(ValueError):
            extract_feature_edge_points(verts, tris, bbox_diagonal=0.0)

    def test_empty_tris_returns_empty(self) -> None:
        verts = np.zeros((0, 3), dtype=np.float64)
        tris = np.zeros((0, 3), dtype=np.int64)
        pts = extract_feature_edge_points(verts, tris, bbox_diagonal=1.0)
        assert pts.shape == (0, 3)

    def test_default_spacing_frac(self) -> None:
        assert DEFAULT_SPACING_FRAC == pytest.approx(0.002)
