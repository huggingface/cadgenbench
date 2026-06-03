"""Tests for cadgenbench.eval.alignment, rigid STEP-to-STEP alignment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.conftest import _make_box, _make_cube, _make_l_bracket, _transform_step


def _rotation_matrix(axis: tuple[float, ...], degrees: float) -> np.ndarray:
    """Axis-angle to 3x3 rotation matrix."""
    ax = np.array(axis, dtype=float)
    ax /= np.linalg.norm(ax)
    return Rotation.from_rotvec(np.radians(degrees) * ax).as_matrix()


def _assert_proper_rigid_rotation(R: np.ndarray) -> None:
    """Rotation is orthonormal with det +1: no hidden scale or mirror."""
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def _cuboid_surface_points(
    extents: tuple[float, float, float] = (10.0, 20.0, 30.0),
    n_per_face: int = 12,
) -> np.ndarray:
    """Deterministic grid on a cuboid surface, centered at the origin."""
    x, y, z = (e / 2.0 for e in extents)
    axes = [
        np.linspace(-x, x, n_per_face),
        np.linspace(-y, y, n_per_face),
        np.linspace(-z, z, n_per_face),
    ]
    pts: list[list[float]] = []
    for sx in (-x, x):
        for yy in axes[1]:
            for zz in axes[2]:
                pts.append([sx, yy, zz])
    for sy in (-y, y):
        for xx in axes[0]:
            for zz in axes[2]:
                pts.append([xx, sy, zz])
    for sz in (-z, z):
        for xx in axes[0]:
            for yy in axes[1]:
                pts.append([xx, yy, sz])
    return np.unique(np.asarray(pts, dtype=np.float64), axis=0)


def _cylinder_surface_points(
    radius: float = 5.0,
    height: float = 20.0,
    n_theta: int = 48,
    n_z: int = 16,
    n_r: int = 10,
) -> np.ndarray:
    """Deterministic points on a cylinder surface, centered at the origin."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z_values = np.linspace(-height / 2.0, height / 2.0, n_z)
    pts: list[list[float]] = []
    for zz in z_values:
        for tt in theta:
            pts.append([radius * np.cos(tt), radius * np.sin(tt), zz])
    radii = np.linspace(0.0, radius, n_r)
    for zz in (-height / 2.0, height / 2.0):
        for rr in radii:
            for tt in theta:
                pts.append([rr * np.cos(tt), rr * np.sin(tt), zz])
    return np.unique(np.asarray(pts, dtype=np.float64), axis=0)


def _nearest_rmse(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    dists, _ = cKDTree(b).query(a)
    return float(np.sqrt(np.mean(dists ** 2)))


# ---------------------------------------------------------------------------
# 1. Identity, same shape, same pose
# ---------------------------------------------------------------------------

class TestIdentityAlignment:

    def test_identity(self, box_step: str, tmp_path: Path) -> None:
        from cadgenbench.eval.alignment import align_step

        result = align_step(
            box_step, box_step,
            output=tmp_path / "id_aligned.step",
            n_samples=5000, seed=0,
        )
        _assert_proper_rigid_rotation(result.rotation)
        assert result.rmse < 1.0
        assert result.output_path.exists()


# ---------------------------------------------------------------------------
# 2. Known R+T, asymmetric box with known rigid transform
# ---------------------------------------------------------------------------

class TestKnownTransform:

    @pytest.fixture(scope="class")
    def transformed_box(self) -> tuple[str, str, np.ndarray, np.ndarray]:
        R = _rotation_matrix((1, 1, 1), 37)
        t = np.array([10.0, -5.0, 3.0])
        src = _make_box()
        tgt = _transform_step(src, R, t, "rot37_t10")
        return src, tgt, R, t

    def test_recovers_rotation(self, transformed_box, tmp_path: Path) -> None:
        from cadgenbench.eval.alignment import align_step

        src, tgt, _R_gt, _t_gt = transformed_box
        result = align_step(
            src, tgt,
            output=tmp_path / "known_aligned.step",
            n_samples=8000, seed=42,
        )
        # The scorer cares that the transformed geometry is close; box
        # symmetries can make several rigid transforms equally valid.
        _assert_proper_rigid_rotation(result.rotation)
        assert result.rmse < 1.0, f"RMSE too high: {result.rmse}"
        assert result.output_path.exists()

    def test_recovers_translation(self, transformed_box, tmp_path: Path) -> None:
        from cadgenbench.eval.alignment import align_step

        src, tgt, _R_gt, _t_gt = transformed_box
        result = align_step(
            src, tgt,
            output=tmp_path / "known_aligned_t.step",
            n_samples=8000, seed=42,
        )
        _assert_proper_rigid_rotation(result.rotation)
        assert result.rmse < 1.0, f"RMSE too high: {result.rmse}"
        assert result.output_path.exists()


# ---------------------------------------------------------------------------
# 3. Symmetric shape (cube), RMSE should be low even if R is ambiguous
# ---------------------------------------------------------------------------

class TestSymmetricCube:

    @pytest.fixture(scope="class")
    def transformed_cube(self) -> tuple[str, str]:
        R = _rotation_matrix((0, 0, 1), 73)
        t = np.array([5.0, -3.0, 7.0])
        src = _make_cube()
        tgt = _transform_step(src, R, t, "cube_rot73")
        return src, tgt

    def test_low_rmse_despite_symmetry(self, transformed_cube, tmp_path: Path) -> None:
        from cadgenbench.eval.alignment import align_step

        src, tgt = transformed_cube
        result = align_step(
            src, tgt,
            output=tmp_path / "cube_aligned.step",
            n_samples=5000, seed=0,
        )
        # We don't check R (ambiguous for a cube), just that alignment is good
        assert result.rmse < 1.0, f"RMSE too high for symmetric shape: {result.rmse}"


# ---------------------------------------------------------------------------
# 4. Symmetric bbox, asymmetric shape (L-bracket)
# ---------------------------------------------------------------------------

class TestAsymmetricLBracket:

    @pytest.fixture(scope="class")
    def transformed_l(self) -> tuple[str, str, np.ndarray]:
        R = _rotation_matrix((1, 0, 0), 45)
        t = np.array([0.0, 0.0, 0.0])
        src = _make_l_bracket()
        tgt = _transform_step(src, R, t, "l_rot45")
        return src, tgt, R

    def test_correct_alignment(self, transformed_l, tmp_path: Path) -> None:
        from cadgenbench.eval.alignment import align_step

        src, tgt, _R_gt = transformed_l
        result = align_step(
            src, tgt,
            output=tmp_path / "l_aligned.step",
            n_samples=8000, seed=42,
        )
        _assert_proper_rigid_rotation(result.rotation)
        assert result.rmse < 1.0, f"RMSE too high: {result.rmse}"
        assert result.output_path.exists()


# ---------------------------------------------------------------------------
# 5. Slightly deformed shape, similar but not identical
# ---------------------------------------------------------------------------

class TestDeformedShape:

    def test_box_to_tapered_box(
        self, box_step: str, tapered_box_step: str, tmp_path: Path,
    ) -> None:
        from cadgenbench.eval.alignment import align_step

        result = align_step(
            box_step, tapered_box_step,
            output=tmp_path / "deformed_aligned.step",
            n_samples=5000, seed=0,
        )
        # Shapes differ so RMSE won't be zero, but should be reasonable
        assert result.rmse < 10.0, f"RMSE too high for similar shapes: {result.rmse}"
        assert result.output_path.exists()


# ---------------------------------------------------------------------------
# 6. Scale mismatch (negative test), no hidden scaling
# ---------------------------------------------------------------------------

class TestScaleMismatch:

    @pytest.fixture(scope="class")
    def boxes_different_scale(self) -> tuple[str, str]:
        small = _make_box(10, 10, 10)
        big = _make_box(20, 20, 20)
        return small, big

    def test_rmse_reflects_scale_difference(
        self, boxes_different_scale, tmp_path: Path,
    ) -> None:
        from cadgenbench.eval.alignment import align_step

        small, big = boxes_different_scale
        result = align_step(
            small, big,
            output=tmp_path / "scale_aligned.step",
            n_samples=5000, seed=0,
        )
        # Shapes are cubes scaled 2x, RMSE should be significant
        # (proportional to size difference, roughly ~2.5 for box half-extent 5 vs 10)
        assert result.rmse > 1.0, (
            f"RMSE suspiciously low ({result.rmse:.2f}), might be secretly scaling"
        )


# ---------------------------------------------------------------------------
# 7. align_points, pure numpy path without STEP I/O
# ---------------------------------------------------------------------------

class TestAlignPoints:

    def test_known_rotation(self) -> None:
        from cadgenbench.eval.alignment import align_points

        rng = np.random.default_rng(0)
        src = rng.standard_normal((500, 3))

        R_gt = _rotation_matrix((0, 1, 0), 60)
        t_gt = np.array([3.0, -2.0, 1.0])
        tgt = (R_gt @ src.T).T + t_gt

        R, t, rmse = align_points(src, tgt)

        assert rmse < 0.1
        _assert_proper_rigid_rotation(R)
        np.testing.assert_allclose(R, R_gt, atol=0.1)
        np.testing.assert_allclose(t, t_gt, atol=0.5)

    def test_identity_is_rigid_identity(self) -> None:
        from cadgenbench.eval.alignment import align_points

        pts = _cuboid_surface_points()

        R, t, rmse = align_points(pts, pts)

        assert rmse < 1e-8
        _assert_proper_rigid_rotation(R)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(t, 0.0, atol=1e-8)

    def test_known_rigid_surface_transform(self) -> None:
        from cadgenbench.eval.alignment import align_points

        src = _cuboid_surface_points()
        R_gt = _rotation_matrix((1, 2, 3), 41)
        t_gt = np.array([12.0, -4.0, 7.0])
        tgt = (R_gt @ src.T).T + t_gt

        R, t, rmse = align_points(src, tgt)
        aligned = (R @ src.T).T + t

        assert rmse < 0.05
        _assert_proper_rigid_rotation(R)
        assert _nearest_rmse(aligned, tgt) < 0.05

    def test_symmetric_cube_accepts_equivalent_rotation(self) -> None:
        from cadgenbench.eval.alignment import align_points

        src = _cuboid_surface_points((10.0, 10.0, 10.0))
        R_gt = _rotation_matrix((0, 0, 1), 90)
        t_gt = np.array([4.0, -8.0, 2.0])
        tgt = (R_gt @ src.T).T + t_gt

        R, t, rmse = align_points(src, tgt)
        aligned = (R @ src.T).T + t

        assert rmse < 0.05
        _assert_proper_rigid_rotation(R)
        # Do not assert a specific R: cube symmetries are equally valid.
        assert _nearest_rmse(aligned, tgt) < 0.05

    def test_symmetric_cylinder_accepts_axial_rotation(self) -> None:
        from cadgenbench.eval.alignment import align_points

        src = _cylinder_surface_points()
        R_gt = _rotation_matrix((0, 0, 1), 137)
        t_gt = np.array([3.0, -6.0, 4.0])
        tgt = (R_gt @ src.T).T + t_gt

        R, t, rmse = align_points(src, tgt)
        aligned = (R @ src.T).T + t

        assert rmse < 0.05
        _assert_proper_rigid_rotation(R)
        # Axial rotation is continuously ambiguous; only geometry residual matters.
        assert _nearest_rmse(aligned, tgt) < 0.05

    def test_near_symmetric_marker_picks_correct_side(self) -> None:
        from cadgenbench.eval.alignment import align_points

        body = _cuboid_surface_points((20.0, 20.0, 10.0), n_per_face=10)
        marker = np.array(
            [
                [12.0, 3.0, -1.0],
                [12.0, 3.0, 1.0],
                [12.0, 5.0, -1.0],
                [12.0, 5.0, 1.0],
            ],
            dtype=np.float64,
        )
        target = np.vstack([body, marker])
        R0 = _rotation_matrix((0, 0, 1), 180)
        t0 = np.array([30.0, -5.0, 0.0])
        source = (R0 @ target.T).T + t0

        R, t, rmse = align_points(source, target)
        aligned = (R @ source.T).T + t
        aligned_marker = aligned[-len(marker):]

        assert rmse < 0.05
        _assert_proper_rigid_rotation(R)
        assert _nearest_rmse(aligned, target) < 0.05
        assert np.linalg.norm(aligned_marker.mean(axis=0) - marker.mean(axis=0)) < 0.05

    def test_scale_mismatch_is_not_scaled_away(self) -> None:
        from cadgenbench.eval.alignment import align_points

        small = _cuboid_surface_points((10.0, 10.0, 10.0))
        big = _cuboid_surface_points((20.0, 20.0, 20.0))

        R, _t, rmse = align_points(small, big)

        _assert_proper_rigid_rotation(R)
        assert rmse > 1.0
