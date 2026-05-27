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
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=0.1)
        np.testing.assert_allclose(result.translation, 0, atol=0.5)
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

        src, tgt, R_gt, t_gt = transformed_box
        result = align_step(
            src, tgt,
            output=tmp_path / "known_aligned.step",
            n_samples=8000, seed=42,
        )
        # R_recovered should be close to R_gt
        assert result.rmse < 1.0, f"RMSE too high: {result.rmse}"

        # The rotation should map src axes to tgt axes
        R_err = result.rotation @ R_gt.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert np.degrees(angle) < 10, f"Rotation error: {np.degrees(angle):.1f}°"

    def test_recovers_translation(self, transformed_box, tmp_path: Path) -> None:
        from cadgenbench.eval.alignment import align_step

        src, tgt, R_gt, t_gt = transformed_box
        result = align_step(
            src, tgt,
            output=tmp_path / "known_aligned_t.step",
            n_samples=8000, seed=42,
        )
        t_err = np.linalg.norm(result.translation - t_gt)
        assert t_err < 3.0, f"Translation error: {t_err:.2f}"


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

        src, tgt, R_gt = transformed_l
        result = align_step(
            src, tgt,
            output=tmp_path / "l_aligned.step",
            n_samples=8000, seed=42,
        )
        assert result.rmse < 1.0, f"RMSE too high: {result.rmse}"

        # L-bracket is asymmetric, so R should be recoverable
        R_err = result.rotation @ R_gt.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert np.degrees(angle) < 15, f"Rotation error: {np.degrees(angle):.1f}°"


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
        np.testing.assert_allclose(R, R_gt, atol=0.1)
        np.testing.assert_allclose(t, t_gt, atol=0.5)
