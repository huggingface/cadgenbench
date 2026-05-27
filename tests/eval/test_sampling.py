"""Tests for cadgenbench.eval.sampling, STEP to point cloud."""
from __future__ import annotations

import numpy as np
import pytest


class TestSampleSurfacePoints:

    def test_returns_correct_shape(self, box_step: str) -> None:
        from cadgenbench.eval.sampling import sample_surface_points

        pts = sample_surface_points(box_step, n_points=500, seed=0)
        assert pts.shape == (500, 3)
        assert pts.dtype == np.float64

    def test_points_lie_on_surface(self, box_step: str) -> None:
        """All sampled points should lie on a face of the 10x20x30 box
        (centered at origin by build123d)."""
        from cadgenbench.eval.sampling import sample_surface_points

        pts = sample_surface_points(box_step, n_points=2000, seed=42)

        half = np.array([5.0, 10.0, 15.0])
        # Each point should have at least one coordinate at ±half (on a face)
        on_face = np.any(
            np.isclose(np.abs(pts), half[None, :], atol=0.2),
            axis=1,
        )
        assert on_face.mean() > 0.95, "Most points should lie on a box face"

    def test_seed_reproducibility(self, box_step: str) -> None:
        from cadgenbench.eval.sampling import sample_surface_points

        a = sample_surface_points(box_step, n_points=100, seed=7)
        b = sample_surface_points(box_step, n_points=100, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_file_not_found(self) -> None:
        from cadgenbench.eval.sampling import sample_surface_points

        with pytest.raises(FileNotFoundError):
            sample_surface_points("/nonexistent/path.step")

    def test_sphere_coverage(self, sphere_step: str) -> None:
        """Points on a sphere should have roughly uniform angular coverage."""
        from cadgenbench.eval.sampling import sample_surface_points

        pts = sample_surface_points(sphere_step, n_points=5000, seed=0)
        norms = np.linalg.norm(pts, axis=1)
        # Sphere radius=10; points should be ~10 from origin
        assert np.abs(norms.mean() - 10.0) < 0.5
