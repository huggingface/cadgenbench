"""Sidecar-aware alignment: trusted meshes are aligned, never re-tessellated.

Covers the fix for the bug where ``compute_edit_baseline`` round-tripped a
trusted ``input.step`` through an aligned STEP and re-meshed it (bypassing the
supplied-mesh sidecar), which blew up on inputs our mesher can't tessellate.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from cadgenbench.common.artifacts import StepArtifacts
from cadgenbench.common.mesh import Mesh
from cadgenbench.eval.alignment import align_cached_mesh
from cadgenbench.eval.shape_similarity import compare_step_files


def _box_mesh() -> Mesh:
    """A 10x20x30 box (distinct extents → unambiguous principal axes)."""
    v = np.array(
        [
            [0, 0, 0], [10, 0, 0], [10, 20, 0], [0, 20, 0],
            [0, 0, 30], [10, 0, 30], [10, 20, 30], [0, 20, 30],
        ],
        dtype=np.float64,
    )
    t = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5], [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=v, triangles=t, linear_deflection_mm=0.3)


class _FakeArtifacts:
    """Minimal stand-in exposing ``.mesh()`` (the only thing align_cached_mesh uses)."""

    def __init__(self, mesh: Mesh) -> None:
        self._mesh = mesh

    def mesh(self) -> Mesh:
        return self._mesh


def _write_sidecar(step_path: Path, mesh: Mesh) -> None:
    sidecar = step_path.with_name(step_path.stem + ".mesh.npz")
    with sidecar.open("wb") as fh:
        np.savez(
            fh,
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            triangles=np.asarray(mesh.triangles, dtype=np.int64),
            linear_deflection_mm=np.asarray(float(mesh.linear_deflection_mm)),
        )


def _assert_proper_rigid_rotation(R: np.ndarray) -> None:
    """Rotation is orthonormal with det +1: no hidden scale or mirror."""
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# align_cached_mesh: recovers a rigid transform from cached meshes only
# ---------------------------------------------------------------------------


def test_align_cached_mesh_recovers_rigid_transform() -> None:
    target = _box_mesh()
    # Rotate 90° about X and translate; align must undo it from points alone.
    R0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t0 = np.array([5.0, -3.0, 2.0])
    source = Mesh(
        vertices=target.vertices @ R0.T + t0,
        triangles=target.triangles,
        linear_deflection_mm=target.linear_deflection_mm,
    )

    res = align_cached_mesh(_FakeArtifacts(source), _FakeArtifacts(target))

    # Proper rigid transform (no scale / mirroring) and a tight fit back onto the target.
    _assert_proper_rigid_rotation(res.rotation)
    assert res.rmse < 1.0
    # Connectivity is untouched (only vertices move).
    assert np.array_equal(res.mesh.triangles, source.triangles)
    # The aligned box recovers the target's extents, up to sampled-ICP tolerance.
    ext = np.sort(res.mesh.vertices.max(axis=0) - res.mesh.vertices.min(axis=0))
    assert ext == pytest.approx([10.0, 20.0, 30.0], abs=0.05)
    # Aligned cloud sits on the target (centroid back near the target's).
    assert np.linalg.norm(res.mesh.vertices.mean(0) - target.vertices.mean(0)) < 1.0


# ---------------------------------------------------------------------------
# compare_step_files: trusted candidate skips the STEP-export / re-tessellate path
# ---------------------------------------------------------------------------


def test_trusted_candidate_exact_match_scores_near_perfect_without_align_step(
    box_step: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the synthetic exact-match sidecar case that scored ~0.95."""
    candidate = tmp_path / "candidate.step"
    gt = tmp_path / "gt.step"
    shutil.copyfile(box_step, candidate)
    shutil.copyfile(box_step, gt)
    # Give the candidate a trusted sidecar (its own mesh).
    _write_sidecar(candidate, StepArtifacts(candidate).mesh())

    # If the trusted path is taken, align_step must never be called.
    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("align_step must not run for a trusted candidate")

    monkeypatch.setattr("cadgenbench.eval.alignment.align_step", _boom)

    result = compare_step_files(candidate, gt, align=True)

    assert result.aligned_step is None  # no aligned STEP written on the trusted path
    assert result.scores.get("shape_similarity_score") is not None
    # Identical box vs box → shape similarity saturates.
    assert result.scores["shape_similarity_score"] > 0.99
    assert result.scores["shape_volume_iou"] > 0.99
    # No aligned STEP artifact was left behind next to the candidate.
    assert not (tmp_path / "candidate_aligned.step").exists()


def test_untrusted_candidate_still_uses_step_alignment(
    box_step: str, tmp_path: Path,
) -> None:
    # No sidecar → the candidate is untrusted (a real submission), so the
    # legacy STEP-export + re-tessellate alignment path must still run.
    candidate = tmp_path / "candidate.step"
    gt = tmp_path / "gt.step"
    shutil.copyfile(box_step, candidate)
    shutil.copyfile(box_step, gt)

    result = compare_step_files(candidate, gt, align=True)

    assert result.aligned_step is not None
    assert Path(result.aligned_step).exists()
    assert result.scores.get("shape_similarity_score") is not None
