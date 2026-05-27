"""Tests for :mod:`cadgenbench.common.mesh`.

Two layers of coverage:

1. **Unit tests on :func:`validate_mesh`** using the hand-built
   :mod:`tests.geometry.adversarial_meshes` fixtures. Each adversarial
   mesh trips exactly one gate (manifold, closed, or
   orientation-consistent); the happy-path cube passes all three.

2. **End-to-end tests on :func:`tessellate_and_validate`** using the
   small build123d STEP fixtures from :mod:`tests.geometry.conftest`.
   These prove the welding + tessellation pipeline produces a mesh
   that survives the gate on real BREP input.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from cadgenbench.common.mesh import (
    DEFLECTION_MAX_MM,
    DEFLECTION_MIN_MM,
    Mesh,
    MeshSanityError,
    deflection_for_bbox,
    tessellate_and_validate,
    tessellate_step,
    validate_mesh,
)
from tests.common.adversarial_meshes import (
    cube_mesh,
    flipped_winding_mesh,
    nonmanifold_t_mesh,
    open_tetrahedron_mesh,
)


# ---------------------------------------------------------------------------
# Gate: manifold + closed + orientation-consistent
# ---------------------------------------------------------------------------


class TestValidateMeshHappyPath:
    """A closed orientable manifold passes all three gates silently."""

    def test_cube_passes(self) -> None:
        validate_mesh(cube_mesh())  # must not raise

    def test_cube_has_expected_euler_chi(self) -> None:
        """Sanity: V − E + F = 2 for a sphere-topology mesh."""
        m = cube_mesh()
        # 12 triangles → 18 unique edges, 8 vertices → chi = 8 - 18 + 12 = 2.
        from collections import Counter

        edge_count: Counter[tuple[int, int]] = Counter()
        for a, b, c in m.triangles.tolist():
            for u, v in ((a, b), (b, c), (c, a)):
                edge_count[(min(u, v), max(u, v))] += 1
        chi = m.n_vertices - len(edge_count) + m.n_triangles
        assert chi == 2


class TestNonManifoldGate:
    """3 triangles sharing one edge must trip the manifold gate."""

    def test_t_junction_raises(self) -> None:
        with pytest.raises(MeshSanityError, match=re.compile(r"non[- ]?manifold", re.I)):
            validate_mesh(nonmanifold_t_mesh())

    def test_error_mentions_offending_edge(self) -> None:
        try:
            validate_mesh(nonmanifold_t_mesh())
        except MeshSanityError as exc:
            assert "3 triangles" in str(exc) or "shared by 3" in str(exc), exc
        else:  # pragma: no cover - sanity
            pytest.fail("expected MeshSanityError")


class TestClosedGate:
    """A tetrahedron with one face removed must trip the closed gate."""

    def test_open_tetrahedron_raises(self) -> None:
        with pytest.raises(MeshSanityError, match=re.compile(r"not closed", re.I)):
            validate_mesh(open_tetrahedron_mesh())

    def test_error_quantifies_open_edges(self) -> None:
        try:
            validate_mesh(open_tetrahedron_mesh())
        except MeshSanityError as exc:
            assert "open edges" in str(exc), exc
        else:  # pragma: no cover - sanity
            pytest.fail("expected MeshSanityError")


class TestOrientationGate:
    """A cube with one triangle's winding flipped must trip orientation."""

    def test_flipped_triangle_raises(self) -> None:
        with pytest.raises(MeshSanityError, match=re.compile(r"orientation", re.I)):
            validate_mesh(flipped_winding_mesh())


class TestEmptyMesh:
    def test_zero_triangle_mesh_raises(self) -> None:
        empty = Mesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            triangles=np.zeros((0, 3), dtype=np.int64),
            linear_deflection_mm=0.01,
        )
        with pytest.raises(MeshSanityError, match="empty"):
            validate_mesh(empty)


# ---------------------------------------------------------------------------
# Deflection policy
# ---------------------------------------------------------------------------


class TestDeflectionForBbox:
    def test_relative_to_bbox(self) -> None:
        # 100 mm diagonal → 0.1 mm deflection (relative regime).
        assert deflection_for_bbox(100.0) == pytest.approx(0.1)

    def test_clamped_min(self) -> None:
        # Tiny part, would compute 0.001 mm, clamped up.
        assert deflection_for_bbox(1.0) == DEFLECTION_MIN_MM

    def test_clamped_max(self) -> None:
        # Huge part, would compute 1.0 mm, clamped down.
        assert deflection_for_bbox(1000.0) == DEFLECTION_MAX_MM


# ---------------------------------------------------------------------------
# End-to-end: tessellate real STEP → must survive the gate
# ---------------------------------------------------------------------------


class TestTessellateAndValidate:
    """Real BREP inputs from the geometry-test fixtures must pass."""

    def test_box(self, box_step: str) -> None:
        m = tessellate_and_validate(box_step, 0.1)
        assert m.n_triangles > 0
        assert m.n_vertices >= 8

    def test_sphere(self, sphere_step: str) -> None:
        # Sphere exercises periodic-surface seam handling.
        m = tessellate_and_validate(sphere_step, 0.5)
        assert m.n_triangles > 0

    def test_l_bracket(self, l_bracket_step: str) -> None:
        m = tessellate_and_validate(l_bracket_step, 0.1)
        assert m.n_triangles > 0


class TestTessellateStepFailures:
    def test_missing_file_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            tessellate_step("/no/such/file.step", 0.1)

    def test_zero_deflection_rejected(self, box_step: str) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            tessellate_step(box_step, 0.0)
