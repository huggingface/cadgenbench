"""The interface-overlay disagreement region uses manifold3d, not OCCT.

Regression test for the hang where ``evaluate_result`` wedged on a raw
OCCT BREP Boolean (``R & part``) reached via the interface-overlay PNG.
The Boolean is now a ``manifold3d`` mesh op (sub-ms, bounded), and these
tests pin that contract:

1. The disagreement geometry / volume is computed correctly by
   :func:`cadgenbench.eval.interface_match_viz._disagreement_mesh` using
   manifold inputs.
2. :func:`cadgenbench.eval.booleans.manifold_to_mesh` round-trips a
   manifold back to a mesh without losing volume.
3. Neither ``interface_match`` nor ``interface_match_viz`` references
   ``build123d`` any more, so an OCCT Boolean cannot silently return.

All headless: manifold3d only, no VTK render and no OCCT tessellation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

m3d = pytest.importorskip("manifold3d")

from cadgenbench.eval.booleans import (
    manifold_to_mesh,
    manifold_volume,
    mesh_to_manifold,
)
from cadgenbench.eval.interface_match_viz import _disagreement_mesh

# R = 10 mm cube at the origin (vol 1000). The candidate part is an
# identical cube shifted +5 mm on each axis, so the two overlap in a
# 5 mm cube => 125 mm^3 of shared material, well above the 1.0 mm^3
# disagreement epsilon.
_R_SIZE = 10.0
_SHIFT = 5.0
_OVERLAP_VOL = (_R_SIZE - _SHIFT) ** 3        # 125.0
_R_VOL = _R_SIZE ** 3                          # 1000.0


def _cube(size: float, origin=(0.0, 0.0, 0.0)):
    return m3d.Manifold.cube([size, size, size], center=False).translate(list(origin))


def test_disagreement_kor_returns_intersection_volume() -> None:
    """KOR disagreement = R ∩ candidate (material that should be empty)."""
    R = _cube(_R_SIZE)
    part = _cube(_R_SIZE, origin=(_SHIFT, _SHIFT, _SHIFT))

    volume, mesh = _disagreement_mesh(part, R, "KOR")

    assert volume == pytest.approx(_OVERLAP_VOL, rel=1e-3)
    assert mesh is not None
    # The returned mesh is the same region, so it round-trips to the
    # same volume via the manifold kernel.
    assert manifold_volume(mesh_to_manifold(mesh)) == pytest.approx(
        _OVERLAP_VOL, rel=1e-3,
    )


def test_disagreement_kir_returns_subtract_volume() -> None:
    """KIR disagreement = R \\ candidate (material that should be present)."""
    R = _cube(_R_SIZE)
    part = _cube(_R_SIZE, origin=(_SHIFT, _SHIFT, _SHIFT))

    volume, mesh = _disagreement_mesh(part, R, "KIR")

    assert volume == pytest.approx(_R_VOL - _OVERLAP_VOL, rel=1e-3)
    assert mesh is not None


def test_disagreement_below_epsilon_has_no_mesh() -> None:
    """A sub-epsilon overlap is numerical noise: volume reported, mesh None."""
    R = _cube(_R_SIZE)
    # Shift so the cubes barely overlap: 0.5 mm cube => 0.125 mm^3 < 1.0.
    part = _cube(_R_SIZE, origin=(9.5, 9.5, 9.5))

    volume, mesh = _disagreement_mesh(part, R, "KOR")

    assert 0.0 < volume < 1.0
    assert mesh is None


def test_disjoint_kor_is_empty() -> None:
    """No overlap => zero disagreement and no mesh (empty manifold)."""
    R = _cube(_R_SIZE)
    part = _cube(_R_SIZE, origin=(100.0, 0.0, 0.0))

    volume, mesh = _disagreement_mesh(part, R, "KOR")

    assert volume == pytest.approx(0.0, abs=1e-6)
    assert mesh is None


def test_manifold_to_mesh_roundtrip_preserves_volume() -> None:
    """manifold_to_mesh ∘ mesh_to_manifold is volume-preserving."""
    cube = _cube(_R_SIZE)
    mesh = manifold_to_mesh(cube)
    assert mesh.n_triangles > 0
    assert manifold_volume(mesh_to_manifold(mesh)) == pytest.approx(_R_VOL, rel=1e-3)


def test_manifold_to_mesh_rejects_empty() -> None:
    empty = _cube(_R_SIZE) ^ _cube(_R_SIZE, origin=(100.0, 0.0, 0.0))
    assert empty.is_empty()
    with pytest.raises(ValueError):
        manifold_to_mesh(empty)


def test_interface_modules_have_no_occt_booleans() -> None:
    """Neither module may reference build123d: that is the only way an
    OCCT Boolean (the original hang) could re-enter this path."""
    import cadgenbench.eval.interface_match as im
    import cadgenbench.eval.interface_match_viz as imv

    for module in (im, imv):
        src = Path(module.__file__).read_text()
        assert "build123d" not in src, (
            f"{module.__name__} references build123d; the interface path "
            "must stay manifold3d-only (no OCCT Booleans)."
        )
