"""Shared session-scoped pytest fixtures.

Generates STEP files once per session via build123d.  Each fixture
creates a STEP file under ``tests/fixtures/geometry/`` that persists
for the full pytest session and is reused on re-runs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "geometry"


def _ensure_fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


# ---------------------------------------------------------------------------
# Shape generators (build123d)
# ---------------------------------------------------------------------------


def _make_box(x: float = 10, y: float = 20, z: float = 30) -> str:
    from build123d import Box, BuildPart, export_step

    path = _ensure_fixtures_dir() / f"box_{x}_{y}_{z}.step"
    if not path.exists():
        with BuildPart() as p:
            Box(x, y, z)
        export_step(p.part, str(path))
    return str(path)


def _make_cube(size: float = 10) -> str:
    return _make_box(size, size, size)


def _make_sphere(r: float = 10) -> str:
    from build123d import BuildPart, Sphere, export_step

    path = _ensure_fixtures_dir() / f"sphere_{r}.step"
    if not path.exists():
        with BuildPart() as p:
            Sphere(r)
        export_step(p.part, str(path))
    return str(path)


def _make_l_bracket() -> str:
    """L-shaped bracket: asymmetric mass distribution but near-cubic bbox."""
    from build123d import Box, BuildPart, Locations, Mode, export_step

    path = _ensure_fixtures_dir() / "l_bracket.step"
    if not path.exists():
        with BuildPart() as p:
            Box(20, 20, 5)
            with Locations([(0, 7.5, 7.5)]):
                Box(20, 5, 20)
        export_step(p.part, str(path))
    return str(path)


def _make_tapered_box() -> str:
    """A box-like shape tapered along Z, similar to but not identical to a box."""
    from build123d import BuildPart, BuildSketch, Plane, Rectangle, export_step, loft

    path = _ensure_fixtures_dir() / "tapered_box.step"
    if not path.exists():
        with BuildPart() as p:
            with BuildSketch(Plane.XY.offset(-10)):
                Rectangle(10, 10)
            with BuildSketch(Plane.XY.offset(10)):
                Rectangle(8, 8)
            loft()
        export_step(p.part, str(path))
    return str(path)


def _transform_step(src_path: str, R, t, suffix: str) -> str:
    """Apply a rigid transform to an existing STEP file and save a new one."""
    import numpy as np
    from cadgenbench.eval.sampling import _load_occ_shape
    from cadgenbench.eval.alignment import _apply_and_export

    out = _ensure_fixtures_dir() / f"{Path(src_path).stem}_{suffix}.step"
    if not out.exists():
        shape = _load_occ_shape(Path(src_path))
        _apply_and_export(shape, np.asarray(R), np.asarray(t), out)
    return str(out)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def box_step() -> str:
    return _make_box()


@pytest.fixture(scope="session")
def cube_step() -> str:
    return _make_cube()


@pytest.fixture(scope="session")
def sphere_step() -> str:
    return _make_sphere()


@pytest.fixture(scope="session")
def l_bracket_step() -> str:
    return _make_l_bracket()


@pytest.fixture(scope="session")
def tapered_box_step() -> str:
    return _make_tapered_box()
