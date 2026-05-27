"""Unit tests for STEP geometric measurements."""
import math
from pathlib import Path

import pytest

from cadgenbench.common.measurements import BBox, Measurements, measure_step

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _ensure_fixtures() -> None:
    from tests.fixtures.generate_fixtures import ALL_GENERATORS

    for fn in ALL_GENERATORS:
        fn()


class TestBBox:
    """Sanity checks on the BBox dataclass helpers."""

    def test_sizes(self) -> None:
        bb = BBox(0, 10, 0, 20, 0, 30)
        assert bb.size_x == 10
        assert bb.size_y == 20
        assert bb.size_z == 30

    def test_diagonal(self) -> None:
        bb = BBox(0, 3, 0, 4, 0, 0)
        assert bb.diagonal == pytest.approx(5.0)


class TestBox:
    """Box(10, 20, 30)."""

    PATH = FIXTURES_DIR / "box.step"

    def test_solid_count(self) -> None:
        assert measure_step(self.PATH).solid_count == 1

    def test_face_count(self) -> None:
        assert measure_step(self.PATH).face_count == 6

    def test_volume(self) -> None:
        assert measure_step(self.PATH).volume == pytest.approx(6000.0, rel=1e-4)

    def test_bounding_box_dimensions(self) -> None:
        bb = measure_step(self.PATH).bounding_box
        assert bb.size_x == pytest.approx(10.0, rel=1e-4)
        assert bb.size_y == pytest.approx(20.0, rel=1e-4)
        assert bb.size_z == pytest.approx(30.0, rel=1e-4)


class TestOpenShell:
    """Single face, measurement still works, volume is 0."""

    PATH = FIXTURES_DIR / "open_shell.step"

    def test_zero_solids(self) -> None:
        assert measure_step(self.PATH).solid_count == 0

    def test_zero_volume(self) -> None:
        assert measure_step(self.PATH).volume == pytest.approx(0.0, abs=1e-9)


class TestTwoSolids:

    PATH = FIXTURES_DIR / "two_solids.step"

    def test_solid_count(self) -> None:
        assert measure_step(self.PATH).solid_count == 2

    def test_volume(self) -> None:
        assert measure_step(self.PATH).volume == pytest.approx(2000.0, rel=1e-4)


class TestSphere:
    """Sphere r=10 exercises volume on curved NURBS."""

    PATH = FIXTURES_DIR / "sphere.step"
    EXPECTED_VOLUME = (4 / 3) * math.pi * 10**3

    def test_volume(self) -> None:
        assert measure_step(self.PATH).volume == pytest.approx(
            self.EXPECTED_VOLUME, rel=1e-4,
        )

    def test_face_count(self) -> None:
        assert measure_step(self.PATH).face_count == 1


class TestCylinderWithHole:
    """Hollow cylinder, tests boolean subtraction and inner face topology."""

    PATH = FIXTURES_DIR / "cylinder_with_hole.step"
    EXPECTED_VOLUME = math.pi * (20**2 - 5**2) * 40

    def test_volume(self) -> None:
        assert measure_step(self.PATH).volume == pytest.approx(
            self.EXPECTED_VOLUME, rel=1e-4,
        )

    def test_face_count(self) -> None:
        assert measure_step(self.PATH).face_count == 4


class TestErrorHandling:

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            measure_step("/nonexistent/path.step")

    def test_corrupted_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.step"
        bad.write_text("this is not a STEP file")
        with pytest.raises(RuntimeError):
            measure_step(bad)

    def test_returns_measurements_type(self) -> None:
        r = measure_step(FIXTURES_DIR / "box.step")
        assert isinstance(r, Measurements)
