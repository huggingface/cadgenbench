"""Unit tests for the CAD validity category.

Covers:
  - :func:`validate_step`, validity only (is_valid, is_watertight, errors).
  - :func:`analyze_step`, validity + measurements bundled into
    :class:`ValidityResult` from a single STEP load.

Measurements alone are covered in ``tests/geometry/test_measurement.py``.
"""
from pathlib import Path

import pytest

from cadgenbench.common.validity import (
    ValidationResult,
    ValidityResult,
    analyze_step,
    validate_step,
)
from cadgenbench.common.measurements import Measurements

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _ensure_fixtures() -> None:
    from tests.fixtures.generate_fixtures import ALL_GENERATORS

    for fn in ALL_GENERATORS:
        fn()


class TestWatertightSolid:
    """Box(10, 20, 30), a valid, watertight solid."""

    PATH = FIXTURES_DIR / "box.step"

    def test_is_valid(self) -> None:
        assert validate_step(self.PATH).is_valid

    def test_is_watertight(self) -> None:
        assert validate_step(self.PATH).is_watertight

    def test_no_topology_errors(self) -> None:
        assert validate_step(self.PATH).topology_errors == ()


class TestOpenShell:
    """A single face, geometry parses but the shape is not watertight.

    Since watertightness is now a hard validity requirement (a non-closed
    shell isn't a usable solid for any downstream metric), ``is_valid``
    is False and ``topology_errors`` carries the descriptive reason.
    """

    PATH = FIXTURES_DIR / "open_shell.step"

    def test_is_not_valid(self) -> None:
        assert not validate_step(self.PATH).is_valid

    def test_not_watertight(self) -> None:
        assert not validate_step(self.PATH).is_watertight

    def test_topology_errors_explain_watertight_failure(self) -> None:
        errors = validate_step(self.PATH).topology_errors
        assert any("watertight" in e.lower() for e in errors), errors


class TestTwoSolids:
    """Two separate boxes, still watertight per-solid."""

    PATH = FIXTURES_DIR / "two_solids.step"

    def test_is_watertight(self) -> None:
        assert validate_step(self.PATH).is_watertight


class TestSphere:
    """Sphere r=10, single periodic face exercises seam-aware closure check."""

    PATH = FIXTURES_DIR / "sphere.step"

    def test_is_watertight(self) -> None:
        assert validate_step(self.PATH).is_watertight


class TestCylinderWithHole:
    """Hollow cylinder, tests boolean subtraction and inner face topology."""

    PATH = FIXTURES_DIR / "cylinder_with_hole.step"

    def test_is_watertight(self) -> None:
        assert validate_step(self.PATH).is_watertight


class TestErrorHandling:

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_step("/nonexistent/path.step")

    def test_corrupted_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.step"
        bad.write_text("this is not a STEP file")
        with pytest.raises(RuntimeError):
            validate_step(bad)

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.step"
        empty.write_bytes(b"")
        with pytest.raises(RuntimeError):
            validate_step(empty)

    def test_returns_validation_result_type(self) -> None:
        r = validate_step(FIXTURES_DIR / "box.step")
        assert isinstance(r, ValidationResult)


class TestAnalyzeStep:
    """``analyze_step`` returns validity + measurements together."""

    PATH = FIXTURES_DIR / "box.step"

    def test_returns_validity_result_type(self) -> None:
        r = analyze_step(self.PATH)
        assert isinstance(r, ValidityResult)
        assert isinstance(r.validation, ValidationResult)
        assert isinstance(r.measurements, Measurements)

    def test_validity_matches_validate_step(self) -> None:
        """``analyze_step`` and ``validate_step`` must agree on validity."""
        a = analyze_step(self.PATH)
        v = validate_step(self.PATH)
        assert a.validation == v

    def test_measurements_first_class(self) -> None:
        """Measurements are populated and usable without a second STEP load."""
        a = analyze_step(self.PATH)
        m = a.measurements
        assert m.solid_count == 1
        assert m.shell_count >= 1
        assert m.face_count == 6
        assert m.volume > 0
        bb = m.bounding_box
        assert bb.size_x > 0 and bb.size_y > 0 and bb.size_z > 0

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            analyze_step("/nonexistent/path.step")
