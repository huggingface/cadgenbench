"""Unit tests for the CAD validity category.

Covers:
  - :func:`validate_step`, validity only (is_valid, is_watertight, errors).
  - :func:`analyze_step`, validity + measurements bundled into
    :class:`ValidityResult` from a single STEP load.
  - :func:`parse_step`, parseability gate only (no BRepCheck, no mesh).

Measurements alone are covered in ``tests/geometry/test_measurement.py``.
"""
from pathlib import Path

import pytest

from cadgenbench.common.validity import (
    ValidationResult,
    ValidityResult,
    analyze_step,
    parse_step,
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


class TestGeometryQualityGate:
    """Fixed geometry floors: edge length, face area, BREP tolerance."""

    def test_normal_box_is_clean(self) -> None:
        from build123d import Box

        from cadgenbench.common.validity import _collect_geometry_quality_errors

        assert _collect_geometry_quality_errors(Box(10, 20, 30).wrapped) == []

    def test_near_degenerate_edge_flagged(self) -> None:
        """A plate 0.0005 mm thick has sub-micron edges (< 0.001 mm)."""
        from build123d import Box

        from cadgenbench.common.validity import _collect_geometry_quality_errors

        errs = _collect_geometry_quality_errors(Box(400, 300, 0.0005).wrapped)
        assert any("shorter than" in e for e in errs), errs

    def test_gate_rejects_degenerate_part_with_clear_reason(self) -> None:
        """A thin plate is a watertight BREP but fails the quality gate."""
        from build123d import Box

        from cadgenbench.common.validity import _validate_wrapped

        result = _validate_wrapped(Box(400, 300, 0.0005).wrapped)
        assert not result.is_valid
        assert result.is_watertight, "the BREP itself is well-formed"
        assert any("shorter than" in e for e in result.topology_errors), (
            result.topology_errors
        )


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


class TestParseStep:
    """``parse_step`` is the cheap-load gate: parseable yes/no, no validity."""

    def test_valid_step_does_not_raise(self) -> None:
        parse_step(FIXTURES_DIR / "box.step")

    def test_parseable_but_invalid_step_does_not_raise(self) -> None:
        """Files that parse but fail the validity gate must still pass.

        Surfacing per-fixture validity is :func:`analyze_step`'s job, not
        :func:`parse_step`'s, so an open-shell STEP is "parseable" here
        even though it isn't watertight.
        """
        parse_step(FIXTURES_DIR / "open_shell.step")

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_step("/nonexistent/path.step")

    def test_corrupted_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.step"
        bad.write_text("this is not a STEP file")
        with pytest.raises(RuntimeError):
            parse_step(bad)

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.step"
        empty.write_bytes(b"")
        with pytest.raises(RuntimeError):
            parse_step(empty)

    def test_returns_none(self) -> None:
        """``parse_step`` is a gate; its successful return value is None."""
        assert parse_step(FIXTURES_DIR / "box.step") is None
