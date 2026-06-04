"""Unit tests for the mesh safeguards in :mod:`cadgenbench.common.validity`.

Three bounds on meshing cost, applied identically to GT and submissions:

  1. **File-size pre-filter** (deterministic) — oversized STEP rejected
     before it is parsed.
  2. **Triangle-count ceiling** (deterministic) — a mesh above the ceiling
     fails the gate.
  3. **Per-mesh process-kill timeout** (machine-dependent backstop) — a mesh
     that overruns is declared invalid on the first overrun (no retry), with
     the offending STEP saved for debugging. A part that fails to mesh once
     is memoised so later stages do not re-pay the cost.

For ground truth every violation escalates to a loud exception (GT must
clear all ceilings) rather than a silent ``is_valid=False``.

The deterministic ceilings are exercised through the in-process meshing
path (``MESH_TIMEOUT_S <= 0``) so the assertions do not depend on a child
process inheriting a monkeypatched module attribute; a separate test proves
the real killable-subprocess round-trip works end to end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cadgenbench.common import validity
from cadgenbench.common.mesh import MeshSanityError, deflection_for_bbox
from cadgenbench.common.validity import (
    MeshTimeoutError,
    analyze_step,
    parse_step,
    safeguarded_tessellate,
    validate_step,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
BOX = FIXTURES_DIR / "box.step"


@pytest.fixture(autouse=True, scope="session")
def _ensure_fixtures() -> None:
    from tests.fixtures.generate_fixtures import ALL_GENERATORS

    for fn in ALL_GENERATORS:
        fn()


@pytest.fixture(autouse=True)
def _clear_mesh_failure_memo() -> None:
    """Isolate the per-process failed-mesh memo across tests.

    Tests reuse ``box.step``; a test that forces a timeout/ceiling failure
    would otherwise poison the memo for every later test on the same file.
    """
    validity._FAILED_MESH_CACHE.clear()
    yield
    validity._FAILED_MESH_CACHE.clear()


def _box_wrapped():  # type: ignore[no-untyped-def]
    from build123d import Box

    return Box(10, 20, 30).wrapped


# ---------------------------------------------------------------------------
# 1. File-size pre-filter
# ---------------------------------------------------------------------------


class TestFileSizeCeiling:
    """An oversized STEP is rejected before the (expensive) parse."""

    def test_oversized_is_invalid_with_reason(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_STEP_FILE_BYTES", 10)
        result = validate_step(BOX)
        assert not result.is_valid
        assert not result.is_watertight
        assert any("ceiling" in e for e in result.topology_errors), (
            result.topology_errors
        )

    def test_analyze_step_oversized_zeroed_measurements(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_STEP_FILE_BYTES", 10)
        a = analyze_step(BOX)
        assert not a.validation.is_valid
        # The file is deliberately never loaded, so measurements are zeroed.
        assert a.measurements.solid_count == 0
        assert a.measurements.volume == 0.0

    def test_within_ceiling_unaffected(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_STEP_FILE_BYTES", 50_000_000)
        assert validate_step(BOX).is_valid

    def test_parse_step_rejects_oversized(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_STEP_FILE_BYTES", 10)
        with pytest.raises(RuntimeError, match="ceiling"):
            parse_step(BOX)

    def test_ground_truth_oversized_raises_loudly(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_STEP_FILE_BYTES", 10)
        with pytest.raises(RuntimeError, match="GROUND TRUTH"):
            validate_step(BOX, is_ground_truth=True)


# ---------------------------------------------------------------------------
# 2. Triangle-count ceiling (deterministic)
# ---------------------------------------------------------------------------


class TestTriangleCeiling:
    """A mesh above the triangle ceiling fails the gate, deterministically."""

    @pytest.fixture(autouse=True)
    def _in_process_meshing(self, monkeypatch) -> None:
        # Force the in-process path so the monkeypatched ceiling is seen by
        # the checker (a spawned child would re-read the env default).
        monkeypatch.setattr(validity, "MESH_TIMEOUT_S", 0.0)

    def test_box_exceeds_tiny_ceiling(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_TRIANGLES", 1)
        wrapped = _box_wrapped()
        with pytest.raises(MeshSanityError, match="triangle ceiling"):
            safeguarded_tessellate(None, 0.5, wrapped=wrapped)

    def test_validate_step_reports_ceiling_reason(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_TRIANGLES", 1)
        result = validate_step(BOX)
        assert not result.is_valid
        assert any("triangle ceiling" in e for e in result.topology_errors), (
            result.topology_errors
        )

    def test_generous_ceiling_passes(self, monkeypatch) -> None:
        monkeypatch.setattr(validity, "MAX_TRIANGLES", 1_000_000)
        assert validate_step(BOX).is_valid

    def test_ground_truth_ceiling_breach_raises(self, monkeypatch) -> None:
        # Trusted parts skip the validity mesh gate (they are pre-verified),
        # so the triangle ceiling is enforced where a GT is actually meshed,
        # on demand. A breach there must still raise loudly rather than pass.
        monkeypatch.setattr(validity, "MAX_TRIANGLES", 1)
        with pytest.raises(MeshSanityError, match="triangle ceiling"):
            safeguarded_tessellate(None, 0.5, wrapped=_box_wrapped(), is_ground_truth=True)


# ---------------------------------------------------------------------------
# 3. Per-mesh process-kill timeout (first overrun is the verdict, no retry)
# ---------------------------------------------------------------------------


class TestMeshTimeout:
    """The timeout backstop: one overrun -> invalid + save the STEP, no retry."""

    def test_timeout_raises_on_first_overrun_and_saves(
        self, monkeypatch, tmp_path,
    ) -> None:
        calls = {"n": 0}

        def _always_timeout(step_path, deflection, ladder, *, angular=0.5):
            calls["n"] += 1
            raise MeshTimeoutError("exceeded test wall-clock")

        monkeypatch.setattr(validity, "MESH_TIMEOUT_S", 5.0)
        monkeypatch.setattr(validity, "_mesh_in_subprocess", _always_timeout)
        monkeypatch.setenv("CADGENBENCH_TIMEOUT_DEBUG_DIR", str(tmp_path))

        with pytest.raises(MeshTimeoutError, match="mesh timeout"):
            safeguarded_tessellate(BOX, 0.5)

        assert calls["n"] == 1, "must not retry a timed-out mesh"
        saved = list(tmp_path.glob("box-*.step"))
        assert saved, "offending STEP should be copied aside for debugging"

    def test_ground_truth_timeout_raises_runtimeerror(self, monkeypatch, tmp_path) -> None:
        def _always_timeout(step_path, deflection, ladder, *, angular=0.5):
            raise MeshTimeoutError("exceeded test wall-clock")

        monkeypatch.setattr(validity, "MESH_TIMEOUT_S", 5.0)
        monkeypatch.setattr(validity, "_mesh_in_subprocess", _always_timeout)
        monkeypatch.setenv("CADGENBENCH_TIMEOUT_DEBUG_DIR", str(tmp_path))

        with pytest.raises(RuntimeError, match="GROUND TRUTH"):
            safeguarded_tessellate(BOX, 0.5, is_ground_truth=True)

    def test_failed_mesh_is_memoised_not_reattempted(
        self, monkeypatch, tmp_path,
    ) -> None:
        """A known-bad part re-raises its verdict without re-running the mesh."""
        calls = {"n": 0}

        def _always_timeout(step_path, deflection, ladder, *, angular=0.5):
            calls["n"] += 1
            raise MeshTimeoutError("exceeded test wall-clock")

        monkeypatch.setattr(validity, "MESH_TIMEOUT_S", 5.0)
        monkeypatch.setattr(validity, "_mesh_in_subprocess", _always_timeout)
        monkeypatch.setenv("CADGENBENCH_TIMEOUT_DEBUG_DIR", str(tmp_path))

        with pytest.raises(MeshTimeoutError):
            safeguarded_tessellate(BOX, 0.5)
        # Second call (e.g. a later stage) must hit the memo, not the mesher.
        with pytest.raises(MeshTimeoutError):
            safeguarded_tessellate(BOX, 0.25)
        assert calls["n"] == 1, "memoised failure must not re-tessellate"


# ---------------------------------------------------------------------------
# End-to-end: the real killable subprocess round-trip
# ---------------------------------------------------------------------------


def test_subprocess_meshing_roundtrip_returns_valid_mesh() -> None:
    """The default (timeout > 0) path meshes a box in the killable worker.

    Proves the spawn-Pool + IPC + Mesh round-trip works and that a normal
    part is unaffected by the safeguards.
    """
    mesh = safeguarded_tessellate(BOX, deflection_for_bbox(40.0))
    assert mesh.n_triangles > 0
    assert mesh.vertices.shape[1] == 3
