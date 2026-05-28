"""Per-fixture validity failures land as status=invalid, not exceptions.

Regression test for the worker-vs-CLI contract used by the leaderboard
Space: an invalid candidate (non-watertight, mesh non-manifold,
BRepCheck errors) must NOT bubble out of :func:`evaluate_result` as an
exception. It must write a clean ``status: invalid`` + ``cad_score:
0.0`` ``result.json`` and return. This keeps "per-fixture validity
failures" a *score signal* (class-3), separate from "the eval
pipeline crashed" (class-2).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cadgenbench.eval.evaluate import (
    STATUS_INVALID,
    STATUS_MISSING,
    evaluate_result,
)

TESTS_DIR = Path(__file__).parent.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _ensure_fixtures() -> None:
    from tests.fixtures.generate_fixtures import ALL_GENERATORS

    for fn in ALL_GENERATORS:
        fn()


def _gt_dir(tmp_path: Path) -> Path:
    """Build a minimal GT dir containing only ``ground_truth.step``.

    Uses ``box.step`` as the GT; the test only cares that
    :func:`evaluate_result` short-circuits before touching the GT-side
    metric stack on an invalid candidate.
    """
    gt = tmp_path / "gt"
    gt.mkdir()
    shutil.copy(FIXTURES_DIR / "box.step", gt / "ground_truth.step")
    return gt


def test_invalid_candidate_writes_status_invalid_and_returns(tmp_path: Path) -> None:
    """``open_shell.step`` parses but fails the validity gate (non-watertight).

    Before the early-validity fix this raised ``MeshSanityError`` (or
    similar) from the downstream metric stack; the leaderboard worker
    treated that as a class-2 hard failure.
    """
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(FIXTURES_DIR / "open_shell.step", run / "output.step")

    out = evaluate_result(run, _gt_dir(tmp_path))

    # No exception. Returned scores dict is empty for invalid candidates.
    assert out == {}

    result = json.loads((run / "result.json").read_text())
    assert result["status"] == STATUS_INVALID
    assert result["cad_score"] == 0.0
    assert "validation" in result
    assert result["validation"]["is_valid"] is False


def test_missing_candidate_writes_status_missing(tmp_path: Path) -> None:
    """Sanity: the existing missing-candidate fast path still fires."""
    run = tmp_path / "run"
    run.mkdir()
    # No output.step.

    out = evaluate_result(run, _gt_dir(tmp_path))

    assert out == {}
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == STATUS_MISSING
    assert result["cad_score"] == 0.0


# The "valid candidate runs the full pipeline" case lives in
# tests/eval/test_cli_parallelism.py and test_evaluate_interface.py;
# it depends on the renderer (Chromium via Playwright), which is
# sandbox-blocked. The invalid + missing fast paths above are the
# only branches the early-validity refactor changed.
