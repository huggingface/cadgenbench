"""Smoke tests for the parallel-fixture dispatch in ``cadgenbench evaluate``.

The actual eval pipeline (renders Chromium, runs OCC alignment, etc.) is
too heavy to exercise here; these tests stub :func:`evaluate_result` and
verify two things:

1. The orchestrator's results are deterministic and order-preserving --
   parallel dispatch with N>1 workers produces the same per-fixture
   output order, success/failure shape, and run_summary as the
   sequential (workers=1) path.
2. :func:`_eval_one` is module-level and picklable, which is the
   precondition for ``ProcessPoolExecutor`` to find it across the spawn
   boundary on macOS.

Real determinism of :func:`evaluate_result` itself between sequential
and parallel runs is a property of the eval pipeline (no shared state
across fixtures, deterministic sampling), not the orchestrator. It's
verified manually by ``cadgenbench evaluate <run_dir> --workers 1`` vs
``--workers 8`` on a real fixture set.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from cadgenbench.eval import _cli


def _stub_evaluate_result(result_dir, gt_dir, **kwargs):  # type: ignore[no-untyped-def]
    """Deterministic stub: writes a result.json with a synthetic cad_score
    derived from the fixture name, no Chromium / OCC.

    Used in place of the real ``cadgenbench.eval.evaluate.evaluate_result``
    so the orchestrator can run end-to-end without the heavy deps.
    """
    name = Path(result_dir).name
    cad_score = round(0.5 + 0.01 * (sum(ord(c) for c in name) % 50), 4)
    payload = {
        "status": "valid",
        "cad_score": cad_score,
        "gt_metrics": {"shape_similarity_score": cad_score},
        "validation": {"is_valid": True, "is_watertight": True},
    }
    (Path(result_dir) / "result.json").write_text(json.dumps(payload, indent=2))
    return payload["gt_metrics"]


@pytest.fixture
def fake_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Build a results dir + a sibling data/{inputs,gt}/ tree for 5 stub fixtures.

    Each fixture dir gets an empty ``output.step`` placeholder; each
    inputs dir gets a ``description.yaml`` (read by
    :func:`run_summary.write_run_summary` for the per-fixture
    ``task_type`` field); each GT dir gets an empty
    ``ground_truth.step`` placeholder. The stubbed ``evaluate_result``
    ignores those files entirely.

    Sets ``CADGENBENCH_DATA_DIR`` so the run-summary writer can find
    its ``data/inputs/`` lookup without hitting the Hub.
    """
    run_dir = tmp_path / "results" / "stub-run"
    data_dir = tmp_path / "data"
    inputs_dir = data_dir / "inputs"
    gt_dir = data_dir / "gt"
    run_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    for name in (
        "fix-aaa",
        "fix-bbb",
        "fix-ccc",
        "fix-ddd",
        "fix-eee",
    ):
        fix = run_dir / name
        fix.mkdir()
        (fix / "output.step").write_text("ISO-10303-21;\n")
        inp = inputs_dir / name
        inp.mkdir()
        (inp / "description.yaml").write_text(
            "task_type: generation\ndescription: stub\n"
        )
        gt = gt_dir / name
        gt.mkdir()
        (gt / "ground_truth.step").write_text("ISO-10303-21;\n")

    monkeypatch.setenv("CADGENBENCH_DATA_DIR", str(data_dir))
    monkeypatch.delenv("CADGENBENCH_DATA_REPO", raising=False)
    monkeypatch.delenv("CADGENBENCH_DATA_GT_REPO", raising=False)
    monkeypatch.setattr(
        "cadgenbench.eval.evaluate.evaluate_result", _stub_evaluate_result,
    )
    return run_dir, gt_dir


def _collect_results(run_dir: Path) -> list[tuple[str, dict]]:
    """Read every per-fixture result.json under run_dir, in name order."""
    out: list[tuple[str, dict]] = []
    for fixture_dir in sorted(d for d in run_dir.iterdir() if d.is_dir()):
        rj = fixture_dir / "result.json"
        if rj.exists():
            out.append((fixture_dir.name, json.loads(rj.read_text())))
    return out


def test_eval_one_is_picklable() -> None:
    """ProcessPoolExecutor on macOS uses spawn; the worker entry point
    must be picklable for the pool to find it across the boundary."""
    pickled = pickle.dumps(_cli._eval_one)
    restored = pickle.loads(pickled)
    assert restored is _cli._eval_one


def test_sequential_dispatch_produces_per_fixture_results(
    fake_run: tuple[Path, Path],
) -> None:
    """workers=1 path runs the stubbed eval on every fixture and
    persists a result.json per fixture."""
    run_dir, gt_dir = fake_run
    failures = _cli._process_run(run_dir, gt_dir, force=False, workers=1)
    assert failures == 0
    results = _collect_results(run_dir)
    names = [n for n, _ in results]
    assert names == ["fix-aaa", "fix-bbb", "fix-ccc", "fix-ddd", "fix-eee"]
    for _, payload in results:
        assert payload["status"] == "valid"
        assert 0.5 <= payload["cad_score"] <= 1.0


def test_parallel_dispatch_matches_sequential(
    fake_run: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """workers>1 path produces identical per-fixture result.json + identical
    run_summary.json to the workers=1 path.

    Process-pool subprocesses on macOS use ``spawn``, which re-imports
    every module fresh and so doesn't see ``monkeypatch`` mutations.
    The orchestrator-level guarantee we want to verify is "dispatching
    over a thread/process pool doesn't reorder, drop, or duplicate
    work" -- which is independent of the executor flavour. The test
    therefore swaps :class:`ProcessPoolExecutor` for a thread-based
    executor inside :mod:`cadgenbench.eval._cli`, lets the stub
    ``evaluate_result`` apply across the threads, and asserts the two
    paths produce byte-identical results + run_summary.

    Real determinism of the heavy pipeline across ``ProcessPoolExecutor``
    is a separate property (verified manually on a real run dir, e.g.
    ``cadgenbench evaluate <run> --workers 1`` vs ``--workers 8``).
    """
    from concurrent.futures import ThreadPoolExecutor

    run_dir, gt_dir = fake_run

    seq_failures = _cli._process_run(run_dir, gt_dir, force=False, workers=1)
    seq_results = _collect_results(run_dir)
    seq_summary = json.loads((run_dir / "run_summary.json").read_text())

    for fixture_dir in run_dir.iterdir():
        if fixture_dir.is_dir():
            (fixture_dir / "result.json").unlink()
    (run_dir / "run_summary.json").unlink()

    class _ThreadExecutorAcceptingMpContext(ThreadPoolExecutor):
        """Thread-pool shim that accepts (and drops) the mp_context kwarg.

        Production code passes ``mp_context=spawn`` to ProcessPoolExecutor
        for VTK fork-safety; ThreadPoolExecutor doesn't take that kwarg.
        """

        def __init__(self, *args, mp_context=None, **kwargs):
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(_cli, "ProcessPoolExecutor", _ThreadExecutorAcceptingMpContext)
    monkeypatch.setattr(
        "cadgenbench.eval.evaluate.evaluate_result", _stub_evaluate_result,
    )

    par_failures = _cli._process_run(run_dir, gt_dir, force=False, workers=4)
    par_results = _collect_results(run_dir)
    par_summary = json.loads((run_dir / "run_summary.json").read_text())

    assert par_failures == seq_failures
    assert [n for n, _ in par_results] == [n for n, _ in seq_results]
    assert [p for _, p in par_results] == [p for _, p in seq_results]
    assert par_summary == seq_summary


def test_one_failed_fixture_doesnt_stop_others(
    fake_run: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker raising on one fixture only fails that fixture's row;
    siblings still get evaluated."""
    run_dir, gt_dir = fake_run

    def _flaky(result_dir, gt_dir_, **_kwargs):  # type: ignore[no-untyped-def]
        if Path(result_dir).name == "fix-ccc":
            raise RuntimeError("synthetic failure")
        return _stub_evaluate_result(result_dir, gt_dir_)

    monkeypatch.setattr(
        "cadgenbench.eval.evaluate.evaluate_result", _flaky,
    )
    failures = _cli._process_run(run_dir, gt_dir, force=False, workers=1)
    assert failures == 1
    results = _collect_results(run_dir)
    names = {n for n, _ in results}
    assert {"fix-aaa", "fix-bbb", "fix-ddd", "fix-eee"} <= names
    assert "fix-ccc" not in names
