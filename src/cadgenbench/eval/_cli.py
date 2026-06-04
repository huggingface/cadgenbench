# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``cadgenbench evaluate`` subcommand handler.

(Re)compute metrics for every fixture in one or more result directories.
Each fixture is passed through :func:`cadgenbench.eval.evaluate.evaluate_result`,
which aligns the candidate STEP (reusing a cached ``aligned/output_aligned.step``
when available), fills in any missing renders, and rewrites ``gt_metrics``,
``validation``, and ``interface_metrics`` inside ``result.json``.

Same code path the live agent loop calls at the end of each run.

Per-fixture eval is independent (no shared in-memory state, each fixture
writes its own ``result.json``), so this CLI dispatches across a
``ProcessPoolExecutor`` by default. Override the worker count with
``--workers N`` (``--workers 1`` reverts to sequential). The leaderboard
Space's ``cpu-upgrade`` tier (8 vCPU) gets a near-linear speedup on
real-GT-scale runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# VTK / Mesa GL state preloaded in the parent (via the renderer's
# pyvista import) is not fork-safe on Linux: forking once any VTK
# module has touched a GL context preloader leaves the child with a
# half-initialised libGL, which terminates abruptly the first time
# the worker tries to render and surfaces as BrokenProcessPool in
# the parent. Spawn re-imports cadgenbench in a fresh interpreter
# per worker, no inherited state. macOS already defaults to spawn
# since 3.8; this line is the Linux fix.
_MP_CONTEXT = mp.get_context("spawn")

DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

# Native thread-pool knobs read by Open3D (OpenMP) and NumPy/SciPy (BLAS).
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _cap_worker_threads(n_workers: int) -> int:
    """Cap each spawned worker's native thread pools to avoid CPU oversubscription.

    Open3D (OpenMP) and NumPy/SciPy (BLAS) each default to *all* cores, so
    ``n_workers`` worker processes on a ``C``-vCPU box spawn ~``n_workers * C``
    threads. That oversubscription — not the algorithm — was the dominant cost
    in the eval ``align`` phase (per-fixture align was flat regardless of mesh
    size and ~4-5x faster when a worker ran alone). The pool uses a spawn
    context, so setting these in the parent before dispatch is inherited and
    read fresh by each worker's native libs at import. User-set values win.
    Returns the per-worker thread budget actually applied.
    """
    cpu = os.cpu_count() or 1
    per_worker = max(1, cpu // max(1, n_workers))
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(per_worker))
    return per_worker


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench evaluate`` subcommand."""
    p = subparsers.add_parser(
        "evaluate",
        help="(Re)compute metrics for a result directory.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="One or more results/<run_name>/ directories.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Realign and re-render even when cached artefacts are fresh.",
    )
    p.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=(
            f"Number of parallel workers across fixtures (default "
            f"{DEFAULT_WORKERS} = min(8, os.cpu_count())); pass 1 for "
            "fully sequential."
        ),
    )
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench evaluate``."""
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s",
    )
    from cadgenbench.common.paths import data_gt_dir as _data_gt_dir
    try:
        data_gt_dir = _data_gt_dir()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    workers = max(1, args.workers)
    total_failures = 0
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            print(f"Not a directory: {run_dir}", file=sys.stderr)
            total_failures += 1
            continue
        total_failures += _process_run(
            run_dir, data_gt_dir, force=args.force, workers=workers,
        )

    return 0 if total_failures == 0 else 1


def _eval_one(args: tuple[Path, Path, bool]) -> tuple[str, dict | None, str | None]:
    """Worker entry point for a single fixture.

    Lives at module level so it picks correctly across the ProcessPool's
    spawn context. Returns ``(fixture_name, scores, error_str)``; the
    caller logs both branches in input order to keep output stable
    regardless of completion order.
    """
    fixture_path, gt_dir, force = args
    name = fixture_path.name
    # Live per-fixture markers, flushed so a running shard is observable in
    # streamed job logs (results otherwise print only after the whole shard
    # finishes). The START line is what reveals *which* fixture is stuck.
    t0 = time.perf_counter()
    pid = os.getpid()
    print(f"[eval] start pid={pid} {name}", file=sys.stderr, flush=True)
    try:
        from cadgenbench.eval.evaluate import evaluate_result  # noqa: PLC0415

        scores = evaluate_result(fixture_path, gt_dir, force_align=force)
        print(
            f"[eval] done  pid={pid} {name} {time.perf_counter() - t0:.1f}s",
            file=sys.stderr, flush=True,
        )
        return (name, scores, None)
    except Exception as exc:  # noqa: BLE001 - return the error rather than raising
        print(
            f"[eval] fail  pid={pid} {name} {time.perf_counter() - t0:.1f}s "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True,
        )
        return (name, None, f"{type(exc).__name__}: {exc}")


def _gt_dir_for(data_gt_dir: Path, fixture_name: str) -> Path | None:
    candidate = data_gt_dir / fixture_name
    return candidate if candidate.exists() else None


def _format_scores(scores: dict[str, float | None]) -> str:
    from cadgenbench.eval.shape_similarity import METRIC_DISPLAY  # noqa: PLC0415  -- heavy import deferred to runtime

    parts = []
    for k, v in scores.items():
        if v is None:
            parts.append(f"{k}=n/a")
            continue
        meta = METRIC_DISPLAY.get(k)
        parts.append(
            f"{meta.label}={format(v, meta.fmt)}{meta.suffix}" if meta else f"{k}={v:.3f}",
        )
    return ", ".join(parts) or "(no metrics)"


def _format_interface_metrics(result_json: Path) -> str:
    if not result_json.exists():
        return ""
    data = json.loads(result_json.read_text())
    interface = data.get("interface_metrics") or {}
    score = interface.get("score")
    if score is None:
        return ""
    return f"interface={float(score):.3f}"


def _process_run(
    run_dir: Path, data_gt_dir: Path, *, force: bool, workers: int = 1,
) -> int:
    """Re-evaluate every fixture under *run_dir* and write the run summary.

    Discovers fixtures by directory rather than by existing ``result.json``,
    so fixtures that produced no STEP still get a fresh ``status="missing"``
    entry written. With *workers > 1*, dispatches eval across a
    ``ProcessPoolExecutor``; output stays in the original fixture order
    regardless of completion order. Returns the failure count.
    """
    from cadgenbench.eval.run_summary import write_run_summary  # noqa: PLC0415

    fixtures = sorted(d for d in run_dir.iterdir() if d.is_dir())
    if not fixtures:
        print(f"No fixture directories under {run_dir}", file=sys.stderr)
        return 1

    work: list[tuple[Path, Path, bool]] = []
    skipped: list[str] = []
    for fixture in fixtures:
        gt_dir = _gt_dir_for(data_gt_dir, fixture.name)
        if gt_dir is None:
            skipped.append(fixture.name)
        else:
            work.append((fixture, gt_dir, force))

    n_workers = max(1, min(workers, len(work))) if work else 1
    label = "worker" if n_workers == 1 else "workers"
    print(
        f"\n=== {run_dir} ({len(work)} fixtures, "
        f"{n_workers} {label}{' [sequential]' if n_workers == 1 else ''}) ==="
    )
    for name in skipped:
        print(f"  {name}: no GT sources, skipping")

    if not work:
        try:
            summary_path = write_run_summary(run_dir)
            print(f"  Wrote {summary_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  run_summary FAILED ({exc})", file=sys.stderr)
            return 1
        return 0

    if n_workers == 1:
        results = [_eval_one(w) for w in work]
    else:
        per_worker_threads = _cap_worker_threads(n_workers)
        print(
            f"  (capping native threads to {per_worker_threads}/worker "
            f"across {os.cpu_count()} vCPUs to avoid oversubscription)"
        )
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=_MP_CONTEXT,
        ) as ex:
            results = list(ex.map(_eval_one, work))

    failures = 0
    for (fixture, _, _), (name, scores, error) in zip(work, results):
        if error is not None:
            failures += 1
            print(f"  {name}: FAILED ({error})", file=sys.stderr)
            logging.getLogger(__name__).warning(
                "evaluate_result failed for %s: %s", fixture, error,
            )
            continue
        parts = [_format_scores(scores or {})]
        interface = _format_interface_metrics(fixture / "result.json")
        if interface:
            parts.append(interface)
        print(f"  {name}: {', '.join(parts)}")

    try:
        summary_path = write_run_summary(run_dir)
        print(f"  Wrote {summary_path.name}")
    except Exception as exc:
        failures += 1
        print(f"  run_summary FAILED ({exc})", file=sys.stderr)
        logging.getLogger(__name__).exception(
            "write_run_summary failed for %s", run_dir,
        )
    return failures
