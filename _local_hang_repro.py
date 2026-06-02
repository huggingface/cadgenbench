"""Local, network-free reproduction of the compare-llms nested-pool hang.

Runs the REAL ``compare_llms.run`` over a single fixture for 3 fake "models"
with parallel fixtures and a 3s wall-clock budget, so renders are abandoned
mid-flight at the timeout exactly like a real run. No Docker / GPU / API keys.

A healthy run prints "[repro] FINISHED rc=..." within a few seconds. A hang
means the parent's as_completed never returns -> the bug.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import tempfile
import threading
import time
from pathlib import Path


def _watchdog(seconds: float) -> None:
    """Hard-stop the whole process tree on hang, after dumping stacks."""
    def _kill() -> None:
        sys.stderr.write(f"\n[watchdog] {seconds:.0f}s elapsed -> HANG. Thread stacks:\n")
        sys.stderr.flush()
        faulthandler.dump_traceback()
        os._exit(99)
    t = threading.Timer(seconds, _kill)
    t.daemon = True
    t.start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--fixtures", type=int, default=2)
    ap.add_argument("--max-duration", type=float, default=3.0)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--watchdog", type=float, default=25.0)
    args = ap.parse_args()

    os.environ["CADGENBENCH_FAKE_LLM"] = "1"
    _watchdog(args.watchdog)

    from cadgenbench.baseline import compare_llms

    tmp = Path(tempfile.mkdtemp(prefix="hang_repro_"))
    # Fabricate fixture task dicts directly (skip data/ discovery): generation
    # tasks with no GT and no input files exercise the full agent+render path.
    tasks = [
        {"name": f"fix_{i}", "description": f"Build part {i}.", "task_type": "generation"}
        for i in range(args.fixtures)
    ]
    compare_llms._discover_fixtures = lambda *a, **k: tasks  # type: ignore[attr-defined]

    # Patch path helpers so run() doesn't require a real data/ tree.
    import cadgenbench.common.paths as paths
    paths.data_inputs_dir = lambda: tmp  # type: ignore[attr-defined]
    paths.data_gt_dir = lambda: tmp  # type: ignore[attr-defined]

    models = ["fake/model-%d" % i for i in range(args.models)]
    # Resolve via LLMClient is fine (no network); just returns the model str.

    ns = argparse.Namespace(
        fixtures=[], all=False, limit=None,
        models=models, labels=None,
        output=tmp / "compare.html", output_dir=tmp / "results",
        max_iter=args.max_iter, max_tokens=1_000_000, max_tokens_per_call=4096,
        max_duration=args.max_duration, llm_timeout=30.0, temperature=0.0,
        reasoning_effort=None, timeout=60, parallel=args.parallel,
        fixture_retries=0,
    )

    t0 = time.monotonic()
    rc = compare_llms.run(ns)
    print(f"[repro] FINISHED rc={rc} in {time.monotonic() - t0:.1f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
