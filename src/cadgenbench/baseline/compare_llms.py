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

"""``cadgenbench baseline compare-llms`` subcommand handler.

Run the baseline agent once per LLM model on the same fixture set, then
emit a side-by-side HTML comparison via the existing report tool.

One command spins up N independent agent runs (one per ``--models``
entry, all other config knobs shared), drops each into its own
``results/<timestamp>_<model_slug>/`` directory, and finally produces
a single comparison HTML at ``compare_<timestamp>.html``.

Equivalent to running ``cadgenbench baseline run`` N times with
different ``--model`` values, then ``cadgenbench report compare`` on
the resulting run directories -- but in one step.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from cadgenbench.baseline._cli import (
    _DEFAULT_OUTPUT_REL,
    _discover_fixtures,
    _print_summary,
    _run_all,
)
from cadgenbench.baseline.llm import LLMClient
from cadgenbench.baseline.types import AgentConfig
from cadgenbench.common.baseline_models import (
    DEFAULT_COMPARE_LABELS,
    DEFAULT_COMPARE_MODELS,
)
from cadgenbench.eval.report.compare_runs import (
    _discover_run,
    _run_label,
    generate_html,
)

# The default LLM trio for ``compare-llms`` (and the orchestrator's
# ``run_baselines`` fan-out) lives in ``cadgenbench.common.baseline_models``
# so it can be imported without the heavy ``[baseline]`` extras. Re-exported
# here to keep the existing ``compare_llms.DEFAULT_COMPARE_*`` import sites
# working. ``__all__`` advertises them as part of this module's surface.
__all__ = ["DEFAULT_COMPARE_MODELS", "DEFAULT_COMPARE_LABELS"]


def _shutdown_child_pools() -> None:
    """Force-close leaked module-global worker pools in this child process.

    ``run_agent`` lazily creates a module-global ``_RENDER_POOL``
    (``ProcessPoolExecutor``) and the validity gate a ``_MESH_POOL``
    (``mp.Pool``); neither is ever shut down. When models run under the
    outer ``ProcessPoolExecutor`` here, those leaked pools keep the model
    child process alive at teardown (a render abandoned at the wall-clock
    timeout makes the implicit join hang), so the parent's ``as_completed``
    blocks forever and the comparison HTML never gets generated. Tearing
    them down (and terminating any lingering workers) lets the child exit
    promptly so its future resolves. Best-effort: never raises.
    """
    try:
        from cadgenbench.baseline import agent as _agent
        pool = getattr(_agent, "_RENDER_POOL", None)
        if pool is not None:
            for proc in list((getattr(pool, "_processes", None) or {}).values()):
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            _agent._RENDER_POOL = None
    except Exception:  # noqa: BLE001
        pass
    try:
        from cadgenbench.common import validity as _validity
        mesh_pool = getattr(_validity, "_MESH_POOL", None)
        if mesh_pool is not None:
            try:
                mesh_pool.terminate()
            except Exception:  # noqa: BLE001
                pass
            _validity._MESH_POOL = None
    except Exception:  # noqa: BLE001
        pass


def _terminate_pool_workers(pool: ProcessPoolExecutor) -> None:
    """Hard-kill a model pool's worker processes without joining them.

    A safety net for the comparison parent: if a model child somehow fails to
    return its future (a teardown deadlock), the ``with`` block's implicit
    ``shutdown(wait=True)`` would re-hang on the wedged child. We instead
    ``terminate()`` the worker processes directly so the parent always makes
    progress to the HTML step. Best-effort; never raises.
    """
    procs = getattr(pool, "_processes", None) or {}
    for proc in list(procs.values()):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def _model_pool_backstop_s(
    *,
    n_fixtures: int,
    parallel: int,
    max_duration_s: float,
    llm_timeout: float,
    runner_timeout: float,
) -> float:
    """Generous wall-clock ceiling for a single model's full fixture run.

    Sized so it only ever trips on a genuine teardown hang, never on a slow-
    but-healthy run: per-fixture cap (wall-clock budget plus one stuck LLM/
    script call plus render/eval/upload slack), times the number of
    sequential fixture waves, times a safety factor, plus a fixed
    floor. Models run concurrently, so this also bounds the whole step.
    """
    per_fixture = max_duration_s + max(llm_timeout, runner_timeout) + 300.0
    waves = math.ceil(max(1, n_fixtures) / max(1, parallel))
    return per_fixture * waves * 2.0 + 600.0


def _run_model_process(
    *,
    idx: int,
    total_models: int,
    model: str,
    tasks: list[dict],
    output_dir: Path,
    base_timestamp: str,
    parallel: int,
    config_kwargs: dict,
) -> tuple[int, str, Path, list[tuple[str, object]]]:
    """Run one model's full fixture set in a separate process."""
    config = AgentConfig(model=model, **config_kwargs)
    model_slug = model.split("/")[-1]
    run_dir = output_dir / f"{base_timestamp}_{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "params.json").write_text(json.dumps({
        "timestamp": base_timestamp,
        "fixtures": [t["name"] for t in tasks],
        "config": asdict(config),
        "parallel": parallel,
    }, indent=2))
    try:
        results = _run_all(
            tasks,
            config=config,
            run_dir=run_dir,
            parallel=parallel,
        )
    finally:
        # Close leaked render/mesh pools so this child exits cleanly and the
        # parent's as_completed loop does not hang waiting on it.
        _shutdown_child_pools()
    return idx, f"[{idx + 1}/{total_models}] {model}", run_dir, results


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench baseline compare-llms`` subcommand."""
    default_models_str = " ".join(DEFAULT_COMPARE_MODELS)
    p = subparsers.add_parser(
        "compare-llms",
        help="Run baseline on N LLMs and emit a comparison HTML.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # use the built-in default trio (Opus 4.7, Gemini 3.1 Pro, GPT-5.5):\n"
            "  cadgenbench baseline compare-llms --all\n\n"
            "  # custom models + labels:\n"
            "  cadgenbench baseline compare-llms --all \\\n"
            "    --models openai/gpt-5.5 anthropic/claude-sonnet-4-6 \\\n"
            "    --label 'GPT-5.5' --label 'Sonnet 4.6'"
        ),
    )

    # AgentConfig defaults are the single source of truth for runtime knobs.
    defaults = AgentConfig()

    # --- Fixture selection -------------------------------------------------
    p.add_argument("fixtures", nargs="*",
                   help="Fixture name(s) from data/inputs/")
    p.add_argument("--all", action="store_true",
                   help="Run all fixtures")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of fixtures to run")

    # --- Models being compared --------------------------------------------
    p.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help=(
            "Two or more LiteLLM model strings to compare. "
            f"Defaults to: {default_models_str}"
        ),
    )
    p.add_argument(
        "--label", action="append", dest="labels", metavar="LABEL",
        help=(
            "Custom label per --models entry (repeat once per model, in order). "
            "Defaults match the default --models trio."
        ),
    )

    # --- Output -----------------------------------------------------------
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Comparison HTML output path (default: compare_<timestamp>.html)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Per-model results dir (default: ./{_DEFAULT_OUTPUT_REL}/)")

    # --- Shared AgentConfig knobs (applied uniformly to every model) ------
    p.add_argument("--max-iter", type=int, default=defaults.max_iterations,
                   help=f"Max agent turns (default: {defaults.max_iterations})")
    p.add_argument("--max-tokens", type=int, default=defaults.max_total_tokens,
                   help=f"Max total tokens per fixture (default: {defaults.max_total_tokens})")
    p.add_argument("--max-tokens-per-call", type=int, default=defaults.max_tokens,
                   help=f"Max completion tokens per LLM call (default: {defaults.max_tokens})")
    p.add_argument("--max-duration", type=float, default=defaults.max_duration_s,
                   help=f"Max wall-clock seconds per fixture (default: {defaults.max_duration_s:.0f})")
    p.add_argument("--llm-timeout", type=float, default=defaults.llm_timeout,
                   help=f"Per-LLM-call timeout in seconds (default: {defaults.llm_timeout:.0f})")
    p.add_argument("--temperature", type=float, default=defaults.temperature)
    p.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"],
        default=defaults.reasoning_effort,
        help="Cross-provider reasoning/thinking budget (shared across models).",
    )
    p.add_argument("--timeout", type=int, default=defaults.runner_timeout,
                   help=f"Per-script execution timeout (default: {defaults.runner_timeout}s)")
    p.add_argument("--parallel", type=int, default=3, metavar="N",
                   help="Fixtures in parallel within each model's run "
                        "(default: 3).")

    p.set_defaults(handler=run)


# ---------------------------------------------------------------------------
# Subcommand entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench baseline compare-llms``."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Apply the default trio when neither --models nor --label is supplied.
    # If the user passes --models but no --label, fall through and let the
    # downstream auto-label kick in (run timestamp + model slug).
    if args.models is None:
        args.models = list(DEFAULT_COMPARE_MODELS)
        if not args.labels:
            args.labels = list(DEFAULT_COMPARE_LABELS)

    if len(args.models) < 2:
        print("Need at least 2 --models entries to compare.", file=sys.stderr)
        return 2

    if args.labels and len(args.labels) != len(args.models):
        print(
            f"Got {len(args.labels)} --label flags but {len(args.models)} --models.",
            file=sys.stderr,
        )
        return 2

    from cadgenbench.common.paths import (
        data_inputs_dir as _data_inputs_dir,
        data_gt_dir as _data_gt_dir,
    )
    try:
        data_inputs_dir = _data_inputs_dir()
        data_gt_dir = _data_gt_dir()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    output_dir = args.output_dir if args.output_dir is not None else Path.cwd() / _DEFAULT_OUTPUT_REL

    tasks = _discover_fixtures(
        data_inputs_dir, data_gt_dir,
        names=args.fixtures or None,
        run_all=args.all,
        limit=args.limit,
    )

    # Resolve each model string via LLMClient so unset env-vars surface here
    # (before we burn cycles running anything).
    resolved_models = [LLMClient(model=m).model for m in args.models]
    parallel = max(1, args.parallel)

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Models to compare: {len(resolved_models)}")
    for m in resolved_models:
        print(f"  - {m}")
    print(f"Fixtures: {len(tasks)}")
    print(f"Config: max_iter={args.max_iter}, max_tokens={args.max_tokens}, "
          f"max_duration={args.max_duration:.0f}s")
    if args.reasoning_effort:
        print(f"Reasoning effort: {args.reasoning_effort}")
    if parallel > 1:
        print(f"Parallel: {parallel} fixtures within each model's run")
    print()

    config_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens_per_call,
        "max_total_tokens": args.max_tokens,
        "max_iterations": args.max_iter,
        "max_duration_s": args.max_duration,
        "runner_timeout": args.timeout,
        "llm_timeout": args.llm_timeout,
        "reasoning_effort": args.reasoning_effort,
    }

    # Models run concurrently in separate processes. This avoids the GIL for
    # CPU-heavy steps (e.g., tessellation/eval) while still overlapping LLM IO.
    print(f"Running {len(resolved_models)} models in parallel\n")
    run_dirs: list[Path | None] = [None] * len(resolved_models)

    # Safety net: bound how long we wait on the model pool so a wedged child
    # can never hang the whole comparison. The deterministic render-pool
    # teardown in ``run_agent`` is the real fix; this guarantees the parent
    # still produces an HTML from whatever finished even if a child wedges.
    backstop_s = _model_pool_backstop_s(
        n_fixtures=len(tasks),
        parallel=parallel,
        max_duration_s=args.max_duration,
        llm_timeout=args.llm_timeout,
        runner_timeout=args.timeout,
    )

    pool = ProcessPoolExecutor(max_workers=len(resolved_models))
    try:
        futures = {
            pool.submit(
                _run_model_process,
                idx=i,
                total_models=len(resolved_models),
                model=m,
                tasks=tasks,
                output_dir=output_dir,
                base_timestamp=base_timestamp,
                parallel=parallel,
                config_kwargs=config_kwargs,
            ): i
            for i, m in enumerate(resolved_models)
        }
        pending = set(futures)
        try:
            for future in as_completed(futures, timeout=backstop_s):
                pending.discard(future)
                i = futures[future]
                try:
                    idx, label, run_dir, results = future.result()
                    print(f"=== {label} done ===", flush=True)
                    _print_summary(results)
                    run_dirs[idx] = run_dir
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Model %s raised; excluding it from the comparison",
                        resolved_models[i],
                    )
        except TimeoutError:
            abandoned = [resolved_models[futures[f]] for f in pending]
            print(
                f"\n⚠️ Backstop: {len(abandoned)} model(s) did not return within "
                f"{backstop_s:.0f}s; force-terminating and proceeding with "
                f"{sum(p is not None for p in run_dirs)} completed run(s). "
                f"Abandoned: {abandoned}",
                file=sys.stderr,
            )
    finally:
        # Never block on a wedged child: hard-kill the workers first (while
        # ``_processes`` is still populated), then cancel queued work without
        # the ``with`` block's blocking join.
        _terminate_pool_workers(pool)
        pool.shutdown(wait=False, cancel_futures=True)

    run_dirs = [p for p in run_dirs if p is not None]
    if not run_dirs:
        print("All model runs failed; nothing to compare.", file=sys.stderr)
        return 1

    # --- Comparison HTML --------------------------------------------------
    print(f"\n=== Generating comparison HTML across {len(run_dirs)} runs ===")
    runs = [_discover_run(p) for p in run_dirs]
    labels = args.labels or [_run_label(r, all_runs=runs) for r in runs]

    fixture_sets = [set(r["fixtures"].keys()) for r in runs]
    common_names = sorted(set.intersection(*fixture_sets))
    entries = [
        {"name": name, "runs": [r["fixtures"].get(name) for r in runs]}
        for name in common_names
    ]

    if not entries:
        print("No common fixtures across runs; nothing to compare.", file=sys.stderr)
        return 1

    html_out = generate_html(runs, labels, entries, mode="intersection")
    out_path = args.output if args.output else Path.cwd() / f"compare_{base_timestamp}.html"
    out_path.write_text(html_out)
    print(
        f"Wrote {out_path} ({len(entries)} fixtures, "
        f"{out_path.stat().st_size // 1024} KB)",
    )
    return 0
