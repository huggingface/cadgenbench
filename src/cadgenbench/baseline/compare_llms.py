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
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from cadgenbench.eval.report.compare_runs import (
    _discover_run,
    _run_label,
    generate_html,
)


# Three-way default LLM trio for ``compare-llms`` when ``--models`` is
# omitted: current flagship from each of Anthropic, Google, OpenAI as of
# May 2026. Override with ``--models`` to pick something else. Kept here
# (not in default_config.yaml) because it's specific to the compare-llms
# subcommand, not a general AgentConfig knob.
DEFAULT_COMPARE_MODELS: tuple[str, ...] = (
    "anthropic/claude-opus-4-7",
    "gemini/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
)
DEFAULT_COMPARE_LABELS: tuple[str, ...] = (
    "Claude Opus 4.7",
    "Gemini 3.1 Pro",
    "GPT-5.5",
)


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
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Fixtures in parallel within each model's run.")
    p.add_argument("--fixture-retries", type=int, default=1, metavar="N",
                   help="Retries per fixture on exhausted LLM retries (default: 1).")

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
    fixture_retries = max(0, args.fixture_retries)

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

    def _run_model(idx: int, model: str) -> Path:
        print(f"=== [{idx + 1}/{len(resolved_models)}] {model} starting ===", flush=True)
        config = AgentConfig(
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens_per_call,
            max_total_tokens=args.max_tokens,
            max_iterations=args.max_iter,
            max_duration_s=args.max_duration,
            runner_timeout=args.timeout,
            llm_timeout=args.llm_timeout,
            reasoning_effort=args.reasoning_effort,
        )
        model_slug = model.split("/")[-1]
        run_dir = output_dir / f"{base_timestamp}_{model_slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "params.json").write_text(json.dumps({
            "timestamp": base_timestamp,
            "fixtures": [t["name"] for t in tasks],
            "config": asdict(config),
            "parallel": parallel,
        }, indent=2))
        results = _run_all(
            tasks, config=config, run_dir=run_dir,
            parallel=parallel, fixture_retries=fixture_retries,
        )
        print(f"=== [{idx + 1}/{len(resolved_models)}] {model} done ===", flush=True)
        _print_summary(results)
        return run_dir

    # Models run concurrently: each is a separate provider/subscription, so
    # there's no shared rate limit, and wall-clock is dominated by LLM latency
    # (the GIL is released during the network wait). Mirrors the
    # ThreadPoolExecutor that _run_all already uses for fixtures.
    print(f"Running {len(resolved_models)} models in parallel\n")
    run_dirs: list[Path | None] = [None] * len(resolved_models)
    with ThreadPoolExecutor(max_workers=len(resolved_models)) as pool:
        futures = {
            pool.submit(_run_model, i, m): i
            for i, m in enumerate(resolved_models)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                run_dirs[i] = future.result()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Model %s raised; excluding it from the comparison",
                    resolved_models[i],
                )
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
