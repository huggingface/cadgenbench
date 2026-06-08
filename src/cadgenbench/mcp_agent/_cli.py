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

"""``cadgenbench mcp-agent run`` subcommand handler.

Run the MCP-agent on one or more benchmark fixtures.  Uses the build123d-mcp
server's native tool-calling interface rather than code-block extraction.

Usage::

    cadgenbench mcp-agent run 101
    cadgenbench mcp-agent run --all --parallel 4
    cadgenbench mcp-agent run --all --model openai/gpt-4o --mcp-server uvx --mcp-args build123d-mcp
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from cadgenbench.mcp_agent.types import McpAgentConfig, McpAgentResult
from cadgenbench.eval.evaluate import evaluate_candidate_only, evaluate_result

GT_STEP_NAME = "ground_truth.step"

_DEFAULT_OUTPUT_REL = Path("results")


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench mcp-agent run`` subcommand."""
    p = subparsers.add_parser(
        "run",
        help="Run the MCP-agent on one or more fixtures.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s 101
              %(prog)s --all --parallel 4
              %(prog)s --all --model anthropic/claude-opus-4-8
              %(prog)s 101 --mcp-server uvx --mcp-args build123d-mcp
        """),
    )

    defaults = McpAgentConfig()

    p.add_argument("fixtures", nargs="*", help="Fixture name(s) from data/inputs/")
    p.add_argument("--all", action="store_true", help="Run all fixtures")
    p.add_argument("--limit", type=int, default=None, help="Max number of fixtures to run")
    p.add_argument("--model", default=defaults.model, help="LiteLLM model string")
    p.add_argument("--max-iter", type=int, default=defaults.max_iterations,
                   help=f"Max agent turns (default: {defaults.max_iterations})")
    p.add_argument("--max-tokens", type=int, default=defaults.max_total_tokens,
                   help=f"Max total tokens (default: {defaults.max_total_tokens})")
    p.add_argument("--max-tokens-per-call", type=int, default=defaults.max_tokens,
                   help=f"Max completion tokens per LLM call (default: {defaults.max_tokens})")
    p.add_argument("--max-duration", type=float, default=defaults.max_duration_s,
                   help=f"Max wall-clock seconds (default: {defaults.max_duration_s:.0f})")
    p.add_argument("--llm-timeout", type=float, default=defaults.llm_timeout,
                   help=f"Per-LLM-call timeout in seconds (default: {defaults.llm_timeout:.0f})")
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Run N fixtures in parallel (default: 1)")
    p.add_argument("--mcp-server", default=defaults.mcp_server_command,
                   help=(f"Command to launch the build123d-mcp server "
                         f"(default: {defaults.mcp_server_command!r}). "
                         "Use 'uvx' with --mcp-args build123d-mcp for uvx installs."))
    p.add_argument("--mcp-args", nargs="*", default=defaults.mcp_server_args,
                   help="Extra args passed to the MCP server command")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Results directory (default: ./{_DEFAULT_OUTPUT_REL}/)")

    p.set_defaults(handler=run)


# ---------------------------------------------------------------------------
# Subcommand entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench mcp-agent run``."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    from cadgenbench.common.paths import (
        data_inputs_dir as _data_inputs_dir,
        data_gt_dir as _data_gt_dir,
    )

    try:
        data_inputs_dir = _data_inputs_dir()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        data_gt_dir = _data_gt_dir()
    except FileNotFoundError:
        data_gt_dir = None
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "Ground-truth dataset not accessible (%s: %s); skipping local scoring.",
            type(e).__name__, e,
        )
        data_gt_dir = None

    output_dir = args.output_dir or Path.cwd() / _DEFAULT_OUTPUT_REL

    from cadgenbench.baseline._cli import _discover_fixtures  # lazy: needs litellm
    tasks = _discover_fixtures(
        data_inputs_dir, data_gt_dir,
        names=args.fixtures or None,
        run_all=args.all,
        limit=args.limit,
    )

    config = McpAgentConfig(
        model=args.model,
        max_tokens=args.max_tokens_per_call,
        max_total_tokens=args.max_tokens,
        max_iterations=args.max_iter,
        max_duration_s=args.max_duration,
        llm_timeout=args.llm_timeout,
        mcp_server_command=args.mcp_server,
        mcp_server_args=args.mcp_args or [],
    )

    parallel = max(1, args.parallel)

    print(f"Fixtures: {', '.join(t['name'] for t in tasks)}")
    print(f"Model:    {config.model}")
    print(f"MCP:      {config.mcp_server_command} {' '.join(config.mcp_server_args)}")
    print(f"Config:   max_iter={config.max_iterations}, max_tokens={config.max_total_tokens}, "
          f"max_duration={config.max_duration_s}s")
    if parallel > 1:
        print(f"Parallel: {parallel} fixtures")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = config.model.split("/")[-1]
    run_dir = output_dir / f"{timestamp}_{model_slug}_mcp"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "params.json").write_text(json.dumps({
        "timestamp": timestamp,
        "fixtures": [t["name"] for t in tasks],
        "config": asdict(config),
        "parallel": parallel,
        "agent": "mcp",
    }, indent=2))

    all_results = _run_all(tasks, config=config, run_dir=run_dir, parallel=parallel)
    _print_summary(all_results)
    return 0


# ---------------------------------------------------------------------------
# Per-fixture run + evaluation
# ---------------------------------------------------------------------------

def _run_one_task(
    task: dict,
    *,
    config: McpAgentConfig,
    run_dir: Path,
) -> tuple[str, McpAgentResult, list[str]]:
    name = task["name"]
    description = task["description"]
    task_type = task.get("task_type", "generation")

    input_files = None
    raw_files = task.get("input_files")
    if raw_files:
        inputs_dir = Path(task["_inputs_dir"])
        input_files = [inputs_dir / f for f in raw_files]

    logs = [
        f"\n--- {name} ({task_type}) ---",
        f"Task: {description[:100]}…",
    ]
    if input_files:
        logs.append(f"Input files: {', '.join(f.name for f in input_files)}")

    from cadgenbench.mcp_agent.agent import run_mcp_agent  # lazy: needs litellm+mcp
    out_dir = run_dir / name
    result = run_mcp_agent(
        description,
        config=config,
        input_files=input_files,
        output_dir=out_dir,
    )
    logs.append(f"Results: {out_dir}")
    logs.append(
        f"  Turns: {len(result.turns)}, tokens: {result.total_tokens}, "
        f"stop: {result.stopped_reason}, done: {result.completed}"
    )

    candidate = result.candidate_step
    gt_dir = Path(task["_gt_dir"]) if task.get("_gt_dir") else None
    gt_step = gt_dir / GT_STEP_NAME if gt_dir else None

    if gt_step and gt_step.exists() and candidate and candidate.exists():
        evaluate_result(out_dir, gt_dir, candidate_step=candidate)
        data = json.loads((out_dir / "result.json").read_text())
        from cadgenbench.baseline._cli import _metric_summary_lines  # lazy
        logs.extend(_metric_summary_lines(data))
    elif candidate and candidate.exists():
        evaluate_candidate_only(candidate, out_dir)
    else:
        logs.append("  Evaluation skipped: no output.step produced")

    return name, result, logs


def _run_all(
    tasks: list[dict],
    *,
    config: McpAgentConfig,
    run_dir: Path,
    parallel: int,
) -> list[tuple[str, McpAgentResult]]:
    all_results: list[tuple[str, McpAgentResult]] = []

    if parallel <= 1:
        for task in tasks:
            try:
                name, result, logs = _run_one_task(task, config=config, run_dir=run_dir)
                for line in logs:
                    print(line)
                all_results.append((name, result))
            except Exception:
                print(f"\n--- {task['name']} FAILED ---")
                logging.getLogger(__name__).exception(
                    "Fixture %s raised an exception", task["name"]
                )
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_run_one_task, task, config=config, run_dir=run_dir): task["name"]
                for task in tasks
            }
            for future in as_completed(futures):
                fixture_name = futures[future]
                try:
                    name, result, logs = future.result()
                    for line in logs:
                        print(line)
                    all_results.append((name, result))
                except Exception:
                    print(f"\n--- {fixture_name} FAILED ---")
                    logging.getLogger(__name__).exception(
                        "Fixture %s raised an exception", fixture_name
                    )

    task_order = {t["name"]: i for i, t in enumerate(tasks)}
    all_results.sort(key=lambda r: task_order.get(r[0], 0))

    try:
        from cadgenbench.eval.run_summary import write_run_summary
        summary_path = write_run_summary(run_dir)
        print(f"Wrote {summary_path.name}")
    except Exception:
        logging.getLogger(__name__).exception("write_run_summary failed for %s", run_dir)

    return all_results


def _print_summary(results: list[tuple[str, McpAgentResult]]) -> None:
    header = f"{'fixture':<25} {'turns':>5} {'tokens':>8} {'time':>7} {'stop':>12} {'done':>5}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, result in results:
        done = "yes" if result.completed else "no"
        print(
            f"{name:<25} {len(result.turns):>5} "
            f"{result.total_tokens:>8} "
            f"{result.total_duration_s:>6.1f}s {result.stopped_reason:>12} {done:>5}"
        )
    print("=" * len(header) + "\n")
