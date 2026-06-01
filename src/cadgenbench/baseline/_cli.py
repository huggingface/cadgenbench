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

"""``cadgenbench baseline run`` subcommand handler.

Run the cadgenbench baseline agent on one or more benchmark fixtures.

The handler reads fixtures from ``data/inputs/<fixture>/`` and ``data/gt/<fixture>/``
in the current working directory, runs the LLM agent loop on each, and writes
results to ``results/<timestamp>_<model_slug>/<fixture>/``.

Usage::

    cadgenbench baseline run jig-01-single-hole-plate
    cadgenbench baseline run jig-01-single-hole-plate \\
        --model anthropic/claude-sonnet-4-6
    cadgenbench baseline run --all --parallel 4
    cadgenbench baseline run --all --max-tokens 100000 --max-iter 50
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

import yaml

from cadgenbench.baseline import AgentConfig, AgentResult, run_agent
from cadgenbench.baseline.llm import LLMClient
from cadgenbench.eval.evaluate import evaluate_candidate_only, evaluate_result
from cadgenbench.eval.shape_similarity import METRIC_DISPLAY

# Output dir is always relative to cwd (the repo root).  Users that need a
# different location pass --output-dir.
_DEFAULT_OUTPUT_REL = Path("results")
GT_STEP_NAME = "ground_truth.step"

# Per-fixture display knob: which gt_metrics keys make it into the live
# stdout summary line.  Everything else is diagnostic and lives in
# result.json.
DISPLAYED_METRICS: tuple[str, ...] = (
    "cad_score",
    "shape_similarity_score",
    "shape_point_cloud_f1",
    "shape_volume_iou",
    "shape_feature_edge_f1",
)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench baseline run`` subcommand."""
    p = subparsers.add_parser(
        "run",
        help="Run the baseline agent on one or more fixtures.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s jig-01-single-hole-plate
              %(prog)s --all --parallel 4
              %(prog)s --all --limit 3
              %(prog)s --all --max-tokens 100000 --max-iter 50
        """),
    )

    # Argparse defaults come straight from AgentConfig(), which itself reads
    # default_config.yaml at import time.  Single source of truth.
    defaults = AgentConfig()

    p.add_argument("fixtures", nargs="*",
                   help="Fixture name(s) from data/inputs/")
    p.add_argument("--all", action="store_true",
                   help="Run all fixtures")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of fixtures to run")
    p.add_argument("--max-iter", type=int, default=defaults.max_iterations,
                   help=f"Max agent turns (default: {defaults.max_iterations})")
    p.add_argument("--max-tokens", type=int, default=defaults.max_total_tokens,
                   help=f"Max total tokens (default: {defaults.max_total_tokens})")
    p.add_argument("--max-tokens-per-call", type=int, default=defaults.max_tokens,
                   help=(f"Max completion tokens per LLM call (default: "
                         f"{defaults.max_tokens}). Increase for reasoning models "
                         "that consume the output budget on thinking."))
    p.add_argument("--max-duration", type=float, default=defaults.max_duration_s,
                   help=f"Max wall-clock seconds (default: {defaults.max_duration_s:.0f})")
    p.add_argument("--llm-timeout", type=float, default=defaults.llm_timeout,
                   help=f"Per-LLM-call timeout in seconds (default: {defaults.llm_timeout:.0f})")
    p.add_argument(
        "--fixture-retries", type=int, default=1, metavar="N",
        help=(
            "Number of additional attempts per fixture if it fails from "
            "exhausted LLM retries (default: 1). A fixture is retried from "
            "turn 0, wasting the prior attempt's tokens but surviving "
            "provider outages longer than the per-call retry budget. "
            "Set to 0 to disable fixture-level retry."
        ),
    )
    p.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Run N fixtures in parallel (default: 1)")
    p.add_argument("--model", default=defaults.model,
                   help="LiteLLM model string")
    p.add_argument("--temperature", type=float, default=defaults.temperature)
    p.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"],
        default=defaults.reasoning_effort,
        help=("Cross-provider reasoning/thinking budget. Maps to OpenAI "
              "reasoning_effort, Anthropic extended-thinking budget_tokens, "
              "and Gemini thinking_budget via LiteLLM. 'high' = max effort."),
    )
    p.add_argument("--timeout", type=int, default=defaults.runner_timeout,
                   help=f"Per-script execution timeout (default: {defaults.runner_timeout}s)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Results directory (default: ./{_DEFAULT_OUTPUT_REL}/)")

    p.set_defaults(handler=run)


# ---------------------------------------------------------------------------
# Subcommand entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench baseline run``."""
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

    parallel = max(1, args.parallel)
    fixture_retries = max(0, args.fixture_retries)
    resolved_model = LLMClient(model=args.model).model

    config = AgentConfig(
        model=resolved_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens_per_call,
        max_total_tokens=args.max_tokens,
        max_iterations=args.max_iter,
        max_duration_s=args.max_duration,
        runner_timeout=args.timeout,
        llm_timeout=args.llm_timeout,
        reasoning_effort=args.reasoning_effort,
    )

    print(f"Fixtures: {', '.join(t['name'] for t in tasks)}")
    print(f"Config: max_iter={config.max_iterations}, max_tokens={config.max_total_tokens}, "
          f"max_duration={config.max_duration_s}s")
    print(f"Model: {config.model}")
    if config.reasoning_effort:
        print(f"Reasoning effort: {config.reasoning_effort}")
    if parallel > 1:
        print(f"Parallel: {parallel} fixtures")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = resolved_model.split("/")[-1]
    run_name = f"{timestamp}_{model_slug}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_params = {
        "timestamp": timestamp,
        "fixtures": [t["name"] for t in tasks],
        "config": asdict(config),
        "parallel": parallel,
    }
    (run_dir / "params.json").write_text(json.dumps(run_params, indent=2))

    all_results = _run_all(
        tasks, config=config, run_dir=run_dir,
        parallel=parallel, fixture_retries=fixture_retries,
    )
    _print_summary(all_results)
    return 0


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------

def _discover_fixtures(
    data_inputs_dir: Path,
    data_gt_dir: Path,
    *,
    names: list[str] | None = None,
    run_all: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Find fixtures under ``data/inputs/<fixture>/description.yaml``."""
    fixtures = []
    for desc_path in sorted(data_inputs_dir.glob("*/description.yaml")):
        data = yaml.safe_load(desc_path.read_text())
        name = desc_path.parent.name
        data["name"] = name
        data["_inputs_dir"] = desc_path.parent
        data["_gt_dir"] = data_gt_dir / name
        fixtures.append(data)

    if names:
        available = {f["name"] for f in fixtures}
        unknown = set(names) - available
        if unknown:
            print(f"Unknown fixtures: {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(available))}", file=sys.stderr)
            sys.exit(1)
        fixtures = [f for f in fixtures if f["name"] in names]

    if not run_all and not names:
        print("Specify fixture name(s) or --all to select fixtures.", file=sys.stderr)
        sys.exit(1)

    if not fixtures:
        print("No fixtures matched the given filters.", file=sys.stderr)
        sys.exit(1)

    if limit is not None and len(fixtures) > limit:
        print(f"Limiting to {limit} of {len(fixtures)} fixtures.", file=sys.stderr)
        fixtures = fixtures[:limit]

    return fixtures


# ---------------------------------------------------------------------------
# Per-fixture run + evaluation
# ---------------------------------------------------------------------------

def _find_candidate_artifact(result: AgentResult) -> Path | None:
    """Find the agent's STEP output in the work dir, if any."""
    if not result.work_dir:
        return None
    for name in ("output.step", "output.stp"):
        candidate = result.work_dir / name
        if candidate.exists():
            return candidate
    return None


def _print_metric_summary(data: dict, *, printer) -> None:
    """Print CAD score + per-component scores + interface + validity for one fixture."""
    for line in _metric_summary_lines(data):
        printer(line)


def _metric_summary_lines(data: dict) -> list[str]:
    """Build summary lines for one fixture's metric output."""
    lines: list[str] = []
    validation = data.get("validation") or {}
    is_valid = validation.get("is_valid")
    is_watertight = validation.get("is_watertight")
    lines.append(f"  CAD validity: valid={is_valid}, watertight={is_watertight}")

    gt_metrics = data.get("gt_metrics") or {}
    cad_score = data.get("cad_score")
    parts: list[str] = []
    if cad_score is not None:
        meta = METRIC_DISPLAY.get("cad_score")
        parts.append(
            f"{meta.label}={format(cad_score, meta.fmt)}{meta.suffix}"
            if meta else f"cad_score={cad_score:.3f}",
        )
    for key in DISPLAYED_METRICS:
        if key == "cad_score":
            continue
        value = gt_metrics.get(key)
        if value is None:
            continue
        meta = METRIC_DISPLAY.get(key)
        parts.append(
            f"{meta.label}={format(value, meta.fmt)}{meta.suffix}"
            if meta else f"{key}={value:.3f}",
        )
    if parts:
        lines.append(f"  GT metrics: {', '.join(parts)}")

    interface = data.get("interface_metrics") or {}
    iface_score = interface.get("score")
    if iface_score is not None:
        ctx_count = len(interface.get("contexts") or {})
        lines.append(f"  Interface: score={iface_score:.3f} (contexts={ctx_count})")

    topology = data.get("topology_metrics") or {}
    topo_score = topology.get("score")
    if topo_score is not None:
        cand = topology.get("candidate") or {}
        gt = topology.get("gt") or {}
        cand_sig = f"({cand.get('b0')}, {cand.get('b1')}, {cand.get('b2')})"
        gt_sig = f"({gt.get('b0')}, {gt.get('b1')}, {gt.get('b2')})"
        lines.append(
            f"  Topo: score={topo_score:.3f} "
            f"betti(cand)={cand_sig} vs betti(gt)={gt_sig}",
        )
    return lines


def _run_one_task(
    task: dict,
    *,
    config: AgentConfig,
    run_dir: Path,
) -> tuple[str, AgentResult, list[str]]:
    """Run one fixture and return its logs plus baseline result."""
    name = task["name"]
    description = task["description"]
    task_type = task.get("task_type", "generation")

    input_files = None
    raw_files = task.get("input_files")
    if raw_files:
        inputs_dir = task["_inputs_dir"]
        input_files = [Path(inputs_dir) / f for f in raw_files]

    logs = [
        f"\n--- {name} ({task_type}) ---",
        f"Task: {description[:100]}...",
    ]
    if input_files:
        logs.append(f"Input files: {', '.join(f.name for f in input_files)}")

    out_dir = run_dir / name
    result = run_agent(
        description,
        config=config,
        input_files=input_files,
        output_dir=out_dir,
    )
    logs.append(f"Results saved to: {out_dir}")

    candidate = _find_candidate_artifact(result)
    gt_dir = Path(task["_gt_dir"]) if task.get("_gt_dir") else None
    gt_step = gt_dir / GT_STEP_NAME if gt_dir else None

    if gt_step and gt_step.exists() and candidate is not None:
        evaluate_result(out_dir, gt_dir, candidate_step=candidate)
        data = json.loads((out_dir / "result.json").read_text())
        logs.extend(_metric_summary_lines(data))
    elif candidate is not None:
        evaluate_candidate_only(candidate, out_dir)
    else:
        logs.append("  Evaluation skipped: no output.step produced")

    return name, result, logs


def _run_one_with_retry(
    task: dict,
    *,
    config: AgentConfig,
    run_dir: Path,
    fixture_retries: int,
) -> tuple[str, AgentResult, list[str]]:
    """Run fixture; on exhausted LLM retries, restart from turn 0."""
    name = task["name"]
    attempts = fixture_retries + 1
    retry_logs: list[str] = []
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            fixture_name, result, logs = _run_one_task(
                task,
                config=config,
                run_dir=run_dir,
            )
            return fixture_name, result, [*retry_logs, *logs]
        except RuntimeError as exc:
            if "LLM call failed after" not in str(exc):
                raise
            last_exc = exc
            if attempt < attempts:
                retry_logs.append(
                    f"\n--- {name} fixture retry "
                    f"({attempt}/{fixture_retries}) after LLM "
                    f"exhaustion; restarting from turn 0 ---",
                )
    assert last_exc is not None
    raise last_exc


def _run_all(
    tasks: list[dict],
    *,
    config: AgentConfig,
    run_dir: Path,
    parallel: int,
    fixture_retries: int,
) -> list[tuple[str, AgentResult]]:
    """Execute every task, sequentially or in parallel, with per-fixture retry."""
    all_results: list[tuple[str, AgentResult]] = []
    if parallel <= 1:
        for task in tasks:
            try:
                name, result, logs = _run_one_with_retry(
                    task,
                    config=config,
                    run_dir=run_dir,
                    fixture_retries=fixture_retries,
                )
                for line in logs:
                    print(line)
                all_results.append((name, result))
            except Exception:
                print(f"\n--- {task['name']} FAILED ---")
                logging.getLogger(__name__).exception(
                    "Fixture %s raised an exception", task["name"],
                )
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(
                    _run_one_with_retry,
                    task,
                    config=config,
                    run_dir=run_dir,
                    fixture_retries=fixture_retries,
                ): task["name"]
                for task in tasks
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    fixture_name, result, logs = future.result()
                    for line in logs:
                        print(line)
                    all_results.append((fixture_name, result))
                except Exception:
                    print(f"\n--- {name} FAILED ---")
                    logging.getLogger(__name__).exception(
                        "Fixture %s raised an exception", name,
                    )

    task_order = {t["name"]: i for i, t in enumerate(tasks)}
    all_results.sort(key=lambda r: task_order.get(r[0], 0))

    try:
        from cadgenbench.eval.run_summary import write_run_summary  # noqa: PLC0415
        summary_path = write_run_summary(run_dir)
        print(f"Wrote {summary_path.name}")
    except Exception:
        logging.getLogger(__name__).exception(
            "write_run_summary failed for %s", run_dir,
        )

    return all_results


def _print_summary(results: list[tuple[str, AgentResult]]) -> None:
    """Print a comparison table to stdout."""
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
