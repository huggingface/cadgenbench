"""``cadgenbench evaluate`` subcommand handler.

(Re)compute metrics for every fixture in one or more result directories.
Each fixture is passed through :func:`cadgenbench.eval.evaluate.evaluate_result`,
which aligns the candidate STEP (reusing a cached ``aligned/output_aligned.step``
when available), fills in any missing renders, and rewrites ``gt_metrics``,
``validation``, and ``interface_metrics`` inside ``result.json``.

Same code path the live agent loop calls at the end of each run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


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

    total_failures = 0
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            print(f"Not a directory: {run_dir}", file=sys.stderr)
            total_failures += 1
            continue
        total_failures += _process_run(run_dir, data_gt_dir, force=args.force)

    return 0 if total_failures == 0 else 1


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


def _process_run(run_dir: Path, data_gt_dir: Path, *, force: bool) -> int:
    """Re-evaluate every fixture under *run_dir* and write the run summary.

    Discovers fixtures by directory rather than by existing ``result.json``,
    so fixtures that produced no STEP still get a fresh ``status="missing"``
    entry written. Returns the failure count.
    """
    from cadgenbench.eval.evaluate import evaluate_result  # noqa: PLC0415
    from cadgenbench.eval.run_summary import write_run_summary  # noqa: PLC0415

    fixtures = sorted(d for d in run_dir.iterdir() if d.is_dir())
    if not fixtures:
        print(f"No fixture directories under {run_dir}", file=sys.stderr)
        return 1

    print(f"\n=== {run_dir} ({len(fixtures)} fixtures) ===")
    failures = 0
    for fixture in fixtures:
        name = fixture.name
        gt_dir = _gt_dir_for(data_gt_dir, name)
        if gt_dir is None:
            print(f"  {name}: no GT sources, skipping")
            continue
        try:
            scores = evaluate_result(fixture, gt_dir, force_align=force)
            parts = [_format_scores(scores)]
            interface = _format_interface_metrics(fixture / "result.json")
            if interface:
                parts.append(interface)
            print(f"  {name}: {', '.join(parts)}")
        except Exception as exc:
            failures += 1
            print(f"  {name}: FAILED ({exc})", file=sys.stderr)
            logging.getLogger(__name__).exception(
                "evaluate_result failed for %s", fixture,
            )

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
