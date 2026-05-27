"""``cadgenbench`` console script entry point.

Routes ``cadgenbench <subcommand>`` (or its short alias ``cgb``) to the
right handler module. Each subcommand handler exposes
``add_subparser(subparsers)`` and ``run(args) -> int``; this module
collects them, dispatches, and returns the exit code.

Heavy imports (litellm, scipy, manifold3d, ...) live inside the
individual handlers, so ``cadgenbench --help`` stays fast and ``import
cadgenbench`` is light.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``project.scripts.cadgenbench``."""
    parser = argparse.ArgumentParser(
        prog="cadgenbench",
        description="CADGenBench: benchmark for AI-driven CAD generation and editing.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # cadgenbench evaluate <run_dir>
    from cadgenbench.eval._cli import add_subparser as add_evaluate
    add_evaluate(subparsers)

    # cadgenbench baseline run|compare-llms
    baseline_p = subparsers.add_parser(
        "baseline", help="Reference baseline agent commands.",
    )
    baseline_sub = baseline_p.add_subparsers(dest="baseline_action")
    from cadgenbench.baseline._cli import add_subparser as add_baseline_run
    from cadgenbench.baseline.compare_llms import add_subparser as add_baseline_compare_llms
    add_baseline_run(baseline_sub)
    add_baseline_compare_llms(baseline_sub)

    # cadgenbench report single|compare
    report_p = subparsers.add_parser(
        "report", help="HTML reports from result directories.",
    )
    report_sub = report_p.add_subparsers(dest="report_action")
    from cadgenbench.eval.report.single_run import add_subparser as add_report_single
    from cadgenbench.eval.report.compare_runs import add_subparser as add_report_compare
    add_report_single(report_sub)
    add_report_compare(report_sub)

    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        # Either no command at all, or a parent command (e.g. `baseline`,
        # `report`) without its action.
        if args.command is None:
            parser.print_help()
        elif args.command == "baseline":
            baseline_p.print_help()
        elif args.command == "report":
            report_p.print_help()
        else:
            parser.print_help()
        return 0

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
