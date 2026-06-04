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

"""``cadgenbench`` console script entry point.

Routes ``cadgenbench <subcommand>`` (or its short alias ``cgb``) to the
right handler module. Each subcommand handler exposes
``add_subparser(subparsers)`` and ``run(args) -> int``; this module
collects them, dispatches, and returns the exit code.

Heavy imports (litellm, scipy, manifold3d, ...) live inside the
individual handlers, so ``cadgenbench --help`` stays fast and ``import
cadgenbench`` is light.

The ``baseline`` subcommand requires the ``[baseline]`` optional
dependencies (litellm, python-dotenv). Its registration is wrapped in
a try/except so an eval-only install (e.g. the leaderboard Space, which
runs ``cadgenbench evaluate`` against user submissions and has no need
for the reference agent) doesn't fail to import the CLI when those
optional deps are missing.
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

    # cadgenbench baseline run|package (needs [baseline] extra).
    baseline_p = _register_baseline_subcommand(subparsers)

    # cadgenbench report single
    report_p = subparsers.add_parser(
        "report", help="HTML reports from result directories.",
    )
    report_sub = report_p.add_subparsers(dest="report_action")
    from cadgenbench.eval.report.single_run import add_subparser as add_report_single
    add_report_single(report_sub)

    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        # Either no command at all, or a parent command (e.g. `baseline`,
        # `report`) without its action.
        if args.command is None:
            parser.print_help()
        elif args.command == "baseline" and baseline_p is not None:
            baseline_p.print_help()
        elif args.command == "report":
            report_p.print_help()
        else:
            parser.print_help()
        return 0

    return args.handler(args)


def _register_baseline_subcommand(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser | None:
    """Register the ``baseline`` subcommand if its extras are installed.

    The baseline agent transitively imports litellm + python-dotenv,
    which only ship with the ``cadgenbench[baseline]`` extra. An
    eval-only install (e.g. a leaderboard Space) doesn't need them,
    so ``ImportError`` here is non-fatal: the subcommand is simply
    not registered, and ``cadgenbench --help`` won't list it.
    """
    try:
        from cadgenbench.baseline._cli import add_subparser as add_baseline_run
        from cadgenbench.baseline.package import (
            add_subparser as add_baseline_package,
        )
    except ImportError:
        return None

    baseline_p = subparsers.add_parser(
        "baseline", help="Reference baseline agent commands.",
    )
    baseline_sub = baseline_p.add_subparsers(dest="baseline_action")
    add_baseline_run(baseline_sub)
    add_baseline_package(baseline_sub)
    return baseline_p


if __name__ == "__main__":
    sys.exit(main())
