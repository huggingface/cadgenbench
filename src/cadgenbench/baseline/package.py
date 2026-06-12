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

"""``cadgenbench baseline package`` subcommand handler.

Turn a baseline run directory (as produced by ``cadgenbench baseline run``)
into a leaderboard-ready submission zip, so a local user can run a baseline
and submit it without hand-assembling anything.

The submission contract (mirrored from the leaderboard's ``submit.py``):

- a top-level ``meta.json`` with keys ``submitter_name``, ``submission_name``,
  ``agent_url``, ``notes``, ``agree_to_publish``;
- one folder per fixture. A folder may omit an ``output.*`` candidate; the evaluator
  records that fixture as ``status="missing"`` and scores it zero.

A baseline run dir already materialises one directory per attempted fixture
(see :meth:`AgentResult.save`), so packaging preserves those directories and
copies any produced candidate files into the zip.

Usage::

    cadgenbench baseline package results/20260602_153012_gpt-5.5
    cadgenbench baseline package <run_dir> --submitter "Jane" \\
        --name "my agent v2" --agree -o my_submission.zip
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Mirror of cadgenbench-leaderboard/submit.py REQUIRED_META_KEYS. Kept inline
# (not imported) because the leaderboard is a separate repo; the contract is
# documented in docs/benchmark/submission.md. STEP/BREP wins when both are
# present, matching evaluator and leaderboard-submit candidate discovery.
_CANDIDATE_NAMES = (
    "output.step",
    "output.stp",
    "output.3mf",
    "output.obj",
    "output.off",
    "output.ply",
    "output.stl",
)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench baseline package`` subcommand."""
    p = subparsers.add_parser(
        "package",
        help="Bundle a baseline run dir into a leaderboard submission zip.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path,
                   help="A baseline run directory (contains <fixture>/output.*).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output zip path (default: <run_dir>.zip next to the run dir).")
    p.add_argument("--submitter", default=None,
                   help="meta.json submitter_name (default: current OS user).")
    p.add_argument("--name", dest="submission_name", default=None,
                   help="meta.json submission_name "
                        "(default: 'HF build123d baseline (<model>)').")
    p.add_argument("--agent-url", default=None,
                   help="meta.json agent_url (optional).")
    p.add_argument("--notes", default=None,
                   help="meta.json notes (optional, <=500 chars).")
    p.add_argument("--agree", action="store_true",
                   help="Set agree_to_publish=true (required before the "
                        "leaderboard accepts the zip). Omitted => false stub.")
    p.set_defaults(handler=run)


def _discover_fixture_entries(run_dir: Path) -> list[tuple[str, Path | None]]:
    """Return ``(fixture_name, output_candidate_path_or_none)`` for each fixture.

    A fixture is any immediate subdirectory. If it has an accepted ``output.*``
    candidate at its root, that file is packaged under the same filename. If it
    has no candidate, the directory is still preserved so the evaluator can
    record ``status="missing"`` and assign zero credit for that fixture.
    """
    found: list[tuple[str, Path | None]] = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        output_candidate: Path | None = None
        for name in _CANDIDATE_NAMES:
            candidate = child / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                output_candidate = candidate
                break
        found.append((child.name, output_candidate))
    return found


def _default_submission_name(run_dir: Path) -> str:
    """Derive a submission name from the run's params.json model, if present."""
    model = None
    params = run_dir / "params.json"
    if params.is_file():
        try:
            model = json.loads(params.read_text()).get("config", {}).get("model")
        except (json.JSONDecodeError, OSError):
            model = None
    label = model or run_dir.name
    return f"HF build123d baseline ({label})"


def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench baseline package``."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s")

    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        return 2

    fixtures = _discover_fixture_entries(run_dir)
    if not fixtures:
        print(
            f"No fixture directories found under {run_dir}. "
            "Did the baseline run write any fixture outputs?",
            file=sys.stderr,
        )
        return 1

    submitter = args.submitter or getpass.getuser()
    submission_name = args.submission_name or _default_submission_name(run_dir)
    meta = {
        "submitter_name": submitter,
        "submission_name": submission_name,
        "agent_url": args.agent_url,
        "notes": args.notes,
        "agree_to_publish": bool(args.agree),
    }

    out_path = args.output or run_dir.with_suffix(".zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta, indent=2) + "\n")
        for fixture_name, candidate in fixtures:
            # Explicit directory entry preserves missing-output fixtures after
            # extraction; without it, an empty fixture folder disappears.
            zf.writestr(f"{fixture_name}/", "")
            if candidate is not None:
                zf.write(candidate, arcname=f"{fixture_name}/{candidate.name}")

    size_kb = out_path.stat().st_size // 1024
    n_with_candidates = sum(1 for _, candidate in fixtures if candidate is not None)
    n_missing = len(fixtures) - n_with_candidates
    print(
        f"Wrote {out_path} ({len(fixtures)} fixtures, "
        f"{n_with_candidates} with candidate, {n_missing} missing, {size_kb} KB)"
    )
    print(f"  submitter_name : {submitter}")
    print(f"  submission_name: {submission_name}")
    print(f"  fixtures       : {', '.join(name for name, _ in fixtures)}")
    if not args.agree:
        print(
            "\n  NOTE: agree_to_publish=false. The leaderboard will reject the "
            "zip until you consent.\n  Re-run with --agree (or edit meta.json) "
            "once you're ready to submit.",
            file=sys.stderr,
        )
    return 0
