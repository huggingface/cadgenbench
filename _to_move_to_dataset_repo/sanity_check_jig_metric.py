#!/usr/bin/env python
"""Sanity-check interface_match discrimination on committed jig fixtures.

This script runs :func:`cadgenbench.eval.interface_match.interface_score`
over every ``tests/fixtures/jig_metric/test_*`` fixture and asserts:

- ``candidates/correct.step`` scores >= threshold
- every ``candidates/broken_*.step`` scores < threshold

It prints a compact table for quick regression inspection and exits
non-zero on any failed assertion.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from cadgenbench.eval.interface_match import DEFAULT_IOU_THRESHOLD, interface_score

DEFAULT_ROOT = Path("tests/fixtures/jig_metric")


@dataclass(frozen=True)
class Row:
    fixture: str
    candidate: str
    expected: str
    score: float
    ok: bool


def _discover_fixture_dirs(root: Path) -> list[Path]:
    fixtures = sorted(d for d in root.glob("test_*") if d.is_dir())
    if not fixtures:
        raise FileNotFoundError(f"No test_* fixture directories under {root}")
    return fixtures


def _collect_rows(root: Path, threshold: float, n_samples: int, workers: int) -> list[Row]:
    rows: list[Row] = []
    for fixture_dir in _discover_fixture_dirs(root):
        fixture_name = fixture_dir.name
        correct = fixture_dir / "candidates" / "correct.step"
        if not correct.exists():
            raise FileNotFoundError(f"Missing candidate: {correct}")

        correct_score = interface_score(
            correct,
            fixture_dir,
            n_samples=n_samples,
            workers=workers,
        )
        rows.append(
            Row(
                fixture=fixture_name,
                candidate="correct.step",
                expected=f">= {threshold:.2f}",
                score=correct_score,
                ok=correct_score >= threshold,
            ),
        )

        broken_steps = sorted((fixture_dir / "candidates").glob("broken_*.step"))
        if not broken_steps:
            raise FileNotFoundError(f"No broken_*.step candidates in {fixture_dir / 'candidates'}")
        for broken in broken_steps:
            broken_score = interface_score(
                broken,
                fixture_dir,
                n_samples=n_samples,
                workers=workers,
            )
            rows.append(
                Row(
                    fixture=fixture_name,
                    candidate=broken.name,
                    expected=f"< {threshold:.2f}",
                    score=broken_score,
                    ok=broken_score < threshold,
                ),
            )
    return rows


def _print_table(rows: list[Row]) -> None:
    headers = ("fixture", "candidate", "expected", "score", "status")
    fixture_w = max(len(headers[0]), *(len(r.fixture) for r in rows))
    candidate_w = max(len(headers[1]), *(len(r.candidate) for r in rows))
    expected_w = max(len(headers[2]), *(len(r.expected) for r in rows))
    score_w = len(headers[3])
    status_w = len(headers[4])

    line = (
        f"{headers[0]:<{fixture_w}}  "
        f"{headers[1]:<{candidate_w}}  "
        f"{headers[2]:<{expected_w}}  "
        f"{headers[3]:>{score_w}}  "
        f"{headers[4]:<{status_w}}"
    )
    print(line)
    print("-" * len(line))
    for r in rows:
        print(
            f"{r.fixture:<{fixture_w}}  "
            f"{r.candidate:<{candidate_w}}  "
            f"{r.expected:<{expected_w}}  "
            f"{r.score:>{score_w}.3f}  "
            f"{'OK' if r.ok else 'FAIL':<{status_w}}",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Fixture root containing test_* dirs (default: tests/fixtures/jig_metric)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Pass/fail threshold for candidate assertions (default: {DEFAULT_IOU_THRESHOLD})",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Pose samples per context (default: 32)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process count for pose search (default: 1)",
    )
    args = ap.parse_args()

    rows = _collect_rows(
        root=args.root.resolve(),
        threshold=args.threshold,
        n_samples=args.n_samples,
        workers=args.workers,
    )
    _print_table(rows)

    failures = [r for r in rows if not r.ok]
    if failures:
        failing_labels = ", ".join(f"{r.fixture}/{r.candidate}" for r in failures)
        raise AssertionError(f"{len(failures)} sanity checks failed: {failing_labels}")

    print(f"\nAll sanity checks passed ({len(rows)} candidate checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
