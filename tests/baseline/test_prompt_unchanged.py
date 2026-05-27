"""Snapshot test: the default system prompt must remain byte-identical.

A snapshot of the prompt was taken at HEAD immediately before the
multi-backend refactor and stored under
``tests/snapshots/baseline_system_prompt.txt``. The collapse of the backend
abstraction (back to BREP-only) inlined the same prompt strings into
``cadgenbench.baseline.prompt``, so the assembled prompt should still
match byte-for-byte. Any drift fails this test.
"""
from __future__ import annotations

import difflib
from pathlib import Path

from cadgenbench.baseline.prompt import assemble_system_prompt

SNAPSHOT = (
    Path(__file__).resolve().parent.parent
    / "snapshots"
    / "baseline_system_prompt.txt"
)


def _diff(a: str, b: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(), b.splitlines(),
            fromfile="snapshot", tofile="current", lineterm="",
        )
    )


def test_default_prompt_matches_snapshot() -> None:
    current = assemble_system_prompt()
    expected = SNAPSHOT.read_text()
    assert current == expected, _diff(expected, current)
