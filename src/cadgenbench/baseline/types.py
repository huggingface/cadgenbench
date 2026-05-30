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

"""Data types for the cadgenbench baseline agent.

Defines the configuration, per-execution / per-turn records, and the
aggregate result object persisted to ``result.json``.

Default values for :class:`AgentConfig` live in
``default_config.yaml`` next to this module: that file is the single
source of truth for the baseline's tuneable parameters. The CLI in
:mod:`cadgenbench.baseline._cli` reads the same dataclass for argparse
defaults, so the dataclass and the CLI cannot drift.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Load defaults from the YAML next to this module. Done once at import
# time; values become AgentConfig field defaults below.
_DEFAULTS: dict[str, Any] = yaml.safe_load(
    (Path(__file__).parent / "default_config.yaml").read_text()
)


def _find_latest_turn_step(fixture_dir: Path) -> Path | None:
    """Return the highest-numbered ``turn_<N>/output.step`` (or ``.stp``).

    Mirrors the selection logic the evaluator used before becoming
    turn-unaware (see :func:`cadgenbench.eval.evaluate._find_candidate_step`
    pre-refactor): of all per-iteration STEP snapshots the baseline kept,
    the latest-turn one wins. Used to materialise the canonical candidate
    at the fixture root so historical scoring is preserved across the
    refactor.
    """
    found: list[tuple[int, Path]] = []
    for td in fixture_dir.iterdir():
        if not (td.is_dir() and td.name.startswith("turn_")):
            continue
        try:
            idx = int(td.name.split("_", 1)[1])
        except ValueError:
            continue
        for name in ("output.step", "output.stp"):
            p = td / name
            if p.exists():
                found.append((idx, p))
                break
    if not found:
        return None
    found.sort(reverse=True)
    return found[0][1]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """All tuneable knobs for an agent run.

    Defaults come from ``default_config.yaml``. Every field is saved to
    ``params.json`` for reproducibility.

    ``reasoning_effort`` is a cross-provider knob. LiteLLM maps it to
    OpenAI's ``reasoning_effort``, Anthropic's extended-thinking
    ``budget_tokens``, and Gemini's ``thinking_config.thinking_budget``.
    ``None`` means "provider default" (no override).
    """

    model: str | None = _DEFAULTS["model"]
    temperature: float = _DEFAULTS["temperature"]
    max_tokens: int = _DEFAULTS["max_tokens"]
    max_total_tokens: int = _DEFAULTS["max_total_tokens"]
    max_iterations: int = _DEFAULTS["max_iterations"]
    max_duration_s: float = _DEFAULTS["max_duration_s"]
    runner_timeout: int = _DEFAULTS["runner_timeout"]
    llm_timeout: float = _DEFAULTS["llm_timeout"]
    reasoning_effort: str | None = _DEFAULTS["reasoning_effort"]


# ---------------------------------------------------------------------------
# Per-execution record
# ---------------------------------------------------------------------------

@dataclass
class CodeExecution:
    """Result of running one ```python block."""

    code: str
    success: bool
    stdout: str
    stderr: str
    files_produced: dict[str, int]  # filename -> size in bytes
    duration_s: float


# ---------------------------------------------------------------------------
# Per-turn record
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """Everything that happened in one agent turn."""

    turn: int
    assistant_message: str
    code_executions: list[CodeExecution]
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    reasoning_tokens: int | None = None


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Full output of an agent run."""

    task_description: str
    config: AgentConfig
    turns: list[TurnRecord]
    total_tokens: int
    total_duration_s: float
    completed: bool
    stopped_reason: str  # "done" | "max_tokens" | "max_iterations" | "timeout" | "threshold"
    work_dir: Path | None = None

    def save(self, output_dir: str | Path) -> Path:
        """Persist baseline-run artefacts to disk.

        Layout::

            output_dir/
              baseline_debug.json -- baseline-only debug info
                                     (stopped_reason, total_duration_s).
                                     Report tools never read this.
              turn_0/
                code_0.py         -- executed code blocks
                stdout_0.txt      -- stdout per execution
                stderr_0.txt      -- stderr per execution (if non-empty)
                *.png             -- per-iteration renders the agent saved
                output.step       -- per-iteration STEP snapshot (debug)
              turn_1/
                ...
              output.step         -- canonical candidate at the fixture
                                     root, per the submission contract;
                                     copied from the highest-numbered
                                     turn_<N>/output.step (or .stp).

        Note: this method does NOT write ``result.json``. That file is
        the evaluator's pure output (status + metrics) and lives in
        :func:`cadgenbench.eval.evaluate.evaluate_result`, which the
        agent loop calls after :meth:`save` so the metrics land on the
        same candidate STEP.

        Returns:
            The resolved output_dir path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for rec in self.turns:
            turn_dir = out / f"turn_{rec.turn}"
            turn_dir.mkdir(exist_ok=True)

            for i, exe in enumerate(rec.code_executions):
                (turn_dir / f"code_{i}.py").write_text(exe.code)
                if exe.stdout:
                    (turn_dir / f"stdout_{i}.txt").write_text(exe.stdout)
                if exe.stderr:
                    (turn_dir / f"stderr_{i}.txt").write_text(exe.stderr)

        if self.work_dir and self.work_dir.exists() and self.turns:
            last_turn_dir = out / f"turn_{self.turns[-1].turn}"
            # STEP artifact + any PNG renders the agent saved.
            preserve_suffixes = {".step", ".stp", ".png"}
            for f in self.work_dir.iterdir():
                if f.suffix.lower() in preserve_suffixes:
                    dest = last_turn_dir / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)

        # Materialise the canonical candidate STEP at the fixture root.
        # The evaluator (and external readers) look ONLY at
        # <fixture>/output.step per the submission contract
        # (docs/benchmark/submission.md). Selection mirrors the
        # pre-refactor evaluator (highest-numbered turn with a STEP wins)
        # so historical scores are preserved across the eval becoming
        # turn-unaware. Per-iteration snapshots in turn_<N>/ are
        # untouched - they remain available for debugging.
        canonical = _find_latest_turn_step(out)
        if canonical is not None:
            shutil.copy2(canonical, out / canonical.name)

        debug: dict[str, Any] = {
            "stopped_reason": self.stopped_reason,
            "total_duration_s": round(self.total_duration_s, 2),
        }
        (out / "baseline_debug.json").write_text(json.dumps(debug, indent=2))
        return out


def save_conversation(
    messages: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save the full conversation history to disk.

    Images are replaced with placeholders to keep the file readable.
    """
    sanitized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if block.get("type") == "image_url":
                    blocks.append({"type": "image_url", "image_url": {"url": "[base64 image omitted]"}})
                else:
                    blocks.append(block)
            sanitized.append({**msg, "content": blocks})
        else:
            sanitized.append(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversation.json").write_text(
        json.dumps(sanitized, indent=2, default=str)
    )
