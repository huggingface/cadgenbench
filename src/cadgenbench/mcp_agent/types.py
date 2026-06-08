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

"""Data types for the MCP-agent strategy."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Per-tool-call record
# ---------------------------------------------------------------------------

@dataclass
class McpToolCall:
    """Record of one LLM-initiated tool call."""

    tool_name: str
    arguments: dict[str, Any]
    result_text: str
    had_image: bool
    duration_s: float


# ---------------------------------------------------------------------------
# Per-turn record
# ---------------------------------------------------------------------------

@dataclass
class McpTurnRecord:
    """Everything that happened in one agent turn (one LLM completion)."""

    turn: int
    assistant_text: str        # text portion of the assistant message
    tool_calls: list[McpToolCall]
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    reasoning_tokens: int | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class McpAgentConfig:
    """Tuneable parameters for the MCP-agent run."""

    model: str = "anthropic/claude-sonnet-4-6"
    temperature: float = 1.0
    max_tokens: int = 16384           # per-LLM-call completion budget
    max_total_tokens: int = 500_000   # global token budget
    max_iterations: int = 60          # agent turn cap
    max_duration_s: float = 1800.0    # wall-clock timeout (seconds)
    llm_timeout: float = 300.0        # per-LLM-call timeout (seconds)
    # Command + args to launch the build123d-mcp server over stdio.
    # Defaults assume `build123d-mcp` is on PATH (pip-installed).
    # Alternative: mcp_server_command="uvx", mcp_server_args=["build123d-mcp"]
    mcp_server_command: str = "build123d-mcp"
    mcp_server_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class McpAgentResult:
    """Full output of an MCP-agent run."""

    task_description: str
    config: McpAgentConfig
    turns: list[McpTurnRecord]
    total_tokens: int
    total_duration_s: float
    completed: bool
    stopped_reason: str   # "done" | "max_tokens" | "max_iterations" | "timeout"
    work_dir: Path | None = None
    candidate_step: Path | None = None

    def save(self, output_dir: str | Path) -> Path:
        """Persist artefacts to disk in the same layout as the baseline.

        The canonical ``output.step`` at the fixture root is required by
        the evaluator. Per-turn tool call logs land in ``turn_<N>/``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for rec in self.turns:
            turn_dir = out / f"turn_{rec.turn}"
            turn_dir.mkdir(exist_ok=True)
            for i, tc in enumerate(rec.tool_calls):
                (turn_dir / f"tool_{i:02d}_{tc.tool_name}.txt").write_text(
                    f"tool: {tc.tool_name}\n"
                    f"args: {json.dumps(tc.arguments, indent=2)}\n\n"
                    f"result ({tc.duration_s:.1f}s, image={tc.had_image}):\n"
                    f"{tc.result_text}"
                )

        # Materialise canonical candidate STEP at the fixture root.
        if self.candidate_step and self.candidate_step.exists():
            dest = out / "output.step"
            if dest.resolve() != self.candidate_step.resolve():
                shutil.copy2(self.candidate_step, dest)

        debug: dict[str, Any] = {
            "stopped_reason": self.stopped_reason,
            "total_duration_s": round(self.total_duration_s, 2),
            "total_tokens": self.total_tokens,
            "turns": len(self.turns),
        }
        (out / "mcp_agent_debug.json").write_text(json.dumps(debug, indent=2))
        return out


def save_mcp_conversation(turns: list[McpTurnRecord], output_dir: Path) -> None:
    """Write a JSON log of the full conversation for debugging."""
    log = []
    for rec in turns:
        log.append({
            "turn": rec.turn,
            "assistant_text": rec.assistant_text,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "result_preview": (
                        tc.result_text[:400] + "…"
                        if len(tc.result_text) > 400 else tc.result_text
                    ),
                    "had_image": tc.had_image,
                    "duration_s": round(tc.duration_s, 2),
                }
                for tc in rec.tool_calls
            ],
            "tokens": {
                "prompt": rec.prompt_tokens,
                "completion": rec.completion_tokens,
            },
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conversation.json").write_text(
        json.dumps(log, indent=2, default=str)
    )
