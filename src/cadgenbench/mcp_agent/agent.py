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

"""MCP-agent loop: task description → LLM tool calls → build123d-mcp → STEP.

The agent uses native LLM tool-calling (via LiteLLM) instead of the
baseline's code-block extraction pattern. The build123d-mcp server handles
execution, rendering, and measurement; a synthetic ``signal_done`` tool is
the completion signal.

Stopping conditions (agent is unaware of the budget/iteration limits):
  - Model calls ``signal_done`` and ``output.step`` exists
  - Token budget exhausted (``max_total_tokens``)
  - Iteration cap (``max_iterations``)
  - Wall-clock timeout (``max_duration_s``)
"""
from __future__ import annotations

import base64
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from cadgenbench.baseline.llm import LLMClient
from cadgenbench.mcp_agent.mcp_client import McpSession
from cadgenbench.mcp_agent.prompt import SIGNAL_DONE_TOOL, assemble_messages
from cadgenbench.mcp_agent.types import (
    McpAgentConfig,
    McpAgentResult,
    McpToolCall,
    McpTurnRecord,
    save_mcp_conversation,
)

logger = logging.getLogger(__name__)

# Name of the output artifact the evaluator requires.
ARTIFACT_FILENAME = "output.step"

# Only expose a focused subset of MCP tools to the LLM by default.
# This keeps the tool list short, which helps model attention and reduces
# token cost. Advanced tools (clearance, cross_sections, etc.) can be added
# by extending this set.
_ALLOWED_MCP_TOOLS = {
    "execute",
    "render_view",
    "measure",
    "export",
    "import_step",
}


def run_mcp_agent(
    task_description: str,
    config: McpAgentConfig = McpAgentConfig(),
    input_files: list[Path] | None = None,
    work_dir: Path | None = None,
    output_dir: Path | None = None,
) -> McpAgentResult:
    """Run the MCP-agent CAD generation loop.

    Args:
        task_description: What the user wants built.
        config: All tuneable parameters.
        input_files: Optional image/STEP files for the initial prompt.
            STEP files are copied into work_dir so the MCP server can
            access them via import_step().
        work_dir: Directory where output.step will land. Created as a
            temp dir if not provided.
        output_dir: If provided, saves results incrementally each turn.

    Returns:
        McpAgentResult with per-turn records, totals, and stopping reason.
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="cadgenbench_mcp_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy STEP inputs into work_dir so import_step() can find them.
    if input_files:
        for src in input_files:
            if src.suffix.lower() in {".step", ".stp"}:
                dest = work_dir / src.name
                if dest.resolve() != src.resolve():
                    shutil.copy2(src, dest)

    # The model is told to export to this exact path.  work_dir is under
    # tempfile.gettempdir(), so build123d-mcp's safe_output_path() allows it.
    output_step = work_dir / ARTIFACT_FILENAME

    # LLMClient with streaming disabled — tool-call reassembly via streaming
    # adds complexity; direct non-streaming calls are simpler and reliable
    # for Anthropic/OpenAI targets.
    client = LLMClient(model=config.model, timeout=config.llm_timeout, stream=False)

    turns: list[McpTurnRecord] = []
    total_tokens = 0
    stopped_reason = "max_iterations"
    t0 = time.monotonic()

    def _build_result(step: Path | None = None) -> McpAgentResult:
        return McpAgentResult(
            task_description=task_description,
            config=config,
            turns=turns,
            total_tokens=total_tokens,
            total_duration_s=time.monotonic() - t0,
            completed=stopped_reason == "done",
            stopped_reason=stopped_reason,
            work_dir=work_dir,
            candidate_step=step,
        )

    def _save(step: Path | None = None) -> None:
        if output_dir is None:
            return
        try:
            _build_result(step).save(output_dir)
            save_mcp_conversation(turns, output_dir)
        except Exception:
            logger.warning("Incremental save failed", exc_info=True)

    with McpSession(config.mcp_server_command, config.mcp_server_args) as mcp:
        # Build the tool list: filtered MCP tools + synthetic signal_done.
        tools = [
            t for t in mcp.tools
            if t["function"]["name"] in _ALLOWED_MCP_TOOLS
        ] + [SIGNAL_DONE_TOOL]

        messages = assemble_messages(
            task_description,
            output_step_path=str(output_step),
            input_files=input_files,
        )

        # Inject initial render of any seeded STEP so the model sees the
        # starting geometry without having to render it itself.
        step_inputs = [
            f for f in (input_files or []) if f.suffix.lower() in {".step", ".stp"}
        ]
        if step_inputs:
            _append_seed_render(messages, step_inputs, mcp)

        for turn_idx in range(config.max_iterations):
            elapsed = time.monotonic() - t0
            if elapsed >= config.max_duration_s:
                stopped_reason = "timeout"
                print(
                    f"  [turn {turn_idx}] Timeout ({elapsed:.0f}s >= "
                    f"{config.max_duration_s:.0f}s)",
                    flush=True,
                )
                break

            if total_tokens >= config.max_total_tokens:
                stopped_reason = "max_tokens"
                print(
                    f"  [turn {turn_idx}] Token budget exhausted "
                    f"({total_tokens} >= {config.max_total_tokens})",
                    flush=True,
                )
                break

            tag = f"[turn {turn_idx}]"
            turn_t0 = time.monotonic()

            print(f"  {tag} Calling LLM…", end="", flush=True)
            remaining_s = max(0.0, config.max_duration_s - (time.monotonic() - t0))
            if remaining_s < 1.0:
                stopped_reason = "timeout"
                break

            completion = client.complete(
                messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                tools=tools,
                tool_choice="auto",
            )
            total_tokens += completion.total_tokens
            print(
                f" {completion.total_tokens} tok "
                f"({completion.prompt_tokens}+{completion.completion_tokens})",
                flush=True,
            )

            assistant_text = completion.content or ""
            raw_tool_calls = completion.tool_calls or []

            # --- Dispatch tool calls -----------------------------------------

            tool_records: list[McpToolCall] = []
            done_signaled = False

            # Build the assistant message including any tool_call metadata so
            # the conversation is valid for the next round.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or None,
            }
            if raw_tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_tool_calls
                ]
            messages.append(assistant_msg)

            if not raw_tool_calls:
                # No tool calls — remind the model to use tools.
                print(f"  {tag} No tool calls in response", flush=True)
                messages.append({
                    "role": "user",
                    "content": (
                        "Please use the available tools to make progress on the task. "
                        "Call execute() with build123d code, then verify with measure() "
                        "and render_view(), then export() when done."
                    ),
                })
                turns.append(McpTurnRecord(
                    turn=turn_idx,
                    assistant_text=assistant_text,
                    tool_calls=[],
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    duration_s=time.monotonic() - turn_t0,
                    reasoning_tokens=completion.reasoning_tokens,
                ))
                _save()
                continue

            # Process each tool call and append tool result messages.
            for tc in raw_tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if tool_name == "signal_done":
                    done_signaled = True
                    result_text = "Completion signal received."
                    had_image = False
                    duration_s = 0.0
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
                else:
                    t_call = time.monotonic()
                    print(f"  {tag} → {tool_name}(…)", end="", flush=True)
                    try:
                        result_text, png_bytes = mcp.call_tool(tool_name, args)
                    except Exception as exc:
                        result_text = f"Tool error: {exc}"
                        png_bytes = None
                    duration_s = time.monotonic() - t_call
                    had_image = png_bytes is not None
                    status = " [+img]" if had_image else ""
                    print(f" {duration_s:.1f}s{status}", flush=True)

                    # Build tool result content.  For Anthropic models,
                    # include the image inline; for others use text only.
                    tool_content: Any
                    if had_image and _supports_image_tool_results(config.model):
                        b64 = base64.b64encode(png_bytes).decode()  # type: ignore[arg-type]
                        tool_content = [
                            {"type": "text", "text": result_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ]
                    else:
                        tool_content = result_text
                        if had_image:
                            tool_content += "\n(image rendered but not shown for this provider)"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    })

                tool_records.append(McpToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    result_text=result_text,
                    had_image=had_image,
                    duration_s=duration_s,
                ))

            turns.append(McpTurnRecord(
                turn=turn_idx,
                assistant_text=assistant_text,
                tool_calls=tool_records,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                duration_s=time.monotonic() - turn_t0,
                reasoning_tokens=completion.reasoning_tokens,
            ))

            # --- Check done signal -------------------------------------------

            if done_signaled:
                if not output_step.exists():
                    # Reject done: no artifact yet. Tell model to export first.
                    print(f"  {tag} signal_done rejected — no {ARTIFACT_FILENAME}", flush=True)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"signal_done rejected: `{output_step}` does not exist. "
                            f"Call `export(\"{output_step}\", format=\"step\")` first, "
                            "then call signal_done()."
                        ),
                    })
                    _save()
                    continue

                stopped_reason = "done"
                print(f"  {tag} Done — {ARTIFACT_FILENAME} found", flush=True)
                _save(output_step)
                break

            if total_tokens >= config.max_total_tokens:
                stopped_reason = "max_tokens"
                _save(output_step if output_step.exists() else None)
                break

            _save(output_step if output_step.exists() else None)

    result = _build_result(output_step if output_step.exists() else None)

    if output_dir is not None:
        result.save(output_dir)
        save_mcp_conversation(turns, output_dir)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _supports_image_tool_results(model: str) -> bool:
    """True for providers known to accept image content in tool results."""
    return "anthropic" in model or "claude" in model


def _append_seed_render(
    messages: list[dict[str, Any]],
    step_paths: list[Path],
    mcp: McpSession,
) -> None:
    """For editing tasks: render the input STEP(s) and append to the first
    user message so the model sees the starting geometry immediately."""
    blocks: list[dict[str, Any]] = []
    for step_path in step_paths:
        # Ask the MCP server to render the seeded file.
        try:
            text, png = mcp.call_tool(
                "execute",
                {"code": (
                    f"from build123d import import_step\n"
                    f"_seed = import_step('{step_path.name}')\n"
                    f"show(_seed, 'input')\n"
                    f"print('Loaded input STEP:', '{step_path.name}')"
                )},
            )
            render_text, render_png = mcp.call_tool(
                "render_view",
                {"direction": "iso", "objects": "input"},
            )
        except Exception as exc:
            blocks.append({"type": "text", "text": f"Could not pre-render {step_path.name}: {exc}"})
            continue

        blocks.append({"type": "text", "text": f"### Starting geometry: {step_path.name}\n{render_text}"})
        if render_png is not None:
            b64 = base64.b64encode(render_png).decode()
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    if not blocks:
        return

    user_msg = messages[-1]
    content = user_msg["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    user_msg["content"] = content + blocks
