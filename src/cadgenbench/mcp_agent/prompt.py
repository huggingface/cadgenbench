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

"""System prompt assembly for the MCP-agent strategy.

Unlike the baseline (which teaches code-block extraction), this prompt is
oriented around tool-calling: execute(), measure(), render_view(), export(),
signal_done(). The build123d cheat sheet is shared from the baseline package.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

_CHEAT_SHEET = (
    (Path(__file__).parent.parent / "baseline" / "build123d_cheat_sheet.md")
    .read_text()
    .strip()
)

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_STEP_SUFFIXES = {".step", ".stp"}


def build_system_prompt(output_step_path: str) -> str:
    """Return the full system prompt, embedding the target export path."""
    return f"""\
You are an expert CAD engineer. You build precise 3D models using build123d, \
a Python CAD framework built on OpenCascade.

You work through a set of tools that connect to a persistent build123d session. \
All named objects created with show() survive across tool calls — you never need \
to rebuild geometry from scratch within a run.

## Workflow

1. **Build geometry** — call `execute(code)` with build123d Python code.
   Use `show(shape, "name")` to register named shapes.
2. **Verify topology** — call `measure()` after every boolean operation (cut, \
fuse, intersect). Confirm face count changed as expected; a stale count means \
the operation silently failed.
3. **Render** — call `render_view(direction="iso")` to visually inspect. Use \
`measure()` first — it is faster and unambiguous.
4. **Export** — when satisfied, call:
   `export("{output_step_path}", format="step")`
   This exact path is required; the evaluator reads it from there.
5. **Signal done** — call `signal_done()`. Only after a successful export.

## Code in execute()

- Each execute() call shares the same persistent session. Variables and imports \
persist across calls.
- Always `show(shape, "name")` after creating geometry so render_view and \
measure can reference it.
- After boolean operations call measure() before rendering — rendering confirms \
appearance, measure() confirms geometry.
- Use named dimension variables (`wall_t = 3`) rather than magic numbers.
- No stubs, no pass, no TODOs — write complete working code.

## Editing tasks

If a starting STEP is provided: call `import_step("input.step")` in your first \
execute() call to load it into the session. Apply the requested changes, then \
export as above.

## build123d reference

{_CHEAT_SHEET}
"""


# Synthetic tool definition added to the MCP tool list on the client side.
# The model calls this to signal it is done; it is never forwarded to the
# MCP server.
SIGNAL_DONE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "signal_done",
        "description": (
            "Signal that the CAD model is complete. Call ONLY after you have "
            "successfully exported output.step via the export() tool and verified "
            "the geometry with measure(). The run ends when this is called."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Optional brief summary of the completed model.",
                }
            },
            "required": [],
        },
    },
}


def assemble_messages(
    task_description: str,
    output_step_path: str,
    input_files: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Build the initial messages list for the MCP agent.

    STEP input files are not inlined (too large); the prompt tells the model
    they are available in the MCP server's session. Images are inlined as
    base64 content blocks.
    """
    system_prompt = build_system_prompt(output_step_path)

    step_files = [f for f in (input_files or []) if f.suffix.lower() in _STEP_SUFFIXES]
    image_files = [
        f for f in (input_files or []) if f.suffix.lower() in _IMAGE_MIME_TYPES
    ]

    user_text = task_description
    if step_files:
        names = ", ".join(f"`{p.name}`" for p in step_files)
        user_text = (
            f"{task_description}\n\n"
            f"Starting STEP file(s) {names} have been placed in the MCP server "
            f"working directory. Load with `import_step(\"<filename>\")` inside "
            f"an `execute()` call."
        )

    if not image_files:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img in image_files:
        content.append(_image_block(img))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _image_block(path: Path) -> dict[str, Any]:
    mime = _IMAGE_MIME_TYPES[path.suffix.lower()]
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
