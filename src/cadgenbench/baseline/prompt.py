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

"""System prompt assembly for the baseline agent.

The baseline agent communicates via Python code blocks, has a persistent
working directory, and signals completion with ``[DONE]``.  The prompt is
hard-wired to the build123d / OpenCascade BREP pipeline -- the only kernel
CADGenBench currently supports.

The build123d API cheat sheet is loaded from ``build123d_cheat_sheet.md``
next to this module (ships with the package install).
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

# Loaded once at import time; the file ships inside the baseline package.
_CHEAT_SHEET = (
    (Path(__file__).parent / "build123d_cheat_sheet.md").read_text().strip()
)

_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_STEP_SUFFIXES = {".step", ".stp"}


# ---------------------------------------------------------------------------
# Prompt sections
# ---------------------------------------------------------------------------

_ROLE = """\
You are an expert CAD engineer and software developer.  You create precise, \
parametric 3D models using build123d, a Python CAD framework built on the \
OpenCascade kernel.

You have a **persistent working directory**, every file you create is \
available in subsequent turns.  Your only tool is Python code execution: \
write ```python blocks and they will be executed.  You see stdout, stderr, \
and any PNG images produced.

The build123d API reference is included below."""

_WORKFLOW = """\
## Workflow

1. **Build geometry**, write a build123d script that exports to `output.step`
2. **Review auto-feedback**, after every successful export you will \
automatically receive:
   - Validation results (watertight, topology, volume, bounding box)
   - An iso render of the current model
3. **Iterate**, fix issues based on the feedback; re-export each time you \
make changes
4. **Signal done**, when the model looks correct, write `[DONE]`

You can also write validation or render scripts yourself for deeper inspection.

### Editing an existing STEP

If a STEP file is already present in your working directory (typically \
`input.step`), this is an **editing task**: load that file with \
`from build123d import import_step; shape = import_step("input.step")`, \
apply the requested modification, then export the result as \
`output.step` exactly as you would for a generation task. The same \
auto-validation + iso render runs on `output.step` regardless of \
whether you authored it from scratch or derived it from `input.step`."""

_CODE_GUIDELINES = """\
## Code

Provide a single, self-contained ```python code block per response.

The code MUST:
- Import everything it needs
- Be a complete, runnable script, no wrapper functions, no `if __name__`
- Print useful diagnostics to stdout so you can see what happened

When building geometry, always save to `output.step` using `export_step`.  \
The auto-feedback system triggers on `output.step`, no manual render code \
needed for basic verification.

You can still write scripts for extra inspection (e.g. additional render \
angles, cross-sections, bounding-box checks), any PNG you save to the \
working directory will be shown to you in the next turn.

Code quality:
- Define key dimensions as named variables at the top (parametric style)
- Use clear variable names (`flange_thickness`, not `t`)
- No stubs, no `# TODO`, no `pass`, write every line"""

_RENDER_EXAMPLE = """\
## Rendering the model

Render your STEP file to PNG images from multiple camera angles:

```python
from cadgenbench.common.viewer import render_step

images = render_step("output.step", views=["iso", "front", "top", "right"])
for img in images:
    with open(f"{img.name}.png", "wb") as f:
        f.write(img.data)
    print(f"Saved {img.name}.png ({len(img.data)} bytes)")
```

Available camera presets: `iso`, `front`, `rear`, `left`, `right`, `top`, \
`bottom`.  You can also adjust `width` and `height` (defaults: 1024×768).

Any PNG files written to the working directory will be shown to you as \
images in the next turn's feedback."""

_DONE_SIGNAL = """\
## Signaling completion

When you are satisfied with the model, include the text `[DONE]` in your \
response (outside of code blocks).

Requirements before signaling done:
- `output.step` must exist in the working directory (required, `[DONE]` \
will be rejected otherwise)
- The geometry builds without errors
- Validation passes (watertight, valid topology)
- The iso render looks correct"""

_LIBRARIES_HEADER = """\
## build123d reference"""


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_system_prompt() -> str:
    """Build the full system prompt for the baseline agent."""
    return "\n\n".join([
        _ROLE,
        _WORKFLOW,
        _CODE_GUIDELINES,
        _RENDER_EXAMPLE,
        _DONE_SIGNAL,
        _LIBRARIES_HEADER,
        _CHEAT_SHEET,
    ])


def assemble_messages(
    task_description: str,
    input_files: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Build the initial message list for the baseline agent.

    `input_files` entries are dispatched by extension:
      - image suffixes are inlined as base64 ``image_url`` content blocks.
      - ``.step`` / ``.stp`` files are NOT inlined into the message
        (they would burn tokens and are better read with
        ``import_step``). Instead the caller is expected to have copied
        them into the agent's working directory; we just prepend a
        short note to the task text telling the agent the file is there.

    Returns a two-message list: system prompt + user content.
    """
    system = {"role": "system", "content": assemble_system_prompt()}

    if not input_files:
        return [system, {"role": "user", "content": task_description}]

    step_files: list[Path] = []
    image_files: list[Path] = []
    for file_path in input_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix in _STEP_SUFFIXES:
            step_files.append(file_path)
        elif suffix in _IMAGE_MIME_TYPES:
            image_files.append(file_path)
        else:
            raise ValueError(
                f"Unsupported input file type '{suffix}' for {file_path}. "
                f"Supported: {', '.join(sorted(_IMAGE_MIME_TYPES | _STEP_SUFFIXES))}.",
            )

    user_text = task_description
    if step_files:
        names = ", ".join(f"`{p.name}`" for p in step_files)
        user_text = (
            f"{task_description}\n\n"
            f"The starting STEP file(s) {names} have been copied into your "
            f"working directory. Load with `import_step(...)`, apply the "
            f"requested edit, then export the result as `output.step`."
        )

    if not image_files:
        return [system, {"role": "user", "content": user_text}]

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img_path in image_files:
        content.append(_image_to_content_block(img_path))

    return [system, {"role": "user", "content": content}]


def _image_to_content_block(path: Path) -> dict[str, Any]:
    """Convert an image file to an OpenAI-compatible ``image_url`` content block."""
    suffix = path.suffix.lower()
    mime = _IMAGE_MIME_TYPES.get(suffix)
    if mime is None:
        raise ValueError(
            f"Unsupported image type '{suffix}' for {path}. "
            f"Supported: {', '.join(sorted(_IMAGE_MIME_TYPES))}."
        )
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
