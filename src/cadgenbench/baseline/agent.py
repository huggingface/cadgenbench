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

"""Baseline agent loop: description -> LLM -> code execution -> iterate.

The agent has one tool, Python code execution in a persistent working
directory.  It writes build123d code, validates geometry, renders views,
and decides when it is done (``[DONE]`` marker).

Stopping conditions (agent is unaware of these):
  - Agent says ``[DONE]``
  - Token budget exhausted (``max_total_tokens``)
  - Iteration cap (``max_iterations``)
  - Wall-clock timeout (``max_duration_s``)
"""
from __future__ import annotations

import base64
import logging
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cadgenbench.baseline.prompt import assemble_messages
from cadgenbench.baseline.types import (
    AgentConfig,
    AgentResult,
    CodeExecution,
    TurnRecord,
    save_conversation,
)
from cadgenbench.common.validity import analyze_step
from cadgenbench.baseline.llm import LLMClient
from cadgenbench.common.viewer import render_mesh, render_step

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_DONE_RE = re.compile(r"\[DONE\]")
# Any fenced code block (``` ... ```), language-agnostic. Stripped before
# searching for the [DONE] marker so a literal "[DONE]" inside code (e.g.
# print("[DONE]") or a comment) never triggers completion.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _has_done_signal(text: str) -> bool:
    """True iff ``[DONE]`` appears outside of any fenced code block."""
    return bool(_DONE_RE.search(_FENCE_RE.sub("", text)))

# Hardcoded for the BREP / build123d pipeline (the only kernel supported).
ARTIFACT_FILENAME = "output.step"

# Extensions in input_files that should be copied verbatim into the
# agent's working directory rather than inlined into the prompt
# (editing tasks: the agent loads the seed STEP with ``import_step``).
_WORKDIR_SEED_SUFFIXES = {".step", ".stp"}


class AgentTimeoutError(TimeoutError):
    """Raised when an agent step exceeds the fixture wall-clock budget."""


class _WallClockAlarm:
    """SIGALRM-backed deadline for blocking provider calls."""

    def __init__(self, seconds: float) -> None:
        self.seconds = max(0.0, float(seconds))
        self._old_handler = None
        self._old_timer = None
        self._enabled = (
            self.seconds > 0
            and threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
        )

    def __enter__(self):
        if not self._enabled:
            return self
        self._old_handler = signal.getsignal(signal.SIGALRM)
        self._old_timer = signal.getitimer(signal.ITIMER_REAL)

        def _raise_timeout(_signum, _frame):
            raise AgentTimeoutError(
                f"LLM call exceeded wall-clock budget ({self.seconds:.0f}s)"
            )

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            if self._old_timer and self._old_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *self._old_timer)
        return False


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> list[str]:
    """Extract all ```python blocks from LLM output."""
    return [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(text)]


def extract_code(text: str) -> str | None:
    """Extract the first ```python block from LLM output, or None."""
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _snapshot_files(directory: Path) -> dict[str, tuple[float, int]]:
    """Return {filename: (mtime, size)} for all files in directory.

    Tracks size alongside mtime so a same-second overwrite that changes the
    file's size is still detected on coarse-resolution filesystems (where
    mtime alone would compare equal).
    """
    result: dict[str, tuple[float, int]] = {}
    if directory.exists():
        for f in directory.iterdir():
            if f.is_file() and not f.name.startswith("_"):
                st = f.stat()
                result[f.name] = (st.st_mtime, st.st_size)
    return result


def execute_code(
    code: str,
    work_dir: Path,
    *,
    timeout: int = 120,
    script_index: int = 0,
) -> CodeExecution:
    """Execute a Python script in the persistent working directory."""
    script_path = work_dir / f"_script_{script_index}.py"
    script_path.write_text(code)

    before = _snapshot_files(work_dir)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
        )
        duration = time.monotonic() - t0
        success = result.returncode == 0
        stdout = result.stdout
        stderr = result.stderr
        error_text = None if success else (stderr.strip() or f"Exit code {result.returncode}")
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - t0
        success = False
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        # Keep any partial stderr captured before the kill: a traceback in
        # progress often shows where the script hung.
        error_text = f"Script timed out after {timeout}s"
        partial = stderr.strip()
        if partial:
            error_text += f"\n--- partial stderr before timeout ---\n{partial}"

    script_path.unlink(missing_ok=True)

    after = _snapshot_files(work_dir)
    new_or_modified = {}
    for name, (mtime, size) in after.items():
        prev = before.get(name)
        if prev is None or mtime > prev[0] or size != prev[1]:
            new_or_modified[name] = size

    return CodeExecution(
        code=code,
        success=success,
        stdout=stdout,
        stderr=stderr if success else (error_text or stderr),
        files_produced=new_or_modified,
        duration_s=duration,
    )


# ---------------------------------------------------------------------------
# Auto validation + render (injected into feedback after every code run)
# ---------------------------------------------------------------------------

# Validate + render runs in a worker process so OCC tessellation (CPU- and
# GIL-bound) parallelizes across concurrent fixtures/models instead of
# serializing on the GIL, and each render gets its own VTK/GL context. The
# pool is created lazily and shared across all agent threads in the process.
_RENDER_POOL = None
_RENDER_POOL_LOCK = threading.Lock()
_RENDER_POOL_ATEXIT_DONE = False


def _shutdown_render_pool() -> None:
    """Force-terminate the render pool's workers so the owning process can exit.

    A plain ``ProcessPoolExecutor`` registers an ``atexit`` that *joins* its
    workers (``wait=True``); if a render was abandoned (e.g. the agent hit its
    wall-clock timeout mid-render) that join blocks forever, so the process
    never exits. Under the nested pools used by ``baseline compare-llms``
    (model pool -> fixture pool -> this render pool) that stalls the parent's
    ``as_completed`` and the comparison HTML is never produced. We terminate
    the worker processes outright instead of joining. Registered *after* the
    executor's own atexit (see ``_get_render_pool``) so LIFO ordering runs this
    first, leaving the executor's join nothing live to wait on. Best-effort.
    """
    global _RENDER_POOL
    pool = _RENDER_POOL
    _RENDER_POOL = None
    if pool is None:
        return
    # Terminate workers BEFORE shutdown: ``shutdown`` clears ``_processes`` to
    # None, so killing first is the only way the terminate actually lands.
    for proc in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass


def _get_render_pool():  # type: ignore[no-untyped-def]
    global _RENDER_POOL, _RENDER_POOL_ATEXIT_DONE
    if _RENDER_POOL is None:
        with _RENDER_POOL_LOCK:
            if _RENDER_POOL is None:
                import atexit
                import multiprocessing as mp
                import os
                from concurrent.futures import ProcessPoolExecutor

                _RENDER_POOL = ProcessPoolExecutor(
                    max_workers=min(os.cpu_count() or 4, 8),
                    mp_context=mp.get_context("spawn"),
                )
                # Register AFTER the executor import above so this runs first
                # at interpreter exit (atexit is LIFO) and kills workers before
                # the executor's own blocking join can hang.
                if not _RENDER_POOL_ATEXIT_DONE:
                    atexit.register(_shutdown_render_pool)
                    _RENDER_POOL_ATEXIT_DONE = True
    return _RENDER_POOL


def _validate_and_render_blob(step_path_str: str) -> tuple[str, bytes | None]:
    """Validate a STEP file and iso-render it, tessellating only **once**.

    The validity gate already tessellates the boundary for its mesh check;
    we hand it a ``mesh_cache`` and reuse that exact mesh for the render
    instead of meshing the part a second time. Designed to run inside the
    render pool's worker process. Returns ``(text_block, iso_png_bytes)``.
    """
    step_path = Path(step_path_str)
    name = step_path.name
    lines: list[str] = [f"### Auto-validation of {name}"]
    mesh_cache: dict[float, object] = {}
    try:
        a = analyze_step(step_path, mesh_cache=mesh_cache)
        v, m = a.validation, a.measurements
        bb = m.bounding_box
        lines += [
            f"Valid:      {v.is_valid}",
            f"Watertight: {v.is_watertight}",
            f"Solids:     {m.solid_count}",
            f"Faces:      {m.face_count}",
            f"Volume:     {m.volume:.1f} mm³",
            f"BBox:       "
            f"{bb.size_x:.1f} × "
            f"{bb.size_y:.1f} × "
            f"{bb.size_z:.1f} mm",
        ]
        if v.topology_errors:
            lines.append(f"Errors:     {v.topology_errors[:3]}")
    except Exception as exc:
        lines.append(f"Validation error: {exc}")

    iso_bytes: bytes | None = None
    lines.append("")
    lines.append(f"### Iso render of {name}")
    try:
        # Reuse the validity gate's tessellation when it produced one (valid
        # parts); otherwise tessellate here (the part never reached the mesh
        # gate, e.g. an invalid BREP).
        cached_mesh = next(iter(mesh_cache.values()), None)
        if cached_mesh is not None:
            images = render_mesh(cached_mesh, views=["iso"])
        else:
            images = render_step(step_path, views=["iso"])
        if images:
            iso_bytes = images[0].data
            lines.append("(see attached image below)")
        else:
            lines.append("⚠️ Render returned no images.")
    except Exception as exc:
        lines.append(f"⚠️ Render failed: {exc}")

    return "\n".join(lines), iso_bytes


def _validate_and_render_step(step_path: Path) -> tuple[str, bytes | None]:
    """Validate + iso-render a STEP file. Caller guarantees ``step_path`` exists.

    Dispatches to a shared process pool so the CPU-heavy tessellation runs in
    parallel across concurrent agent threads rather than serializing on the
    GIL. Falls back to running inline if the pool is unavailable. Used both
    for per-turn ``output.step`` feedback and the startup snapshot of any
    seeded ``input.step`` (editing tasks).
    """
    try:
        return _get_render_pool().submit(
            _validate_and_render_blob, str(step_path),
        ).result()
    except Exception:
        return _validate_and_render_blob(str(step_path))


def _auto_validate_and_render(
    work_dir: Path,
    last_exe: CodeExecution | None,
) -> tuple[str, bytes | None]:
    """After a code execution, validate the STEP artifact and render iso if it exists.

    Returns (text_block, iso_png_bytes).  iso_png_bytes is None if no render
    was produced.  Always returns a human-readable explanation of what happened.
    """
    artifact_path = work_dir / ARTIFACT_FILENAME

    if not artifact_path.exists():
        if last_exe is None:
            return "", None
        if not last_exe.success:
            return (
                f"⚠️ No `{ARTIFACT_FILENAME}` found (script failed, see error above).",
                None,
            )
        return (
            f"⚠️ No `{ARTIFACT_FILENAME}` found, your script ran without errors but "
            f"did not export the expected file.  Make sure your script writes "
            f"`{ARTIFACT_FILENAME}` at the end.",
            None,
        )

    return _validate_and_render_step(artifact_path)


def _seed_step_feedback_blocks(step_paths: list[Path]) -> list[dict[str, Any]]:
    """Build initial-message content blocks describing each seeded STEP.

    Parallel to the per-turn auto-feedback for ``output.step``: a
    validation summary plus an iso render. Used on agent startup for
    editing tasks so the agent sees a structured summary and a render
    of the starting geometry on turn 0 without having to render it
    itself.
    """
    blocks: list[dict[str, Any]] = []
    for path in step_paths:
        text, iso_png = _validate_and_render_step(path)
        blocks.append({"type": "text", "text": text})
        if iso_png is not None:
            b64 = base64.b64encode(iso_png).decode()
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    return blocks



def _file_listing(work_dir: Path) -> str:
    """Human-readable file listing with sizes and modification times."""
    files = sorted(
        (f for f in work_dir.iterdir() if f.is_file() and not f.name.startswith("_")),
        key=lambda f: f.name,
    )
    if not files:
        return "Working directory is empty."

    lines = ["Files in working directory:"]
    for f in files:
        stat = f.stat()
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        ts = mtime.strftime("%H:%M:%S")
        if size >= 1024 * 1024:
            sz = f"{size / (1024*1024):.1f}MB"
        elif size >= 1024:
            sz = f"{size / 1024:.1f}KB"
        else:
            sz = f"{size}B"
        lines.append(f"  {f.name:<30s} {sz:>8s}  modified {ts}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------

def _truncate_middle(text: str, limit: int) -> str:
    """Truncate *text* to ~*limit* chars, keeping the head and the tail.

    Used for stderr: Python tracebacks put the actual ``ExceptionType:
    message`` line at the very end, so a head-only truncation would drop
    the single most useful line. Keep a small head (the failing call site)
    plus a larger tail (the exception itself).
    """
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return (
        f"{text[:head]}\n"
        f"... (truncated, {len(text)} chars total) ...\n"
        f"{text[-tail:]}"
    )


def format_execution_feedback(
    execution: CodeExecution | None,
    work_dir: Path,
    auto_text: str = "",
    auto_iso: bytes | None = None,
) -> list[dict[str, Any]]:
    """Build a multimodal user message for one turn's execution + auto feedback.

    The agent runs exactly one code block per turn, so this takes a single
    :class:`CodeExecution` (or ``None`` when the turn produced no runnable
    code).
    """
    content: list[dict[str, Any]] = []
    text_parts: list[str] = []

    if execution is not None:
        exe = execution
        status = "SUCCESS" if exe.success else "FAILED"
        text_parts.append(f"### Execution: {status} ({exe.duration_s:.1f}s)")

        if exe.stdout.strip():
            stdout_truncated = exe.stdout[:3000]
            if len(exe.stdout) > 3000:
                stdout_truncated += f"\n... (truncated, {len(exe.stdout)} chars total)"
            text_parts.append(f"stdout:\n```\n{stdout_truncated}\n```")

        if not exe.success and exe.stderr.strip():
            stderr_truncated = _truncate_middle(exe.stderr, 2000)
            text_parts.append(f"stderr:\n```\n{stderr_truncated}\n```")

        if exe.files_produced:
            produced = ", ".join(
                f"{name} ({sz}B)" for name, sz in sorted(exe.files_produced.items())
            )
            text_parts.append(f"Files created/modified: {produced}")

    text_parts.append("")
    text_parts.append(_file_listing(work_dir))

    if auto_text:
        text_parts.append("")
        text_parts.append(auto_text)

    content.append({"type": "text", "text": "\n\n".join(text_parts)})

    # Manually-saved PNGs the agent rendered *this turn*. Restrict to the
    # files this turn's execution actually created/modified (not everything
    # on disk) so stale one-off renders from earlier turns don't keep riding
    # along and multiplying the request size.
    files_produced = execution.files_produced if execution is not None else {}
    produced_pngs = sorted(
        name for name in files_produced if name.lower().endswith(".png")
    )
    for name in produced_pngs:
        f = work_dir / name
        if not f.is_file():
            continue
        try:
            data = base64.b64encode(f.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            })
        except Exception:
            logger.warning("Failed to read PNG %s", f, exc_info=True)

    # Auto iso render (in-memory, not written to disk)
    if auto_iso is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(auto_iso).decode()}"},
        })

    return content


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_agent(
    task_description: str,
    config: AgentConfig = AgentConfig(),
    client: LLMClient | None = None,
    input_files: list[Path] | None = None,
    work_dir: Path | None = None,
    output_dir: Path | None = None,
) -> AgentResult:
    """Run the baseline single-agent CAD generation loop.

    Args:
        task_description: What the user wants built.
        config: All tuneable parameters.
        client: LLM client.  Created from ``config.model`` if not provided.
        input_files: Optional image files to include in the initial prompt.
        work_dir: Persistent working directory.  Created as a temp dir
            if not provided.
        output_dir: If provided, save results incrementally after each turn.

    Returns:
        AgentResult with per-turn records, totals, and stopping reason.
    """
    if client is None:
        client = LLMClient(model=config.model, timeout=config.llm_timeout)

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="cadgenbench_agent_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # Editing tasks ship a starting STEP in input_files; copy it into
    # the work dir so the agent can load it with ``import_step``. Keep
    # the filename verbatim (e.g. ``input.step``) since the assembled
    # message references it by name.
    seeded_step_paths: list[Path] = []
    if input_files:
        for src in input_files:
            if src.suffix.lower() in _WORKDIR_SEED_SUFFIXES:
                dest = work_dir / src.name
                if dest.resolve() != src.resolve():
                    shutil.copy2(src, dest)
                seeded_step_paths.append(dest)

    messages = assemble_messages(task_description, input_files=input_files)

    # Editing tasks: auto-validate + iso-render every seeded STEP and
    # attach the result to the initial user message, parallel to the
    # per-turn feedback the agent gets on ``output.step``. Gives the
    # agent a structured summary (bbox, faces, volume) plus a render
    # of the starting geometry on turn 0 without it having to render
    # ``input.step`` itself.
    if seeded_step_paths:
        seed_blocks = _seed_step_feedback_blocks(seeded_step_paths)
        user_msg = messages[-1]
        content = user_msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        user_msg["content"] = content + seed_blocks
    turns: list[TurnRecord] = []
    total_tokens = 0
    stopped_reason = "max_iterations"
    # Whether the agent has been shown the auto validation + render of
    # output.step at least once. Gates [DONE]: the first time the artifact is
    # reviewed we defer and require re-confirmation; after that a repeated
    # [DONE] is accepted even if the agent re-ran code that turn (otherwise a
    # model that always emits a code block could never satisfy the gate).
    artifact_feedback_shown = False
    t0 = time.monotonic()

    def _build_result() -> AgentResult:
        return AgentResult(
            task_description=task_description,
            config=config,
            turns=turns,
            total_tokens=total_tokens,
            total_duration_s=time.monotonic() - t0,
            completed=stopped_reason == "done",
            stopped_reason=stopped_reason,
            work_dir=work_dir,
        )

    def _save_incremental() -> None:
        if output_dir is not None:
            try:
                _build_result().save(output_dir)
                save_conversation(messages, output_dir)
            except Exception:
                logger.warning("Incremental save failed", exc_info=True)

    try:
      for turn_idx in range(config.max_iterations):
        elapsed = time.monotonic() - t0
        if elapsed >= config.max_duration_s:
            stopped_reason = "timeout"
            print(f"  [turn {turn_idx}] Wall-clock timeout ({elapsed:.0f}s >= {config.max_duration_s:.0f}s)", flush=True)
            break

        if total_tokens >= config.max_total_tokens:
            stopped_reason = "max_tokens"
            print(f"  [turn {turn_idx}] Token budget exhausted ({total_tokens} >= {config.max_total_tokens})", flush=True)
            break

        tag = f"[turn {turn_idx}]"
        turn_t0 = time.monotonic()

        print(f"  {tag} Calling LLM...", end="", flush=True)
        complete_kwargs: dict[str, Any] = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        if config.reasoning_effort is not None:
            complete_kwargs["reasoning_effort"] = config.reasoning_effort
        remaining_s = max(0.0, config.max_duration_s - (time.monotonic() - t0))
        call_deadline_s = min(config.llm_timeout, remaining_s)
        try:
            with _WallClockAlarm(call_deadline_s):
                completion = client.complete(messages, **complete_kwargs)
        except AgentTimeoutError as exc:
            stopped_reason = "timeout"
            print(f" TIMEOUT ({exc})", flush=True)
            _save_incremental()
            break
        total_tokens += completion.total_tokens
        print(f" {completion.total_tokens} tok ({completion.prompt_tokens}+{completion.completion_tokens})", flush=True)

        budget_exceeded = total_tokens >= config.max_total_tokens

        assistant_text = completion.content

        code_blocks = extract_code_blocks(assistant_text)
        code = code_blocks[0] if code_blocks else None
        # Single-block contract: we execute only the first ```python block.
        # If the model emitted more, run the first and tell it so the dropped
        # blocks aren't silently lost (the prompt asks for one self-contained
        # block per turn).
        multi_block_note = ""
        if len(code_blocks) > 1:
            multi_block_note = (
                f"⚠️ Note: your response contained {len(code_blocks)} ```python "
                "blocks, but only the first was executed.  Send exactly one "
                "self-contained code block per turn."
            )
            print(f"  {tag} {len(code_blocks)} code blocks; ran first only", flush=True)
        executions: list[CodeExecution] = []

        if code is not None:
            print(f"  {tag} Running code...", end="", flush=True)
            exe = execute_code(code, work_dir, timeout=config.runner_timeout)
            status = "ok" if exe.success else "FAILED"
            print(f" {status} ({exe.duration_s:.1f}s)", flush=True)
            executions.append(exe)

            # Auto validate + render iso whenever the artifact exists after a run
            t_render = time.monotonic()
            auto_text, auto_iso = _auto_validate_and_render(work_dir, exe)
            render_s = time.monotonic() - t_render
            if auto_iso is not None:
                print(f"  {tag} Auto-rendered iso ({render_s:.1f}s)", flush=True)
                if output_dir is not None:
                    turn_out = output_dir / f"turn_{turn_idx}"
                    turn_out.mkdir(parents=True, exist_ok=True)
                    (turn_out / "auto_render_iso.png").write_bytes(auto_iso)
        else:
            print(f"  {tag} No code block in response", flush=True)
            auto_text, auto_iso = "", None

        done_signaled = _has_done_signal(assistant_text)

        # Record this turn once, then append the assistant message so the
        # persisted conversation is always complete before any save.
        turns.append(TurnRecord(
            turn=turn_idx,
            assistant_message=assistant_text,
            code_executions=executions,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            reasoning_tokens=completion.reasoning_tokens,
            duration_s=time.monotonic() - turn_t0,
        ))
        messages.append({"role": "assistant", "content": assistant_text})

        artifact_exists = (work_dir / ARTIFACT_FILENAME).exists()
        last_exe_failed = bool(executions) and not executions[-1].success

        def _send_feedback(note: str | None = None) -> None:
            content = format_execution_feedback(
                executions[0] if executions else None,
                work_dir, auto_text=auto_text, auto_iso=auto_iso,
            )
            prefix = note or ""
            if multi_block_note:
                prefix += multi_block_note + "\n\n"
            if prefix:
                if content and content[0].get("type") == "text":
                    content[0]["text"] = prefix + content[0]["text"]
                else:
                    content.insert(0, {"type": "text", "text": prefix})
            messages.append({"role": "user", "content": content})

        if done_signaled:
            # Hard gate: require output.step before accepting [DONE].
            if not artifact_exists:
                print(f"  {tag} [DONE] rejected (no {ARTIFACT_FILENAME})", flush=True)
                messages.append({"role": "user", "content": (
                    f"⚠️ `[DONE]` rejected, `{ARTIFACT_FILENAME}` was not found in the "
                    "working directory.  Make sure your script writes "
                    f"`{ARTIFACT_FILENAME}` before signaling done.\n\n"
                    + _file_listing(work_dir)
                )})
                _save_incremental()
                continue

            # Don't finish on a failed script: the artifact may be stale (the
            # failed run didn't regenerate it). Show the failure and let the
            # agent fix it, then re-confirm on a clean run.
            if last_exe_failed:
                print(f"  {tag} [DONE] deferred, last script failed", flush=True)
                _send_feedback(
                    "Your `[DONE]` signal was received, but the script above failed, "
                    f"so `{ARTIFACT_FILENAME}` may be stale.  Fix the error and "
                    "re-export, then signal `[DONE]` again.\n\n"
                )
                _save_incremental()
                continue

            # Require the agent to have reviewed the auto validation + render of
            # output.step at least once before accepting [DONE]. Fires only on
            # the first review; afterwards a repeated [DONE] is accepted even if
            # the agent re-ran code, so a model that always emits a code block
            # can still finish.
            if (auto_text or auto_iso is not None) and not artifact_feedback_shown:
                print(f"  {tag} [DONE] deferred, sending unseen auto-feedback first", flush=True)
                _send_feedback(
                    "Your `[DONE]` signal was received, but you haven't yet reviewed "
                    "the automatic validation and render below.  "
                    "Please inspect them and respond with `[DONE]` again if you are "
                    "satisfied, or continue iterating if you see issues.\n\n"
                )
                artifact_feedback_shown = True
                _save_incremental()
                continue

            stopped_reason = "done"
            print(f"  {tag} Agent signaled [DONE]", flush=True)
            _save_incremental()
            break

        if budget_exceeded:
            stopped_reason = "max_tokens"
            print(f"  Token budget reached after turn {turn_idx} ({total_tokens} >= {config.max_total_tokens})", flush=True)
            _save_incremental()
            break

        if executions:
            _send_feedback()
            # Mark the artifact as reviewed once we've shown its validation/
            # render, so a later [DONE] doesn't get deferred indefinitely.
            if auto_text or auto_iso is not None:
                artifact_feedback_shown = True
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Your response contained no ```python code blocks. "
                    "Please write Python code to make progress on the task.\n\n"
                    + _file_listing(work_dir)
                ),
            })

        _save_incremental()
    finally:
        # Deterministic teardown of this process's shared render pool while we
        # are still alive. The wall-clock timeout can abandon work mid-render,
        # leaving the spawn-based ``_RENDER_POOL`` with live workers; relying on
        # ``atexit`` does NOT help inside a ``ProcessPoolExecutor`` worker
        # (fixture grandchild), which is force-killed on shutdown and never
        # runs atexit. That left the inner fixture pool's join — and therefore
        # ``compare-llms``' ``as_completed`` — deadlocked forever. Killing the
        # render workers here, before the worker is asked to exit, lets every
        # fixture future resolve and the join complete. Best-effort.
        _shutdown_render_pool()

    result = _build_result()

    if output_dir is not None:
        result.save(output_dir)
        save_conversation(messages, output_dir)

    return result
