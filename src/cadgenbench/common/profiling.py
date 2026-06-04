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

"""Lightweight, env-gated phase profiling for the eval hot path.

The evaluator runs each fixture inside a ``ProcessPoolExecutor`` worker
and only prints results once the whole shard finishes, so a running shard
is otherwise a black box. These helpers emit one *flushed* stderr line per
timed block: worker stderr streams live into the HF Job logs, so enabling
the gate turns the black box into a per-phase trace.

On by default so any slow or large submission is debuggable without a
redeploy; the cost is a handful of flushed stderr lines per fixture (more
when a part re-meshes through the deflection ladder), with negligible
compute overhead. Opt out with ``CADGENBENCH_EVAL_PROFILE=0`` (also accepts
``false``/``no``/``off``). Lines are prefixed ``[eval:phase]`` so they are
trivial to grep out of a job log.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from collections.abc import Iterator

_FALSE = frozenset({"0", "false", "no", "off"})


def profile_enabled() -> bool:
    """True unless ``CADGENBENCH_EVAL_PROFILE`` is explicitly set falsy."""
    return os.environ.get("CADGENBENCH_EVAL_PROFILE", "").strip().lower() not in _FALSE


@contextmanager
def phase(label: str, *, tag: str | None = None) -> Iterator[None]:
    """Time a block and emit ``[eval:phase] pid=<pid> <tag> <label> <secs>s``.

    A no-op (single env read) unless :func:`profile_enabled`. The line is
    flushed so it appears live in streamed job logs. ``pid`` lets you group
    a single worker's interleaved phases (8 run concurrently); *tag* is
    typically the fixture name when the caller has it.
    """
    if not profile_enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        tag_part = f"{tag} " if tag else ""
        print(
            f"[eval:phase] pid={os.getpid()} {tag_part}{label} {dt:.2f}s",
            file=sys.stderr, flush=True,
        )


def note(message: str, *, tag: str | None = None) -> None:
    """Emit a one-off ``[eval:phase]`` annotation line (env-gated like :func:`phase`).

    For logging a measured value (e.g. mesh size) rather than a duration, so
    slow runs can be correlated with geometry without a redeploy.
    """
    if not profile_enabled():
        return
    tag_part = f"{tag} " if tag else ""
    print(
        f"[eval:phase] pid={os.getpid()} {tag_part}{message}",
        file=sys.stderr, flush=True,
    )
