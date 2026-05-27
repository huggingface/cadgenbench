"""Resolve the cadgenbench fixtures + ground-truth directory.

Single source of truth for "where is ``data/``", so the rest of the
codebase stays agnostic to whether the fixtures live in-tree (dev
workflow), bundled in the installed wheel (e.g. an HF Space pip-install),
or downloaded from an HF dataset repo (future, see
``space-setup/migration.md``).

Resolution order, first match wins:

1. ``$CADGENBENCH_DATA_DIR`` env var - explicit override. Useful for
   tests, CI, and configuring a custom location once
   ``cadgenbench-data`` is downloaded from the Hub.
2. ``./data/`` relative to the current working directory - in-repo dev
   workflow; backwards-compatible with how every CLI was wired before
   this helper existed.
3. ``<package>/_data/`` - the copy bundled into the wheel by
   ``[tool.hatch.build.targets.wheel.force-include]`` in
   ``pyproject.toml``. Lets ``pip install cadgenbench`` carry the
   fixtures along with the code, so an HF Space or any other
   installed-only consumer works out of the box without a separate
   data download step.

A future fourth branch will pull from a Hub dataset repo when
``$CADGENBENCH_DATA_REPO`` is set; that lands when
``cadgenbench-data`` exists as its own HF dataset.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "CADGENBENCH_DATA_DIR"


def data_dir() -> Path:
    """Resolve the canonical ``data/`` directory. Raises if not found."""
    env = os.environ.get(_ENV_VAR)
    if env:
        p = Path(env).expanduser()
        if (p / "inputs").is_dir() or (p / "gt").is_dir():
            return p
        raise FileNotFoundError(
            f"{_ENV_VAR}={env!r} but neither {p}/inputs nor {p}/gt exists."
        )

    cwd_candidate = Path.cwd() / "data"
    if (cwd_candidate / "inputs").is_dir() or (cwd_candidate / "gt").is_dir():
        return cwd_candidate

    bundled = Path(__file__).resolve().parent.parent / "_data"
    if (bundled / "inputs").is_dir() or (bundled / "gt").is_dir():
        return bundled

    raise FileNotFoundError(
        "Could not locate cadgenbench data directory. Tried "
        f"${_ENV_VAR}, ./data/ (CWD), and the bundled <package>/_data/. "
        "If you installed cadgenbench from a wheel, make sure the wheel "
        "was built with `force-include` for `data/` (see pyproject.toml). "
        "If you cloned the source, run cadgenbench commands from the "
        "repo root."
    )


def data_inputs_dir() -> Path:
    """Resolve ``data/inputs/`` (where fixture description.yaml live)."""
    return data_dir() / "inputs"


def data_gt_dir() -> Path:
    """Resolve ``data/gt/`` (where ground-truth STEPs + sub-volumes live)."""
    return data_dir() / "gt"
