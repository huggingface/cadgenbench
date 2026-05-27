"""Resolve the cadgenbench fixture inputs + ground-truth directories.

Single source of truth for "where are the fixtures". Two helpers
callers use: :func:`data_inputs_dir` (fixture inputs:
``description.yaml``, ``input.png``, optional ``input.step``) and
:func:`data_gt_dir` (ground truth: ``ground_truth.step``, optional
sub-volume jig STEPs, optional renders).

Each helper independently resolves through this order, first match wins:

1. **Hub dataset repo** (``$CADGENBENCH_DATA_REPO`` for inputs,
   ``$CADGENBENCH_DATA_GT_REPO`` for ground truth). When set, the
   helper calls :func:`huggingface_hub.snapshot_download` and returns
   the local snapshot directory; the snapshot root *is* the fixtures
   root (each top-level entry is a fixture directory). Private repos
   need ``HF_TOKEN`` in the environment; ``snapshot_download`` picks
   it up automatically. This is the production path for both the
   leaderboard Space and local dev.
2. **Explicit dir override** (``$CADGENBENCH_DATA_DIR``). Pointer to
   a local dir laid out like the legacy in-repo tree (``inputs/`` +
   ``gt/`` subdirectories). Useful for tests, CI, or pinning to a
   local snapshot.
3. **Current-working-directory ``./data/``**. Backward compat path
   for users who manually drop a ``data/`` dir into their CWD; the
   in-repo ``cadgenbench/data/`` was removed once the Hub datasets
   came online.

The Hub branch is independent per helper because inputs and ground
truth live in two separate dataset repos (one public-at-launch, one
permanently-private). Branches 2-3 share :func:`data_dir`, which
returns the local parent containing both ``inputs/`` and ``gt/``
subdirectories; it's only meaningful when neither Hub env var is set.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_DATA_DIR = "CADGENBENCH_DATA_DIR"
_ENV_DATA_REPO = "CADGENBENCH_DATA_REPO"
_ENV_DATA_GT_REPO = "CADGENBENCH_DATA_GT_REPO"


def data_dir() -> Path:
    """Resolve the local fixtures parent dir (with ``inputs/`` + ``gt/``).

    Tries ``$CADGENBENCH_DATA_DIR`` then ``./data/`` (CWD). Does NOT
    consult the Hub branch -- when inputs and ground truth live in
    separate Hub repos there's no single parent dir; callers should
    use :func:`data_inputs_dir` and :func:`data_gt_dir` directly.

    Raises:
        FileNotFoundError: when neither candidate contains ``inputs/``
            or ``gt/`` subdirectories. The Hub-dataset path
            (``$CADGENBENCH_DATA_REPO`` / ``$CADGENBENCH_DATA_GT_REPO``)
            is the recommended dev workflow.
    """
    env = os.environ.get(_ENV_DATA_DIR)
    if env:
        p = Path(env).expanduser()
        if (p / "inputs").is_dir() or (p / "gt").is_dir():
            return p
        raise FileNotFoundError(
            f"{_ENV_DATA_DIR}={env!r} but neither {p}/inputs nor {p}/gt exists."
        )

    cwd_candidate = Path.cwd() / "data"
    if (cwd_candidate / "inputs").is_dir() or (cwd_candidate / "gt").is_dir():
        return cwd_candidate

    raise FileNotFoundError(
        "Could not locate a local cadgenbench data directory. Tried "
        f"${_ENV_DATA_DIR} and ./data/ (CWD). To use the Hub-hosted "
        f"datasets instead (recommended), set ${_ENV_DATA_REPO} "
        f"(inputs) and/or ${_ENV_DATA_GT_REPO} (ground truth, requires "
        "HF_TOKEN with read access)."
    )


def data_inputs_dir() -> Path:
    """Resolve the fixture inputs directory.

    See module docstring for the full resolution order. Returns a
    directory whose immediate children are fixture directories
    (each containing ``description.yaml`` + ``input.png`` etc.).
    """
    repo = os.environ.get(_ENV_DATA_REPO)
    if repo:
        return _snapshot_dataset(repo)
    return data_dir() / "inputs"


def data_gt_dir() -> Path:
    """Resolve the ground-truth directory.

    See module docstring for the full resolution order. Returns a
    directory whose immediate children are fixture directories
    (each containing ``ground_truth.step`` + optional sub-volumes
    + renders).

    The Hub branch downloads a private dataset repo; the caller must
    have ``HF_TOKEN`` set with read access.
    """
    repo = os.environ.get(_ENV_DATA_GT_REPO)
    if repo:
        return _snapshot_dataset(repo)
    return data_dir() / "gt"


def _snapshot_dataset(repo_id: str) -> Path:
    """Snapshot-download a dataset repo and return the local cache path.

    Lazy-imports ``huggingface_hub`` so the dependency only loads when
    a caller explicitly opts into the Hub branch via an env var.
    Subsequent calls for the same repo are fast: snapshot_download
    checks revisions against the local cache.
    """
    from huggingface_hub import snapshot_download

    logger.info("Resolving cadgenbench dataset from Hub: %s", repo_id)
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
