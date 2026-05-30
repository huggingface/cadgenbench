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

"""Run-level aggregation: rolls per-fixture ``result.json`` files into a
single ``run_summary.json`` at the root of a results directory.

Schema produced (a sibling of ``params.json`` inside ``results/<run>/``)::

    {
        "aggregate_score":   float in [0, 1],     # mean cad_score over all fixtures
        "validity_rate":     float in [0, 1],     # n_valid / n_fixtures
        "n_fixtures":        int,
        "n_valid":           int,
        "n_invalid":         int,
        "n_missing":         int,
        "score_by_task_type": {
            "generation": float in [0, 1],
            "editing":    float in [0, 1]
        },
        "per_task_scores": {
            "<task_type>": {
                "score":         float,
                "validity_rate": float,
                "n_fixtures":    int,
                "n_valid":       int,
                "n_invalid":     int,
                "n_missing":     int
            }
        },
        "per_fixture_scores": {
            "<fixture_name>": {
                "status":     "valid" | "invalid" | "missing",
                "cad_score":  float in [0, 1],
                "task_type":  "generation" | "editing"
            }
        }
    }

Design rules:
- Aggregate **includes zeros** from invalid and missing fixtures (this is
  the leaderboard convention; gating by validity must not let bad runs
  silently inflate by averaging only successes).
- Reads exclusively from ``result.json`` per fixture + the fixture's
  authored ``description.yaml`` (for ``task_type``). Never touches
  baseline-only debug files.
- One unknown task type ⇒ it gets its own bucket in
  ``per_task_scores`` automatically; no schema change needed.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from cadgenbench.eval.evaluate import STATUS_INVALID, STATUS_MISSING, STATUS_VALID

logger = logging.getLogger(__name__)

RUN_SUMMARY_NAME = "run_summary.json"
_KNOWN_TASK_TYPES: tuple[str, ...] = ("generation", "editing")


def write_run_summary(
    run_dir: Path,
    *,
    data_inputs_dir: Path | None = None,
) -> Path:
    """Walk *run_dir*, aggregate every per-fixture ``result.json``, and
    persist ``run_summary.json`` at the run-dir root.

    Args:
        run_dir: ``results/<run_name>/`` directory.
        data_inputs_dir: Override for ``data/inputs/`` lookups (handy in
            tests). Defaults to
            :func:`cadgenbench.common.paths.data_inputs_dir`.

    Returns:
        The path to the written ``run_summary.json``.
    """
    run_dir = Path(run_dir).resolve()
    if data_inputs_dir is None:
        from cadgenbench.common.paths import data_inputs_dir as _data_inputs_dir
        data_inputs_dir = _data_inputs_dir()

    summary = _build_run_summary(run_dir, data_inputs_dir)
    out = run_dir / RUN_SUMMARY_NAME
    out.write_text(json.dumps(summary, indent=2))
    return out


def _build_run_summary(run_dir: Path, data_inputs_dir: Path) -> dict[str, Any]:
    """Read every fixture result.json under *run_dir* and assemble the
    aggregate summary dict. Pure function; no IO besides reads."""

    per_fixture: dict[str, dict[str, Any]] = {}
    fixture_scores: list[float] = []
    status_counts: dict[str, int] = {
        STATUS_VALID: 0, STATUS_INVALID: 0, STATUS_MISSING: 0,
    }
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "scores": [],
        "n_valid": 0, "n_invalid": 0, "n_missing": 0,
    })

    for fixture_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        result_json = fixture_dir / "result.json"
        if not result_json.exists():
            continue
        try:
            result = json.loads(result_json.read_text())
        except Exception:
            logger.warning("Skipping unreadable %s", result_json, exc_info=True)
            continue

        name = fixture_dir.name
        status = result.get("status") or STATUS_MISSING
        # Invalid + missing both contribute zero (leaderboard convention).
        cad_score = float(result.get("cad_score") or 0.0)
        task_type = _read_task_type(data_inputs_dir / name)

        per_fixture[name] = {
            "status": status,
            "cad_score": round(cad_score, 4),
            "task_type": task_type,
        }
        fixture_scores.append(cad_score)
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1

        bucket = by_task[task_type]
        bucket["scores"].append(cad_score)
        status_key = f"n_{status}" if status in {STATUS_VALID, STATUS_INVALID, STATUS_MISSING} else None
        if status_key is not None:
            bucket[status_key] = bucket.get(status_key, 0) + 1

    n_fixtures = len(per_fixture)
    aggregate = sum(fixture_scores) / n_fixtures if n_fixtures else 0.0
    n_valid = status_counts.get(STATUS_VALID, 0)

    per_task_scores: dict[str, dict[str, Any]] = {}
    score_by_task_type: dict[str, float] = {}
    for task_type, bucket in by_task.items():
        scores = bucket["scores"]
        n = len(scores)
        valid = bucket.get("n_valid", 0)
        per_task_scores[task_type] = {
            "score": round(sum(scores) / n, 4) if n else 0.0,
            "validity_rate": round(valid / n, 4) if n else 0.0,
            "n_fixtures": n,
            "n_valid": valid,
            "n_invalid": bucket.get("n_invalid", 0),
            "n_missing": bucket.get("n_missing", 0),
        }
        score_by_task_type[task_type] = per_task_scores[task_type]["score"]

    # Keep the headline ``score_by_task_type`` in a stable order: the
    # known types first (in declaration order), then any unknown task
    # types alphabetically.
    ordered_headline = {
        t: score_by_task_type[t] for t in _KNOWN_TASK_TYPES if t in score_by_task_type
    }
    for t in sorted(score_by_task_type):
        if t not in ordered_headline:
            ordered_headline[t] = score_by_task_type[t]

    return {
        "aggregate_score": round(aggregate, 4),
        "validity_rate": round(n_valid / n_fixtures, 4) if n_fixtures else 0.0,
        "n_fixtures": n_fixtures,
        "n_valid": n_valid,
        "n_invalid": status_counts.get(STATUS_INVALID, 0),
        "n_missing": status_counts.get(STATUS_MISSING, 0),
        "score_by_task_type": ordered_headline,
        "per_task_scores": per_task_scores,
        "per_fixture_scores": per_fixture,
    }


def _read_task_type(fixture_inputs_dir: Path) -> str:
    """Return ``description.yaml["task_type"]`` (defaulting to
    ``"generation"`` when the field is absent or the file is missing).
    """
    desc_path = fixture_inputs_dir / "description.yaml"
    if not desc_path.exists():
        return "generation"
    try:
        data = yaml.safe_load(desc_path.read_text()) or {}
    except Exception:
        logger.warning("Could not parse %s", desc_path, exc_info=True)
        return "generation"
    return str(data.get("task_type") or "generation")
