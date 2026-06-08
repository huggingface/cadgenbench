"""Unit tests for cadgenbench.eval.run_summary.write_run_summary.

The aggregator reads only per-fixture result.json + description.yaml
(for task_type). It must never depend on baseline-only files.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from cadgenbench.eval.run_summary import write_run_summary


def _make_fixture_dir(
    run_dir: Path,
    name: str,
    *,
    status: str,
    cad_score: float,
) -> None:
    fd = run_dir / name
    fd.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "cad_score": cad_score}
    if status != "missing":
        payload["validation"] = {"is_valid": status == "valid"}
    (fd / "result.json").write_text(json.dumps(payload))


def _make_description(
    inputs_dir: Path,
    name: str,
    *,
    task_type: str = "generation",
) -> None:
    d = inputs_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "description.yaml").write_text(yaml.safe_dump({
        "description": "x",
        "task_type": task_type,
        "input_files": ["input.png"],
    }))


def test_includes_zeros_in_aggregate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    inputs_dir = tmp_path / "inputs"
    # 1 valid (0.9), 1 invalid (0.0), 1 missing (0.0).
    _make_fixture_dir(run_dir, "a", status="valid", cad_score=0.9)
    _make_fixture_dir(run_dir, "b", status="invalid", cad_score=0.0)
    _make_fixture_dir(run_dir, "c", status="missing", cad_score=0.0)
    for n in ("a", "b", "c"):
        _make_description(inputs_dir, n, task_type="generation")

    write_run_summary(run_dir, data_inputs_dir=inputs_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text())

    # Aggregate is mean over ALL 3 fixtures (0.9 + 0 + 0) / 3 = 0.3.
    assert summary["aggregate_score"] == 0.3
    assert summary["n_samples"] == 3
    assert summary["n_valid"] == 1
    assert summary["n_invalid"] == 1
    assert summary["n_missing"] == 1
    assert summary["validity_rate"] == round(1 / 3, 4)


def test_per_task_scores_split(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    inputs_dir = tmp_path / "inputs"
    # 2 generation (0.8, 0.6 mean=0.7), 1 editing (0.4).
    _make_fixture_dir(run_dir, "gen1", status="valid", cad_score=0.8)
    _make_fixture_dir(run_dir, "gen2", status="valid", cad_score=0.6)
    _make_fixture_dir(run_dir, "edit1", status="valid", cad_score=0.4)
    _make_description(inputs_dir, "gen1", task_type="generation")
    _make_description(inputs_dir, "gen2", task_type="generation")
    _make_description(inputs_dir, "edit1", task_type="editing")

    write_run_summary(run_dir, data_inputs_dir=inputs_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text())

    assert summary["score_by_task_type"]["generation"] == 0.7
    assert summary["score_by_task_type"]["editing"] == 0.4
    assert summary["per_task_scores"]["generation"]["n_samples"] == 2
    assert summary["per_task_scores"]["editing"]["n_samples"] == 1


def test_missing_description_defaults_to_generation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    inputs_dir = tmp_path / "inputs"  # exists but empty: no per-fixture description.yaml
    inputs_dir.mkdir()
    _make_fixture_dir(run_dir, "orphan", status="valid", cad_score=0.5)

    write_run_summary(run_dir, data_inputs_dir=inputs_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text())

    assert summary["per_sample_scores"]["orphan"]["task_type"] == "generation"
    assert summary["score_by_task_type"]["generation"] == 0.5


def test_empty_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    write_run_summary(run_dir, data_inputs_dir=inputs_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text())

    assert summary["n_samples"] == 0
    assert summary["aggregate_score"] == 0.0
    assert summary["validity_rate"] == 0.0
