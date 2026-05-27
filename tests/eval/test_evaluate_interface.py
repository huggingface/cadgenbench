"""Focused tests for evaluate.py interface-metrics integration helper."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadgenbench.eval.evaluate import _interface_metrics_dict

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "jig_metric"


def test_interface_metrics_empty_when_no_jig_files(tmp_path: Path) -> None:
    got = _interface_metrics_dict(
        aligned_candidate_step=tmp_path / "candidate.step",
        fixture_dir=tmp_path,
        gt_step=tmp_path / "ground_truth.step",
        n_samples=4,
    )
    assert got == {}


def test_interface_metrics_for_correct_candidate() -> None:
    fixture = FIXTURES_DIR / "test_3"
    got = _interface_metrics_dict(
        aligned_candidate_step=fixture / "candidates" / "correct.step",
        fixture_dir=fixture,
        gt_step=fixture / "gt.step",
        n_samples=4,
    )
    assert got["score"] == pytest.approx(1.0, abs=1e-3)
    assert set(got["contexts"]) == {"1"}
    assert got["contexts"]["1"]["score"] == pytest.approx(1.0, abs=1e-3)
    assert len(got["contexts"]["1"]["sub_volumes"]) == 5


def test_interface_metrics_missing_hole_scores_zero() -> None:
    fixture = FIXTURES_DIR / "test_2"
    got = _interface_metrics_dict(
        aligned_candidate_step=fixture / "candidates" / "broken_2_missing_hole.step",
        fixture_dir=fixture,
        gt_step=fixture / "gt.step",
        n_samples=4,
    )
    assert got["score"] == pytest.approx(0.0, abs=1e-6)
    assert got["contexts"]["1"]["score"] == pytest.approx(0.0, abs=1e-6)
