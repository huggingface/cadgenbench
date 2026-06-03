"""Tests for the editing-task no-op baseline + shape-axis renormalization.

Pure-logic tests (renorm, weighting, staleness, file I/O) avoid the
renderer; one geometry smoke test exercises ``compute_edit_baseline``
through the real shape-similarity path (no rendering involved).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cadgenbench import __version__
from cadgenbench.eval.edit_baseline import (
    EDIT_HEADROOM_FLOOR,
    EDITING_AXIS_WEIGHTS,
    check_baseline_fresh,
    compute_edit_baseline,
    read_edit_baseline,
    renormalize_shape,
    write_edit_baseline,
)
from cadgenbench.eval.evaluate import GENERATION_AXIS_WEIGHTS, _cad_score

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "jig_metric"


# ---------------------------------------------------------------------------
# renormalize_shape
# ---------------------------------------------------------------------------


def test_renorm_maps_noop_to_zero() -> None:
    assert renormalize_shape(0.86, 0.86) == pytest.approx(0.0)


def test_renorm_maps_perfect_to_one() -> None:
    assert renormalize_shape(1.0, 0.86) == pytest.approx(1.0)


def test_renorm_midpoint() -> None:
    b = 0.8
    assert renormalize_shape((b + 1.0) / 2.0, b) == pytest.approx(0.5)


def test_renorm_floors_worse_than_noop_at_zero() -> None:
    assert renormalize_shape(0.5, 0.86) == 0.0
    assert renormalize_shape(0.0, 0.86) == 0.0


def test_renorm_zero_headroom_defensive() -> None:
    # b_shape == 1 should never reach scoring (authoring gate rejects it),
    # but the function must not divide by zero.
    assert renormalize_shape(1.0, 1.0) == 0.0
    assert renormalize_shape(0.5, 1.0) == 0.0


# ---------------------------------------------------------------------------
# _cad_score weighting
# ---------------------------------------------------------------------------

_VALID = {"is_valid": True}


def test_cad_score_no_weights_is_equal_mean() -> None:
    # weights=None ⇒ plain mean over present axes (the function default).
    got = _cad_score(
        scores={"shape_similarity_score": 0.6},
        interface_metrics={"score": 0.9},
        topology_metrics={"score": 0.3},
        validation=_VALID,
    )
    assert got == pytest.approx((0.6 + 0.9 + 0.3) / 3)


def test_cad_score_generation_weights() -> None:
    # Generation: shape 0.4 / interface 0.4 / topology 0.2.
    got = _cad_score(
        scores={"shape_similarity_score": 0.6},
        interface_metrics={"score": 0.9},
        topology_metrics={"score": 0.3},
        validation=_VALID,
        weights=GENERATION_AXIS_WEIGHTS,
    )
    assert got == pytest.approx(0.4 * 0.6 + 0.4 * 0.9 + 0.2 * 0.3)


def test_cad_score_editing_weights_and_shape_override() -> None:
    # Editing: shape axis is overridden with the renormalized value and
    # weighted 0.5 / 0.3 / 0.2 (shape / interface / topology).
    got = _cad_score(
        scores={"shape_similarity_score": 0.95},  # raw, must be ignored
        interface_metrics={"score": 0.8},
        topology_metrics={"score": 1.0},
        validation=_VALID,
        shape_score=0.0,  # renormalized no-op
        weights=EDITING_AXIS_WEIGHTS,
    )
    assert got == pytest.approx(0.5 * 0.0 + 0.3 * 0.8 + 0.2 * 1.0)


def test_cad_score_editing_reweights_when_interface_absent() -> None:
    # No interface axis: shape 0.5 + topo 0.2 renormalize over 0.7.
    got = _cad_score(
        scores={"shape_similarity_score": 0.95},
        interface_metrics={},
        topology_metrics={"score": 0.9},
        validation=_VALID,
        shape_score=0.4,
        weights=EDITING_AXIS_WEIGHTS,
    )
    assert got == pytest.approx((0.5 * 0.4 + 0.2 * 0.9) / 0.7)


def test_cad_score_invalid_is_zero_regardless_of_weights() -> None:
    got = _cad_score(
        scores={"shape_similarity_score": 0.95},
        interface_metrics={"score": 1.0},
        topology_metrics={"score": 1.0},
        validation={"is_valid": False},
        shape_score=1.0,
        weights=EDITING_AXIS_WEIGHTS,
    )
    assert got == 0.0


# ---------------------------------------------------------------------------
# file I/O + staleness
# ---------------------------------------------------------------------------


def test_read_baseline_absent_returns_none(tmp_path: Path) -> None:
    assert read_edit_baseline(tmp_path) is None


def test_write_read_roundtrip(tmp_path: Path) -> None:
    payload = {"shape_similarity_score": 0.8, "cadgenbench_version": __version__}
    write_edit_baseline(tmp_path, payload)
    got = read_edit_baseline(tmp_path)
    assert got == payload


def test_check_baseline_fresh_passes_on_match() -> None:
    check_baseline_fresh({"cadgenbench_version": __version__}, "fix")


def test_check_baseline_fresh_raises_on_mismatch() -> None:
    with pytest.raises(RuntimeError, match="stale"):
        check_baseline_fresh({"cadgenbench_version": "0.0.0-old"}, "fix")


# ---------------------------------------------------------------------------
# compute_edit_baseline (real geometry, no renderer)
# ---------------------------------------------------------------------------


def test_compute_baseline_identical_input_gt_scores_near_perfect() -> None:
    # input == GT (a degenerate "no edit") scores ~1 on shape similarity:
    # the no-op problem in its most extreme form. It does not hit exactly
    # 1.0 because of tessellation residue on volume IoU, so the tiny
    # EDIT_HEADROOM_FLOOR (numerical-stability guard, not a "meaningful
    # edit" threshold) does not by itself reject a no-edit.
    gt = FIXTURES_DIR / "test_1" / "gt.step"
    baseline = compute_edit_baseline(gt, gt)
    assert baseline["cadgenbench_version"] == __version__
    assert baseline["shape_similarity_score"] > 0.99


def test_compute_baseline_real_edit_has_resolvable_headroom() -> None:
    # A genuinely different part (gt vs a broken candidate) leaves real
    # shape headroom: well above the authoring floor, below 1.
    gt = FIXTURES_DIR / "test_1" / "gt.step"
    other = FIXTURES_DIR / "test_1" / "candidates" / "broken_3_no_hole.step"
    baseline = compute_edit_baseline(other, gt)
    b = baseline["shape_similarity_score"]
    assert 0.0 < b < 1.0
    assert (1.0 - b) >= EDIT_HEADROOM_FLOOR
