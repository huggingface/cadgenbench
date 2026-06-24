"""Tests for the editing-task no-op baseline + per-axis renormalization.

Pure-logic tests (renorm, gate, role classification, weighted composition,
staleness, file I/O) avoid the renderer; one geometry smoke test exercises
``compute_edit_baseline`` through the real align + metric path (no rendering).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cadgenbench import __version__
from cadgenbench.eval.edit_baseline import (
    EDIT_HEADROOM_FLOOR,
    EDITING_AXIS_WEIGHTS,
    PRESERVE_HEADROOM_FLOOR,
    ROLE_ABSENT,
    ROLE_PRESERVE,
    ROLE_TARGET,
    check_baseline_fresh,
    classify_axis,
    compute_edit_baseline,
    preserve_gate,
    read_edit_baseline,
    renormalize,
    write_edit_baseline,
)
from cadgenbench.eval.evaluate import (
    GENERATION_AXIS_WEIGHTS,
    _cad_score,
    _edit_cad_score,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "jig_metric"

_VALID = {"is_valid": True}


# ---------------------------------------------------------------------------
# renormalize (target axes)
# ---------------------------------------------------------------------------


def test_renorm_maps_noop_to_zero() -> None:
    assert renormalize(0.86, 0.86) == pytest.approx(0.0)


def test_renorm_maps_perfect_to_one() -> None:
    assert renormalize(1.0, 0.86) == pytest.approx(1.0)


def test_renorm_midpoint() -> None:
    b = 0.8
    assert renormalize((b + 1.0) / 2.0, b) == pytest.approx(0.5)


def test_renorm_floors_worse_than_noop_at_zero() -> None:
    assert renormalize(0.5, 0.86) == 0.0
    assert renormalize(0.0, 0.86) == 0.0


def test_renorm_zero_headroom_defensive() -> None:
    assert renormalize(1.0, 1.0) == 0.0
    assert renormalize(0.5, 1.0) == 0.0


# ---------------------------------------------------------------------------
# preserve_gate (one-sided break gate)
# ---------------------------------------------------------------------------


def test_gate_at_or_above_baseline_is_one() -> None:
    assert preserve_gate(1.0, 1.0) == pytest.approx(1.0)
    assert preserve_gate(0.99, 0.98) == pytest.approx(1.0)  # clamped


def test_gate_below_baseline_penalizes() -> None:
    assert preserve_gate(0.4, 1.0) == pytest.approx(0.4)
    assert preserve_gate(0.5, 1.0) == pytest.approx(0.5)


def test_gate_degenerate_baseline_is_one() -> None:
    assert preserve_gate(0.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# classify_axis
# ---------------------------------------------------------------------------


def test_classify_shape_always_target() -> None:
    # Even a tiny-headroom shape edit stays the anchor target axis.
    assert classify_axis("shape", 1.0 - EDIT_HEADROOM_FLOOR) == ROLE_TARGET


def test_classify_absent_when_baseline_none() -> None:
    assert classify_axis("interface", None) == ROLE_ABSENT


def test_classify_topo_target_above_floor_preserve_below() -> None:
    big = 1.0 - (PRESERVE_HEADROOM_FLOOR * 5)
    small = 1.0 - (PRESERVE_HEADROOM_FLOOR / 5)
    assert classify_axis("topology", big) == ROLE_TARGET
    assert classify_axis("topology", small) == ROLE_PRESERVE
    assert classify_axis("interface", 1.0) == ROLE_PRESERVE  # unchanged => gate


# ---------------------------------------------------------------------------
# _edit_cad_score (per-axis renorm + gate composition)
# ---------------------------------------------------------------------------


def _meta(shape_b, topo_b, iface_b):
    """Build an axes-meta dict with auto-classified roles."""
    out = {}
    for axis, b in (("shape", shape_b), ("topology", topo_b), ("interface", iface_b)):
        out[axis] = {"baseline": b, "headroom": None if b is None else 1.0 - b,
                     "role": classify_axis(axis, b)}
    return out


def test_edit_noop_scores_zero() -> None:
    # The no-op: raw == baseline on every axis. Targets renorm to 0, preserve
    # gates pass at 1 => overall 0.
    meta = _meta(0.95, 1.0, 1.0)  # shape target, topo+iface preserve
    raw = {"shape": 0.95, "topology": 1.0, "interface": 1.0}
    score, _ = _edit_cad_score(raw, meta, _VALID)
    assert score == pytest.approx(0.0)


def test_edit_perfect_scores_one() -> None:
    meta = _meta(0.95, 1.0, 1.0)
    raw = {"shape": 1.0, "topology": 1.0, "interface": 1.0}
    score, _ = _edit_cad_score(raw, meta, _VALID)
    assert score == pytest.approx(1.0)


def test_edit_only_shape_changes_scores_shape_renorm() -> None:
    # topo + iface unchanged (preserve, gates=1) => score == shape renorm.
    meta = _meta(0.8, 1.0, 1.0)
    raw = {"shape": 0.9, "topology": 1.0, "interface": 1.0}
    score, _ = _edit_cad_score(raw, meta, _VALID)
    assert score == pytest.approx(renormalize(0.9, 0.8))  # 0.5


def test_edit_breaking_preserved_axis_penalizes() -> None:
    # Nails the shape edit but cracks a preserved axis (raw 0.4 vs baseline 1).
    meta = _meta(0.8, 1.0, 1.0)
    raw = {"shape": 1.0, "topology": 0.4, "interface": 1.0}
    score, _ = _edit_cad_score(raw, meta, _VALID)
    assert score == pytest.approx(1.0 * 0.4)


def test_edit_two_targets_weighted_mean() -> None:
    # shape + topo both targets; iface preserve (gate 1). Weighted mean over
    # the two targets with EDITING_AXIS_WEIGHTS, renormalized over present.
    meta = _meta(0.8, 0.5, 1.0)  # topo headroom 0.5 >> floor => target
    raw = {"shape": 0.9, "topology": 0.75, "interface": 1.0}
    s_r = renormalize(0.9, 0.8)   # 0.5
    t_r = renormalize(0.75, 0.5)  # 0.5
    w_s, w_t = EDITING_AXIS_WEIGHTS["shape"], EDITING_AXIS_WEIGHTS["topology"]
    expected = (w_s * s_r + w_t * t_r) / (w_s + w_t)
    score, _ = _edit_cad_score(raw, meta, _VALID)
    assert score == pytest.approx(expected)


def test_edit_absent_axis_dropped() -> None:
    # interface absent (no jig). Score is the shape target alone (topo preserve).
    meta = _meta(0.8, 1.0, None)
    raw = {"shape": 0.9, "topology": 1.0, "interface": None}
    score, metrics = _edit_cad_score(raw, meta, _VALID)
    assert score == pytest.approx(renormalize(0.9, 0.8))
    assert metrics["axes"]["interface"]["role"] == ROLE_ABSENT


def test_edit_invalid_is_zero() -> None:
    meta = _meta(0.8, 1.0, 1.0)
    raw = {"shape": 1.0, "topology": 1.0, "interface": 1.0}
    score, _ = _edit_cad_score(raw, meta, {"is_valid": False})
    assert score == 0.0


# ---------------------------------------------------------------------------
# _cad_score weighting (generation path, unchanged)
# ---------------------------------------------------------------------------


def test_cad_score_no_weights_is_equal_mean() -> None:
    got = _cad_score(
        scores={"shape_similarity_score": 0.6},
        interface_metrics={"score": 0.9},
        topology_metrics={"score": 0.3},
        validation=_VALID,
    )
    assert got == pytest.approx((0.6 + 0.9 + 0.3) / 3)


def test_cad_score_generation_weights() -> None:
    got = _cad_score(
        scores={"shape_similarity_score": 0.6},
        interface_metrics={"score": 0.9},
        topology_metrics={"score": 0.3},
        validation=_VALID,
        weights=GENERATION_AXIS_WEIGHTS,
    )
    assert got == pytest.approx(0.4 * 0.6 + 0.4 * 0.9 + 0.2 * 0.3)


def test_cad_score_invalid_is_zero() -> None:
    got = _cad_score(
        scores={"shape_similarity_score": 0.95},
        interface_metrics={"score": 1.0},
        topology_metrics={"score": 1.0},
        validation={"is_valid": False},
        weights=GENERATION_AXIS_WEIGHTS,
    )
    assert got == 0.0


# ---------------------------------------------------------------------------
# file I/O + staleness
# ---------------------------------------------------------------------------


def test_read_baseline_absent_returns_none(tmp_path: Path) -> None:
    assert read_edit_baseline(tmp_path) is None


def test_write_read_roundtrip(tmp_path: Path) -> None:
    payload = {
        "axes": {"shape": {"baseline": 0.8, "headroom": 0.2, "role": ROLE_TARGET}},
        "cadgenbench_version": __version__,
    }
    write_edit_baseline(tmp_path, payload)
    assert read_edit_baseline(tmp_path) == payload


def test_check_baseline_fresh_passes_on_match() -> None:
    check_baseline_fresh({"cadgenbench_version": __version__}, "fix")


def test_check_baseline_fresh_raises_on_mismatch() -> None:
    with pytest.raises(RuntimeError, match="stale"):
        check_baseline_fresh({"cadgenbench_version": "0.0.0-old"}, "fix")


# ---------------------------------------------------------------------------
# compute_edit_baseline (real geometry, no renderer)
# ---------------------------------------------------------------------------


def _gt_dir_with(tmp_path: Path) -> Path:
    """A GT dir holding ground_truth.step (no jig => interface absent)."""
    src = FIXTURES_DIR / "test_1" / "gt.step"
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    shutil.copy(src, gt_dir / "ground_truth.step")
    return gt_dir


def test_compute_baseline_identical_input_gt(tmp_path: Path) -> None:
    # input == GT: near-perfect shape, identical topology, no jig.
    gt_dir = _gt_dir_with(tmp_path)
    baseline = compute_edit_baseline(gt_dir / "ground_truth.step", gt_dir)
    axes = baseline["axes"]
    assert baseline["cadgenbench_version"] == __version__
    assert axes["shape"]["baseline"] > 0.97
    assert axes["shape"]["role"] == ROLE_TARGET
    assert axes["topology"]["role"] == ROLE_PRESERVE  # unchanged topology
    assert axes["interface"]["role"] == ROLE_ABSENT   # no jig sub-volumes


def test_compute_baseline_real_edit_has_resolvable_headroom(tmp_path: Path) -> None:
    gt_dir = _gt_dir_with(tmp_path)
    other = FIXTURES_DIR / "test_1" / "candidates" / "broken_3_no_hole.step"
    baseline = compute_edit_baseline(other, gt_dir)
    b = baseline["axes"]["shape"]["baseline"]
    assert 0.0 < b < 1.0
    assert (1.0 - b) >= EDIT_HEADROOM_FLOOR
