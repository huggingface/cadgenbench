"""Tests for the interface-match metric category.

Two layers:

1. **Public-API contract** -- :class:`SubVolume`, :func:`discover_sub_volumes`,
   :func:`iou_at_pose`, :func:`best_iou_in_context`,
   :func:`interface_score_iou`, :func:`interface_score`. These are the primitives the
   metric, the visualiser, and any future verifier all share.
2. **Discrimination check** on the committed jig_metric fixtures
   (``tests/fixtures/jig_metric/test_*``):
   - GT and ``correct.step`` -- disagreement ≈ 0 / IoU = 1.0 across
     every sub-volume.
   - every ``broken_*.step`` -- disagreement above the epsilon (or
     IoU below the threshold) for at least one sub-volume (the metric
     must flag the deliberate failure even after the bounded pose
     search).

The same parametrisation works on any future dataset that follows the
``jig_<context_id>__<index>__<fit_type>.step`` naming -- no test
scaffolding changes required, only a new fixtures root.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _multiprocessing_available() -> bool:
    """Return ``True`` if :class:`ProcessPoolExecutor` can be created here.

    The Cursor agent sandbox blocks ``os.sysconf("SC_SEM_NSEMS_MAX")``,
    which the stdlib uses to vet named-semaphore support before
    spawning workers. Outside the sandbox this returns ``True`` on
    macOS / Linux. Used to skip the multiprocessing test cleanly
    instead of crashing on a sandbox limitation.
    """
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except (OSError, ValueError, PermissionError):
        return False
    return True

from cadgenbench.eval.interface_match import (
    DEFAULT_IOU_THRESHOLD,
    FIT_TYPES,
    SubVolume,
    best_iou_in_context,
    discover_sub_volumes,
    interface_score,
    interface_score_iou,
    iou_at_pose,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
JIG_FIXTURES_ROOT = FIXTURES_DIR / "jig_metric"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

# Expected sub-volume counts per fixture (sanity-checks the committed data).
_EXPECTED_SUBVOL_COUNT = {
    "test_1": 1,
    "test_2": 4,
    "test_3": 5,
    "test_4": 3,
}


@pytest.fixture(scope="session")
def fixture_dirs() -> list[Path]:
    dirs = sorted(d for d in JIG_FIXTURES_ROOT.glob("test_*") if d.is_dir())
    if not dirs:
        pytest.skip(f"No jig_metric fixtures under {JIG_FIXTURES_ROOT}")
    return dirs


class TestDiscovery:

    def test_finds_all_fixtures(self, fixture_dirs: list[Path]) -> None:
        assert {d.name for d in fixture_dirs} == set(_EXPECTED_SUBVOL_COUNT)

    @pytest.mark.parametrize("test_name, expected_n", sorted(_EXPECTED_SUBVOL_COUNT.items()))
    def test_subvolume_count_matches_spec(self, test_name: str, expected_n: int) -> None:
        svs = discover_sub_volumes(JIG_FIXTURES_ROOT / test_name)
        assert len(svs) == expected_n, [sv.path.name for sv in svs]

    def test_subvolume_fields_well_formed(self, fixture_dirs: list[Path]) -> None:
        for d in fixture_dirs:
            for sv in discover_sub_volumes(d):
                assert isinstance(sv, SubVolume)
                assert sv.context_id >= 1
                assert sv.index >= 1
                assert sv.fit_type in FIT_TYPES
                assert sv.path.exists()
                assert sv.name == f"{sv.index}__{sv.fit_type}"

    def test_test_3_one_context_mixed_fit_types(self) -> None:
        """test_3 is the canonical mixed-context fixture (bolts + boss)."""
        svs = discover_sub_volumes(JIG_FIXTURES_ROOT / "test_3")
        assert {sv.context_id for sv in svs} == {1}
        assert {sv.fit_type for sv in svs} == {"KOR", "KIR"}

    def test_test_4_three_independent_contexts(self) -> None:
        """test_4 has three mechanically independent interfaces."""
        svs = discover_sub_volumes(JIG_FIXTURES_ROOT / "test_4")
        assert {sv.context_id for sv in svs} == {1, 2, 3}


# ---------------------------------------------------------------------------
# Discrimination fixtures (shared parametrisation)
# ---------------------------------------------------------------------------

# ``(fixture_dir, candidate_path, expect_pass)``: GT + correct.step pass,
# every broken_*.step fails.
def _discrimination_cases() -> list[tuple[Path, Path, bool]]:
    cases: list[tuple[Path, Path, bool]] = []
    for fixture_dir in sorted(JIG_FIXTURES_ROOT.glob("test_*")):
        if not fixture_dir.is_dir():
            continue
        cases.append((fixture_dir, fixture_dir / "gt.step", True))
        cases.append((fixture_dir, fixture_dir / "candidates" / "correct.step", True))
        for broken in sorted((fixture_dir / "candidates").glob("broken_*.step")):
            cases.append((fixture_dir, broken, False))
    return cases


_CASES = _discrimination_cases()
_CASE_IDS = [
    f"{fd.name}/{cp.name}" for fd, cp, _ in _CASES
] if _CASES else []


# ---------------------------------------------------------------------------
# IoU-based discrimination (the v1 metric primitive)
# ---------------------------------------------------------------------------

class TestIoUDiscrimination:
    """Same fixtures × candidates parametrisation as the disagreement-based
    check, but asserts on the pose-searched IoU per the v1 spec:
    correct -> IoU >= 0.95, broken -> IoU < 0.95 for at least one
    sub-volume.
    """

    @pytest.mark.parametrize("fixture_dir, candidate, expect_pass", _CASES, ids=_CASE_IDS)
    def test_iou_discrimination(
        self,
        fixture_dir: Path,
        candidate: Path,
        expect_pass: bool,
    ) -> None:
        ious = interface_score_iou(candidate, fixture_dir)
        worst = min(ious.values()) if ious else 1.0
        if expect_pass:
            assert worst >= DEFAULT_IOU_THRESHOLD, (
                f"{candidate.name} expected PASS but worst IoU is "
                f"{worst:.3f} (threshold {DEFAULT_IOU_THRESHOLD}); "
                f"ious={ious}"
            )
        else:
            assert worst < DEFAULT_IOU_THRESHOLD, (
                f"{candidate.name} expected FAIL but worst IoU is "
                f"{worst:.3f} (>= threshold {DEFAULT_IOU_THRESHOLD}); "
                f"ious={ious}"
            )

    def test_correct_candidate_perfect_iou(self) -> None:
        """A GT-identical correct.step must score IoU = 1.0 per sub-volume."""
        d = JIG_FIXTURES_ROOT / "test_3"
        correct = d / "candidates" / "correct.step"
        gt_step = d / "gt.step"
        for sv in discover_sub_volumes(d):
            iou = iou_at_pose(correct, sv, gt_step)
            assert iou == pytest.approx(1.0, abs=1e-3), (
                f"{sv.name}: expected IoU=1.0, got {iou:.4f}"
            )

    def test_score_iou_keys_match_sub_volume_names(
        self, fixture_dirs: list[Path],
    ) -> None:
        for fd in fixture_dirs:
            expected = {sv.name for sv in discover_sub_volumes(fd)}
            actual = set(interface_score_iou(fd / "gt.step", fd))
            assert actual == expected, fd


# ---------------------------------------------------------------------------
# Pose search invariants
# ---------------------------------------------------------------------------

class TestPoseSearch:
    """Properties the bounded deterministic pose search must hold.

    1. Pose-search IoU >= GT-pose IoU per sub-volume (the zero pose is
       always sampled, so ``max`` over samples can't go below it).
    2. Multiprocessing path matches the single-process path bit-for-bit
       because the pose set is deterministic and generated on the main process.
    3. Regression: obvious missing-feature cases (no hole / missing hole)
       must still hit an exact zero IoU for at least one sub-volume.
    """

    @pytest.fixture
    def t3_inputs(self) -> tuple[Path, Path, list]:
        d = JIG_FIXTURES_ROOT / "test_3"
        correct = d / "candidates" / "broken_3_shifted_holes.step"
        gt_step = d / "gt.step"
        # Single context (test_3 -> all 5 sub-volumes share context_id=1).
        svs = discover_sub_volumes(d)
        return correct, gt_step, svs

    def test_pose_search_at_least_gt_pose(
        self, t3_inputs: tuple[Path, Path, list],
    ) -> None:
        candidate, gt_step, svs = t3_inputs
        searched = best_iou_in_context(
            candidate, svs, gt_step,
            n_samples=8,
        )
        for sv in svs:
            gt_pose = iou_at_pose(candidate, sv, gt_step)
            assert searched[sv.name] >= gt_pose - 1e-6, (
                f"{sv.name}: pose search dropped below zero-pose IoU "
                f"({searched[sv.name]:.4f} < {gt_pose:.4f})"
            )

    @pytest.mark.skipif(
        not _multiprocessing_available(),
        reason="multiprocessing semaphores unavailable (sandbox)",
    )
    def test_workers_match_single_process(
        self, t3_inputs: tuple[Path, Path, list],
    ) -> None:
        candidate, gt_step, svs = t3_inputs
        single = best_iou_in_context(
            candidate, svs, gt_step,
            n_samples=4, workers=1,
        )
        parallel = best_iou_in_context(
            candidate, svs, gt_step,
            n_samples=4, workers=2,
        )
        assert single.keys() == parallel.keys()
        for name in single:
            assert single[name] == pytest.approx(parallel[name], abs=1e-6), (
                f"{name}: single={single[name]:.4f}, parallel={parallel[name]:.4f}"
            )

    def test_repeated_calls_are_identical(self, t3_inputs: tuple[Path, Path, list]) -> None:
        candidate, gt_step, svs = t3_inputs
        first = best_iou_in_context(candidate, svs, gt_step, n_samples=8)
        second = best_iou_in_context(candidate, svs, gt_step, n_samples=8)
        assert first == second

    def test_no_hole_candidate_stays_exact_zero(self) -> None:
        d = JIG_FIXTURES_ROOT / "test_1"
        candidate = d / "candidates" / "broken_3_no_hole.step"
        svs = discover_sub_volumes(d)
        got = best_iou_in_context(candidate, svs, d / "gt.step", n_samples=32)
        assert len(got) == 1
        only_value = next(iter(got.values()))
        assert only_value == pytest.approx(0.0, abs=1e-6), got

    def test_missing_hole_has_at_least_one_exact_zero_iou(self) -> None:
        d = JIG_FIXTURES_ROOT / "test_2"
        candidate = d / "candidates" / "broken_2_missing_hole.step"
        ious = interface_score_iou(candidate, d)
        assert min(ious.values()) == pytest.approx(0.0, abs=1e-6), ious


# ---------------------------------------------------------------------------
# Aggregated single-number score (mean of per-context mins)
# ---------------------------------------------------------------------------

# Pinned scores. Any future math change should be a conscious decision,
# not silent drift. Values are reproducible from interface_score() with the
# deterministic default sampler, default n_samples=32, default ±1% of GT
# bounding-box diagonal translation budget, ±1° rotation budget, and the
# mean-of-context-mins aggregation rule.
_EXPECTED_SCORES = {
    ("test_1", "gt.step"):                                 1.000,
    ("test_1", "candidates/correct.step"):                 1.000,
    # 0.809 (was 0.804): the sub-volume is now meshed once at the parent-GT
    # deflection (the override), instead of being pre-tessellated at its own
    # finer deflection by the validity gate and then left uncoarsened by
    # BRepMesh. This is the override working as designed.
    ("test_1", "candidates/broken_1_small_hole.step"):     0.809,
    ("test_1", "candidates/broken_2_offset_hole.step"):    0.348,
    ("test_1", "candidates/broken_3_no_hole.step"):        0.000,
    ("test_2", "gt.step"):                                 1.000,
    ("test_2", "candidates/correct.step"):                 1.000,
    ("test_2", "candidates/broken_1_wrong_spacing.step"):  0.522,
    ("test_2", "candidates/broken_2_missing_hole.step"):   0.000,
    ("test_2", "candidates/broken_3_wrong_diameter.step"): 0.640,
    ("test_3", "gt.step"):                                 1.000,
    ("test_3", "candidates/correct.step"):                 1.000,
    ("test_3", "candidates/broken_1_cylinder_boss.step"):  0.860,
    ("test_3", "candidates/broken_2_rotated_boss.step"):   0.897,
    ("test_3", "candidates/broken_3_shifted_holes.step"):  0.718,
    ("test_4", "gt.step"):                                 1.000,
    ("test_4", "candidates/correct.step"):                 1.000,
    ("test_4", "candidates/broken_1_narrow_slot.step"):    0.923,
    ("test_4", "candidates/broken_2_slot_offset.step"):    0.750,
}


class TestAggregatedScore:
    """Pin the single-number aggregated score for each fixture x candidate.

    interface_score() = mean over contexts of (min over the context's sub-volume IoUs),
    with each per-sub-volume IoU itself the max over the bounded
    deterministic pose search.
    """

    @pytest.mark.parametrize(
        "test_name, candidate_rel, expected",
        [(t, p, s) for (t, p), s in _EXPECTED_SCORES.items()],
        ids=[f"{t}/{p}" for (t, p) in _EXPECTED_SCORES],
    )
    def test_score_matches_expected(
        self, test_name: str, candidate_rel: str, expected: float,
    ) -> None:
        fixture_dir = JIG_FIXTURES_ROOT / test_name
        candidate = fixture_dir / candidate_rel
        got = interface_score(candidate, fixture_dir)
        assert got == pytest.approx(expected, abs=0.005), (
            f"{test_name}/{candidate_rel}: expected score={expected}, "
            f"got {got:.4f}"
        )

    def test_score_in_unit_interval(self, fixture_dirs: list[Path]) -> None:
        for fd in fixture_dirs:
            for candidate in [fd / "gt.step", *(fd / "candidates").iterdir()]:
                if candidate.suffix.lower() != ".step":
                    continue
                s = interface_score(candidate, fd, n_samples=4)
                assert 0.0 <= s <= 1.0, (candidate, s)
