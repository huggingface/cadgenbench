"""Tests for :mod:`cadgenbench.eval.topo_match`.

Coverage:

- Identity case: candidate == GT must score 1.0 with every per-axis
  score also 1.0.
- Known Betti shapes: box (1, 0, 0), plate-with-through-hole (1, 1, 0),
  plate-with-4-holes (1, 4, 0), hollow-ball (1, 0, 1), two-separate-cubes
  (2, 0, 0).
- Mismatch case: when one axis drifts, the aggregate stays strictly
  between 0 and 1 and the corresponding per-axis fuzzy log-ratio score
  drops below 1 while the other two axes still score 1.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from cadgenbench.eval.topo_match import (
    BettiResult,
    compute_betti_for_step,
    topo_match,
    topo_match_score,
)


def _expected_axis_score(b_cand: int, b_gt: int) -> float:
    """Reference implementation of the fuzzy log-ratio for one Betti axis."""
    from cadgenbench.eval.topo_match import BETTI_SHARPNESS

    return math.exp(-BETTI_SHARPNESS * abs(math.log((b_cand + 1) / (b_gt + 1))))


# ---------------------------------------------------------------------------
# build123d shape fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_steps(tmp_path_factory) -> dict[str, Path]:
    """Build a suite of known-Betti shapes once per module."""
    from build123d import (
        BuildPart,
        Box,
        Cylinder,
        Hole,
        Locations,
        Mode,
        Sphere,
        export_step,
    )

    tmp = tmp_path_factory.mktemp("topo_match_steps")
    out: dict[str, Path] = {}

    # Solid box: (1, 0, 0)
    with BuildPart() as p:
        Box(20, 20, 5)
    out["box"] = tmp / "box.step"
    export_step(p.part, str(out["box"]))

    # Plate with one through-hole: (1, 1, 0)
    with BuildPart() as p:
        Box(20, 20, 5)
        with Locations((0, 0, 2.5)):
            Hole(radius=3, depth=5)
    out["plate_1hole"] = tmp / "plate_1hole.step"
    export_step(p.part, str(out["plate_1hole"]))

    # Plate with four through-holes: (1, 4, 0)
    with BuildPart() as p:
        Box(40, 40, 5)
        with Locations(
            (-12, -12, 2.5), (12, -12, 2.5),
            (-12, 12, 2.5), (12, 12, 2.5),
        ):
            Hole(radius=3, depth=5)
    out["plate_4holes"] = tmp / "plate_4holes.step"
    export_step(p.part, str(out["plate_4holes"]))

    # Hollow ball (sphere with smaller sphere subtracted): (1, 0, 1)
    with BuildPart() as p:
        Sphere(10)
        Sphere(8, mode=Mode.SUBTRACT)
    out["hollow_ball"] = tmp / "hollow_ball.step"
    export_step(p.part, str(out["hollow_ball"]))

    # Two separate cubes, distinct solid bodies: (2, 0, 0)
    with BuildPart() as p:
        with Locations((-10, 0, 0)):
            Box(5, 5, 5)
        with Locations((10, 0, 0)):
            Box(5, 5, 5)
    out["two_cubes"] = tmp / "two_cubes.step"
    export_step(p.part, str(out["two_cubes"]))

    return out


# ---------------------------------------------------------------------------
# Betti for known shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("box", (1, 0, 0)),
        ("plate_1hole", (1, 1, 0)),
        ("plate_4holes", (1, 4, 0)),
        ("hollow_ball", (1, 0, 1)),
        ("two_cubes", (2, 0, 0)),
    ],
)
def test_betti_for_known_shape(
    tmp_steps: dict[str, Path],
    name: str,
    expected: tuple[int, int, int],
) -> None:
    r = compute_betti_for_step(tmp_steps[name])
    assert (r.b0, r.b1, r.b2) == expected, (
        f"{name}: got ({r.b0}, {r.b1}, {r.b2}), expected {expected}"
    )


# ---------------------------------------------------------------------------
# topo_match score logic
# ---------------------------------------------------------------------------


class TestTopoMatchIdentity:
    """Comparing a shape against itself must produce a perfect score."""

    def test_identity_scores_one(self, tmp_steps: dict[str, Path]) -> None:
        r = topo_match(tmp_steps["plate_1hole"], tmp_steps["plate_1hole"])
        assert r.score == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0) for v in r.per_axis_scores.values())


class TestTopoMatchMismatch:
    """Different shapes score in (0, 1) and surface per-axis decay."""

    def test_plate_1hole_vs_plate_4holes(self, tmp_steps: dict[str, Path]) -> None:
        # (1, 1, 0) vs (1, 4, 0): only b1 differs (1+1=2 vs 4+1=5 → 2/5).
        r = topo_match(tmp_steps["plate_1hole"], tmp_steps["plate_4holes"])
        expected_b1 = _expected_axis_score(1, 4)
        assert r.per_axis_scores["b0"] == pytest.approx(1.0)
        assert r.per_axis_scores["b1"] == pytest.approx(expected_b1)
        assert r.per_axis_scores["b2"] == pytest.approx(1.0)
        assert r.score == pytest.approx(expected_b1)

    def test_box_vs_two_cubes(self, tmp_steps: dict[str, Path]) -> None:
        # (1, 0, 0) vs (2, 0, 0): only b0 differs (2/3).
        r = topo_match(tmp_steps["box"], tmp_steps["two_cubes"])
        expected_b0 = _expected_axis_score(1, 2)
        assert r.per_axis_scores["b0"] == pytest.approx(expected_b0)
        assert r.per_axis_scores["b1"] == pytest.approx(1.0)
        assert r.per_axis_scores["b2"] == pytest.approx(1.0)
        assert r.score == pytest.approx(expected_b0)

    def test_box_vs_hollow_ball(self, tmp_steps: dict[str, Path]) -> None:
        # (1, 0, 0) vs (1, 0, 1): only b2 differs (1/2).
        r = topo_match(tmp_steps["box"], tmp_steps["hollow_ball"])
        expected_b2 = _expected_axis_score(0, 1)
        assert r.per_axis_scores["b0"] == pytest.approx(1.0)
        assert r.per_axis_scores["b1"] == pytest.approx(1.0)
        assert r.per_axis_scores["b2"] == pytest.approx(expected_b2)
        assert r.score == pytest.approx(expected_b2)


class TestTopoMatchScoreFn:
    """`topo_match_score` is a pure function, exercise its boundaries."""

    @staticmethod
    def _mk(b0: int, b1: int, b2: int) -> BettiResult:
        return BettiResult(
            b0=b0, b1=b1, b2=b2,
            chi_surface=2 * (b0 - b1 + b2),
            n_components=1,
            n_triangles=12,
            n_vertices=8,
            linear_deflection_mm=0.01,
        )

    def test_all_match(self) -> None:
        score, m = topo_match_score(self._mk(1, 3, 0), self._mk(1, 3, 0))
        assert score == pytest.approx(1.0)
        assert m == pytest.approx({"b0": 1.0, "b1": 1.0, "b2": 1.0})

    def test_all_differ(self) -> None:
        # (2, 1, 1) vs (1, 0, 0): every axis drifts; the score is the
        # product of three fuzzy log-ratios, never zero (no axis is exact).
        score, m = topo_match_score(self._mk(2, 1, 1), self._mk(1, 0, 0))
        expected = {
            "b0": _expected_axis_score(2, 1),
            "b1": _expected_axis_score(1, 0),
            "b2": _expected_axis_score(1, 0),
        }
        assert m == pytest.approx(expected)
        assert score == pytest.approx(
            expected["b0"] * expected["b1"] * expected["b2"],
        )
        assert 0.0 < score < 1.0

    def test_one_differs(self) -> None:
        # (1, 5, 0) vs (1, 4, 0): b1 off by one (6/5 ratio). With the
        # product aggregate the two exact axes pass through and the score
        # equals the single non-exact axis.
        score, m = topo_match_score(self._mk(1, 5, 0), self._mk(1, 4, 0))
        expected_b1 = _expected_axis_score(5, 4)
        assert m == pytest.approx(
            {"b0": 1.0, "b1": expected_b1, "b2": 1.0},
        )
        assert score == pytest.approx(expected_b1)

    def test_negative_betti_scores_zero_without_crashing(self) -> None:
        # A degenerate candidate (b1 = -1 from a broken mesh) must not crash
        # the log-ratio; its topology axis is zeroed, which collapses the
        # product to 0.
        score, m = topo_match_score(self._mk(1, -1, 0), self._mk(1, 1, 0))
        assert m["b1"] == 0.0
        assert m["b0"] == pytest.approx(1.0)
        assert m["b2"] == pytest.approx(1.0)
        assert score == pytest.approx(0.0)

    def test_axis_score_is_symmetric(self) -> None:
        # Swapping candidate and GT must leave each per-axis score and
        # the aggregate unchanged.
        score_a, axes_a = topo_match_score(
            self._mk(1, 3, 2), self._mk(2, 8, 0),
        )
        score_b, axes_b = topo_match_score(
            self._mk(2, 8, 0), self._mk(1, 3, 2),
        )
        assert score_a == pytest.approx(score_b)
        assert axes_a == pytest.approx(axes_b)
