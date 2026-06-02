"""Unit tests for shape-similarity metrics."""
from __future__ import annotations

from pathlib import Path

import pytest

from cadgenbench.eval.shape_similarity import (
    MetricContext,
    compute_metrics,
    shape_feature_edge_f1,
    shape_point_cloud_f1,
    shape_volume_iou,
)
from cadgenbench.common.measurements import BBox, Measurements


def _meas(*, volume: float = 1000.0, bbox: BBox | None = None) -> Measurements:
    if bbox is None:
        bbox = BBox(0, 10, 0, 10, 0, 10)
    return Measurements(
        solid_count=1, shell_count=1, face_count=6,
        volume=volume, bounding_box=bbox,
    )


def _make_step_box(x: float, y: float, z: float) -> Path:
    """Create a box STEP file, cached by dimensions."""
    from build123d import Box, BuildPart, export_step

    fixtures = Path(__file__).parent / "fixtures"
    fixtures.mkdir(exist_ok=True)
    path = fixtures / f"cd_box_{x}_{y}_{z}.step"
    if not path.exists():
        with BuildPart() as p:
            Box(x, y, z)
        export_step(p.part, str(path))
    return path


def _ctx_with_step(x: float, y: float, z: float) -> MetricContext:
    """MetricContext with step, validation, and measurements (for bbox diagonal)."""
    from cadgenbench.common.validity import analyze_step

    path = _make_step_box(x, y, z)
    a = analyze_step(path)
    return MetricContext(
        step_path=path, validation=a.validation, measurements=a.measurements,
    )


# ---------------------------------------------------------------------------
# shape_point_cloud_f1
# ---------------------------------------------------------------------------

class TestShapePointCloudF1:

    def test_identical_shape_near_one(self) -> None:
        a = _ctx_with_step(10, 20, 30)
        score = shape_point_cloud_f1(a, a)
        assert score is not None
        assert score > 0.95

    def test_symmetric(self) -> None:
        a = _ctx_with_step(10, 20, 30)
        b = _ctx_with_step(12, 22, 32)
        ab = shape_point_cloud_f1(a, b)
        ba = shape_point_cloud_f1(b, a)
        assert ab is not None and ba is not None
        assert ab == pytest.approx(ba, rel=0.15)

    def test_dissimilar_shapes_lower_score(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        b = _ctx_with_step(10, 10, 50)
        score = shape_point_cloud_f1(a, b)
        self_score = shape_point_cloud_f1(a, a)
        assert score is not None
        assert self_score is not None
        assert score < self_score - 0.05

    def test_returns_float_in_range(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        score = shape_point_cloud_f1(a, a)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_returns_none_without_step(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        b = MetricContext()
        assert shape_point_cloud_f1(a, b) is None
        assert shape_point_cloud_f1(b, a) is None

    def test_returns_none_without_gt_step(self) -> None:
        """Without a STEP path on GT the metric can't sample, returns None."""
        path = _make_step_box(10, 10, 10)
        a = MetricContext(step_path=path, measurements=_meas())
        b = MetricContext()
        assert shape_point_cloud_f1(a, b) is None

    def test_normal_gate_accepts_flipped_normals_via_absdot(self) -> None:
        """Inverted normals at coincident points are ACCEPTED (|dot| gate).

        The hit gate matches on ``|n_cand · n_gt|`` (orientation-insensitive),
        so a fully-inverted candidate at identical positions still scores ~1.
        This is intentional: flipped/winding-convention differences are not a
        shape difference, and genuine "wrong side" geometry is excluded upstream
        by the watertight + manifold + winding gate, not here.
        """
        import numpy as np

        a = _ctx_with_step(10, 10, 10)
        baseline = shape_point_cloud_f1(a, a)
        assert baseline is not None and baseline > 0.95

        b = _ctx_with_step(10, 10, 10)
        _ = b.point_cloud
        b._pc_normals = -1.0 * b._pc_normals  # noqa: SLF001
        flipped = shape_point_cloud_f1(a, b)
        assert flipped is not None
        assert flipped > 0.95, flipped  # |dot| -> flip accepted
        sanity = float(np.linalg.norm(
            a.point_cloud[:5] - b.point_cloud[:5], axis=1,
        ).max())
        assert sanity < 1e-9, sanity


# ---------------------------------------------------------------------------
# shape_volume_iou
# ---------------------------------------------------------------------------

class TestShapeVolumeIoU:

    def test_identical_shape_one(self) -> None:
        a = _ctx_with_step(10, 20, 30)
        score = shape_volume_iou(a, a)
        assert score is not None
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_different_shapes_lower_score(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        b = _ctx_with_step(10, 10, 50)
        score = shape_volume_iou(a, b)
        assert score is not None
        assert score < 0.5

    def test_returns_none_without_step(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        b = MetricContext()
        assert shape_volume_iou(a, b) is None
        assert shape_volume_iou(b, a) is None

    def test_returns_float_in_range(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        score = shape_volume_iou(a, a)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# shape_feature_edge_f1
# ---------------------------------------------------------------------------

class TestShapeFeatureEdgeF1:

    def test_identical_box_one(self) -> None:
        """Same box on both sides -> all real edges match, F1 ≈ 1."""
        a = _ctx_with_step(10, 20, 30)
        score = shape_feature_edge_f1(a, a)
        assert score is not None
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_sphere_pair_both_empty_one(self) -> None:
        """Sphere has zero sharp edges; both-empty must return 1.0 per spec."""
        from pathlib import Path as _P

        sphere = _P(__file__).parent.parent / "fixtures" / "sphere.step"
        if not sphere.exists():
            pytest.skip("sphere.step fixture missing")
        from cadgenbench.common.validity import analyze_step

        a_analysis = analyze_step(sphere)
        ctx = MetricContext(
            step_path=sphere,
            validation=a_analysis.validation,
            measurements=a_analysis.measurements,
        )
        score = shape_feature_edge_f1(ctx, ctx)
        assert score is not None
        assert score == pytest.approx(1.0)

    def test_mismatched_geometry_lower_score(self) -> None:
        """A 10x10x10 cube vs a 10x10x50 column -> some edges line up, many don't."""
        a = _ctx_with_step(10, 10, 10)
        b = _ctx_with_step(10, 10, 50)
        score = shape_feature_edge_f1(a, b)
        self_score = shape_feature_edge_f1(a, a)
        assert score is not None and self_score is not None
        assert score < self_score - 0.05

    def test_returns_none_without_gt_step(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        b = MetricContext()
        assert shape_feature_edge_f1(a, b) is None


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_identical_step_full_context(self) -> None:
        """Same STEP on both sides -> all three shape sub-metrics report."""
        a = _ctx_with_step(10, 10, 10)
        scores, errors = compute_metrics(a, a)
        assert "shape_point_cloud_f1" in scores
        assert "shape_volume_iou" in scores
        assert "shape_feature_edge_f1" in scores
        assert errors == {}

    def test_empty_context(self) -> None:
        a = MetricContext()
        scores, errors = compute_metrics(a, a)
        assert scores == {}
        assert errors == {}

    def test_custom_subset(self) -> None:
        a = _ctx_with_step(10, 10, 10)
        scores, _ = compute_metrics(a, a, metrics={"pc_f1": shape_point_cloud_f1})
        assert list(scores.keys()) == ["pc_f1"]

    def test_custom_metric_function(self) -> None:
        def face_ratio(c: MetricContext, g: MetricContext) -> float | None:
            if c.measurements is None or g.measurements is None:
                return None
            return min(c.measurements.face_count, g.measurements.face_count) / max(
                c.measurements.face_count, g.measurements.face_count,
            )

        a = MetricContext(measurements=_meas())  # face_count=6
        b = MetricContext(measurements=Measurements(
            solid_count=1, shell_count=1, face_count=12,
            volume=1000, bounding_box=BBox(0, 10, 0, 10, 0, 10),
        ))
        scores, _ = compute_metrics(a, b, metrics={"face_ratio": face_ratio})
        assert scores["face_ratio"] == pytest.approx(0.5)

    def test_failing_metric_scored_zero_and_recorded(self) -> None:
        def broken(c: MetricContext, g: MetricContext) -> float | None:
            raise RuntimeError("nope")

        a = MetricContext(measurements=_meas())
        scores, errors = compute_metrics(
            a, a,
            metrics={"broken": broken, "pc_f1": shape_point_cloud_f1},
        )
        assert scores["broken"] == 0.0
        assert "broken" in errors
        assert "nope" in errors["broken"]
