"""Tests for the shared camera preset module."""
from __future__ import annotations

import numpy as np
import pytest

from cadgenbench.common.camera_presets import (
    CAMERA_PRESETS,
    DEFAULT_VIEWS,
    camera_placement,
    validate_views,
)


class TestPresetSet:
    def test_all_standard_presets_present(self) -> None:
        assert CAMERA_PRESETS == frozenset(
            ("iso", "front", "rear", "left", "right", "top", "bottom"),
        )

    def test_default_views_are_subset(self) -> None:
        assert set(DEFAULT_VIEWS).issubset(CAMERA_PRESETS)


class TestValidateViews:
    def test_valid(self) -> None:
        validate_views(("iso", "front"))

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            validate_views(("iso", "nope"))


class TestCameraPlacement:
    bbox_min = np.array([-1.0, -2.0, -3.0])
    bbox_max = np.array([1.0, 2.0, 3.0])

    def test_target_is_bbox_center(self) -> None:
        _eye, target, _up = camera_placement("iso", self.bbox_min, self.bbox_max)
        np.testing.assert_allclose(target, np.array([0.0, 0.0, 0.0]))

    @pytest.mark.parametrize("preset", sorted(CAMERA_PRESETS))
    def test_eye_outside_bbox(self, preset: str) -> None:
        eye, target, _up = camera_placement(preset, self.bbox_min, self.bbox_max)
        # Eye must be at least as far from target as the bbox diagonal.
        diag = float(np.linalg.norm(self.bbox_max - self.bbox_min))
        assert np.linalg.norm(eye - target) >= diag

    def test_front_direction_is_negative_y(self) -> None:
        eye, target, _ = camera_placement("front", self.bbox_min, self.bbox_max)
        direction = (eye - target) / np.linalg.norm(eye - target)
        np.testing.assert_allclose(direction, [0.0, -1.0, 0.0], atol=1e-6)

    def test_top_direction_is_positive_z(self) -> None:
        eye, target, _ = camera_placement("top", self.bbox_min, self.bbox_max)
        direction = (eye - target) / np.linalg.norm(eye - target)
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-6)

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError):
            camera_placement("oblique", self.bbox_min, self.bbox_max)
