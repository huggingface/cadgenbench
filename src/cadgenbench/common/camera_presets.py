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

"""Shared camera presets for both tcv (BREP) and pyrender (mesh) renderers.

``tcv_screenshots`` uses three-cad-viewer's built-in ``setView`` for named
presets ("iso", "front", etc.); the exact orientation/distance is therefore
fixed by that library. This module mirrors those presets for the pyrender
mesh renderer so mesh renders frame the model the same way tcv does.

The conventions assumed here (Z-up, metric units, right-handed coords) match
build123d / OpenCascade defaults and three-cad-viewer's coordinate system.
The camera positions are expressed as ``(direction, up)`` unit vectors; the
renderer positions the camera at ``center + direction * distance`` where
``distance`` is ``DISTANCE_FACTOR * bbox_diagonal`` and the bounding box
center is the look-at target. ``DISTANCE_FACTOR`` was chosen empirically so
a unit cube centered at the origin roughly fills 70% of a 1024x768 frame
with a 30 degree vertical field of view, matching tcv's framing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_PRESETS: frozenset[str] = frozenset(
    ("front", "rear", "left", "right", "top", "bottom", "iso")
)

DEFAULT_VIEWS: tuple[str, ...] = ("iso", "front", "top", "right")

# 30-degree vertical FOV matches three-cad-viewer's default perspective
# projection well enough for similar framing on typical CAD parts.
DEFAULT_FOV_Y_DEG: float = 30.0

# Empirically tuned so a unit cube at the origin renders at roughly the same
# size as three-cad-viewer's default fit.
DISTANCE_FACTOR: float = 2.6


@dataclass(frozen=True)
class CameraPreset:
    """Direction the camera looks *from* plus its up-vector.

    ``direction`` is a unit vector *from the look-at target toward the camera*.
    That is, the camera is placed at ``target + direction * distance``.
    """
    direction: tuple[float, float, float]
    up: tuple[float, float, float]


_INV_SQRT3 = 1.0 / (3.0 ** 0.5)

PRESETS: dict[str, CameraPreset] = {
    # Standard orthographic-like directions; Z-up.
    "front":  CameraPreset(direction=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0)),
    "rear":   CameraPreset(direction=(0.0,  1.0, 0.0), up=(0.0, 0.0, 1.0)),
    "left":   CameraPreset(direction=(-1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)),
    "right":  CameraPreset(direction=( 1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)),
    "top":    CameraPreset(direction=(0.0,  0.0, 1.0), up=(0.0, 1.0, 0.0)),
    "bottom": CameraPreset(direction=(0.0,  0.0, -1.0), up=(0.0, -1.0, 0.0)),
    # Front-right-top iso corner, matching three-cad-viewer's default iso.
    "iso":    CameraPreset(
        direction=(_INV_SQRT3, -_INV_SQRT3, _INV_SQRT3),
        up=(0.0, 0.0, 1.0),
    ),
}


def validate_views(views) -> None:
    """Raise ``ValueError`` if *views* contains unknown preset names."""
    unknown = set(views) - CAMERA_PRESETS
    if unknown:
        raise ValueError(
            f"Unknown camera preset(s): {unknown}. "
            f"Valid: {sorted(CAMERA_PRESETS)}"
        )


def camera_placement(
    preset: str,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    distance_factor: float = DISTANCE_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(eye, target, up)`` for framing *bbox* at *preset*.

    Args:
        preset: One of :data:`CAMERA_PRESETS`.
        bbox_min: ``(3,)`` array with the minimum bounding-box corner.
        bbox_max: ``(3,)`` array with the maximum bounding-box corner.
        distance_factor: Multiplier on the bbox diagonal for camera distance.

    Returns:
        Tuple of numpy arrays ``(eye, target, up)``, each shape ``(3,)``.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown camera preset: {preset!r}. Valid: {sorted(CAMERA_PRESETS)}"
        )
    p = PRESETS[preset]
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    target = (bbox_min + bbox_max) * 0.5
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    distance = max(diagonal * distance_factor, 1e-6)
    direction = np.asarray(p.direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    up = np.asarray(p.up, dtype=np.float64)
    eye = target + direction * distance
    return eye, target, up
