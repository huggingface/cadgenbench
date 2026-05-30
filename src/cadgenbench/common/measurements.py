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

"""Geometric measurements for STEP files.

Computes bounding box, volume, and topology counts on a STEP shape using
the raw OCC kernel. Measurements are taken on solid sub-shapes only so
that PMI annotation geometry (datum arrows, dimension leaders, etc.) does
not inflate bbox / volume.

This module is orthogonal to :mod:`cadgenbench.common.validity`, which
answers "is this BREP well-formed?". Measurements are facts about the
geometry regardless of whether it is valid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def size_x(self) -> float:
        return self.x_max - self.x_min

    @property
    def size_y(self) -> float:
        return self.y_max - self.y_min

    @property
    def size_z(self) -> float:
        return self.z_max - self.z_min

    @property
    def diagonal(self) -> float:
        """Euclidean diagonal length, useful as a scale-invariant unit."""
        return (self.size_x**2 + self.size_y**2 + self.size_z**2) ** 0.5


@dataclass(frozen=True)
class Measurements:
    """Geometric measurements for a BREP shape."""

    solid_count: int
    shell_count: int
    face_count: int
    volume: float
    bounding_box: BBox


def measure_step(step_path: str | Path) -> Measurements:
    """Load a STEP file and compute its geometric measurements.

    Args:
        step_path: Path to a .step or .stp file.

    Returns:
        :class:`Measurements`, topology counts, bbox, and volume.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the STEP file cannot be loaded.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    from build123d import import_step

    try:
        shape = import_step(str(step_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load STEP file: {step_path}") from exc

    if shape is None or not shape.wrapped:
        raise RuntimeError(f"STEP file produced no geometry: {step_path}")

    return _measure_wrapped(shape.wrapped)


def _measure_wrapped(wrapped) -> Measurements:  # type: ignore[no-untyped-def]
    """Compute measurements from a pre-loaded OCC ``TopoDS_Shape``.

    Split from :func:`measure_step` so a caller that already has the raw
    shape loaded (e.g. :func:`cadgenbench.common.validity.analyze_step`)
    can reuse it without paying for a second STEP parse.
    """
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID

    from cadgenbench.eval.sampling import solids_only

    solid_count = _count_subshapes(wrapped, TopAbs_SOLID)
    shell_count = _count_subshapes(wrapped, TopAbs_SHELL)
    face_count = _count_subshapes(wrapped, TopAbs_FACE)

    measurement_shape = solids_only(wrapped)
    return Measurements(
        solid_count=solid_count,
        shell_count=shell_count,
        face_count=face_count,
        volume=_compute_volume(measurement_shape),
        bounding_box=_compute_bbox(measurement_shape),
    )


def _count_subshapes(shape, shape_type) -> int:  # type: ignore[no-untyped-def]
    """Count sub-shapes of the given TopAbs type."""
    from OCP.TopExp import TopExp_Explorer

    count = 0
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _compute_bbox(shape) -> BBox:  # type: ignore[no-untyped-def]
    """Axis-aligned bounding box of a raw OCC ``TopoDS_Shape``."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    bnd = Bnd_Box()
    BRepBndLib.Add_s(shape, bnd, True)
    x_min, y_min, z_min, x_max, y_max, z_max = bnd.Get()
    return BBox(x_min, x_max, y_min, y_max, z_min, z_max)


def _compute_volume(shape) -> float:  # type: ignore[no-untyped-def]
    """Volume of solid sub-shapes of *shape* (0.0 for shells / faces)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())
