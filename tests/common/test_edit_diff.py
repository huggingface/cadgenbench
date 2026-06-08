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

"""Unit tests for the edit-diff surface-deviation field."""
from __future__ import annotations

import numpy as np
import pytest

from cadgenbench.common import edit_diff
from cadgenbench.common.mesh import Mesh


def _box(x: float = 10.0, y: float = 10.0, z: float = 10.0, *, defl: float = 0.5) -> Mesh:
    """Axis-aligned box mesh centred at the origin (outward-consistent winding)."""
    import pyvista as pv

    cube = pv.Cube(x_length=x, y_length=y, z_length=z).triangulate()
    faces = cube.faces.reshape(-1, 4)[:, 1:]
    return Mesh(
        vertices=np.asarray(cube.points, dtype=np.float64),
        triangles=np.ascontiguousarray(faces, dtype=np.int64),
        linear_deflection_mm=defl,
    )


class TestSeverity:
    def test_identical_is_zero(self) -> None:
        # The key correctness property: a candidate equal to GT must paint
        # nothing -- distance 0 and, with original-surface normals, no normal
        # disagreement across folds (the regression that reddened a perfect part).
        box = _box()
        sev = edit_diff._severity(box, box, box, f1_tol=0.05, normal=True)
        assert float(sev.max()) == pytest.approx(0.0, abs=1e-9)

    def test_material_outside_paints(self) -> None:
        # A bigger candidate has material outside the reference -> positive
        # (directional: this is the "extra / red" side).
        small = _box(10, 10, 10)
        big = _box(14, 10, 10)
        sev = edit_diff._severity(big, small, big, f1_tol=0.05, normal=True)
        assert float(sev.max()) > 0.5

    def test_directional_inside_is_zero(self) -> None:
        # A smaller candidate lies INSIDE the reference: nothing is *outside*,
        # so the red side stays 0 (the deficit is the amber/GT side's job).
        small = _box(6, 10, 10)
        big = _box(10, 10, 10)
        sev = edit_diff._severity(small, big, small, f1_tol=0.05, normal=False)
        assert float(sev.max()) == pytest.approx(0.0, abs=1e-9)


class TestZoomBbox:
    def test_none_when_identical(self) -> None:
        box = _box()
        assert edit_diff.edit_region_zoom_bbox(box, box) is None

    def test_found_when_different(self) -> None:
        bbox = edit_diff.edit_region_zoom_bbox(_box(10, 10, 10), _box(14, 10, 10))
        assert bbox is not None
        lo, hi = bbox
        assert np.all(hi > lo)


class TestBuildShapes:
    def test_returns_two_rgba_fields(self) -> None:
        shapes = edit_diff.build_edit_diff_shapes(_box(10, 10, 10), _box(14, 10, 10))
        assert len(shapes) == 2
        for poly, _rgb, _alpha in shapes:
            assert "rgba" in poly.point_data
            assert poly.point_data["rgba"].shape[1] == 4
