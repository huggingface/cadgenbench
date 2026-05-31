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

"""Mesh Boolean operations backed by the ``manifold3d`` kernel.

Used by the two metric-side Boolean call sites:

- :mod:`cadgenbench.eval.shape_similarity._volume_overlap_stats`
- :mod:`cadgenbench.eval.interface_match` (sub-volume IoU)

``manifold3d`` is a combinatorial mesh-Boolean kernel: every input that
is a closed oriented 2-manifold produces a closed oriented 2-manifold
output. Sub-millisecond per op and deterministic. Inputs are guaranteed
to be closed orientable manifolds by the validity gate
(:mod:`cadgenbench.common.validity` + :mod:`cadgenbench.common.mesh`);
invalid candidates short-circuit to ``cad_score = 0`` before reaching
this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cadgenbench.common.mesh import Mesh

if TYPE_CHECKING:  # pragma: no cover - typing only
    import manifold3d as m3d


# ---------------------------------------------------------------------------
# Mesh ↔ Manifold conversion
# ---------------------------------------------------------------------------


def mesh_to_manifold(mesh: Mesh) -> "m3d.Manifold":
    """Convert our :class:`Mesh` to a ``manifold3d.Manifold``.

    The input mesh is assumed to have already passed the strict gate
    in :mod:`cadgenbench.common.mesh` (manifold + closed +
    orientation-consistent). If for some reason ``manifold3d`` still
    cannot convert it (e.g. orientation-inverted relative to its
    convention) we raise :class:`RuntimeError` with the kernel's own
    error status string.
    """
    import manifold3d as m3d

    if mesh.n_triangles == 0:
        raise ValueError("cannot convert empty mesh to Manifold")

    md_mesh = m3d.Mesh(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(mesh.triangles, dtype=np.uint32),
    )
    manifold = m3d.Manifold(md_mesh)
    status = manifold.status
    # The exact enum varies across manifold3d versions; we compare by
    # string name to stay forward-compatible. "NoError" is the success
    # state across published versions.
    if hasattr(status, "name") and status.name != "NoError":
        raise RuntimeError(
            f"manifold3d failed to ingest mesh: status={status!r}, "
            f"F={mesh.n_triangles}, V={mesh.n_vertices}",
        )
    return manifold


# ---------------------------------------------------------------------------
# Boolean ops (typed shorthand)
# ---------------------------------------------------------------------------


def intersect(a: "m3d.Manifold", b: "m3d.Manifold") -> "m3d.Manifold":
    return a ^ b  # manifold3d.Manifold.__xor__ = intersection


def union(a: "m3d.Manifold", b: "m3d.Manifold") -> "m3d.Manifold":
    return a + b  # manifold3d.Manifold.__add__ = union


def subtract(a: "m3d.Manifold", b: "m3d.Manifold") -> "m3d.Manifold":
    return a - b  # manifold3d.Manifold.__sub__ = difference


# ---------------------------------------------------------------------------
# Pose → 4×4 transform matrix
# ---------------------------------------------------------------------------


def pose_to_matrix(
    pose: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Convert a 6-tuple pose ``(tx, ty, tz, rx, ry, rz)`` to a 4×4 matrix.

    Convention matches the previous OCC path: rotate XYZ Euler about
    the origin (in degrees), then translate.
    """
    from scipy.spatial.transform import Rotation as R

    tx, ty, tz, rx, ry, rz = pose
    rot = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = (tx, ty, tz)
    return mat


def apply_pose(manifold: "m3d.Manifold", pose: tuple[float, ...]) -> "m3d.Manifold":
    """Apply a 6-tuple Euler pose to a manifold via ``Manifold.transform``."""
    mat = pose_to_matrix(pose)
    # manifold3d wants a 3×4 affine transform (rotation 3×3 + translation 3×1).
    affine = np.ascontiguousarray(mat[:3, :], dtype=np.float32)
    return manifold.transform(affine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def manifold_volume(manifold: "m3d.Manifold") -> float:
    """Volume of a manifold; 0.0 for an empty result."""
    if manifold.is_empty():
        return 0.0
    return float(manifold.volume())
