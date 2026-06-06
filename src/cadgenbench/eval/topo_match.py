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

"""Topology Match, v1 metric category.

Scores whether a candidate part's 3D topology (connected solid pieces,
through-handles, internal voids) matches the ground truth's. Implements
the design in [`docs/metrics/topo_match.md`](../../docs/metrics/topo_match.md).

Three integer invariants of the solid as a 3-manifold with boundary:

- :math:`b_0`, number of connected solid components.
- :math:`b_1`, number of independent through-handles / through-holes.
- :math:`b_2`, number of enclosed internal voids.

Computed from the tessellated boundary mesh produced by
:mod:`cadgenbench.common.mesh`:

1. Connected components of the mesh (union-find on triangle adjacency).
2. For every component, decide outer-shell-of-solid vs. inner-shell-of-
   void by parity of how many other components contain its interior
   probe (even = outer → contributes to :math:`b_0`, odd = inner →
   contributes to :math:`b_2`). Containment is decided by even/odd ray
   casting.
3. :math:`\\chi_\\text{surface} = V - E + F` of the welded mesh.
4. Apply the topological identity
   :math:`\\chi(\\partial S) = 2\\chi(S) = 2(b_0 - b_1 + b_2)` to recover
   :math:`b_1 = b_0 + b_2 - \\chi_\\text{surface}/2`.

The mesh module's strict gate (manifold + closed + orientation
consistent) is a precondition of this math; if it fails, the caller
should have already short-circuited validity. We re-assert on entry as
a safety net.

Score: per-Betti fuzzy log-ratio in :math:`[0, 1]`, **product** over the
three:

.. code-block:: text

    s_i        = exp(-alpha * |log((b_cand_i + 1) / (b_gt_i + 1))|)   ∈ [0, 1]
    topo_match = s_0 * s_1 * s_2                                      ∈ [0, 1]

The sharpness ``alpha`` (``BETTI_SHARPNESS``, currently ``2``) steepens the
per-axis penalty: doubling a count scores 0.36, not 0.60.

Each :math:`s_i` equals ``1`` iff the candidate matches the GT on
that axis, and decays smoothly as the count drifts in either direction
(the :math:`+1` shift makes the score well-defined when either Betti is
zero, and keeps "off by one near zero" from being indistinguishable
from "completely wrong"). The product (rather than the mean) means a
single badly-wrong axis collapses the aggregate toward ``0``: topology
is discrete, so "right on two of three invariants" is not a part that
matches.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cadgenbench.common.artifacts import StepArtifacts
from cadgenbench.common.measurements import measure_step
from cadgenbench.common.mesh import (
    Mesh,
    deflection_for_bbox,
    tessellate_and_validate,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BettiResult:
    """Betti numbers of one solid plus the diagnostic χ that produced them."""

    b0: int
    b1: int
    b2: int
    chi_surface: int
    n_components: int
    n_triangles: int
    n_vertices: int
    linear_deflection_mm: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TopoMatchResult:
    """One candidate vs. one GT, Betti for each plus per-axis scores + mean."""

    candidate: BettiResult
    gt: BettiResult
    per_axis_scores: dict[str, float]
    score: float

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "gt": self.gt.to_dict(),
            "per_axis_scores": {
                k: float(v) for k, v in self.per_axis_scores.items()
            },
            "score": float(self.score),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_betti_for_step(
    step_path: str | Path,
    *,
    linear_deflection_mm: float | None = None,
) -> BettiResult:
    """Tessellate *step_path* and compute its solid Betti numbers.

    The mesh-gate (manifold + closed + orientation-consistent) is a hard
    precondition; failure raises :class:`cadgenbench.common.mesh.MeshSanityError`,
    which the caller is expected to catch and fold into validity.

    Args:
        step_path: Path to a watertight STEP file.
        linear_deflection_mm: Chord-error for the tessellator. When
            ``None``, derived from the part's own bounding-box diagonal
            via :func:`cadgenbench.common.mesh.deflection_for_bbox`. **For
            metric use, the caller should always pass an explicit value
            derived from the GT's bbox** so candidate and GT are
            tessellated at the same scale; the auto path exists for
            stand-alone analysis only.
    """
    step_path = Path(step_path)
    if linear_deflection_mm is None:
        m = measure_step(step_path)
        linear_deflection_mm = deflection_for_bbox(m.bounding_box.diagonal)
    mesh = tessellate_and_validate(step_path, linear_deflection_mm)
    return compute_betti_from_mesh(mesh)


def compute_betti_from_mesh(mesh: Mesh) -> BettiResult:
    """Compute Betti numbers from a pre-validated :class:`Mesh`."""
    chi = _euler_characteristic(mesh)
    components = _triangle_components(mesh)
    b0, b2 = _classify_components_by_containment(mesh, components)
    b1 = b0 + b2 - chi // 2
    return BettiResult(
        b0=int(b0),
        b1=int(b1),
        b2=int(b2),
        chi_surface=int(chi),
        n_components=len(components),
        n_triangles=mesh.n_triangles,
        n_vertices=mesh.n_vertices,
        linear_deflection_mm=mesh.linear_deflection_mm,
    )


def topo_match_score(
    candidate: BettiResult, gt: BettiResult,
) -> tuple[float, dict[str, float]]:
    """Return ``(score, per_axis_scores)`` for one candidate-vs-GT pair.

    Each per-axis score is the fuzzy log-ratio

    .. math::

        s_i = \\exp\\!\\bigl(-\\alpha\\,\\bigl|\\log\\bigl((b_i^{\\text{cand}}+1)/(b_i^{\\text{gt}}+1)\\bigr)\\bigr|\\bigr),

    with sharpness :math:`\\alpha` = ``BETTI_SHARPNESS``. It is symmetric
    in candidate / GT, equals ``1`` iff the two counts agree, and decays
    smoothly to ``0`` as the ratio departs from ``1``. ``score`` is the product over the three axes, so the
    aggregate lives in ``[0, 1]`` and any single badly-wrong axis
    collapses it toward ``0``.
    """
    per_axis = {
        "b0": _per_betti_score(candidate.b0, gt.b0),
        "b1": _per_betti_score(candidate.b1, gt.b1),
        "b2": _per_betti_score(candidate.b2, gt.b2),
    }
    score = per_axis["b0"] * per_axis["b1"] * per_axis["b2"]
    return score, per_axis


# Sharpness exponent on the per-axis fuzzy log-ratio. At ``1.0`` the score is
# ``(min+1)/(max+1)``; raising it steepens every penalty while preserving the
# shape (symmetric, ``1`` on a match, finite at zero). ``2.0`` is deliberately
# strict: doubling a count (e.g. 2 -> 4 holes) scores 0.36 instead of 0.60,
# because a wrong topological count is a real defect, not a near miss.
BETTI_SHARPNESS = 2.0


def _per_betti_score(b_cand: int, b_gt: int) -> float:
    """Fuzzy log-ratio score for one Betti axis, in ``[0, 1]``.

    The :math:`+1` shift keeps the ratio finite when either Betti is
    zero and gives "off by one near zero" graceful (rather than
    catastrophic) decay; for non-negative integers it is equivalent to
    ``((min(b_cand, b_gt) + 1) / (max(b_cand, b_gt) + 1)) ** BETTI_SHARPNESS``.

    A negative Betti is not a real count - it means the candidate's mesh
    is degenerate (not a clean manifold). Score it ``0`` rather than feed
    a non-positive argument into ``log``. GT is always clean, so this only
    ever triggers on a broken candidate.
    """
    if b_cand < 0 or b_gt < 0:
        return 0.0
    return math.exp(-BETTI_SHARPNESS * abs(math.log((b_cand + 1) / (b_gt + 1))))


def topo_match(
    candidate_step: str | Path,
    gt_step: str | Path,
    *,
    candidate_artifacts: StepArtifacts | None = None,
    gt_artifacts: StepArtifacts | None = None,
) -> TopoMatchResult:
    """End-to-end: tessellate both at the **GT's** deflection, score Betti.

    Returns a :class:`TopoMatchResult` with both Betti vectors, the
    per-axis fuzzy log-ratio scores, and the aggregate.
    """
    candidate_step = Path(candidate_step)
    gt_step = Path(gt_step)

    gt_artifacts = gt_artifacts or StepArtifacts(gt_step)
    candidate_artifacts = candidate_artifacts or StepArtifacts(candidate_step)

    gt_betti = gt_artifacts.betti()
    cand_betti = candidate_artifacts.betti()
    score, per_axis_scores = topo_match_score(cand_betti, gt_betti)
    return TopoMatchResult(
        candidate=cand_betti,
        gt=gt_betti,
        per_axis_scores=per_axis_scores,
        score=score,
    )


# ---------------------------------------------------------------------------
# Internals, components, containment, χ
# ---------------------------------------------------------------------------


def _euler_characteristic(mesh: Mesh) -> int:
    """Return χ = V − E + F of the welded mesh (assumes the gate passed)."""
    tris = mesh.triangles
    n_v = int(mesh.n_vertices)
    n_f = int(mesh.n_triangles)
    edges = np.sort(
        np.concatenate(
            [
                tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]],
            ],
            axis=0,
        ),
        axis=1,
    )
    keys = edges[:, 0].astype(np.int64) * (n_v + 1) + edges[:, 1]
    n_e = int(len(np.unique(keys)))
    return n_v - n_e + n_f


def _triangle_components(mesh: Mesh) -> list[np.ndarray]:
    """Return one (Nt_i, 3) triangle array per connected mesh component.

    Triangles are grouped by their union-find root in vertex space -
    valid since the mesh is a clean closed manifold (gate-passing
    precondition), so components correspond directly to vertex-graph
    components.
    """
    tris = mesh.triangles
    n_v = mesh.n_vertices
    parent = list(range(n_v))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    tris_list = tris.tolist()
    for a, b, c in tris_list:
        union(a, b)
        union(b, c)

    groups: dict[int, list[tuple[int, int, int]]] = {}
    for tri in tris_list:
        root = find(tri[0])
        groups.setdefault(root, []).append(tri)
    return [np.asarray(g, dtype=np.int64) for g in groups.values()]


def _classify_components_by_containment(
    mesh: Mesh, components: list[np.ndarray],
) -> tuple[int, int]:
    """Classify each component as outer-shell (b0+1) or void-shell (b2+1).

    For each component:

    1. Pick an interior probe point (centroid of a seed triangle,
       nudged inward by a small step relative to the part bbox).
    2. Count how many *other* components contain that probe (even/odd
       ray casting). Even parity → outer shell of a distinct solid;
       odd parity → inner shell of a void.
    """
    interior_pts = [_interior_point(mesh.vertices, c) for c in components]
    boxes = [_component_aabb(mesh.vertices, c) for c in components]
    b0 = 0
    b2 = 0
    for i, probe in enumerate(interior_pts):
        depth = 0
        for j, comp in enumerate(components):
            if i == j:
                continue
            lo, hi = boxes[j]
            if (probe < lo - 1e-9).any() or (probe > hi + 1e-9).any():
                continue
            if _point_in_mesh(probe, mesh.vertices, comp):
                depth += 1
        if depth % 2 == 0:
            b0 += 1
        else:
            b2 += 1
    return b0, b2


def _component_aabb(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a component's vertices."""
    used = np.unique(tris.reshape(-1))
    pts = verts[used]
    return pts.min(axis=0), pts.max(axis=0)


def _interior_point(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Pick a probe point demonstrably inside the closed mesh component.

    Strategy: take the centroid of a seed triangle and try nudging it
    by a small step along the triangle's normal in both ±directions;
    return whichever side the ray test reports as "inside". If neither
    side is inside (degenerate component), fall back to the centroid.
    """
    used = np.unique(tris.reshape(-1))
    pts = verts[used]
    bbox_diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    step = max(1e-6, 1e-4 * bbox_diag)

    seed_tri = tris[len(tris) // 2]
    a, b, c = verts[seed_tri[0]], verts[seed_tri[1]], verts[seed_tri[2]]
    centroid = (a + b + c) / 3.0
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm <= 1e-12:
        return centroid
    n /= norm

    for sign in (-1.0, 1.0):
        p = centroid + sign * step * n
        if _point_in_mesh(p, verts, tris):
            return p
    return centroid


def _point_in_mesh(point: np.ndarray, verts: np.ndarray, tris: np.ndarray) -> bool:
    """Point-in-closed-mesh via Möller–Trumbore even/odd ray casting.

    Re-rolls the ray direction up to 8 times whenever any triangle is
    *grazed* (a barycentric coord lands within :math:`10^{-9}` of an
    edge), so degenerate intersections never poison the parity count.
    Deterministic across runs (seeded RNG).
    """
    rng = np.random.default_rng(0)
    for attempt in range(8):
        if attempt == 0:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            v = rng.standard_normal(3)
            direction = v / max(np.linalg.norm(v), 1e-12)

        v0 = verts[tris[:, 0]] - point
        v1 = verts[tris[:, 1]] - point
        v2 = verts[tris[:, 2]] - point

        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = np.einsum("ij,ij->i", edge1, h)
        ok = np.abs(a) > 1e-12
        if not ok.any():
            continue
        f = np.zeros_like(a)
        f[ok] = 1.0 / a[ok]
        s = -v0
        u = f * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, edge1)
        v = f * np.einsum("j,ij->i", direction, q)
        t = f * np.einsum("ij,ij->i", edge2, q)

        hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        grazed = ok & (
            (np.abs(u) < 1e-9)
            | (np.abs(u - 1) < 1e-9)
            | (np.abs(v) < 1e-9)
            | (np.abs(u + v - 1) < 1e-9)
        )
        if grazed.any():
            continue
        return bool(int(hit.sum()) % 2 == 1)
    return False
