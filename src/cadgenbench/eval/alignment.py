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

"""Rigid-body alignment of STEP files and trusted meshes.

Alignment is rotation + translation only; scale, shear, and mirrors are
projected away. The pipeline refines a pool of candidate poses (identity and the
24 octahedral PCA orientations) with Open3D multi-scale point-to-plane ICP, then
selects the pose by shape agreement (bidirectional surface F1, capped symmetric
Chamfer, nearest-neighbour RMSE) with a canonical tie-break so near-symmetric
parts resolve to one deterministic pose.

STEP candidates are exported as aligned STEP files. Trusted sidecar meshes are
aligned in memory and never re-tessellated.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cadgenbench.eval.sampling import (
    _load_occ_shape,
    _tessellate,
    _area_weighted_sample,
)

_OPEN3D_ICP_VOXEL_DIVISOR = 150.0

# Poses whose surface F1 ties within this tolerance, and whose capped Chamfer
# then ties within this fraction of the target diagonal, are treated as equally
# good geometric fits; the canonical least-rotation tie-break decides between
# them so near-symmetric parts resolve to one deterministic pose.
_TIE_F1_TOLERANCE = 0.005
_TIE_CHAMFER_FRACTION = 0.005


@dataclass(frozen=True)
class AlignmentResult:
    """Output of :func:`align_step`."""

    rotation: np.ndarray  # (3, 3) orthogonal matrix
    translation: np.ndarray  # (3,) translation vector
    rmse: float  # root-mean-square closest-point distance after alignment
    output_path: Path  # path to the aligned STEP file


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_step(
    source: str | Path,
    target: str | Path,
    output: str | Path | None = None,
    n_samples: int = 10_000,
    icp_max_iter: int = 50,
    icp_tolerance: float = 1e-6,
    seed: int | None = None,
) -> AlignmentResult:
    """Rigidly align *source* STEP file to *target* STEP file.

    Args:
        source: Path to the STEP file to be transformed.
        target: Path to the reference STEP file (stays fixed).
        output: Where to write the aligned STEP.  Defaults to
            ``<source_stem>_aligned.step`` next to *source*.
        n_samples: Number of surface points to sample for alignment.
        icp_max_iter: Maximum Open3D ICP iterations across the multi-scale pass.
        icp_tolerance: Open3D ICP convergence threshold.
        seed: RNG seed for reproducible point sampling.

    Returns:
        :class:`AlignmentResult` with the recovered R, t, RMSE, and output
        path.
    """
    source = Path(source)
    target = Path(target)
    if not source.exists():
        raise FileNotFoundError(f"Source STEP not found: {source}")
    if not target.exists():
        raise FileNotFoundError(f"Target STEP not found: {target}")

    if output is None:
        output = source.parent / f"{source.stem}_aligned.step"
    output = Path(output)

    # --- Sample point clouds ---
    src_shape = _load_occ_shape(source)
    tgt_shape = _load_occ_shape(target)

    src_verts, src_tris = _tessellate(src_shape)
    tgt_verts, tgt_tris = _tessellate(tgt_shape)

    seed_src = seed
    seed_tgt = seed + 1 if seed is not None else None
    src_pts = _area_weighted_sample(src_verts, src_tris, n_samples, seed_src)
    tgt_pts = _area_weighted_sample(tgt_verts, tgt_tris, n_samples, seed_tgt)

    R_total, t_total, rmse = _align_points_with_selector(
        src_pts,
        tgt_pts,
        selector_source=src_pts,
        selector_target=tgt_pts,
        icp_max_iter=icp_max_iter,
        icp_tolerance=icp_tolerance,
    )

    # --- Apply transform to BREP and export ---
    _apply_and_export(src_shape, R_total, t_total, output)

    return AlignmentResult(
        rotation=R_total,
        translation=t_total,
        rmse=rmse,
        output_path=output,
    )


@dataclass(frozen=True)
class CachedAlignmentResult:
    """Output of :func:`align_cached_mesh`: the aligned mesh + transform + RMSE."""

    mesh: object  # cadgenbench.common.mesh.Mesh, rigidly moved into the target frame
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float


def align_cached_mesh(
    source_artifacts,
    target_artifacts,
    *,
    n_samples: int = 10_000,
    seed: int = 0,
) -> CachedAlignmentResult:
    """Rigidly align *source*'s trusted mesh to *target*'s — no re-tessellation.

    Both meshes come from :meth:`StepArtifacts.mesh` (sidecar-aware), so a part
    with a supplied mesh is never re-meshed from its STEP. The transform is
    recovered from area-weighted point clouds via :func:`align_points` and
    applied to the source mesh's vertices; it is a proper rotation (det +1), so
    triangle winding is preserved and only the vertices move.
    """
    from cadgenbench.common.mesh import Mesh
    from cadgenbench.eval.sampling import _area_weighted_sample

    src_mesh = source_artifacts.mesh()
    tgt_mesh = target_artifacts.mesh()
    src_pts = _area_weighted_sample(src_mesh.vertices, src_mesh.triangles, n_samples, seed)
    tgt_pts = _area_weighted_sample(tgt_mesh.vertices, tgt_mesh.triangles, n_samples, seed + 1)
    R, t, rmse = _align_points_with_selector(
        src_pts,
        tgt_pts,
        selector_source=np.asarray(src_mesh.vertices, dtype=np.float64),
        selector_target=np.asarray(tgt_mesh.vertices, dtype=np.float64),
    )
    aligned = Mesh(
        vertices=np.asarray(src_mesh.vertices, dtype=np.float64) @ R.T + t,
        triangles=src_mesh.triangles,
        linear_deflection_mm=src_mesh.linear_deflection_mm,
    )
    return CachedAlignmentResult(mesh=aligned, rotation=R, translation=t, rmse=float(rmse))


def align_points(
    source: np.ndarray,
    target: np.ndarray,
    icp_max_iter: int = 50,
    icp_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align two (N, 3) point clouds.  Returns ``(R, t, rmse)``.

    Candidates are identity and the 24 octahedral PCA orientations, each refined
    with Open3D multi-scale point-to-plane ICP. The pose is chosen by shape
    agreement (F1, capped symmetric Chamfer, RMSE) with a canonical tie-break.
    """
    return _align_points_with_selector(
        source,
        target,
        selector_source=source,
        selector_target=target,
        icp_max_iter=icp_max_iter,
        icp_tolerance=icp_tolerance,
    )


def _align_points_with_selector(
    source: np.ndarray,
    target: np.ndarray,
    *,
    selector_source: np.ndarray,
    selector_target: np.ndarray,
    icp_max_iter: int = 50,
    icp_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine identity + all PCA candidates and pick by shape agreement."""
    identity = (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    candidates = [identity, *_pca_align_candidates(source, target)]
    return _refine_alignment_candidates(
        source,
        target,
        candidates,
        selector_source=selector_source,
        selector_target=selector_target,
        icp_max_iter=icp_max_iter,
        icp_tolerance=icp_tolerance,
    )


def _refine_alignment_candidates(
    source: np.ndarray,
    target: np.ndarray,
    candidates: list[tuple[np.ndarray, np.ndarray]],
    *,
    selector_source: np.ndarray,
    selector_target: np.ndarray,
    icp_max_iter: int,
    icp_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine every candidate with ICP and pick the best pose by shape agreement.

    Selection narrows by surface F1, then by capped Chamfer (geometry), each
    within a tie tolerance; among the poses that remain genuinely tied — the
    near-symmetric cases — the smallest rotation wins. This keeps the pose
    deterministic and stable across point-sampling seeds.
    """
    if not candidates:
        raise ValueError("alignment requires at least one initial candidate")

    scored: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
    for R_c, t_c in candidates:
        R_ref, t_ref = _open3d_multiscale_icp(
            source, target, R_c, t_c, max_iter=icp_max_iter, tolerance=icp_tolerance,
        )
        for R, t in ((R_c, t_c), (R_ref, t_ref)):
            f1, capped_chamfer, _ = _shape_agreement_selector(
                selector_source, selector_target, R, t,
            )
            scored.append((f1, capped_chamfer, _rotation_angle(R), R, t))

    diag = float(np.linalg.norm(np.ptp(selector_target, axis=0)))
    best_f1 = max(s[0] for s in scored)
    f1_ties = [s for s in scored if s[0] >= best_f1 - _TIE_F1_TOLERANCE]
    min_chamfer = min(s[1] for s in f1_ties)
    geom_ties = [s for s in f1_ties if s[1] <= min_chamfer + _TIE_CHAMFER_FRACTION * diag]
    geom_ties.sort(key=lambda s: (s[2], s[1]))  # least rotation, then Chamfer
    _f1, _chamfer, _angle, R_best, t_best = geom_ties[0]
    _, point_rmse = _point_distance_rmse(source, target, R_best, t_best)
    return R_best, t_best, point_rmse


def _rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in radians."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def _open3d_multiscale_icp(
    source: np.ndarray,
    target: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    *,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine one rigid candidate with Open3D multi-scale point-to-plane ICP."""
    import open3d as o3d

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(source, dtype=np.float64)))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(target, dtype=np.float64)))

    diag = float(np.linalg.norm(np.ptp(target, axis=0)))
    voxel = max(diag / _OPEN3D_ICP_VOXEL_DIVISOR, 1e-6)
    scales = (voxel * 4.0, voxel * 2.0, voxel)
    iter_weights = np.asarray([0.50, 0.30, 0.20])
    iterations = np.maximum(1, np.rint(iter_weights * max_iter).astype(int))
    # Preserve the requested budget exactly after rounding.
    iterations[-1] += max(0, max_iter - int(iterations.sum()))

    T = _transform_from_rt(R_init, t_init)
    for scale, n_iter in zip(scales, iterations, strict=True):
        src_d = src.voxel_down_sample(float(scale))
        tgt_d = tgt.voxel_down_sample(float(scale))
        _estimate_o3d_normals(src_d)
        _estimate_o3d_normals(tgt_d)
        kernel = o3d.pipelines.registration.TukeyLoss(k=float(scale))
        reg = o3d.pipelines.registration.registration_icp(
            src_d,
            tgt_d,
            max_correspondence_distance=float(scale * 2.0),
            init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=tolerance,
                relative_rmse=tolerance,
                max_iteration=int(n_iter),
            ),
        )
        T = np.asarray(reg.transformation, dtype=np.float64)

    return _rt_from_transform(T)


def _estimate_o3d_normals(point_cloud) -> None:  # type: ignore[no-untyped-def]
    """Estimate normals in-place for an Open3D point cloud."""
    import open3d as o3d

    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    point_cloud.normalize_normals()


def _shape_agreement_selector(
    source: np.ndarray,
    target: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``(F1, capped symmetric Chamfer, point RMSE)`` for candidate selection."""
    transformed = (R @ source.T).T + t
    diag = float(np.linalg.norm(np.ptp(target, axis=0)))
    threshold = max(1e-6, 0.01 * diag)
    cap = max(1e-6, 0.02 * diag)

    tree_t = cKDTree(target)
    tree_s = cKDTree(transformed)
    s_to_t, _ = tree_t.query(transformed)
    t_to_s, _ = tree_s.query(target)

    precision = float((s_to_t <= threshold).mean())
    recall = float((t_to_s <= threshold).mean())
    f1 = 0.0 if precision + recall == 0.0 else (
        2.0 * precision * recall / (precision + recall)
    )
    capped_chamfer = float(
        (np.minimum(s_to_t, cap).mean() + np.minimum(t_to_s, cap).mean()) / 2.0,
    )
    point_rmse = float(np.sqrt((np.mean(s_to_t ** 2) + np.mean(t_to_s ** 2)) / 2.0))
    return f1, capped_chamfer, point_rmse


def _point_distance_rmse(
    source: np.ndarray,
    target: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[float, float]:
    """Return one-way nearest-neighbor mean distance and RMSE after transform."""
    transformed = (R @ source.T).T + t
    dists, _ = cKDTree(target).query(transformed)
    return float(dists.mean()), float(np.sqrt(np.mean(dists ** 2)))


def _transform_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack ``R @ x + t`` into a homogeneous Open3D transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rt_from_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a homogeneous Open3D transform into ``(R, t)``."""
    R = _project_to_proper_rotation(np.asarray(T[:3, :3], dtype=np.float64))
    return R, np.asarray(T[:3, 3], dtype=np.float64)


def _project_to_proper_rotation(R: np.ndarray) -> np.ndarray:
    """Strip numerical scale/shear and return the nearest det+1 rotation."""
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj


# ---------------------------------------------------------------------------
# PCA coarse alignment
# ---------------------------------------------------------------------------

# The 24 proper rotations of the cube (axis permutations x right-handed sign
# flips).  Pre-computed once.
_OCTAHEDRAL_CANDIDATES: list[np.ndarray] = []


def _build_octahedral_candidates() -> list[np.ndarray]:
    """Return the 24 rotation matrices in the octahedral symmetry group."""
    if _OCTAHEDRAL_CANDIDATES:
        return _OCTAHEDRAL_CANDIDATES

    perms = list(itertools.permutations(range(3)))
    for perm in perms:
        for signs in itertools.product((-1, 1), repeat=3):
            R = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                R[row, col] = s
            # Keep only proper rotations (det = +1)
            if np.linalg.det(R) > 0:
                _OCTAHEDRAL_CANDIDATES.append(R)

    return _OCTAHEDRAL_CANDIDATES


def _pca_align_candidates(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all 24 octahedral PCA candidates sorted by mean closest-point distance.

    Each entry is ``(R, t)`` such that ``R @ src + t ≈ tgt`` (approximately).
    The list is ordered best-first so callers can take ``[:k]`` for multi-start ICP.
    """
    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)

    src_c = src_pts - src_center
    tgt_c = tgt_pts - tgt_center

    src_axes = _principal_axes(src_c)
    tgt_axes = _principal_axes(tgt_c)

    candidates = _build_octahedral_candidates()
    tgt_tree = cKDTree(tgt_c)

    scored: list[tuple[float, np.ndarray, np.ndarray]] = []
    for C in candidates:
        R_cand = tgt_axes @ C @ src_axes.T
        transformed = (R_cand @ src_c.T).T
        dists, _ = tgt_tree.query(transformed)
        cost = dists.mean()
        t_cand = tgt_center - R_cand @ src_center
        scored.append((cost, R_cand, t_cand))

    scored.sort(key=lambda x: x[0])
    return [(R, t) for _, R, t in scored]


def _principal_axes(centered: np.ndarray) -> np.ndarray:
    """Return (3, 3) matrix whose columns are PCA eigenvectors (descending)."""
    cov = (centered.T @ centered) / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; flip to descending
    idx = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, idx]
    # Ensure right-handed frame
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes

# ---------------------------------------------------------------------------
# BREP transform + export
# ---------------------------------------------------------------------------


def export_aligned_shape(
    shape,  # OCC TopoDS_Shape
    rotation: np.ndarray,
    translation: np.ndarray,
    output: str | Path,
) -> None:
    """Apply a rigid transform to an OCC shape and write the aligned STEP.

    A thin public wrapper over :func:`_apply_and_export` for callers that
    recover the transform separately (e.g. :func:`align_cached_mesh`) and
    only need the BREP exported — a cheap geometric transform with no
    tessellation.
    """
    _apply_and_export(shape, rotation, translation, Path(output))


def _apply_and_export(
    shape,  # OCC TopoDS_Shape
    R: np.ndarray,
    t: np.ndarray,
    output_path: Path,
) -> None:
    """Apply rigid transform to an OCC shape and write a STEP file."""
    from build123d import Compound, export_step
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    trsf = gp_Trsf()
    trsf.SetValues(
        float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(t[0]),
        float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(t[1]),
        float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2]),
    )

    builder = BRepBuilderAPI_Transform(shape, trsf, True)
    builder.Build()
    transformed = builder.Shape()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(Compound(transformed), str(output_path))
