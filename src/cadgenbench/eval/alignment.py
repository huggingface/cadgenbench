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

"""Rigid-body alignment of STEP files (rotation + translation, no scaling).

Two-phase approach:
  1. **PCA coarse alignment**, center both point clouds, align principal
     axes.  All 24 candidates from the octahedral rotation group
     (axis permutations x valid sign flips) are scored; the best is kept.
  2. **ICP refinement**, iterative closest-point with a KD-tree to polish
     the rigid transform.

The aligned source shape is exported as a new STEP file.
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
    pca_top_k: int = 5,
    refine: bool = False,
) -> AlignmentResult:
    """Rigidly align *source* STEP file to *target* STEP file.

    Args:
        source: Path to the STEP file to be transformed.
        target: Path to the reference STEP file (stays fixed).
        output: Where to write the aligned STEP.  Defaults to
            ``<source_stem>_aligned.step`` next to *source*.
        n_samples: Number of surface points to sample for alignment.
        icp_max_iter: Maximum ICP iterations (only used when refine=True).
        icp_tolerance: Convergence threshold on RMSE delta (only used when refine=True).
        seed: RNG seed for reproducible point sampling.
        refine: If True, run ICP refinement after PCA coarse alignment.
            If False (default), use the best of the 24 PCA candidates directly
            without ICP, faster and avoids ICP divergence on dissimilar shapes.

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

    # --- Phase 1: PCA coarse alignment → top-K candidates ---
    # Symmetric parts can have several PCA candidates with nearly equal cost.
    # Running ICP from each and keeping the lowest RMSE avoids locking into
    # a symmetry-equivalent but wrong orientation.
    pca_candidates = _pca_align_candidates(src_pts, tgt_pts)
    top_k = pca_candidates[:pca_top_k]

    # --- Phase 2: ICP refinement (optional) ---
    if refine:
        best: tuple[np.ndarray, np.ndarray, float] | None = None
        for R_c, t_c in top_k:
            R_cand, t_cand, rmse_cand = _icp(
                src_pts, tgt_pts, R_c, t_c, icp_max_iter, icp_tolerance,
            )
            if best is None or rmse_cand < best[2]:
                best = (R_cand, t_cand, rmse_cand)
        R_total, t_total, rmse = best  # type: ignore[misc]
    else:
        # Score all top-k PCA candidates by RMSE, pick the best, no ICP
        tgt_tree = cKDTree(tgt_pts)
        best = None
        for R_c, t_c in top_k:
            transformed = (R_c @ src_pts.T).T + t_c
            dists, _ = tgt_tree.query(transformed)
            rmse_c = float(np.sqrt((dists ** 2).mean()))
            if best is None or rmse_c < best[2]:
                best = (R_c, t_c, rmse_c)
        R_total, t_total, rmse = best

    # --- Apply transform to BREP and export ---
    _apply_and_export(src_shape, R_total, t_total, output)

    return AlignmentResult(
        rotation=R_total,
        translation=t_total,
        rmse=rmse,
        output_path=output,
    )


def align_points(
    source: np.ndarray,
    target: np.ndarray,
    icp_max_iter: int = 50,
    icp_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align two (N, 3) point clouds.  Returns ``(R, t, rmse)``."""
    R_coarse, t_coarse = _pca_align(source, target)
    return _icp(source, target, R_coarse, t_coarse, icp_max_iter, icp_tolerance)


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


def _pca_align(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the single best PCA candidate.  Used by :func:`align_points`."""
    return _pca_align_candidates(src_pts, tgt_pts)[0]


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
# ICP refinement
# ---------------------------------------------------------------------------


def _icp(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    R_init: np.ndarray,
    t_init: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP starting from an initial (R, t).

    Returns ``(R, t, rmse)`` where ``R @ src + t ≈ tgt``.
    """
    R = R_init.copy()
    t = t_init.copy()
    tgt_tree = cKDTree(tgt_pts)
    prev_rmse = np.inf

    for _ in range(max_iter):
        transformed = (R @ src_pts.T).T + t

        dists, indices = tgt_tree.query(transformed)
        rmse = np.sqrt((dists ** 2).mean())

        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse

        # Solve for best R, t given correspondences (Kabsch algorithm)
        matched_tgt = tgt_pts[indices]
        src_center = transformed.mean(axis=0)
        tgt_center = matched_tgt.mean(axis=0)

        src_c = transformed - src_center
        tgt_c = matched_tgt - tgt_center

        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1.0, 1.0, d])
        R_delta = Vt.T @ S @ U.T

        t_delta = tgt_center - R_delta @ src_center

        # Compose: new = R_delta @ old_transformed = R_delta @ (R_old @ x + t_old) + t_delta
        R = R_delta @ R
        t = R_delta @ t + t_delta

    # Final RMSE
    final_transformed = (R @ src_pts.T).T + t
    dists, _ = tgt_tree.query(final_transformed)
    rmse = np.sqrt((dists ** 2).mean())

    return R, t, rmse


# ---------------------------------------------------------------------------
# BREP transform + export
# ---------------------------------------------------------------------------


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
