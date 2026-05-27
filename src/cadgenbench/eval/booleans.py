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

A process-level timeout (:func:`with_timeout`) wraps each op as a
safety net.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cadgenbench.common.mesh import Mesh

if TYPE_CHECKING:  # pragma: no cover - typing only
    import manifold3d as m3d


DEFAULT_BOOLEAN_TIMEOUT_S = 30.0


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
# Defensive timeout (process-level, interrupts C-extension code)
# ---------------------------------------------------------------------------


class BooleanTimeoutError(RuntimeError):
    """Raised when a Manifold Boolean call exceeds the timeout."""


def _bool_worker(
    op_name: str,
    a_mesh: tuple[np.ndarray, np.ndarray],
    b_mesh: tuple[np.ndarray, np.ndarray],
    out_q: "mp.Queue[object]",
) -> None:
    """Top-level worker for :func:`with_timeout`. Must be picklable."""
    try:
        import manifold3d as m3d

        def _to_manifold(verts: np.ndarray, tris: np.ndarray) -> "m3d.Manifold":
            mm = m3d.Mesh(
                vert_properties=np.ascontiguousarray(verts, dtype=np.float32),
                tri_verts=np.ascontiguousarray(tris, dtype=np.uint32),
            )
            return m3d.Manifold(mm)

        a = _to_manifold(*a_mesh)
        b = _to_manifold(*b_mesh)
        if op_name == "intersect":
            result = a ^ b
        elif op_name == "union":
            result = a + b
        elif op_name == "subtract":
            result = a - b
        else:
            raise ValueError(f"unknown op: {op_name}")
        # Round-trip back to (verts, tris) so the result is picklable.
        rmesh = result.to_mesh()
        out_q.put(("ok", (np.asarray(rmesh.vert_properties), np.asarray(rmesh.tri_verts))))
    except Exception as exc:  # pragma: no cover - safety net only
        out_q.put(("error", repr(exc)))


@dataclass(frozen=True)
class _BoolResult:
    """Payload returned by :func:`bool_with_timeout`."""

    vertices: np.ndarray
    triangles: np.ndarray


def bool_with_timeout(
    op: str,
    a: Mesh,
    b: Mesh,
    *,
    timeout_s: float = DEFAULT_BOOLEAN_TIMEOUT_S,
) -> _BoolResult:
    """Run a Manifold Boolean in a subprocess with a hard timeout.

    Use for the highest-risk call sites only. In-process
    :func:`intersect` / :func:`union` / :func:`subtract` are
    millisecond-fast and never need this in normal operation, but the
    subprocess wrapper lets us prove "no metric run can hang forever"
    independent of any single library's behaviour.

    Raises:
        BooleanTimeoutError: ``timeout_s`` elapsed without a result.
        RuntimeError: the subprocess raised inside Manifold.
    """
    ctx = mp.get_context("fork" if mp.get_start_method() == "fork" else "spawn")
    out_q: mp.Queue[object] = ctx.Queue()
    proc = ctx.Process(
        target=_bool_worker,
        args=(
            op,
            (np.asarray(a.vertices), np.asarray(a.triangles)),
            (np.asarray(b.vertices), np.asarray(b.triangles)),
            out_q,
        ),
        daemon=True,
    )
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():
            proc.kill()
        raise BooleanTimeoutError(
            f"Manifold {op} did not complete in {timeout_s:.1f}s "
            f"(a: F={a.n_triangles} V={a.n_vertices}, "
            f"b: F={b.n_triangles} V={b.n_vertices})",
        )
    if out_q.empty():
        raise RuntimeError(
            f"Manifold {op} subprocess exited without a result "
            f"(exit code {proc.exitcode})",
        )
    tag, payload = out_q.get()
    if tag == "error":
        raise RuntimeError(f"Manifold {op} failed: {payload}")
    verts, tris = payload
    return _BoolResult(vertices=verts, triangles=tris)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def manifold_volume(manifold: "m3d.Manifold") -> float:
    """Volume of a manifold; 0.0 for an empty result."""
    if manifold.is_empty():
        return 0.0
    return float(manifold.volume())
