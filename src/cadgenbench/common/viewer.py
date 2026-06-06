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

"""Headless STEP renderer using VTK via PyVista.

Tessellates the STEP file with :func:`cadgenbench.common.mesh.tessellate_step`,
hands the welded triangle mesh to a :class:`pyvista.Plotter`, draws
shaded triangles, and returns PNG bytes
per requested view. Camera presets are shared with the rest of the
benchmark via :mod:`cadgenbench.common.camera_presets`. Projection is
parallel (orthographic) across all views, the canonical CAD-drawing
convention; per-view ``parallel_scale`` is fitted to the bbox projected
onto the camera plane so wide/flat parts viewed from a side aren't
dwarfed.

In-process, no subprocesses, no Chromium. VTK picks its OpenGL backend
at runtime:

- macOS dev: Cocoa / NSOpenGL (GPU when present).
- Linux GPU: EGL (GPU when drivers are present).
- Linux CPU-only (HF Space cpu-upgrade and similar): EGL + Mesa
  software rasteriser (``libglx-mesa0`` / ``libegl-mesa0``). No code
  change required.

Per-render cost on a modern macOS dev box: ~40 ms (small jig parts) to
~350 ms (heavy NIST CTC parts) for 1024x768 PNGs, roughly two orders
of magnitude faster than the previous Chromium / three-cad-viewer path.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv
from PIL import Image

from cadgenbench.common.camera_presets import (
    CAMERA_PRESETS,
    DEFAULT_VIEWS,
    DISTANCE_FACTOR,
    camera_placement,
    validate_views,
)
from cadgenbench.common.mesh import (
    Mesh,
    deflection_for_bbox,
    tessellate_step,
)
from cadgenbench.common.profiling import phase

# Re-exported for callers that want the canonical preset / default-view sets.
__all__ = [
    "CAMERA_PRESETS",
    "DEFAULT_VIEWS",
    "DIFF_EXTRA_RGB",
    "DIFF_GHOST_RGB",
    "DIFF_MISSING_RGB",
    "MeshDiff",
    "OVERLAY_PALETTE",
    "RenderedImage",
    "mesh_diff",
    "render_mesh",
    "render_mesh_diff",
    "render_mesh_diff_turntable_webp",
    "render_mesh_overlay",
    "render_mesh_turntable_webp",
    "render_overlay",
    "render_step",
    "render_step_turntable_webp",
    "render_steps",
]


# ---------------------------------------------------------------------------
# Look-and-feel constants
# ---------------------------------------------------------------------------

# Default body colour. Slightly brighter than the OCC ``Color(0.68,0.72,0.76)``
# we used to hand to three-cad-viewer; with VTK's ambient+diffuse model the
# OCC value lands too dark, so brighten the base RGB and let directional
# lighting carve out the shading. Edge colour matches tcv's ``edgeColor: 0x333333``.
DEFAULT_BODY_RGB: tuple[float, float, float] = (0.85, 0.87, 0.90)
BACKGROUND_RGB: tuple[float, float, float] = (1.0, 1.0, 1.0)

# Margin (multiplier on the projected bbox half-extent) when fitting the
# part to the frame. > 1 leaves whitespace around the silhouette.
FRAME_MARGIN: float = 1.10

# rgba palette used by :func:`render_overlay` when no explicit colours are
# given. Matches the previous module's palette so existing report renders
# stay consistent.
OVERLAY_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.18, 0.45, 0.86, 1.00),  # solid blue
    (0.90, 0.30, 0.30, 0.40),  # translucent red
    (1.00, 0.85, 0.00, 1.00),  # solid yellow (good for highlights)
    (0.20, 0.70, 0.30, 0.60),  # translucent green
)

# Edit-diff look (see :func:`render_mesh_diff`). The candidate output is ghosted
# translucent grey -- the neutral "your geometry" body, the same ghost the
# interface overlay uses -- so internal changes show through. Every surface that
# *differs* from the ground truth is painted on top in a *warm* highlight: the
# two directions of difference are kept distinct, but both stay in the
# "alert / wrong" family so neither reads as "correct" (an earlier blue/red split
# coloured "added" blue, which misled viewers into reading it as good). This is
# the standard CAD deviation-map idea (red = excess, cool = deficit) with the
# cool/"looks-fine" end swapped for amber:
#   - extra material the candidate added that the GT lacks  -> red  ("too much")
#   - GT material the candidate is missing                  -> amber ("too little")
# Red is shared with cadgenbench.eval.interface_match_viz's "wrong" colour and
# the grey ghost is shared too, so the two per-fixture reports read as one
# palette (grey = you, red = wrong; amber = the missing-material flavour of wrong).
DIFF_GHOST_RGB: tuple[float, float, float] = (0.74, 0.77, 0.82)
DIFF_EXTRA_RGB: tuple[float, float, float] = (0.90, 0.16, 0.16)    # added by candidate (too much)
DIFF_MISSING_RGB: tuple[float, float, float] = (0.96, 0.60, 0.10)  # missing from candidate (too little)
DIFF_GHOST_ALPHA: float = 0.16
# "A bit of alpha" on the highlight (vs fully opaque) so where extra and missing
# patches stack, or sit over the ghost body, both still read through.
DIFF_HIGHLIGHT_ALPHA: float = 0.85
# Surface-deviation tolerance for the diff: a vertex must lie more than this far
# OUTSIDE the other solid to count as added/removed, which keeps coincident
# surfaces and tessellation noise from lighting up. Expressed as a *fraction of
# the GT bounding-box diagonal* (resolved per fixture in :func:`mesh_diff`) so it
# matches the shape-similarity point-cloud F1 distance gate exactly, instead of a
# fixed millimetre value that a small edit on a large part slips under. A tiny
# absolute floor guards against a degenerate (~0-diagonal) input.
DIFF_TOL_FRACTION: float = 0.005  # 0.5% of the GT bbox diagonal (== shape F1 gate)
DIFF_TOL_FLOOR_MM: float = 1e-4
# Float the highlighted patch out along its own normals by this many mm so it
# sits proud of the ghost shell and never z-fights the coincident base surface.
DIFF_OFFSET_MM: float = 0.2


@dataclass(frozen=True)
class RenderedImage:
    """One rendered view of a STEP file or mesh."""

    name: str
    data: bytes
    width: int
    height: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_mesh(
    mesh: Mesh,
    views: Sequence[str] | None = None,
    *,
    width: int = 1024,
    height: int = 768,
    body_rgb: tuple[float, float, float] = DEFAULT_BODY_RGB,
) -> list[RenderedImage]:
    """Render an already-tessellated :class:`Mesh` from multiple angles.

    The eval pipeline tessellates each candidate / GT once for the
    metric path; this entry point lets the renderer reuse that mesh
    instead of re-tessellating from STEP.

    Args:
        mesh: Welded triangle mesh produced by
            :func:`cadgenbench.common.mesh.tessellate_step` (or any
            equivalent welded mesh in mm-space).
        views: Camera preset names. Defaults to :data:`DEFAULT_VIEWS`.
        width: Image width in pixels.
        height: Image height in pixels.
        body_rgb: Per-render body colour (rgb in ``[0, 1]``).

    Returns:
        One :class:`RenderedImage` per requested view, in input order.

    Raises:
        ValueError: Unknown camera preset, or empty mesh.
    """
    if mesh.n_triangles == 0:
        raise ValueError("render_mesh: mesh has zero triangles")
    if views is None:
        views = DEFAULT_VIEWS
    validate_views(views)

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    body = _mesh_to_polydata(mesh)

    pngs = _render_views(
        shapes=[(body, body_rgb, 1.0)],
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        views=views,
        width=width,
        height=height,
    )
    return [
        RenderedImage(name=view, data=png, width=width, height=height)
        for view, png in zip(views, pngs)
    ]


def render_step(
    step_path: str | Path,
    views: Sequence[str] | None = None,
    *,
    width: int = 1024,
    height: int = 768,
) -> list[RenderedImage]:
    """Render a single STEP file from multiple camera angles.

    Tessellation deflection is derived from the part's own bounding-box
    diagonal, clamped via :func:`cadgenbench.common.mesh.deflection_for_bbox`.

    Args:
        step_path: Path to a .step / .stp file.
        views: Camera preset names. Defaults to :data:`DEFAULT_VIEWS`.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One :class:`RenderedImage` per requested view, in input order.

    Raises:
        FileNotFoundError: STEP file does not exist.
        ValueError: Unknown camera preset.
        RuntimeError: STEP file produced no geometry.
    """
    step_path = Path(step_path)
    mesh = _tessellate(step_path)
    return render_mesh(mesh, views, width=width, height=height)


def render_mesh_turntable_webp(
    mesh: Mesh,
    *,
    frames: int = 120,
    width: int = 512,
    height: int = 384,
    duration_ms: int = 150,
    quality: int = 68,
    body_rgb: tuple[float, float, float] = DEFAULT_BODY_RGB,
) -> bytes:
    """Render a smooth Z-up turntable as an animated WebP for a welded mesh.

    Frames share the exact shaded look of the canonical PNG views (same
    material, lighting and parallel projection as :func:`render_mesh`); the
    only difference is the camera orbits the Z axis at a fixed orthographic
    scale. The animation is encoded as truecolor WebP (no palette banding,
    inter-frame compression) which is both smoother and smaller than the
    GIF equivalent: at the defaults a full orbit is 120 frames over ~18s and
    a typical part lands around ~390 KB.
    """
    if mesh.n_triangles == 0:
        raise ValueError("render_mesh_turntable_webp: mesh has zero triangles")
    if frames < 2:
        raise ValueError("render_mesh_turntable_webp: frames must be >= 2")

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    body = _mesh_to_polydata(mesh)
    pngs = _render_turntable_frames(
        shapes=[(body, body_rgb, 1.0)],
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        frames=frames,
        width=width,
        height=height,
    )
    return _encode_webp(pngs, duration_ms=duration_ms, quality=quality)


# WebP compression effort. Fixed at 3 on purpose: it is the single value
# every caller (eval submissions + one-time GT) encodes with, so the encode
# stays cheap and uniform. method=6 (Pillow's max effort) was ~30s/clip here
# for only ~3% smaller files (363 vs 387 KB) -- ~40x slower for no real gain;
# method=3 keeps the quality and size while encoding in <1s. Do NOT expose
# this as a parameter: a per-caller knob is exactly how an accidental
# method=6 would creep back into the eval hot path.
_WEBP_METHOD = 3


def _encode_webp(pngs: list[bytes], *, duration_ms: int, quality: int) -> bytes:
    """Encode PNG frames into a looping animated WebP.

    WebP is truecolor (no 256-colour palette, so the grey shading ramp keeps
    its contrast) and compresses frame-to-frame deltas like a video codec, so
    a high frame count stays small. Compression effort is fixed at
    :data:`_WEBP_METHOD` (see its note) so every artifact encodes identically.
    """
    frames = [Image.open(io.BytesIO(png)).convert("RGB") for png in pngs]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=quality,
        method=_WEBP_METHOD,
    )
    return buf.getvalue()


def render_step_turntable_webp(
    step_path: str | Path,
    *,
    frames: int = 120,
    width: int = 512,
    height: int = 384,
    duration_ms: int = 150,
    quality: int = 68,
) -> bytes:
    """Render a turntable WebP directly from a STEP file (tessellates once).

    Prefer :func:`render_mesh_turntable_webp` when a welded mesh is already in
    hand (e.g. the eval's aligned candidate mesh, or a GT part loaded from its
    trusted ``.mesh.npz`` sidecar) so no STEP round-trip / re-tessellation is
    paid.
    """
    mesh = _tessellate(Path(step_path))
    return render_mesh_turntable_webp(
        mesh,
        frames=frames,
        width=width,
        height=height,
        duration_ms=duration_ms,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Edit-diff rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshDiff:
    """Classified surface difference between a candidate mesh and the GT.

    Small edits (a moved hole, a shortened internal boss, a 2.5 mm thickening)
    are visually invisible in a plain shaded render, so a no-op candidate that
    echoes the input looks identical to a correct edit. This splits the two
    surfaces into the material that genuinely differs, classified by *signed*
    distance to the other solid so coincident or interior surfaces never light
    up:

    Attributes:
        removed: Sub-mesh of the GT surface lying more than ``tol_mm`` outside
            the candidate solid (present in GT, missing from the candidate).
            ``None`` when nothing qualifies. Drawn red.
        added: Sub-mesh of the candidate surface lying more than ``tol_mm``
            outside the GT solid (present in the candidate, absent from GT).
            ``None`` when nothing qualifies. Drawn blue.
        fraction_removed: Share of GT triangles flagged removed, in ``[0, 1]``.
        fraction_added: Share of candidate triangles flagged added, in
            ``[0, 1]``.
        max_deviation_mm: Largest outward signed distance, in mm, over both
            directions. The magnitude the colour saturation throws away; a
            no-op on a "remove three holes" task reads ~0, a correct edit's
            residual is small, a wrong edit is large.
    """

    removed: Mesh | None
    added: Mesh | None
    fraction_removed: float
    fraction_added: float
    max_deviation_mm: float


def _diff_tol_mm(gt_mesh: Mesh, tol_mm: float | None) -> float:
    """Resolve the diff tolerance: explicit *tol_mm*, else 0.5% of the GT bbox.

    The scale-relative default matches the shape-similarity F1 distance gate
    (:data:`DIFF_TOL_FRACTION`), so the diff lights up the same deviations the
    shape score penalises rather than slipping a small edit on a large part
    under a fixed millimetre threshold.
    """
    if tol_mm is not None:
        return tol_mm
    lo = gt_mesh.vertices.min(axis=0)
    hi = gt_mesh.vertices.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return max(DIFF_TOL_FLOOR_MM, DIFF_TOL_FRACTION * diag)


def mesh_diff(
    gt_mesh: Mesh,
    candidate_mesh: Mesh,
    *,
    tol_mm: float | None = None,
) -> MeshDiff:
    """Classify the added / removed material between candidate and GT.

    Both meshes must already be in the same frame (the eval aligns the
    candidate to GT before scoring; the GT mesh comes from its trusted
    ``.mesh.npz`` sidecar). Signed distance is computed with Open3D's
    ``RaycastingScene``; a positive value means the query point lies outside
    the reference solid.

    Args:
        gt_mesh: Ground-truth welded mesh.
        candidate_mesh: Candidate welded mesh, aligned into the GT frame.
        tol_mm: Outward-distance threshold (mm) above which a vertex counts as
            added / removed. ``None`` (the default) resolves it to
            :data:`DIFF_TOL_FRACTION` of the GT bounding-box diagonal so it
            tracks the shape-F1 distance gate; pass an explicit value to
            override. Below the threshold, coincident surfaces and tessellation
            noise stay unflagged.

    Returns:
        A :class:`MeshDiff`.
    """
    tol_mm = _diff_tol_mm(gt_mesh, tol_mm)
    s_cand = _signed_distance(candidate_mesh.vertices, gt_mesh)
    s_gt = _signed_distance(gt_mesh.vertices, candidate_mesh)
    added, frac_added = _subset_mesh(candidate_mesh, s_cand, tol_mm)
    removed, frac_removed = _subset_mesh(gt_mesh, s_gt, tol_mm)
    max_dev = float(max(s_cand.max(initial=0.0), s_gt.max(initial=0.0)))
    return MeshDiff(
        removed=removed,
        added=added,
        fraction_removed=frac_removed,
        fraction_added=frac_added,
        max_deviation_mm=max_dev,
    )


def render_mesh_diff(
    gt_mesh: Mesh,
    candidate_mesh: Mesh,
    views: Sequence[str] | None = None,
    *,
    width: int = 1024,
    height: int = 768,
    tol_mm: float | None = None,
    ghost_rgb: tuple[float, float, float] = DIFF_GHOST_RGB,
) -> list[RenderedImage]:
    """Render the candidate as a ghost body with added/removed material lit up.

    The candidate is drawn translucent (so internal changes show through) and
    the :func:`mesh_diff` added (blue) / removed (red) sub-meshes are painted
    opaque on top, floated proud of the shell so they never z-fight.

    Args:
        gt_mesh: Ground-truth welded mesh.
        candidate_mesh: Candidate welded mesh, aligned into the GT frame.
        views: Camera preset names. Defaults to :data:`DEFAULT_VIEWS`.
        width: Image width in pixels.
        height: Image height in pixels.
        tol_mm: Surface-deviation tolerance passed to :func:`mesh_diff`.
        ghost_rgb: Body colour for the translucent ghost.

    Returns:
        One :class:`RenderedImage` per requested view, in input order.

    Raises:
        ValueError: Unknown camera preset, or empty candidate mesh.
    """
    if candidate_mesh.n_triangles == 0:
        raise ValueError("render_mesh_diff: candidate mesh has zero triangles")
    if views is None:
        views = DEFAULT_VIEWS
    validate_views(views)

    diff = mesh_diff(gt_mesh, candidate_mesh, tol_mm=tol_mm)
    shapes = _diff_shapes(candidate_mesh, diff, ghost_rgb)
    bbox_min, bbox_max = _diff_bbox(gt_mesh, candidate_mesh)
    pngs = _render_views(
        shapes=shapes,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        views=views,
        width=width,
        height=height,
    )
    return [
        RenderedImage(name=view, data=png, width=width, height=height)
        for view, png in zip(views, pngs)
    ]


def render_mesh_diff_turntable_webp(
    gt_mesh: Mesh,
    candidate_mesh: Mesh,
    *,
    frames: int = 120,
    width: int = 512,
    height: int = 384,
    duration_ms: int = 150,
    quality: int = 68,
    tol_mm: float | None = None,
    ghost_rgb: tuple[float, float, float] = DIFF_GHOST_RGB,
) -> bytes:
    """Render a Z-up turntable of the edit diff as an animated WebP.

    Motion is what sells a small or internal edit: the changed material orbits
    into view from every angle even when a single still would occlude it. Same
    ghost-body + opaque added/removed look as :func:`render_mesh_diff`, encoded
    truecolor (no palette banding) and small (~200 KB) via :func:`_encode_webp`.

    Args:
        gt_mesh: Ground-truth welded mesh.
        candidate_mesh: Candidate welded mesh, aligned into the GT frame.
        frames: Number of orbit frames (>= 2).
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_ms: Per-frame duration in the encoded WebP.
        quality: WebP quality in ``[0, 100]``.
        tol_mm: Surface-deviation tolerance passed to :func:`mesh_diff`.
        ghost_rgb: Body colour for the translucent ghost.

    Returns:
        Encoded animated-WebP bytes.

    Raises:
        ValueError: Empty candidate mesh, or ``frames`` < 2.
    """
    if candidate_mesh.n_triangles == 0:
        raise ValueError("render_mesh_diff_turntable_webp: candidate mesh has zero triangles")
    if frames < 2:
        raise ValueError("render_mesh_diff_turntable_webp: frames must be >= 2")

    diff = mesh_diff(gt_mesh, candidate_mesh, tol_mm=tol_mm)
    shapes = _diff_shapes(candidate_mesh, diff, ghost_rgb)
    bbox_min, bbox_max = _diff_bbox(gt_mesh, candidate_mesh)
    pngs = _render_turntable_frames(
        shapes=shapes,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        frames=frames,
        width=width,
        height=height,
    )
    return _encode_webp(pngs, duration_ms=duration_ms, quality=quality)


def render_steps(
    step_paths: Sequence[str | Path],
    views: Sequence[str] | None = None,
    *,
    width: int = 1024,
    height: int = 768,
) -> dict[str, list[RenderedImage]]:
    """Render multiple STEP files, keyed by stem.

    Each file is tessellated and rendered independently; one VTK
    plotter per (file, view), all in-process.

    Args:
        step_paths: Paths to .step / .stp files.
        views: Camera preset names. Defaults to :data:`DEFAULT_VIEWS`.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``{stem: [RenderedImage, ...]}`` for each input file.

    Raises:
        FileNotFoundError: A STEP file does not exist.
        ValueError: ``step_paths`` is empty, or unknown camera preset.
        RuntimeError: A STEP file produced no geometry.
    """
    if not step_paths:
        raise ValueError("step_paths must not be empty")
    paths = [Path(p) for p in step_paths]
    return {
        p.stem: render_step(p, views, width=width, height=height) for p in paths
    }


def render_overlay(
    step_paths: Sequence[str | Path],
    *,
    colors: Sequence[tuple[float, float, float, float]] | None = None,
    views: Sequence[str] | None = None,
    width: int = 1024,
    height: int = 768,
) -> list[RenderedImage]:
    """Render several STEPs as one composite scene per view.

    Each STEP is tessellated independently (its own deflection from its
    own bbox), assigned its rgba colour, and rendered together. Used
    for metric-development visualisation (GT + jig overlays, alignment
    debugging, etc.). Later shapes are drawn on top of earlier ones,
    matching the call-order convention of the previous viewer.

    Args:
        step_paths: Paths to .step / .stp files.
        colors: Per-shape rgba (each component in ``[0, 1]``). Defaults
            to :data:`OVERLAY_PALETTE` cycled.
        views: Camera preset names. Defaults to ``("iso",)``, a single
            iso view is typically enough for an overlay.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One :class:`RenderedImage` per requested view, in input order.

    Raises:
        FileNotFoundError: A STEP file does not exist.
        ValueError: ``step_paths`` is empty, ``colors`` length mismatch,
            or unknown camera preset.
        RuntimeError: A STEP file produced no geometry.
    """
    if not step_paths:
        raise ValueError("step_paths must not be empty")
    paths = [Path(p) for p in step_paths]
    if colors is None:
        colors = [OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)] for i in range(len(paths))]
    elif len(colors) != len(paths):
        raise ValueError(
            f"colors length ({len(colors)}) must match step_paths length ({len(paths)})"
        )
    meshes = [_tessellate(path) for path in paths]
    return render_mesh_overlay(
        meshes, colors=colors, views=views, width=width, height=height,
    )


def render_mesh_overlay(
    meshes: Sequence[Mesh],
    *,
    colors: Sequence[tuple[float, float, float, float]] | None = None,
    views: Sequence[str] | None = None,
    width: int = 1024,
    height: int = 768,
) -> list[RenderedImage]:
    """Render several already-tessellated meshes as one composite scene per view.

    Mesh-native sibling of :func:`render_overlay`: callers that already
    hold welded meshes (e.g. metric-side tessellations or a manifold
    Boolean result) overlay them directly without a STEP round-trip.
    Later meshes draw on top of earlier ones.

    Args:
        meshes: Welded triangle meshes (mm-space), drawn in order.
        colors: Per-mesh rgba (each component in ``[0, 1]``). Defaults
            to :data:`OVERLAY_PALETTE` cycled.
        views: Camera preset names. Defaults to ``("iso",)``.
        width: Image width in pixels.
        height: Image height in pixels.

    Raises:
        ValueError: ``meshes`` is empty, ``colors`` length mismatch, a
            mesh is empty, or unknown camera preset.
    """
    if not meshes:
        raise ValueError("meshes must not be empty")
    if colors is None:
        colors = [OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)] for i in range(len(meshes))]
    elif len(colors) != len(meshes):
        raise ValueError(
            f"colors length ({len(colors)}) must match meshes length ({len(meshes)})"
        )
    if views is None:
        views = ("iso",)
    validate_views(views)

    shapes: list[tuple[pv.PolyData, tuple[float, float, float], float]] = []
    bbox_min = np.array([np.inf] * 3, dtype=np.float64)
    bbox_max = np.array([-np.inf] * 3, dtype=np.float64)
    for mesh, rgba in zip(meshes, colors):
        if mesh.n_triangles == 0:
            raise ValueError("render_mesh_overlay: a mesh has zero triangles")
        body = _mesh_to_polydata(mesh)
        shapes.append((body, (rgba[0], rgba[1], rgba[2]), float(rgba[3])))
        bbox_min = np.minimum(bbox_min, mesh.vertices.min(axis=0))
        bbox_max = np.maximum(bbox_max, mesh.vertices.max(axis=0))

    pngs = _render_views(
        shapes=shapes,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        views=views,
        width=width,
        height=height,
    )
    return [
        RenderedImage(name=view, data=png, width=width, height=height)
        for view, png in zip(views, pngs)
    ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _tessellate(step_path: Path) -> Mesh:
    """STEP -> welded triangle mesh, deflection from the part's own bbox."""
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")
    from build123d import import_step

    shape = import_step(str(step_path))
    if shape is None or not shape.wrapped:
        raise RuntimeError(f"STEP file produced no geometry: {step_path}")
    bb = shape.bounding_box()
    diag = float(
        np.linalg.norm(
            np.array([bb.max.X - bb.min.X, bb.max.Y - bb.min.Y, bb.max.Z - bb.min.Z]),
        )
    )
    return tessellate_step(step_path, deflection_for_bbox(diag))


def _mesh_to_polydata(mesh: Mesh) -> pv.PolyData:
    """Convert a welded :class:`Mesh` to a :class:`pyvista.PolyData`."""
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    t = np.ascontiguousarray(mesh.triangles, dtype=np.int64)
    n = t.shape[0]
    cells = np.empty((n, 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = t
    return pv.PolyData(v, cells.reshape(-1))


def _signed_distance(points: np.ndarray, target: Mesh) -> np.ndarray:
    """Signed distance from each point to the *target* solid.

    Positive outside the solid, negative inside. Uses Open3D's
    ``RaycastingScene`` (lazy import, matching :mod:`cadgenbench.eval.alignment`).
    """
    import open3d as o3d  # noqa: PLC0415

    tri = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(target.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(
            np.ascontiguousarray(target.triangles, dtype=np.int32),
            dtype=o3d.core.Dtype.Int32,
        ),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tri)
    query = o3d.core.Tensor(
        np.ascontiguousarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32,
    )
    return scene.compute_signed_distance(query).numpy().astype(np.float64)


def _subset_mesh(
    mesh: Mesh, signed: np.ndarray, tol_mm: float,
) -> tuple[Mesh | None, float]:
    """Extract the triangles with any vertex more than *tol_mm* outside.

    Returns the compacted sub-mesh (vertices re-indexed) and the fraction of
    triangles kept. ``(None, 0.0)`` when nothing qualifies.
    """
    cell_far = (signed > tol_mm)[mesh.triangles].any(axis=1)
    if not cell_far.any():
        return None, 0.0
    tris = mesh.triangles[cell_far]
    used = np.unique(tris)
    remap = np.full(mesh.n_vertices, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    sub = Mesh(
        vertices=mesh.vertices[used],
        triangles=remap[tris].astype(np.int64),
        linear_deflection_mm=mesh.linear_deflection_mm,
    )
    return sub, float(cell_far.mean())


def _diff_subpoly(mesh: Mesh) -> pv.PolyData:
    """Polydata for a highlight sub-mesh, floated out along its own normals."""
    poly = _mesh_to_polydata(mesh)
    if DIFF_OFFSET_MM:
        poly = poly.compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True,
        )
        poly.points = poly.points + poly.point_data["Normals"] * DIFF_OFFSET_MM
    return poly


def _diff_shapes(
    candidate_mesh: Mesh,
    diff: MeshDiff,
    ghost_rgb: tuple[float, float, float],
) -> list[tuple[pv.PolyData, tuple[float, float, float], float]]:
    """Build the shape list for the diff renderers: ghost body + warm differences.

    The candidate is the translucent ghost; the differing material is painted on
    top, floated proud of the shell, in two warm tones that both read as "wrong"
    while keeping the direction legible -- red for extra material the candidate
    added, amber for ground-truth material it is missing. See
    :data:`DIFF_EXTRA_RGB` / :data:`DIFF_MISSING_RGB`.
    """
    shapes: list[tuple[pv.PolyData, tuple[float, float, float], float]] = [
        (_mesh_to_polydata(candidate_mesh), ghost_rgb, DIFF_GHOST_ALPHA),
    ]
    if diff.removed is not None:
        shapes.append((_diff_subpoly(diff.removed), DIFF_MISSING_RGB, DIFF_HIGHLIGHT_ALPHA))
    if diff.added is not None:
        shapes.append((_diff_subpoly(diff.added), DIFF_EXTRA_RGB, DIFF_HIGHLIGHT_ALPHA))
    return shapes


def _diff_bbox(gt_mesh: Mesh, candidate_mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Union bounding box of both meshes (so neither is clipped)."""
    bbox_min = np.minimum(gt_mesh.vertices.min(axis=0), candidate_mesh.vertices.min(axis=0))
    bbox_max = np.maximum(gt_mesh.vertices.max(axis=0), candidate_mesh.vertices.max(axis=0))
    return bbox_min, bbox_max


def _parallel_scale_for_view(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    window_w: int,
    window_h: int,
) -> float:
    """Half-height of the visible volume needed to fit the bbox in frame.

    Projects the 8 bbox corners onto the camera's (right, up) plane and
    picks the smallest ``parallel_scale`` that contains every corner,
    accounting for the window aspect ratio.
    """
    view_dir = target - eye
    view_dir /= max(float(np.linalg.norm(view_dir)), 1e-12)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    right = np.cross(view_dir, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, view_dir)

    corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ], dtype=np.float64) - target
    half_w = float(np.max(np.abs(corners @ right)))
    half_h = float(np.max(np.abs(corners @ up)))
    aspect = float(window_w) / float(window_h)
    return max(half_h, half_w / aspect) * FRAME_MARGIN


# VTK/OpenGL is not thread-safe and there is a single GL context per process.
# Baseline fixtures render from threads, so this guards concurrent renders
# within one process.
_RENDER_LOCK = threading.Lock()

# Cross-process render serialisation. The threading lock above only spans one
# process; the eval ProcessPool runs 8 sibling workers that would otherwise
# drive the single GPU's GL context concurrently — the contention class that
# can balloon a render to many seconds (and which the baseline agent already
# works around with its dedicated render pool). An advisory file lock makes
# rendering one-at-a-time machine-wide.
_CROSS_PROC_RENDER_LOCK_PATH = (
    os.environ.get("CADGENBENCH_RENDER_LOCK_FILE")
    or str(Path(tempfile.gettempdir()) / "cadgenbench_render.lock")
)


@contextmanager
def _cross_process_render_lock():
    """Serialise GPU/GL rendering across processes via an ``fcntl`` file lock.

    Disabled with ``CADGENBENCH_RENDER_LOCK=0``; a no-op where ``fcntl`` is
    unavailable (non-POSIX). The lock is advisory and held only for the
    duration of one part's render.
    """
    if os.environ.get("CADGENBENCH_RENDER_LOCK", "1") == "0":
        yield
        return
    try:
        import fcntl  # noqa: PLC0415
    except ImportError:
        yield
        return
    lock_file = open(_CROSS_PROC_RENDER_LOCK_PATH, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _render_views(
    *,
    shapes: list[tuple[pv.PolyData, tuple[float, float, float], float]],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    views: Sequence[str],
    width: int,
    height: int,
) -> list[bytes]:
    """Render every *view* of *shapes* as PNG bytes.

    Use a fresh ``pv.Plotter`` per view. Reusing one plotter and mutating only
    the camera looked faster, but VTK/PyVista can return stale screenshots
    after camera changes on the headless backends we use for eval, causing
    ``front`` / ``top`` / ``right`` files to contain the first (usually ``iso``)
    view. A per-view plotter is slower but makes the output images independent
    by construction.

    Serialised by ``_RENDER_LOCK`` (in-process threads) and
    ``_cross_process_render_lock`` (sibling eval workers); timed as the
    ``render`` phase when profiling is enabled.
    """
    pngs: list[bytes] = []
    with _RENDER_LOCK, _cross_process_render_lock(), phase(
        f"render n={len(views)}",
    ):
        for view in views:
            pl = pv.Plotter(off_screen=True, window_size=(width, height))
            try:
                pl.set_background(BACKGROUND_RGB)
                for body, rgb, alpha in shapes:
                    pl.add_mesh(
                        body,
                        color=rgb,
                        opacity=alpha,
                        smooth_shading=False,
                        show_edges=False,
                        ambient=0.45,
                        diffuse=0.65,
                        specular=0.10,
                        specular_power=15,
                    )

                eye, target, up = camera_placement(view, bbox_min, bbox_max)
                cam = pl.camera
                cam.position = tuple(map(float, eye))
                cam.focal_point = tuple(map(float, target))
                cam.up = tuple(map(float, up))
                cam.parallel_projection = True
                cam.parallel_scale = _parallel_scale_for_view(
                    bbox_min, bbox_max,
                    eye=eye, target=target, up=up,
                    window_w=width, window_h=height,
                )
                arr = np.asarray(
                    pl.screenshot(return_img=True, transparent_background=False)
                )
            finally:
                pl.close()

            # Encode after closing the VTK context so each view is fully
            # independent before the next plotter is created.
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG", optimize=True)
            pngs.append(buf.getvalue())

    return pngs


def _render_turntable_frames(
    *,
    shapes: list[tuple[pv.PolyData, tuple[float, float, float], float]],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    frames: int,
    width: int,
    height: int,
) -> list[bytes]:
    """Render one complete orbit around the Z axis as PNG frame bytes."""
    pngs: list[bytes] = []
    target = (bbox_min + bbox_max) * 0.5
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    distance = max(diagonal * DISTANCE_FACTOR, 1e-6)
    elevation = np.deg2rad(32.0)
    start_azimuth = np.deg2rad(-45.0)
    directions = [
        np.array(
            [
                np.cos(start_azimuth + (2.0 * np.pi * i / frames)) * np.cos(elevation),
                np.sin(start_azimuth + (2.0 * np.pi * i / frames)) * np.cos(elevation),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )
        for i in range(frames)
    ]
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Keep the orthographic scale fixed across the whole orbit. Computing a
    # fresh fit per frame makes long/flat parts appear to breathe toward and
    # away from the camera as their projected bbox changes.
    parallel_scale = max(
        _parallel_scale_for_view(
            bbox_min, bbox_max,
            eye=target + direction * distance,
            target=target,
            up=up,
            window_w=width,
            window_h=height,
        )
        for direction in directions
    )
    with _RENDER_LOCK, _cross_process_render_lock(), phase(
        f"render turntable n={frames}",
    ):
        for direction in directions:
            eye = target + direction * distance

            pl = pv.Plotter(off_screen=True, window_size=(width, height))
            try:
                pl.set_background(BACKGROUND_RGB)
                for body, rgb, alpha in shapes:
                    pl.add_mesh(
                        body,
                        color=rgb,
                        opacity=alpha,
                        smooth_shading=False,
                        show_edges=False,
                        ambient=0.45,
                        diffuse=0.65,
                        specular=0.10,
                        specular_power=15,
                    )

                cam = pl.camera
                cam.position = tuple(map(float, eye))
                cam.focal_point = tuple(map(float, target))
                cam.up = tuple(map(float, up))
                cam.parallel_projection = True
                cam.parallel_scale = parallel_scale
                arr = np.asarray(
                    pl.screenshot(return_img=True, transparent_background=False)
                )
            finally:
                pl.close()

            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG", optimize=True)
            pngs.append(buf.getvalue())
    return pngs
