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
shaded triangles + dihedral-angle feature edges, and returns PNG bytes
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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv
from PIL import Image

from cadgenbench.common.camera_presets import (
    CAMERA_PRESETS,
    DEFAULT_VIEWS,
    camera_placement,
    validate_views,
)
from cadgenbench.common.mesh import (
    Mesh,
    deflection_for_bbox,
    tessellate_step,
)

# Re-exported for callers that want the canonical preset / default-view sets.
__all__ = [
    "CAMERA_PRESETS",
    "DEFAULT_VIEWS",
    "OVERLAY_PALETTE",
    "RenderedImage",
    "render_mesh",
    "render_overlay",
    "render_step",
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
EDGE_RGB: tuple[float, float, float] = (0.20, 0.20, 0.20)
BACKGROUND_RGB: tuple[float, float, float] = (1.0, 1.0, 1.0)

# Dihedral threshold above which a mesh edge counts as a feature edge and
# gets drawn. 30 deg matches :mod:`cadgenbench.eval.feature_edges` (tau_sharp).
FEATURE_EDGE_ANGLE_DEG: float = 30.0
EDGE_LINE_WIDTH: float = 1.5

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
    edges = body.extract_feature_edges(
        feature_angle=FEATURE_EDGE_ANGLE_DEG,
        boundary_edges=True,
        feature_edges=True,
        non_manifold_edges=False,
        manifold_edges=False,
    )

    out: list[RenderedImage] = []
    for view in views:
        png = _render_scene(
            shapes=[(body, body_rgb, 1.0)],
            edge_meshes=[edges],
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            view=view,
            width=width,
            height=height,
        )
        out.append(RenderedImage(name=view, data=png, width=width, height=height))
    return out


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
    if views is None:
        views = ("iso",)
    validate_views(views)

    shapes: list[tuple[pv.PolyData, tuple[float, float, float], float]] = []
    edge_meshes: list[pv.PolyData] = []
    bbox_min = np.array([np.inf] * 3, dtype=np.float64)
    bbox_max = np.array([-np.inf] * 3, dtype=np.float64)
    for path, rgba in zip(paths, colors):
        mesh = _tessellate(path)
        body = _mesh_to_polydata(mesh)
        shapes.append((body, (rgba[0], rgba[1], rgba[2]), float(rgba[3])))
        edge_meshes.append(
            body.extract_feature_edges(
                feature_angle=FEATURE_EDGE_ANGLE_DEG,
                boundary_edges=True,
                feature_edges=True,
                non_manifold_edges=False,
                manifold_edges=False,
            )
        )
        bbox_min = np.minimum(bbox_min, mesh.vertices.min(axis=0))
        bbox_max = np.maximum(bbox_max, mesh.vertices.max(axis=0))

    out: list[RenderedImage] = []
    for view in views:
        png = _render_scene(
            shapes=shapes,
            edge_meshes=edge_meshes,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            view=view,
            width=width,
            height=height,
        )
        out.append(RenderedImage(name=view, data=png, width=width, height=height))
    return out


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


_RENDER_LOCK = threading.Lock()


def _render_scene(
    *,
    shapes: list[tuple[pv.PolyData, tuple[float, float, float], float]],
    edge_meshes: list[pv.PolyData],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    view: str,
    width: int,
    height: int,
) -> bytes:
    """Render one frame: shaded triangles + feature-edge overlay -> PNG bytes."""
    # VTK/OpenGL is not thread-safe and there is a single GL context per
    # process. Baseline fixtures and compare-llms models both run as threads,
    # so without serialisation concurrent renders contend on (or crash) the
    # shared context. A process-global lock makes every render one-at-a-time;
    # each render is short, so this is correct rather than a real bottleneck.
    # (True render parallelism needs process/GPU isolation, e.g. HF Jobs.)
    with _RENDER_LOCK:
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
            for edges in edge_meshes:
                if edges.n_points > 0:
                    pl.add_mesh(edges, color=EDGE_RGB, line_width=EDGE_LINE_WIDTH)

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

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
