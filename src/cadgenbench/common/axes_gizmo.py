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

"""Inline-SVG orientation gizmo for the static iso/front/top/right renders.

The benchmark's prompts (especially the editing tasks) constantly refer to
world directions -- "the far end of the part in the +Y direction", "extend in
the +Z direction". A reader looking at a shaded render has no way to know which
way X/Y/Z point, so this draws a small, exact axis indicator to overlay on each
view tile.

Because every canonical render uses a *parallel* (orthographic) camera with a
fixed preset orientation (see :mod:`cadgenbench.common.camera_presets`), the
screen-space projection of the world axes is **the same for every fixture** --
it depends only on the view name, not the part. So the gizmo is computed once
per view from the shared camera presets (no per-part rendering, no re-rendering
of the cached PNGs) and the report / tasks page just drop the returned SVG into
the corner of each tile.

The look (chosen against a high aesthetic bar): a muted X=red / Y=green /
Z=blue palette, thin lines with a slim arrowhead marking the *positive*
direction, and the axis letter set just past the tip with a white halo (SVG
``paint-order`` stroke) so it stays legible over the part without an opaque box.
An axis pointing straight into / out of the screen collapses to a small dot at
the origin (filled = toward the viewer, hollow = away) with its label beside it.
"""
from __future__ import annotations

import math

from cadgenbench.common.camera_presets import PRESETS

# Muted X/Y/Z palette: the universal red/green/blue convention, desaturated so
# the gizmo reads as a quiet annotation rather than a garish primary-color
# overlay. Shared by the report and the tasks page so both draw identically.
AXIS_COLORS: dict[str, str] = {"X": "#d24b46", "Y": "#3f9d63", "Z": "#3f72c4"}

# Unit world axes, drawn in this order.
_AXES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("X", (1.0, 0.0, 0.0)),
    ("Y", (0.0, 1.0, 0.0)),
    ("Z", (0.0, 0.0, 1.0)),
)

# Internal SVG coordinate space (a square viewBox); display size is set by the
# width/height attributes / CSS so the gizmo scales crisply.
_BOX = 100.0
_CENTER = _BOX / 2.0
_LEN = 30.0               # axis line length from the origin
_HEAD = 9.0              # arrowhead length
_HEAD_HALF_ANGLE = 0.34  # radians, half the arrowhead spread
_LABEL_GAP = 14.0        # label distance past the tip
_INPLANE_EPS = 0.16      # below this, an axis is treated as into/out of screen


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    m = math.sqrt(_dot(a, a)) or 1e-12
    return (a[0] / m, a[1] / m, a[2] / m)


def _screen_axes(view: str):
    """Per-axis screen projection for *view*.

    Yields ``(label, color, sx, sy, depth, in_plane)`` where ``(sx, sy)`` is the
    axis direction in screen space (``+sx`` right, ``+sy`` up), ``depth`` is its
    component along the view direction (``>0`` points into the screen, ``<0``
    toward the viewer) and ``in_plane`` is the in-screen magnitude.
    """
    preset = PRESETS[view]
    # eye = target + direction * d, so the view direction (target -> eye flipped)
    # is -direction; right/up follow the standard look-at basis.
    view_dir = _norm(tuple(-c for c in preset.direction))
    right = _norm(_cross(view_dir, preset.up))
    up_cam = _norm(_cross(right, view_dir))
    for label, axis in _AXES:
        sx = _dot(axis, right)
        sy = _dot(axis, up_cam)
        yield label, AXIS_COLORS[label], sx, sy, _dot(axis, view_dir), math.hypot(sx, sy)


def _halo_text(x: float, y: float, label: str, color: str, fs: float = 15.0) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{fs:.0f}" '
        f'font-weight="600" text-anchor="middle" dominant-baseline="central" '
        f'paint-order="stroke" stroke="#ffffff" stroke-width="3" '
        f'stroke-linejoin="round" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,sans-serif">'
        f"{label}</text>"
    )


def gizmo_svg(view: str, *, size: int = 72) -> str:
    """Return an inline ``<svg>`` orientation gizmo for *view*, or ``""``.

    *view* is a camera-preset name (``iso`` / ``front`` / ``top`` / ``right`` /
    ``bottom`` / ...); an unknown name returns an empty string so callers can
    drop the result in unconditionally. *size* is the rendered pixel size of the
    square gizmo.
    """
    if view not in PRESETS:
        return ""
    cx = cy = _CENTER
    # Draw far-into-screen axes first so nearer ones layer on top.
    axes = sorted(_screen_axes(view), key=lambda a: a[4], reverse=True)
    parts = [
        f'<svg viewBox="0 0 {_BOX:.0f} {_BOX:.0f}" width="{size}" height="{size}" '
        f'aria-hidden="true">'
    ]
    for label, color, sx, sy, depth, in_plane in axes:
        if in_plane < _INPLANE_EPS:
            # Into / out of the screen: a small dot at the origin (filled =
            # toward the viewer, hollow = away) with the label beside it.
            fill = color if depth < 0 else "#ffffff"
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.4" fill="{fill}" '
                f'stroke="{color}" stroke-width="1.8"/>'
            )
            parts.append(_halo_text(cx + 14, cy - 12, label, color, 13))
            continue
        tx, ty = cx + sx * _LEN, cy - sy * _LEN
        bx, by = cx + sx * (_LEN - _HEAD * 0.8), cy - sy * (_LEN - _HEAD * 0.8)
        parts.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="{color}" stroke-width="2.4" stroke-linecap="round"/>'
        )
        ang = math.atan2(ty - cy, tx - cx)
        p1 = (tx + _HEAD * math.cos(ang + math.pi - _HEAD_HALF_ANGLE),
              ty + _HEAD * math.sin(ang + math.pi - _HEAD_HALF_ANGLE))
        p2 = (tx + _HEAD * math.cos(ang + math.pi + _HEAD_HALF_ANGLE),
              ty + _HEAD * math.sin(ang + math.pi + _HEAD_HALF_ANGLE))
        parts.append(
            f'<path d="M{tx:.1f},{ty:.1f} L{p1[0]:.1f},{p1[1]:.1f} '
            f'L{p2[0]:.1f},{p2[1]:.1f} Z" fill="{color}"/>'
        )
        parts.append(
            _halo_text(cx + sx * (_LEN + _LABEL_GAP),
                       cy - sy * (_LEN + _LABEL_GAP), label, color)
        )
    parts.append("</svg>")
    return "".join(parts)
