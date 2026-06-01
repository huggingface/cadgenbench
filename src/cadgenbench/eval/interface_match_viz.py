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

"""Visualisation helpers for the interface-match metric.

Sibling to :mod:`cadgenbench.eval.interface_match`. Dev/debug aid
that overlays a part (GT or candidate) with the metric's sub-volumes:

- the part (GT or candidate) in solid blue,
- each ``__KOR`` (keep-out region) sub-volume in translucent red
  ("candidate should be empty here"),
- each ``__KIR`` (keep-in region) sub-volume in translucent green
  ("candidate should be solid here"),
- the per-sub-volume *disagreement* volume in opaque yellow.

"Disagreement" = the part of R the candidate gets wrong:

- For ``KOR``  R: ``R ∩ candidate_solid``  (candidate has material it shouldn't).
- For ``KIR``  R: ``R \\ candidate_solid`` (candidate is missing material it should have).

Both fire the same yellow highlight, so the eye flags any failure
without thinking about fit_type. For ``correct.step`` the disagreement
volume is ≈ 0 and no yellow appears.

Rendering goes through :func:`cadgenbench.common.viewer.render_mesh_overlay`,
in-process via VTK; safe to call from any host. All geometry (the
disagreement region included) is computed with the ``manifold3d`` mesh
kernel: no OCCT Booleans on this path.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from cadgenbench.common.mesh import Mesh
from cadgenbench.eval.interface_match import (
    DEFAULT_DISAGREEMENT_EPSILON,
    SubVolume,
    discover_sub_volumes,
)
from cadgenbench.common.viewer import RenderedImage, render_mesh_overlay

logger = logging.getLogger(__name__)


DEFAULT_GRID_VIEWS: tuple[str, ...] = ("iso", "top", "left", "rear")

# Colour scheme (rgba).
PART_COLOR: tuple[float, float, float, float] = (0.18, 0.45, 0.86, 1.00)   # solid blue
KOR_COLOR: tuple[float, float, float, float] = (0.90, 0.30, 0.30, 0.40)   # translucent red
KIR_COLOR: tuple[float, float, float, float] = (0.20, 0.70, 0.30, 0.40)   # translucent green
DISAGREEMENT_COLOR: tuple[float, float, float, float] = (1.00, 0.85, 0.00, 1.00)  # opaque yellow


def _color_for(sv: SubVolume) -> tuple[float, float, float, float]:
    return KOR_COLOR if sv.fit_type == "KOR" else KIR_COLOR


def render_part_with_subvolumes(
    part_step: str | Path,
    sub_volumes: list[SubVolume],
    *,
    views: tuple[str, ...] = ("iso",),
    width: int = 1024,
    height: int = 768,
) -> list[RenderedImage]:
    """Render *part_step* with each sub-volume coloured by fit_type and the
    per-sub-volume disagreement highlighted.

    See module docstring for the colour scheme.

    Tessellates the part and each sub-volume exactly once (via
    :class:`~cadgenbench.common.artifacts.StepArtifacts`, at the part's
    deflection so every shape shares one scale), reuses those meshes for
    both the ``manifold3d`` disagreement Boolean and the render, and
    draws everything through :func:`render_mesh_overlay`.
    """
    from cadgenbench.common.artifacts import StepArtifacts

    part_step = Path(part_step).resolve()
    part_artifacts = StepArtifacts(part_step)
    deflection = part_artifacts.deflection()
    part_manifold = part_artifacts.manifold(deflection)

    meshes: list[Mesh] = [part_artifacts.mesh(deflection)]
    colors: list[tuple[float, float, float, float]] = [PART_COLOR]
    total_disagreement_vol = 0.0
    n_disagreements = 0
    for sv in sub_volumes:
        sv_artifacts = StepArtifacts(sv.path)
        meshes.append(sv_artifacts.mesh(deflection))
        colors.append(_color_for(sv))

        vol, disagreement_mesh = _disagreement_mesh(
            part_manifold, sv_artifacts.manifold(deflection), sv.fit_type,
        )
        total_disagreement_vol += vol
        if disagreement_mesh is not None:
            meshes.append(disagreement_mesh)
            colors.append(DISAGREEMENT_COLOR)
            n_disagreements += 1

    if n_disagreements:
        logger.info(
            "Total disagreement volume %.2f mm^3 across %d sub-volume(s)",
            total_disagreement_vol, len(sub_volumes),
        )

    return render_mesh_overlay(
        meshes,
        colors=colors,
        views=views,
        width=width,
        height=height,
    )


def render_fixture(
    fixture_dir: str | Path,
    *,
    candidate: str | None = None,
    views: tuple[str, ...] = ("iso",),
    width: int = 1024,
    height: int = 768,
) -> list[RenderedImage]:
    """Render the GT (or a named candidate) overlaid with every sub-volume.

    Args:
        fixture_dir: Directory matching the jig_metric layout. Must
            contain ``gt.step`` and one or more
            ``jig_<context_id>__<index>__<fit_type>.step`` files.
        candidate: If given, render
            ``candidates/<candidate>.step`` instead of the GT.
        views: Camera presets.
    """
    fixture_dir = Path(fixture_dir).resolve()
    sub_volumes = discover_sub_volumes(fixture_dir)
    if not sub_volumes:
        raise FileNotFoundError(
            f"No jig_<id>__<index>__<fit>.step files found in {fixture_dir}"
        )

    if candidate is None:
        part = fixture_dir / "gt.step"
    else:
        part = fixture_dir / "candidates" / f"{candidate}.step"
    if not part.exists():
        raise FileNotFoundError(f"Part STEP missing: {part}")

    return render_part_with_subvolumes(
        part, sub_volumes, views=views, width=width, height=height,
    )


# ---------------------------------------------------------------------------
# Grid composite
# ---------------------------------------------------------------------------


def composite_grid(
    images: list[RenderedImage],
    *,
    cols: int = 2,
    label_height: int = 32,
    label_font_size: int = 18,
) -> bytes:
    """Lay out *images* in a labelled grid and return PNG bytes."""
    if not images:
        raise ValueError("composite_grid: images must not be empty")

    from PIL import Image, ImageDraw, ImageFont

    tile_w = images[0].width
    tile_h = images[0].height
    rows = (len(images) + cols - 1) // cols

    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(255, 255, 255))

    try:
        font = ImageFont.truetype("Helvetica", label_font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        tile = Image.open(io.BytesIO(img.data)).convert("RGB")
        if tile.size != (tile_w, tile_h):
            tile = tile.resize((tile_w, tile_h))
        strip = Image.new("RGB", (tile_w, label_height), color=(255, 255, 255))
        tile.paste(strip, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (10, max(0, (label_height - label_font_size) // 2)),
            img.name,
            fill=(40, 40, 40),
            font=font,
        )
        grid.paste(tile, (c * tile_w, r * tile_h))

    buf = io.BytesIO()
    grid.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_test_overview(
    test_dir: str | Path,
    *,
    views: tuple[str, ...] = DEFAULT_GRID_VIEWS,
    tile_width: int = 480,
    tile_height: int = 360,
) -> bytes:
    """Render every candidate in a test case as one labelled overview PNG.

    Rows = candidates (correct + brokens). Columns = views. Left label
    column shows the candidate stem, expected PASS/FAIL, and the
    failure-mode hint extracted from the filename. Title row at the top
    shows ``test_dir.name``.
    """
    from PIL import Image, ImageDraw, ImageFont

    test_dir = Path(test_dir).resolve()
    candidates_dir = test_dir / "candidates"
    if not candidates_dir.exists():
        raise FileNotFoundError(f"candidates/ missing in {test_dir}")

    correct = candidates_dir / "correct.step"
    if not correct.exists():
        raise FileNotFoundError(f"correct.step missing in {candidates_dir}")
    brokens = sorted(p for p in candidates_dir.glob("broken_*.step"))
    candidates: list[Path] = [correct, *brokens]

    sub_volumes = discover_sub_volumes(test_dir)
    if not sub_volumes:
        raise FileNotFoundError(
            f"No jig_<id>__<index>__<fit>.step files in {test_dir}"
        )

    try:
        font = ImageFont.truetype("Helvetica", 16)
        font_title = ImageFont.truetype("Helvetica", 22)
        font_small = ImageFont.truetype("Helvetica", 13)
    except OSError:
        font = ImageFont.load_default()
        font_title = font
        font_small = font

    label_col_w = 280
    title_h = 40
    header_h = 28
    n_views = len(views)
    n_rows = len(candidates)

    img_w = label_col_w + n_views * tile_width
    img_h = title_h + header_h + n_rows * tile_height
    overview = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(overview)

    draw.rectangle([(0, 0), (img_w, title_h)], fill=(38, 50, 64))
    draw.text((16, 8), test_dir.name, fill=(255, 255, 255), font=font_title)

    for col, view in enumerate(views):
        x0 = label_col_w + col * tile_width
        draw.rectangle(
            [(x0, title_h), (x0 + tile_width, title_h + header_h)],
            fill=(230, 234, 240),
        )
        draw.text((x0 + 10, title_h + 6), view, fill=(40, 40, 40), font=font)

    for row, cand in enumerate(candidates):
        y0 = title_h + header_h + row * tile_height

        stem = cand.stem
        is_correct = stem == "correct"
        verdict = "PASS" if is_correct else "FAIL"
        verdict_color = (40, 140, 60) if is_correct else (180, 50, 50)

        draw.rectangle(
            [(0, y0), (label_col_w, y0 + tile_height)],
            fill=(248, 249, 251),
            outline=(210, 215, 220),
        )
        draw.text((16, y0 + 14), stem, fill=(20, 20, 20), font=font)
        draw.text(
            (16, y0 + 14 + 22), f"expected: {verdict}",
            fill=verdict_color, font=font,
        )
        if not is_correct:
            parts = stem.split("_", 2)
            hint = parts[2] if len(parts) >= 3 else stem
            draw.text(
                (16, y0 + 14 + 22 + 24), hint.replace("_", " "),
                fill=(80, 80, 80), font=font_small,
            )

        images = render_part_with_subvolumes(
            cand, sub_volumes,
            views=views, width=tile_width, height=tile_height,
        )
        for col, img in enumerate(images):
            x0 = label_col_w + col * tile_width
            tile = Image.open(io.BytesIO(img.data)).convert("RGB")
            if tile.size != (tile_width, tile_height):
                tile = tile.resize((tile_width, tile_height))
            overview.paste(tile, (x0, y0))
            draw.rectangle(
                [(x0, y0), (x0 + tile_width, y0 + tile_height)],
                outline=(210, 215, 220),
            )

    buf = io.BytesIO()
    overview.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _disagreement_mesh(
    part_manifold,
    sub_volume_manifold,
    fit_type: str,
) -> tuple[float, Mesh | None]:
    """Disagreement region between a candidate and one sub-volume.

    The disagreement is the part of ``R`` the candidate gets wrong:

    - ``KOR`` (keep-out): ``R ∩ candidate`` (candidate has material it shouldn't).
    - ``KIR`` (keep-in):  ``R \\ candidate`` (candidate is missing material).

    Computed with the ``manifold3d`` mesh kernel (no OCCT Booleans).
    Returns ``(volume, mesh)``; ``mesh`` is ``None`` when the volume is
    below :data:`cadgenbench.eval.interface_match.DEFAULT_DISAGREEMENT_EPSILON`,
    i.e. nothing visually meaningful to highlight.
    """
    from cadgenbench.eval.booleans import (
        intersect,
        manifold_to_mesh,
        manifold_volume,
        subtract,
    )

    if fit_type == "KOR":
        result = intersect(sub_volume_manifold, part_manifold)
    elif fit_type == "KIR":
        result = subtract(sub_volume_manifold, part_manifold)
    else:  # pragma: no cover - construction-time guarantee
        raise ValueError(f"Unknown fit_type {fit_type!r}")

    volume = manifold_volume(result)
    if volume < DEFAULT_DISAGREEMENT_EPSILON:
        return volume, None
    return volume, manifold_to_mesh(result)
