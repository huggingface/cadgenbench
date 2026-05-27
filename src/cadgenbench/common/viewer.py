"""Headless STEP renderer using tcv-screenshots.

Renders STEP files from multiple camera angles and returns PNG images.
Uses three-cad-viewer via headless Chromium (Playwright), same renderer
as the ocp-vscode CAD viewer.

Chromium is launched once per ``render_steps()`` call regardless of how
many STEP files are rendered.  The convenience wrapper ``render_step()``
handles the single-file case.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

CAMERA_PRESETS: frozenset[str] = frozenset(
    ("front", "rear", "left", "right", "top", "bottom", "iso")
)

DEFAULT_VIEWS: tuple[str, ...] = ("iso", "front", "top", "right")


@dataclass(frozen=True)
class RenderedImage:
    """One rendered view of a STEP file."""

    name: str
    data: bytes
    width: int
    height: int


def _validate_views(views: Sequence[str]) -> None:
    unknown = set(views) - CAMERA_PRESETS
    if unknown:
        raise ValueError(
            f"Unknown camera preset(s): {unknown}. "
            f"Valid: {sorted(CAMERA_PRESETS)}"
        )


def _build_batch_script(
    step_paths: Sequence[Path],
    views: Sequence[str],
    width: int,
    height: int,
) -> str:
    """Generate a single Python script that renders every STEP file.

    All models end up in one ``main()`` so tcv_screenshots launches
    Chromium exactly once for the entire batch.  Output PNGs are named
    ``{stem}__{view}.png`` to avoid collisions.
    """
    import_lines = []
    save_lines = []

    for idx, path in enumerate(step_paths):
        var = f"shape{idx}"
        stem = path.stem
        import_lines.append(
            f"{var} = import_step({str(path)!r})\n"
            f"{var}.color = Color(0.68, 0.72, 0.76)"
        )
        for view in views:
            png_name = f"{stem}__{view}" if len(step_paths) > 1 else view
            save_lines.append(
                f'    save_model({var}, "{png_name}", '
                f'{{"cadWidth": {width}, "height": {height}, '
                f'"reset_camera": "{view}", '
                f'"edgeColor": 0x333333}})'
            )

    imports = "\n".join(import_lines)
    saves = "\n".join(save_lines)

    return (
        f"from build123d import import_step, Color\n"
        f"\n"
        f"{imports}\n"
        f"\n"
        f"def main():\n"
        f"    from tcv_screenshots import save_model, get_saved_models\n"
        f"{saves}\n"
        f"    return get_saved_models()\n"
    )


def _build_overlay_script(
    step_paths: Sequence[Path],
    colors: Sequence[tuple[float, float, float, float]],
    views: Sequence[str],
    width: int,
    height: int,
) -> str:
    """Generate a script that renders all STEPs as one composite scene per view.

    Each STEP is loaded, given its rgba color, and grouped into one
    ``Compound``; the compound is rendered once per requested view.
    """
    import_lines = []
    for idx, (path, rgba) in enumerate(zip(step_paths, colors)):
        r, g, b, a = rgba
        import_lines.append(
            f"shape{idx} = import_step({str(path)!r})\n"
            f"shape{idx}.color = Color({r}, {g}, {b}, {a})"
        )
    compound_args = ", ".join(f"shape{i}" for i in range(len(step_paths)))
    # Notes on the viewer config:
    #   - Skip ``"transparent": True``; the viewer auto-enables transparency
    #     mode as soon as any shape has alpha < 1.0 (see three-cad-viewer.js,
    #     ``if (alpha < 1.0) this.transparent = true;``).
    #   - Force ``defaultOpacity: 1.0``. The viewer computes per-shape opacity
    #     as ``defaultOpacity * alpha`` once it is in transparent mode, and
    #     defaults to 0.5 -- which would also wash out our alpha=1.0 shapes
    #     and turn a "solid" GT translucent.
    save_lines = [
        f'    save_model(scene, "{view}", '
        f'{{"cadWidth": {width}, "height": {height}, '
        f'"reset_camera": "{view}", "theme": "light", '
        f'"edgeColor": 0x333333, "defaultOpacity": 1.0}})'
        for view in views
    ]

    imports = "\n".join(import_lines)
    saves = "\n".join(save_lines)
    return (
        f"from build123d import import_step, Color, Compound\n"
        f"\n"
        f"{imports}\n"
        f"\n"
        f"scene = Compound(children=[{compound_args}])\n"
        f"\n"
        f"def main():\n"
        f"    from tcv_screenshots import save_model, get_saved_models\n"
        f"{saves}\n"
        f"    return get_saved_models()\n"
    )


def render_overlay(
    step_paths: Sequence[str | Path],
    *,
    colors: Sequence[tuple[float, float, float, float]] | None = None,
    views: Sequence[str] | None = None,
    width: int = 1024,
    height: int = 768,
    timeout: int = 180,
) -> list[RenderedImage]:
    """Render several STEPs as one composite scene per view.

    Each STEP is given a per-shape rgba color so different parts of the
    scene are visually distinguishable. Used for metric-development
    visualisation (GT + jig overlays, alignment debugging, etc.).

    Args:
        step_paths: Paths to .step / .stp files. Rendered in the order
            given; later shapes are drawn on top of earlier ones.
        colors: Per-shape rgba colors (each component in [0, 1]). If
            omitted, falls back to a small built-in palette
            (:data:`OVERLAY_PALETTE`).
        views: Camera preset names. Defaults to ``("iso",)`` -- a single
            iso view is typically enough for an overlay.
        width: Image width in pixels.
        height: Image height in pixels.
        timeout: Max seconds for the render subprocess.

    Returns:
        One ``RenderedImage`` per requested view, in input order.

    Raises:
        FileNotFoundError: If any STEP path does not exist.
        ValueError: If ``step_paths`` is empty, ``colors`` length mismatches,
            or an unknown camera preset is requested.
        RuntimeError: If the renderer subprocess fails.
    """
    if not step_paths:
        raise ValueError("step_paths must not be empty")

    resolved: list[Path] = []
    for p in step_paths:
        rp = Path(p).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"STEP file not found: {rp}")
        resolved.append(rp)

    if colors is None:
        colors = [OVERLAY_PALETTE[i % len(OVERLAY_PALETTE)] for i in range(len(resolved))]
    elif len(colors) != len(resolved):
        raise ValueError(
            f"colors length ({len(colors)}) must match step_paths length ({len(resolved)})"
        )

    if views is None:
        views = ("iso",)
    _validate_views(views)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        script = tmp / "_overlay.py"
        script.write_text(
            _build_overlay_script(resolved, colors, views, width, height)
        )
        out_dir = tmp / "out"
        out_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable, "-m", "tcv_screenshots",
                "-f", str(script),
                "-o", str(out_dir),
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"tcv_screenshots failed (exit {result.returncode}):\n{result.stderr}"
            )

        images: list[RenderedImage] = []
        for view in views:
            png_path = out_dir / f"{view}.png"
            if not png_path.exists():
                available = [p.name for p in out_dir.glob("*.png")]
                raise RuntimeError(
                    f"Expected output '{view}.png' not found. Available: {available}"
                )
            data = png_path.read_bytes()
            if not data:
                raise RuntimeError(f"Renderer produced empty image for view '{view}'")
            images.append(
                RenderedImage(name=view, data=data, width=width, height=height)
            )
    return images


# Default rgba palette used by render_overlay when colors are not specified.
# Kept short and distinct; interface_match_viz overrides for its own scheme.
OVERLAY_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.18, 0.45, 0.86, 1.00),   # solid blue
    (0.90, 0.30, 0.30, 0.40),   # translucent red
    (1.00, 0.85, 0.00, 1.00),   # solid yellow (good for highlights)
    (0.20, 0.70, 0.30, 0.60),   # translucent green
)


def render_steps(
    step_paths: Sequence[str | Path],
    views: Sequence[str] | None = None,
    width: int = 1024,
    height: int = 768,
    timeout: int = 300,
) -> dict[str, list[RenderedImage]]:
    """Render multiple STEP files in a single Chromium session.

    Args:
        step_paths: Paths to .step / .stp files.
        views: Camera preset names.  Valid values: front, rear, left, right,
               top, bottom, iso.  Defaults to iso/front/top/right.
        width: Image width in pixels.
        height: Image height in pixels.
        timeout: Max seconds for the entire batch.

    Returns:
        Dict mapping each STEP file's stem name to its list of RenderedImage.

    Raises:
        FileNotFoundError: If any STEP file does not exist.
        ValueError: If an unknown camera preset is requested or paths are empty.
        RuntimeError: If the renderer subprocess fails.
    """
    if not step_paths:
        raise ValueError("step_paths must not be empty")

    resolved: list[Path] = []
    for p in step_paths:
        rp = Path(p).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"STEP file not found: {rp}")
        resolved.append(rp)

    if views is None:
        views = DEFAULT_VIEWS
    _validate_views(views)

    multi = len(resolved) > 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        script = tmp / "_render.py"
        script.write_text(
            _build_batch_script(resolved, views, width, height)
        )

        out_dir = tmp / "out"
        out_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tcv_screenshots",
                "-f",
                str(script),
                "-o",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"tcv_screenshots failed (exit {result.returncode}):\n"
                f"{result.stderr}"
            )

        results: dict[str, list[RenderedImage]] = {}
        for path in resolved:
            stem = path.stem
            images: list[RenderedImage] = []
            for view in views:
                png_name = f"{stem}__{view}" if multi else view
                png_path = out_dir / f"{png_name}.png"
                if not png_path.exists():
                    available = [p.name for p in out_dir.glob("*.png")]
                    raise RuntimeError(
                        f"Expected output '{png_name}.png' not found. "
                        f"Available: {available}"
                    )
                data = png_path.read_bytes()
                if not data:
                    raise RuntimeError(
                        f"Renderer produced empty image for '{png_name}'"
                    )
                images.append(
                    RenderedImage(
                        name=view, data=data, width=width, height=height
                    )
                )
            results[stem] = images

    return results


def render_step(
    step_path: str | Path,
    views: Sequence[str] | None = None,
    width: int = 1024,
    height: int = 768,
    timeout: int = 120,
) -> list[RenderedImage]:
    """Render a single STEP file from multiple angles.

    Convenience wrapper around :func:`render_steps` for the single-file case.
    See that function for full documentation.
    """
    step_path = Path(step_path)
    results = render_steps(
        [step_path], views=views, width=width, height=height, timeout=timeout
    )
    return results[step_path.stem]
