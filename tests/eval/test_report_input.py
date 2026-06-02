"""Report input-column resolution for editing vs generation fixtures.

The per-submission report (``eval.report.single_run``) renders an
Input column. Editing fixtures ship the starting solid as
``input.step``; the raw STEP must never be embedded as an ``<img>``
(it renders broken). Instead the report surfaces the shape's
pre-rendered canonical views from ``inputs/<fixture>/renders/`` —
mirroring the GT column. Generation fixtures keep embedding their
drawing ``input.png`` directly.

These tests exercise :func:`_load_description`'s classification only,
so they need no renderer / GL context.
"""
from __future__ import annotations

from pathlib import Path

from cadgenbench.eval.report.single_run import VIEWS, _load_description


def _make_fixture(
    root: Path,
    name: str,
    *,
    description_yaml: str,
    input_files: dict[str, bytes],
    render_views: list[str] | None = None,
) -> Path:
    """Build a ``gt/<name>`` + ``inputs/<name>`` pair, return the gt dir.

    ``_load_description`` resolves the inputs dir as
    ``gt_dir.parent.parent / "inputs" / gt_dir.name``.
    """
    gt_dir = root / "gt" / name
    inputs_dir = root / "inputs" / name
    gt_dir.mkdir(parents=True)
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "description.yaml").write_text(description_yaml)
    for fname, data in input_files.items():
        (inputs_dir / fname).write_bytes(data)
    if render_views:
        renders = inputs_dir / "renders"
        renders.mkdir()
        for v in render_views:
            (renders / f"{v}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return gt_dir


def test_editing_input_uses_rendered_views_not_raw_step(tmp_path):
    """Editing fixture: input.step is replaced by its shipped render PNGs."""
    gt_dir = _make_fixture(
        tmp_path,
        "example_edit",
        description_yaml=(
            "description: >\n  Double the rib thickness.\n\n"
            "task_type: editing\n"
            "input_files:\n  - input.step\n\n"
            "input_type: text+step\n"
        ),
        input_files={"input.step": b"ISO-10303-21; dummy"},
        render_views=["iso", "front", "top", "right"],
    )
    text, image_files, shape_pngs, wants_shape = _load_description(gt_dir)
    assert text.strip() == "Double the rib thickness."
    # The raw STEP is never offered as an embeddable image.
    assert image_files == []
    assert wants_shape is True
    # Render PNGs are surfaced, ordered by the report's VIEWS list.
    assert [p.stem for p in shape_pngs] == [
        v for v in VIEWS if v in {"iso", "front", "top", "right"}
    ]
    assert all(p.name.endswith(".png") for p in shape_pngs)


def test_editing_input_without_renders_flags_wants_shape(tmp_path):
    """Editing fixture missing render PNGs: no images, but wants_shape stays set.

    Lets the caller render an explicit 'no input renders' note instead
    of a silent blank (and never a broken <img> of the STEP).
    """
    gt_dir = _make_fixture(
        tmp_path,
        "example_edit_norender",
        description_yaml=(
            "description: >\n  Edit it.\n\n"
            "task_type: editing\n"
            "input_files:\n  - input.step\n\n"
        ),
        input_files={"input.step": b"ISO-10303-21; dummy"},
        render_views=None,
    )
    _text, image_files, shape_pngs, wants_shape = _load_description(gt_dir)
    assert image_files == []
    assert shape_pngs == []
    assert wants_shape is True


def test_generation_input_embeds_drawing_png(tmp_path):
    """Generation fixture: the drawing input.png is embedded directly."""
    gt_dir = _make_fixture(
        tmp_path,
        "example_gen",
        description_yaml=(
            "description: >\n  Reproduce the drawing.\n\n"
            "input_files:\n  - input.png\n\n"
            "input_type: text+image\n"
        ),
        input_files={"input.png": b"\x89PNG\r\n\x1a\n"},
    )
    _text, image_files, shape_pngs, wants_shape = _load_description(gt_dir)
    assert [p.name for p in image_files] == ["input.png"]
    assert shape_pngs == []
    assert wants_shape is False
