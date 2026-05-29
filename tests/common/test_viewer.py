"""Unit tests for the headless STEP renderer."""
from pathlib import Path

import pytest

from cadgenbench.common.viewer import (
    CAMERA_PRESETS,
    DEFAULT_VIEWS,
    RenderedImage,
    render_step,
    render_steps,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
BOX_STEP = FIXTURES_DIR / "box.step"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True, scope="module")
def _ensure_fixture() -> None:
    """Generate the box STEP fixture if it doesn't exist."""
    if not BOX_STEP.exists():
        from tests.fixtures.generate_fixtures import generate_box

        generate_box()
    assert BOX_STEP.exists(), f"Fixture missing: {BOX_STEP}"


class TestRenderStep:
    """Single-file rendering (in-process via VTK / PyVista)."""

    def test_default_views_returns_four_images(self) -> None:
        images = render_step(BOX_STEP)
        assert len(images) == len(DEFAULT_VIEWS)

    def test_image_names_match_views(self) -> None:
        images = render_step(BOX_STEP)
        names = [img.name for img in images]
        assert names == list(DEFAULT_VIEWS)

    def test_images_are_non_empty_png(self) -> None:
        images = render_step(BOX_STEP)
        for img in images:
            assert isinstance(img.data, bytes)
            assert len(img.data) > 100, f"Image '{img.name}' suspiciously small"
            assert img.data[:8] == PNG_MAGIC, f"Image '{img.name}' not a valid PNG"

    def test_custom_single_view(self) -> None:
        images = render_step(BOX_STEP, views=["top"])
        assert len(images) == 1
        assert images[0].name == "top"

    def test_image_dimensions_stored(self) -> None:
        w, h = 800, 600
        images = render_step(BOX_STEP, views=["iso"], width=w, height=h)
        assert images[0].width == w
        assert images[0].height == h


class TestRenderSteps:
    """Batch rendering across multiple files."""

    def test_batch_single_file(self) -> None:
        results = render_steps([BOX_STEP], views=["iso"])
        assert "box" in results
        assert len(results["box"]) == 1

    def test_batch_same_file_twice(self) -> None:
        """Two copies of the same STEP to verify multi-file naming."""
        import shutil

        copy = FIXTURES_DIR / "box_copy.step"
        shutil.copy2(BOX_STEP, copy)
        try:
            results = render_steps([BOX_STEP, copy], views=["iso"])
            assert "box" in results
            assert "box_copy" in results
            assert len(results["box"]) == 1
            assert len(results["box_copy"]) == 1
            for imgs in results.values():
                assert imgs[0].data[:8] == PNG_MAGIC
        finally:
            copy.unlink(missing_ok=True)


class TestValidation:
    """Input validation, no renderer needed."""

    def test_missing_step_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            render_step("/nonexistent/path/fake.step")

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown camera preset"):
            render_step(BOX_STEP, views=["diagonal"])

    def test_empty_paths_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            render_steps([])
