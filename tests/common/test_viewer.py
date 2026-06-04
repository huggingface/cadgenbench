"""Unit tests for the headless STEP renderer."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from cadgenbench.common.mesh import Mesh
from cadgenbench.common.viewer import (
    CAMERA_PRESETS,
    DEFAULT_VIEWS,
    RenderedImage,
    mesh_diff,
    render_mesh_diff,
    render_mesh_diff_turntable_webp,
    render_step,
    render_steps,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
BOX_STEP = FIXTURES_DIR / "box.step"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_MAGIC_RIFF = b"RIFF"
WEBP_MAGIC_WEBP = b"WEBP"


def _box_mesh(extents=(10.0, 10.0, 10.0), offset=(0.0, 0.0, 0.0)) -> Mesh:
    """A welded axis-aligned box mesh, optionally translated."""
    import trimesh

    b = trimesh.creation.box(extents=extents)
    return Mesh(
        vertices=np.asarray(b.vertices, dtype=np.float64) + np.asarray(offset),
        triangles=np.asarray(b.faces, dtype=np.int64),
        linear_deflection_mm=0.1,
    )


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

    def test_default_views_are_distinct_images(self) -> None:
        """Camera changes must repaint, not save the first view under every name.

        Regression guard for the headless VTK/PyVista path: a one-plotter
        optimization once produced ``front`` / ``top`` / ``right`` PNG files
        whose bytes were identical to ``iso``. The 10x20x30 box fixture has
        different silhouettes from the default views, so byte-identical PNGs
        indicate a stale screenshot/camera update bug.
        """
        images = render_step(BOX_STEP)
        hashes = {hashlib.sha256(img.data).hexdigest() for img in images}
        assert len(hashes) == len(images)

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


class TestMeshDiff:
    """Signed-distance edit diff classification (no renderer needed)."""

    def test_identical_meshes_have_no_diff(self) -> None:
        box = _box_mesh()
        diff = mesh_diff(box, box)
        assert diff.added is None
        assert diff.removed is None
        assert diff.fraction_added == 0.0
        assert diff.fraction_removed == 0.0
        assert diff.max_deviation_mm < 0.5

    def test_translated_box_flags_added_and_removed(self) -> None:
        """A candidate shifted +X sticks out one side (added) and uncovers the
        other (removed)."""
        gt = _box_mesh()
        candidate = _box_mesh(offset=(2.0, 0.0, 0.0))
        diff = mesh_diff(gt, candidate)
        assert diff.added is not None
        assert diff.removed is not None
        assert diff.fraction_added > 0.0
        assert diff.fraction_removed > 0.0
        # The largest deviation should reflect the 2 mm shift.
        assert diff.max_deviation_mm == pytest.approx(2.0, abs=0.3)

    def test_tolerance_suppresses_small_offset(self) -> None:
        gt = _box_mesh()
        candidate = _box_mesh(offset=(0.2, 0.0, 0.0))
        diff = mesh_diff(gt, candidate, tol_mm=0.5)
        assert diff.added is None
        assert diff.removed is None


class TestRenderMeshDiff:
    """Diff renderers (in-process via VTK / PyVista)."""

    def test_diff_returns_image_per_view(self) -> None:
        gt = _box_mesh()
        candidate = _box_mesh(offset=(2.0, 0.0, 0.0))
        images = render_mesh_diff(gt, candidate, views=["iso", "front"])
        assert [img.name for img in images] == ["iso", "front"]
        for img in images:
            assert img.data[:8] == PNG_MAGIC

    def test_diff_turntable_is_animated_webp(self) -> None:
        gt = _box_mesh()
        candidate = _box_mesh(offset=(2.0, 0.0, 0.0))
        data = render_mesh_diff_turntable_webp(
            gt, candidate, frames=6, width=128, height=128,
        )
        assert data[:4] == WEBP_MAGIC_RIFF
        assert data[8:12] == WEBP_MAGIC_WEBP

    def test_diff_empty_candidate_raises(self) -> None:
        empty = Mesh(
            vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), dtype=np.int64),
            linear_deflection_mm=0.1,
        )
        with pytest.raises(ValueError, match="zero triangles"):
            render_mesh_diff(_box_mesh(), empty)


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
