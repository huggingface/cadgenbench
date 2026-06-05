"""Unit tests for renderer-free image transforms (no VTK/PyVista)."""
import io

from PIL import Image

from cadgenbench.common.imaging import first_frame_png

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _animated_webp(size=(128, 96), n_frames=6) -> bytes:
    """A tiny animated WebP whose frame 0 is solid red (the rest blue), so
    the extracted still is verifiable by its corner pixel."""
    frames = [Image.new("RGB", size, (200, 30, 30))]
    frames += [
        Image.new("RGB", size, (30, 30, 200)) for _ in range(n_frames - 1)
    ]
    buf = io.BytesIO()
    frames[0].save(
        buf, format="WEBP", save_all=True, append_images=frames[1:],
        duration=100, loop=0,
    )
    return buf.getvalue()


def test_first_frame_png_returns_png_at_frame_size() -> None:
    png = first_frame_png(_animated_webp(size=(160, 120)))
    assert png[:8] == PNG_MAGIC
    with Image.open(io.BytesIO(png)) as im:
        assert im.format == "PNG"
        assert im.size == (160, 120)


def test_first_frame_png_is_frame_zero_not_a_later_frame() -> None:
    """Extracts frame 0 (red), never a subsequent frame (blue)."""
    png = first_frame_png(_animated_webp())
    with Image.open(io.BytesIO(png)) as im:
        r, g, b = im.convert("RGB").getpixel((0, 0))
    assert r > 150 and b < 100
