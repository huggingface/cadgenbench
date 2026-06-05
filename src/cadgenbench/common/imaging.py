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

"""Small image transforms with no renderer dependencies.

Kept separate from :mod:`cadgenbench.common.viewer` (which imports VTK /
PyVista at module load) so a pure-Pillow helper like :func:`first_frame_png`
can run anywhere — the bucket backfill tool, tests — without pulling in the
headless-GL stack.
"""
from __future__ import annotations

import io

from PIL import Image


def first_frame_png(webp_bytes: bytes) -> bytes:
    """Return frame 0 of an (animated) WebP as PNG bytes.

    Derives a small static still from a turntable WebP **without
    re-rendering**: the encoder lays the turntable down from a fixed start
    angle, so frame 0 is a stable, representative view. Used as the grid
    thumbnail for the edit-diff (``edit_diff.png`` beside
    ``edit_diff.webp``). The eval renderer (going forward) and the
    one-time bucket backfill both go through here, so the still always
    matches the clip's first frame.
    """
    with Image.open(io.BytesIO(webp_bytes)) as im:
        im.seek(0)
        out = io.BytesIO()
        im.convert("RGB").save(out, format="PNG")
        return out.getvalue()
