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

"""Regenerate the metric-doc illustrations.

These figures are renders of real fixtures through the actual visualizers,
so they track the code: rerun this whenever the overlay / edit-diff colour
scheme changes (see ``interface_match_viz`` and ``common.viewer``) so the
images in ``metrics.md`` / ``interface_match.md`` stay in sync.

Run from a source checkout with ``cadgenbench`` installed (the render extras
pull VTK/PyVista/manifold3d) and an OpenGL context available::

    python docs/metrics/illustrations/generate.py

Shared colour language across both overlays: grey = your geometry,
blue = matches, red = extra material (too much), amber = missing (too little).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests" / "fixtures" / "jig_metric"
ILL = Path(__file__).resolve().parent

# Doc render size (4:3, comfortably crisp inline).
W, H = 900, 680


def _plain_iso(step: Path, dest: Path) -> None:
    """Plain grey iso render of a part (the GT / candidate panels)."""
    from cadgenbench.common.artifacts import StepArtifacts
    from cadgenbench.common.viewer import render_mesh

    img = render_mesh(StepArtifacts(step).mesh(), views=("iso",), width=W, height=H)[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(img.data)
    print("wrote", dest.relative_to(REPO))


def _overlay(test: str, candidate: str, dest: Path) -> None:
    """Interface overlay (grey ghost / blue fits / red too-much / amber too-little)."""
    from cadgenbench.eval.interface_match import discover_sub_volumes
    from cadgenbench.eval.interface_match_viz import render_part_with_subvolumes

    d = FIX / test
    img = render_part_with_subvolumes(
        d / "candidates" / f"{candidate}.step",
        discover_sub_volumes(d),
        gt_step=d / "gt.step",
        views=("iso",),
        width=W,
        height=H,
    )[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(img.data)
    print("wrote", dest.relative_to(REPO))


def _plate_with_boss(boss_xy: tuple[float, float]):
    """80x50x8 plate with a small Ø10 boss (height 5) on the top face."""
    from build123d import Box, BuildPart, Cylinder, Locations

    with BuildPart() as p:
        Box(80.0, 50.0, 8.0)
        with Locations((boss_xy[0], boss_xy[1], 8.0 / 2 + 5.0 / 2)):
            Cylinder(radius=10.0 / 2, height=5.0)
    return p.part


def _edit_diff_small_move(dest: Path) -> None:
    """Edit diff of a small feature moved a little: a boss nudged 6 mm.

    GT boss at (-12, 0), candidate at (-6, 0). The move is small relative to
    the part, so the diff is two localized crescents -- red where the moved
    boss adds material, amber where the original boss is now missing -- which
    reads far cleaner than a large symmetric edit.
    """
    from build123d import export_step
    from cadgenbench.common.artifacts import StepArtifacts
    from cadgenbench.common.viewer import render_mesh_diff

    tmp = Path(tempfile.mkdtemp(prefix="cgb-docedit-"))
    gt_path, cand_path = tmp / "gt.step", tmp / "cand.step"
    export_step(_plate_with_boss((-12.0, 0.0)), str(gt_path))
    export_step(_plate_with_boss((-6.0, 0.0)), str(cand_path))

    gt = StepArtifacts(gt_path)
    cand = StepArtifacts(cand_path, deflection_override=gt.deflection())
    img = render_mesh_diff(gt.mesh(), cand.mesh(), views=("iso",), width=W, height=H)[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(img.data)
    print("wrote", dest.relative_to(REPO))


def main() -> None:
    # metrics.md Example 3: single-group plate (2 holes + central slot), slot
    # shifted. All three panels from one fixture so they stay consistent.
    ex3 = ILL / "example_3_interface"
    _plain_iso(FIX / "test_4" / "gt.step", ex3 / "gt_iso.png")
    _plain_iso(
        FIX / "test_4" / "candidates" / "broken_2_slot_offset.step",
        ex3 / "candidate_iso.png",
    )
    _overlay("test_4", "broken_2_slot_offset", ex3 / "interface_overlay.png")

    # interface_match.md: keep-out (single-hole plate) and keep-in (bracket
    # boss), each fit vs fail.
    _overlay("test_1", "correct", ILL / "kor_fit.png")
    _overlay("test_1", "broken_3_no_hole", ILL / "kor_fail.png")
    _overlay("test_3", "correct", ILL / "kir_fit.png")
    _overlay("test_3", "broken_1_cylinder_boss", ILL / "kir_fail.png")

    # metrics.md editing section: ghost-diff of a small feature moved a little.
    _edit_diff_small_move(ILL / "example_editing" / "edit_diff.png")


if __name__ == "__main__":
    main()
