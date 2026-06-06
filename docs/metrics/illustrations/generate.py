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


def _overlay_for_part(test: str, part_step: Path, dest: Path) -> None:
    """Interface overlay of *part_step* against *test*'s GT + sub-volumes.

    Colours: grey ghost / blue fits / red too-much / amber too-little.
    """
    from cadgenbench.eval.interface_match import discover_sub_volumes
    from cadgenbench.eval.interface_match_viz import render_part_with_subvolumes

    d = FIX / test
    img = render_part_with_subvolumes(
        part_step,
        discover_sub_volumes(d),
        gt_step=d / "gt.step",
        views=("iso",),
        width=W,
        height=H,
    )[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(img.data)
    print("wrote", dest.relative_to(REPO))


def _overlay(test: str, candidate: str, dest: Path) -> None:
    """Overlay of a named ``candidates/<candidate>.step`` fixture."""
    _overlay_for_part(test, FIX / test / "candidates" / f"{candidate}.step", dest)


# L-bracket constants, mirroring tests/fixtures/jig_metric/generate.py, used to
# author the "boss missing" KIR-fail candidate (no such fixture ships).
_BR_FLOOR_LEN, _BR_FLOOR_W, _BR_T, _BR_WALL_H = 80.0, 60.0, 8.0, 70.0
_BR_BOLTS = [(24.0, -15.0), (24.0, 15.0), (64.0, -15.0), (64.0, 15.0)]
_BR_BOLT_DIA = 8.0


def _bracket_without_boss():
    """The test_3 bracket (floor + wall + bolt holes) with the boss left off."""
    from build123d import Box, BuildPart, Cylinder, Locations, Mode

    with BuildPart() as b:
        with Locations((_BR_FLOOR_LEN / 2, 0.0, _BR_T / 2)):
            Box(_BR_FLOOR_LEN, _BR_FLOOR_W, _BR_T)
        wall_h = _BR_WALL_H - _BR_T
        with Locations((_BR_T / 2, 0.0, _BR_T + wall_h / 2)):
            Box(_BR_T, _BR_FLOOR_W, wall_h)
        for hx, hy in _BR_BOLTS:
            with Locations((hx, hy, _BR_T / 2)):
                Cylinder(radius=_BR_BOLT_DIA / 2, height=_BR_T + 1.0, mode=Mode.SUBTRACT)
    return b.part


def _plate_with_boss(boss_xy: tuple[float, float]):
    """80x50x8 plate with a small Ø10 boss (height 5) on the top face."""
    from build123d import Box, BuildPart, Cylinder, Locations

    with BuildPart() as p:
        Box(80.0, 50.0, 8.0)
        with Locations((boss_xy[0], boss_xy[1], 8.0 / 2 + 5.0 / 2)):
            Cylinder(radius=10.0 / 2, height=5.0)
    return p.part


def _edit_diff_moved_feature(dest: Path) -> None:
    """Edit diff of a small feature moved to a new position: a boss relocated.

    GT boss at (-15, 0), candidate at (1, 0) -- moved far enough to fully clear
    its old footprint. The diff colours only what changed, so a clean move
    shows two whole shapes (no carved-open crescents): amber where the feature
    used to be (now missing), red where it moved to (added). The move is still
    small relative to the part.
    """
    from build123d import export_step
    from cadgenbench.common.artifacts import StepArtifacts
    from cadgenbench.common.viewer import render_mesh_diff

    tmp = Path(tempfile.mkdtemp(prefix="cgb-docedit-"))
    gt_path, cand_path = tmp / "gt.step", tmp / "cand.step"
    export_step(_plate_with_boss((-15.0, 0.0)), str(gt_path))
    export_step(_plate_with_boss((1.0, 0.0)), str(cand_path))

    gt = StepArtifacts(gt_path)
    cand = StepArtifacts(cand_path, deflection_override=gt.deflection())
    img = render_mesh_diff(gt.mesh(), cand.mesh(), views=("iso",), width=W, height=H)[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(img.data)
    print("wrote", dest.relative_to(REPO))


def main() -> None:
    # interface_match.md concept image: the ground-truth plate with its three
    # mating regions in blue (the regions the metric checks; the GT satisfies
    # them, hence blue). Replaces the old blue-plate / red-regions figure.
    _overlay("test_4", "correct", ILL / "interface_concept.png")

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
    # KIR fail = boss missing -> one clean amber hex (too little). Authored,
    # since test_3 ships only oversize / rotated boss candidates, whose thin
    # boolean overflow tessellates into jagged slivers.
    from build123d import export_step

    noboss = Path(tempfile.mkdtemp(prefix="cgb-kir-")) / "bracket_no_boss.step"
    export_step(_bracket_without_boss(), str(noboss))
    _overlay_for_part("test_3", noboss, ILL / "kir_fail.png")

    # metrics.md editing section: ghost-diff of a feature moved to a new spot.
    _edit_diff_moved_feature(ILL / "example_editing" / "edit_diff.png")


if __name__ == "__main__":
    main()
