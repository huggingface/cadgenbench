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
# author the KIR-fail / KIR-partial candidates (no such fixtures ship).
_BR_FLOOR_LEN, _BR_FLOOR_W, _BR_T, _BR_WALL_H = 80.0, 60.0, 8.0, 70.0
_BR_BOLTS = [(24.0, -15.0), (24.0, 15.0), (64.0, -15.0), (64.0, 15.0)]
_BR_BOLT_DIA = 8.0
_BR_BOSS_CENTER_YZ = (0.0, 40.0)
_BR_BOSS_FLATS = 15.0
_BR_BOSS_HEIGHT = 8.0


def _bracket(*, boss_dy: float | None):
    """The test_3 bracket. ``boss_dy=None`` omits the hex boss; a float shifts
    it by that much along the wall (+y), for the KIR fail / partial examples."""
    from build123d import (
        Box, BuildPart, BuildSketch, Cylinder, Locations, Mode,
        Plane, RegularPolygon, extrude,
    )

    with BuildPart() as b:
        with Locations((_BR_FLOOR_LEN / 2, 0.0, _BR_T / 2)):
            Box(_BR_FLOOR_LEN, _BR_FLOOR_W, _BR_T)
        wall_h = _BR_WALL_H - _BR_T
        with Locations((_BR_T / 2, 0.0, _BR_T + wall_h / 2)):
            Box(_BR_T, _BR_FLOOR_W, wall_h)
        for hx, hy in _BR_BOLTS:
            with Locations((hx, hy, _BR_T / 2)):
                Cylinder(radius=_BR_BOLT_DIA / 2, height=_BR_T + 1.0, mode=Mode.SUBTRACT)
        if boss_dy is not None:
            by, bz = _BR_BOSS_CENTER_YZ
            with BuildSketch(Plane.YZ) as bs:
                with Locations((by + boss_dy, bz)):
                    RegularPolygon(radius=_BR_BOSS_FLATS / 2.0, side_count=6, major_radius=False)
            extrude(bs.sketch, amount=-_BR_BOSS_HEIGHT, mode=Mode.ADD)
    return b.part


def _plate_one_hole(hole_xy: tuple[float, float]):
    """test_1-style plate (60x40x8) with a single Ø10 through-hole at hole_xy."""
    from build123d import Box, BuildPart, Cylinder, Locations, Mode

    with BuildPart() as p:
        Box(60.0, 40.0, 8.0)
        with Locations((hole_xy[0], hole_xy[1], 0.0)):
            Cylinder(radius=10.0 / 2, height=8.0 + 1.0, mode=Mode.SUBTRACT)
    return p.part


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


def _mating_group_webp(dest: Path) -> None:
    """Animated WebP: a one-piece jig seats into the part as one mating group.

    A carrier block + two pins + one slot key (a single connected solid)
    eases down so the pins seat in the two bolt holes and the key in the
    central slot of the test_4 plate, holds, then lifts -- a smooth loop that
    shows a mating group seating as a unit. Custom studio render (SSAA via 2x
    downscale, depth-peeled glass, glossy steel, 3-point lighting, perspective)
    reusing cadgenbench's WebP encoder.
    """
    import io

    import numpy as np
    import pyvista as pv
    from build123d import (
        Box, BuildPart, BuildSketch, Cylinder, Locations, Plane, SlotOverall,
        export_step, extrude,
    )
    from PIL import Image

    from cadgenbench.common.artifacts import StepArtifacts
    from cadgenbench.common.viewer import _encode_webp

    aw, ah, ss, drop, n_down = 720, 540, 2, 42.0, 32

    tmp = Path(tempfile.mkdtemp(prefix="cgb-mating-"))
    with BuildPart() as carrier:
        with Locations((0.0, 0.0, 10.0)):
            Box(110.0, 70.0, 8.0)
    with BuildPart() as feat:
        for x in (35.0, -35.0):
            with Locations((x, 0.0, 0.0)):
                Cylinder(radius=9.0 / 2, height=14.0)
        with BuildSketch(Plane.XY) as s:
            SlotOverall(width=29.0, height=11.0)
        extrude(s.sketch, amount=7.0, both=True)
    export_step(carrier.part, str(tmp / "c.step"))
    export_step(feat.part, str(tmp / "f.step"))
    plate = StepArtifacts(FIX / "test_4" / "gt.step").mesh()
    car = StepArtifacts(tmp / "c.step", deflection_override=0.5).mesh()
    ft = StepArtifacts(tmp / "f.step", deflection_override=0.25).mesh()

    allv = np.vstack([plate.vertices, car.vertices, ft.vertices,
                      ft.vertices + [0, 0, drop]])
    center = (allv.min(0) + allv.max(0)) * 0.5
    diag = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    d = np.array([1.0, -1.15, 0.78]); d /= np.linalg.norm(d)
    eye = center + d * diag * 1.5

    def poly(mesh, dz):
        v = np.ascontiguousarray(mesh.vertices, dtype=np.float64).copy()
        v[:, 2] += dz
        t = np.ascontiguousarray(mesh.triangles, dtype=np.int64)
        cells = np.empty((t.shape[0], 4), dtype=np.int64)
        cells[:, 0] = 3
        cells[:, 1:] = t
        return pv.PolyData(v, cells.reshape(-1))

    def frame(dz):
        pl = pv.Plotter(off_screen=True, window_size=(aw * ss, ah * ss), lighting="none")
        try:
            pl.set_background((0.95, 0.96, 0.975), top=(0.83, 0.86, 0.91))
            try:
                pl.enable_depth_peeling(12, occlusion_ratio=0.0)
            except Exception:
                pass
            for off, inten in (([0.7, -1.0, 1.3], 1.05), ([-1.2, -0.4, 0.5], 0.45),
                               ([0.0, 1.1, 1.0], 0.65)):
                lt = pv.Light(position=tuple(center + np.array(off) * 220.0),
                              focal_point=tuple(center), color="white")
                lt.positional = False
                lt.intensity = inten
                pl.add_light(lt)
            pl.add_mesh(poly(plate, 0.0), color=(0.17, 0.42, 0.78), opacity=0.40,
                        smooth_shading=True, split_sharp_edges=True, feature_angle=45,
                        ambient=0.22, diffuse=0.7, specular=0.45, specular_power=40)
            pl.add_mesh(poly(car, dz), color=(0.80, 0.83, 0.88), opacity=0.20,
                        smooth_shading=True, split_sharp_edges=True, feature_angle=45,
                        ambient=0.3, diffuse=0.6, specular=0.3, specular_power=30)
            pl.add_mesh(poly(ft, dz), color=(0.58, 0.61, 0.66), opacity=1.0,
                        smooth_shading=True, split_sharp_edges=True, feature_angle=40,
                        ambient=0.18, diffuse=0.62, specular=0.6, specular_power=60)
            pl.camera.position = tuple(eye)
            pl.camera.focal_point = tuple(center)
            pl.camera.up = (0.0, 0.0, 1.0)
            pl.camera.view_angle = 26.0
            pl.enable_anti_aliasing("ssaa")
            arr = np.asarray(pl.screenshot(return_img=True))
        finally:
            pl.close()
        buf = io.BytesIO()
        Image.fromarray(arr).resize((aw, ah), Image.LANCZOS).save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    ts = [0.5 - 0.5 * np.cos(np.pi * i / (n_down - 1)) for i in range(n_down)]
    down = [drop * (1.0 - t) for t in ts]
    offsets = down + [0.0] * 8 + down[::-1]
    pngs = [frame(dz) for dz in offsets]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_encode_webp(pngs, duration_ms=66, quality=80))
    print("wrote", dest.relative_to(REPO), f"({len(pngs)} frames)")


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
    # boss), each fit / partial / fail.
    from build123d import export_step

    scratch = Path(tempfile.mkdtemp(prefix="cgb-iface-"))

    _overlay("test_1", "correct", ILL / "kor_fit.png")
    # KOR partial: hole nudged 3mm -> blue (clearance kept) + red (intruding) +
    # amber (plate cut where it should stay solid), authored (no such fixture).
    kor_partial = scratch / "kor_partial.step"
    export_step(_plate_one_hole((3.0, 0.0)), str(kor_partial))
    _overlay_for_part("test_1", kor_partial, ILL / "kor_partial.png")
    _overlay("test_1", "broken_3_no_hole", ILL / "kor_fail.png")

    _overlay("test_3", "correct", ILL / "kir_fit.png")
    # KIR partial: boss shifted 5mm along the wall -> blue (filled half) + amber
    # (empty half) + red (overshoot into the clearance).
    kir_partial = scratch / "kir_partial.step"
    export_step(_bracket(boss_dy=5.0), str(kir_partial))
    _overlay_for_part("test_3", kir_partial, ILL / "kir_partial.png")
    # KIR fail = boss missing -> one clean amber hex (too little). Authored,
    # since test_3 ships only oversize / rotated boss candidates, whose thin
    # boolean overflow tessellates into jagged slivers.
    noboss = scratch / "bracket_no_boss.step"
    export_step(_bracket(boss_dy=None), str(noboss))
    _overlay_for_part("test_3", noboss, ILL / "kir_fail.png")

    # metrics.md editing section: ghost-diff of a feature moved to a new spot.
    _edit_diff_moved_feature(ILL / "example_editing" / "edit_diff.png")

    # interface_match.md hero: the mating-group seating animation.
    _mating_group_webp(ILL / "mating_group.webp")


if __name__ == "__main__":
    main()
