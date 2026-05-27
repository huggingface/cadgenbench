"""STEP fixtures for the interface-match metric.

Authors four self-contained test cases, each consisting of:

  - ``gt.step``                                       -- ground-truth part.
  - ``jig_<context_id>__<index>__<fit_type>.step``    -- one file per sub-volume
                                                         (= the canonical
                                                         region R the metric
                                                         scores). The file's
                                                         geometry IS R; no
                                                         separate "main jig"
                                                         file, no YAML.
  - ``candidates/correct.step``                       -- candidate equal to GT.
  - ``candidates/broken_*.step``                      -- predictable failures.

Naming convention (see docs/metrics/interface_match.md):

  jig_<context_id>__<index>__<fit_type>.step

- ``context_id`` is a 1-indexed integer. Sub-volumes sharing the same
  ``context_id`` are pose-searched together as one rigid group.
- ``index`` is a 1-indexed integer within that context.
- ``fit_type`` is exactly ``KOR`` -- keep-out region (hole / pocket /
  slot: the candidate's solid must be absent here) or ``KIR`` -- keep-in
  region (boss / protrusion: the candidate's solid must be present here).

Coordinate frame conventions:

- Units: mm.
- Plate GTs are centered at the origin with the plate's midplane at z=0
  (so an 8 mm plate spans z in [-4, +4]).
- The L-bracket in test_3 puts its bend at the origin; the floor runs
  in +x, +/- y; the wall runs in +z, +/- y (see ``_make_bracket``).
- Sub-volumes are at their absolute GT-spec pose (same frame as
  ``gt.step``).

Brief deviations (kept from the original colleague brief; documented
inline in the respective ``generate_test_*`` functions):

- Test 4 broken_1: slot width 9 mm (not 10 mm) so Ø10-equivalent slot
  pins produce a real intersection rather than tangent contact.
- Test 4 broken_2: replaced the brief's ``rotated_slot`` (which a
  cylindrical pin at slot centre can't detect) with ``slot_offset``.
- Test 4 broken_3 (rectangle vs stadium slot): dropped -- the brief
  itself admits its pass/fail depends on metric strictness, which
  violates the contract that every ``broken_*`` should fail predictably.
"""
from __future__ import annotations

from pathlib import Path

from build123d import (
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Cylinder,
    Locations,
    Mode,
    Plane,
    RegularPolygon,
    SlotOverall,
    export_step,
    extrude,
)

FIXTURES_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Sub-volume primitives
# ---------------------------------------------------------------------------
#
# A sub-volume is the canonical reference region R that the metric scores.
# It is the geometry the candidate's free space (KOR) or solid (KIR)
# is compared against. Three primitive shapes cover the v1 scope: cylinder
# (round holes / pins), hex prism (hex bosses / sockets), stadium prism
# (stadium-shaped slots).

def _cylinder_subvolume(
    diameter: float,
    length: float,
    center: tuple[float, float, float],
):
    """Cylindrical sub-volume centered at *center*, axis +z."""
    with BuildPart() as p:
        with Locations(center):
            Cylinder(radius=diameter / 2.0, height=length)
    return p.part


def _hex_subvolume(
    across_flats: float,
    length: float,
    center: tuple[float, float, float],
    axis: str = "z",
):
    """Hex prism sub-volume centered at *center*, axis along ``axis``."""
    cx, cy, cz = center
    with BuildPart() as p:
        if axis == "z":
            plane = Plane.XY.offset(cz)
            uv = (cx, cy)
        elif axis == "x":
            plane = Plane.YZ.offset(cx)
            uv = (cy, cz)
        else:
            raise ValueError(f"_hex_subvolume: unsupported axis {axis!r}")
        with BuildSketch(plane) as s:
            with Locations(uv):
                RegularPolygon(
                    radius=across_flats / 2.0,
                    side_count=6,
                    major_radius=False,
                )
        extrude(s.sketch, amount=length / 2.0, both=True, mode=Mode.ADD)
    return p.part


def _slot_subvolume(
    width: float,
    height: float,
    length: float,
    center: tuple[float, float, float],
    rotation_deg: float = 0.0,
):
    """Stadium-prism sub-volume centered at *center*, axis +z."""
    cx, cy, cz = center
    with BuildPart() as p:
        plane = Plane.XY.offset(cz)
        with BuildSketch(plane) as s:
            with Locations((cx, cy)):
                SlotOverall(width=width, height=height, rotation=rotation_deg)
        extrude(s.sketch, amount=length / 2.0, both=True, mode=Mode.ADD)
    return p.part


def _export_subvolume(
    part,
    test_dir: Path,
    *,
    context_id: int,
    index: int,
    fit_type: str,
) -> None:
    """Write a sub-volume STEP under *test_dir* using the canonical filename."""
    assert fit_type in {"KOR", "KIR"}, fit_type
    name = f"jig_{context_id}__{index}__{fit_type}.step"
    export_step(part, str(test_dir / name))


# ---------------------------------------------------------------------------
# GT / candidate primitives
# ---------------------------------------------------------------------------

def _plate_with_holes(
    plate_w: float,
    plate_l: float,
    plate_t: float,
    *,
    holes: list[tuple[float, float, float]],
):
    """Centered rectangular plate with (x, y, radius) through-holes along +z."""
    with BuildPart() as p:
        Box(plate_w, plate_l, plate_t)
        for x, y, r in holes:
            with Locations((x, y, 0.0)):
                Cylinder(radius=r, height=plate_t + 1.0, mode=Mode.SUBTRACT)
    return p.part


def _plate_with_holes_and_slot(
    plate_w: float,
    plate_l: float,
    plate_t: float,
    *,
    holes: list[tuple[float, float, float]],
    slot_center: tuple[float, float],
    slot_width: float,
    slot_height: float,
    slot_rotation_deg: float,
):
    """Plate with circular through-holes and one stadium through-slot."""
    cx, cy = slot_center
    with BuildPart() as p:
        Box(plate_w, plate_l, plate_t)
        for x, y, r in holes:
            with Locations((x, y, 0.0)):
                Cylinder(radius=r, height=plate_t + 1.0, mode=Mode.SUBTRACT)
        with BuildSketch() as s:
            with Locations((cx, cy)):
                SlotOverall(
                    width=slot_width,
                    height=slot_height,
                    rotation=slot_rotation_deg,
                )
        extrude(
            s.sketch,
            amount=(plate_t + 1.0) / 2.0,
            both=True,
            mode=Mode.SUBTRACT,
        )
    return p.part


# Bracket geometry (used by GT + every Test 3 candidate).
_BR_FLOOR_LEN = 80.0
_BR_FLOOR_W = 60.0
_BR_T = 8.0
_BR_WALL_H = 70.0
_BR_BOLT_PATTERN_XY = [
    (24.0, -15.0), (24.0, 15.0), (64.0, -15.0), (64.0, 15.0),
]
_BR_BOLT_DIA = 8.0
_BR_BOSS_CENTER = (0.0, 0.0, 40.0)
_BR_BOSS_FLATS = 15.0
_BR_BOSS_HEIGHT = 8.0


def _make_bracket(
    *,
    bolt_pattern: list[tuple[float, float]],
    bolt_dia: float,
    boss: str,
    boss_rotation_deg: float = 0.0,
):
    """L-bracket with a bolt pattern on the floor and a boss on the wall.

    Coordinate frame: bend at origin; floor in +x with z in [0, _BR_T];
    wall in +z with x in [0, _BR_T]. Boss on the wall's outside face
    (x = 0 plane), raised in -x by ``_BR_BOSS_HEIGHT``.

    ``boss="hex"`` -> regular hexagonal prism; ``boss="cylinder"``
    -> Ø17 cylinder (too big in the hex-socket's flat directions).
    """
    with BuildPart() as bracket:
        with Locations((_BR_FLOOR_LEN / 2.0, 0.0, _BR_T / 2.0)):
            Box(_BR_FLOOR_LEN, _BR_FLOOR_W, _BR_T)
        wall_height = _BR_WALL_H - _BR_T
        with Locations((_BR_T / 2.0, 0.0, _BR_T + wall_height / 2.0)):
            Box(_BR_T, _BR_FLOOR_W, wall_height)
        for hx, hy in bolt_pattern:
            with Locations((hx, hy, _BR_T / 2.0)):
                Cylinder(radius=bolt_dia / 2.0, height=_BR_T + 1.0, mode=Mode.SUBTRACT)
        boss_plane = Plane.YZ
        _, by, bz = _BR_BOSS_CENTER
        with BuildSketch(boss_plane) as bs:
            with Locations((by, bz)):
                if boss == "hex":
                    RegularPolygon(
                        radius=_BR_BOSS_FLATS / 2.0,
                        side_count=6,
                        major_radius=False,
                        rotation=boss_rotation_deg,
                    )
                elif boss == "cylinder":
                    Circle(radius=17.0 / 2.0)
                else:
                    raise ValueError(f"Unknown boss kind: {boss!r}")
        extrude(bs.sketch, amount=-_BR_BOSS_HEIGHT, mode=Mode.ADD)
    return bracket.part


# ---------------------------------------------------------------------------
# Test 1 -- Plate with one through-hole
# ---------------------------------------------------------------------------

def generate_test_1() -> None:
    """Plate (60x40x8) with a single centered Ø10 through-hole.

    Sub-volume: Ø10 cylinder, length 8 (plate thickness), centered at origin.

    Broken candidates:
      - broken_1_small_hole:   Ø9 hole; R extends outside C_free -> IoU drops.
      - broken_2_offset_hole:  hole at (5, 0); 5 mm offset >> ±0.5 mm pose
                               bound; R has no overlap with C_free -> IoU ~0.
      - broken_3_no_hole:      candidate is solid; C_free empty -> IoU = 0.
    """
    test_dir = FIXTURES_DIR / "test_1"
    candidates_dir = test_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    plate_w, plate_l, plate_t = 60.0, 40.0, 8.0
    hole_d = 10.0

    gt = _plate_with_holes(plate_w, plate_l, plate_t, holes=[(0.0, 0.0, hole_d / 2)])
    export_step(gt, str(test_dir / "gt.step"))

    hole = _cylinder_subvolume(hole_d, plate_t, (0.0, 0.0, 0.0))
    _export_subvolume(hole, test_dir, context_id=1, index=1, fit_type="KOR")

    export_step(gt, str(candidates_dir / "correct.step"))
    broken_small = _plate_with_holes(plate_w, plate_l, plate_t, holes=[(0.0, 0.0, 9.0 / 2)])
    export_step(broken_small, str(candidates_dir / "broken_1_small_hole.step"))
    broken_offset = _plate_with_holes(plate_w, plate_l, plate_t, holes=[(5.0, 0.0, hole_d / 2)])
    export_step(broken_offset, str(candidates_dir / "broken_2_offset_hole.step"))
    broken_no_hole = _plate_with_holes(plate_w, plate_l, plate_t, holes=[])
    export_step(broken_no_hole, str(candidates_dir / "broken_3_no_hole.step"))


# ---------------------------------------------------------------------------
# Test 2 -- Plate with 4-hole bolt pattern (one context, 4 free sub-volumes)
# ---------------------------------------------------------------------------

def generate_test_2() -> None:
    """Plate (100x60x8) with 4 Ø10 holes on a 70x40 pattern.

    Sub-volumes (same context_id=1 so the 4 holes are pose-searched as one
    rigid group; this is what catches "wrong relative spacing"):
      - jig_1__1..4__KOR.step at (+/-35, +/-20)

    Broken candidates:
      - broken_1_wrong_spacing:  holes at (+/-32.5, +/-17.5); no single rigid
                                 pose fits all four within ±0.5 mm bound.
      - broken_2_missing_hole:   one hole solid; that R has no C_free -> IoU=0.
      - broken_3_wrong_diameter: one hole at Ø8; that R extends beyond C_free.
    """
    test_dir = FIXTURES_DIR / "test_2"
    candidates_dir = test_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    plate_w, plate_l, plate_t = 100.0, 60.0, 8.0
    hole_d = 10.0
    pattern = [(35.0, 20.0), (-35.0, 20.0), (35.0, -20.0), (-35.0, -20.0)]

    gt = _plate_with_holes(
        plate_w, plate_l, plate_t,
        holes=[(x, y, hole_d / 2) for x, y in pattern],
    )
    export_step(gt, str(test_dir / "gt.step"))

    for idx, (x, y) in enumerate(pattern, start=1):
        subvol = _cylinder_subvolume(hole_d, plate_t, (x, y, 0.0))
        _export_subvolume(
            subvol, test_dir,
            context_id=1, index=idx, fit_type="KOR",
        )

    export_step(gt, str(candidates_dir / "correct.step"))

    wrong_pattern = [(32.5, 17.5), (-32.5, 17.5), (32.5, -17.5), (-32.5, -17.5)]
    broken_spacing = _plate_with_holes(
        plate_w, plate_l, plate_t,
        holes=[(x, y, hole_d / 2) for x, y in wrong_pattern],
    )
    export_step(broken_spacing, str(candidates_dir / "broken_1_wrong_spacing.step"))

    broken_missing = _plate_with_holes(
        plate_w, plate_l, plate_t,
        holes=[(x, y, hole_d / 2) for x, y in pattern[:-1]],
    )
    export_step(broken_missing, str(candidates_dir / "broken_2_missing_hole.step"))

    mixed_holes = [(x, y, hole_d / 2) for x, y in pattern[:-1]]
    mixed_holes.append((pattern[-1][0], pattern[-1][1], 8.0 / 2))
    broken_dia = _plate_with_holes(plate_w, plate_l, plate_t, holes=mixed_holes)
    export_step(broken_dia, str(candidates_dir / "broken_3_wrong_diameter.step"))


# ---------------------------------------------------------------------------
# Test 3 -- L-bracket with bolt pattern + hex boss
#           One context with mixed fit_types: 4 KOR + 1 KIR
# ---------------------------------------------------------------------------

def generate_test_3() -> None:
    """L-bracket: 4xØ8 bolt holes (40x30 pattern) on the floor plus a hex
    boss (across-flats 15, height 8) on the wall's outside face.

    Sub-volumes (all share context_id=1 -- bolts and boss must align with
    one tool together; pose search moves them rigidly):
      - jig_1__1..4__KOR.step on the floor
      - jig_1__5__KIR.step on the wall's outside face

    Broken candidates:
      - broken_1_cylinder_boss:  Ø17 cylinder boss; R(hex) extends past C_filled.
      - broken_2_rotated_boss:   hex rotated 15° around axis; R(hex) doesn't
                                 align with C_filled.
      - broken_3_shifted_holes:  bolt holes shifted (+1, +1) mm; no single
                                 rigid pose fits both bolt pattern AND boss
                                 within ±0.5 mm bound.
    """
    test_dir = FIXTURES_DIR / "test_3"
    candidates_dir = test_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    gt = _make_bracket(
        bolt_pattern=_BR_BOLT_PATTERN_XY,
        bolt_dia=_BR_BOLT_DIA,
        boss="hex",
    )
    export_step(gt, str(test_dir / "gt.step"))

    for idx, (hx, hy) in enumerate(_BR_BOLT_PATTERN_XY, start=1):
        subvol = _cylinder_subvolume(
            diameter=_BR_BOLT_DIA, length=_BR_T,
            center=(hx, hy, _BR_T / 2.0),
        )
        _export_subvolume(
            subvol, test_dir,
            context_id=1, index=idx, fit_type="KOR",
        )

    boss_subvol = _hex_subvolume(
        across_flats=_BR_BOSS_FLATS,
        length=_BR_BOSS_HEIGHT,
        center=(-_BR_BOSS_HEIGHT / 2.0, _BR_BOSS_CENTER[1], _BR_BOSS_CENTER[2]),
        axis="x",
    )
    _export_subvolume(
        boss_subvol, test_dir,
        context_id=1, index=5, fit_type="KIR",
    )

    export_step(gt, str(candidates_dir / "correct.step"))

    cylinder_boss = _make_bracket(
        bolt_pattern=_BR_BOLT_PATTERN_XY,
        bolt_dia=_BR_BOLT_DIA,
        boss="cylinder",
    )
    export_step(cylinder_boss, str(candidates_dir / "broken_1_cylinder_boss.step"))

    rotated_boss = _make_bracket(
        bolt_pattern=_BR_BOLT_PATTERN_XY,
        bolt_dia=_BR_BOLT_DIA,
        boss="hex",
        boss_rotation_deg=15.0,
    )
    export_step(rotated_boss, str(candidates_dir / "broken_2_rotated_boss.step"))

    shifted_pattern = [(x + 1.0, y + 1.0) for x, y in _BR_BOLT_PATTERN_XY]
    shifted_holes = _make_bracket(
        bolt_pattern=shifted_pattern,
        bolt_dia=_BR_BOLT_DIA,
        boss="hex",
    )
    export_step(shifted_holes, str(candidates_dir / "broken_3_shifted_holes.step"))


# ---------------------------------------------------------------------------
# Test 4 -- Plate with 2 holes + 1 slot
#           Three independent contexts (no rigid relationship enforced)
# ---------------------------------------------------------------------------

def generate_test_4() -> None:
    """Plate (100x60x8) with 2 fixed Ø10 holes and one 30x12 stadium slot.

    Sub-volumes (each in its own context, so they pose-search independently):
      - jig_1__1__KOR.step  at (+35, 0)
      - jig_2__1__KOR.step  at (-35, 0)
      - jig_3__1__KOR.step  30x12 stadium at origin

    Broken candidates:
      - broken_1_narrow_slot:  slot height 9 mm (not 12); R(30x12) extends past
                               C_free (the smaller 30x9 slot) -> IoU drops.
      - broken_2_slot_offset:  slot centre shifted to (0, 8); R covers
                               y in [-6, 6] but C_free is y in [2, 14]; very
                               small overlap -> IoU near 0.
    """
    test_dir = FIXTURES_DIR / "test_4"
    candidates_dir = test_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    plate_w, plate_l, plate_t = 100.0, 60.0, 8.0
    hole_d = 10.0
    slot_w, slot_h = 30.0, 12.0

    gt = _plate_with_holes_and_slot(
        plate_w, plate_l, plate_t,
        holes=[(35.0, 0.0, hole_d / 2), (-35.0, 0.0, hole_d / 2)],
        slot_center=(0.0, 0.0),
        slot_width=slot_w,
        slot_height=slot_h,
        slot_rotation_deg=0.0,
    )
    export_step(gt, str(test_dir / "gt.step"))

    hole_1_sub = _cylinder_subvolume(hole_d, plate_t, (35.0, 0.0, 0.0))
    _export_subvolume(hole_1_sub, test_dir, context_id=1, index=1, fit_type="KOR")

    hole_2_sub = _cylinder_subvolume(hole_d, plate_t, (-35.0, 0.0, 0.0))
    _export_subvolume(hole_2_sub, test_dir, context_id=2, index=1, fit_type="KOR")

    slot_sub = _slot_subvolume(slot_w, slot_h, plate_t, (0.0, 0.0, 0.0))
    _export_subvolume(slot_sub, test_dir, context_id=3, index=1, fit_type="KOR")

    export_step(gt, str(candidates_dir / "correct.step"))

    narrow_slot = _plate_with_holes_and_slot(
        plate_w, plate_l, plate_t,
        holes=[(35.0, 0.0, hole_d / 2), (-35.0, 0.0, hole_d / 2)],
        slot_center=(0.0, 0.0),
        slot_width=slot_w,
        slot_height=9.0,
        slot_rotation_deg=0.0,
    )
    export_step(narrow_slot, str(candidates_dir / "broken_1_narrow_slot.step"))

    offset_slot = _plate_with_holes_and_slot(
        plate_w, plate_l, plate_t,
        holes=[(35.0, 0.0, hole_d / 2), (-35.0, 0.0, hole_d / 2)],
        slot_center=(0.0, 8.0),
        slot_width=slot_w,
        slot_height=slot_h,
        slot_rotation_deg=0.0,
    )
    export_step(offset_slot, str(candidates_dir / "broken_2_slot_offset.step"))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_GENERATORS = [
    generate_test_1,
    generate_test_2,
    generate_test_3,
    generate_test_4,
]


def main() -> None:
    for fn in ALL_GENERATORS:
        fn()
        print(f"Generated {fn.__name__}")


if __name__ == "__main__":
    main()
