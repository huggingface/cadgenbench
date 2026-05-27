"""One-off builder for the jig-01-edit-double-hole fixture (geometry only).

Author-only: writes the STEP files; rendering is NOT done here because
the headless renderer (Playwright / Chromium) does not run inside the
Cursor sandbox. PNGs are produced by a separate command the user runs
locally (see the trailing instructions in chat).

Creates:

- data/gt/jig-01-edit-double-hole/ground_truth.step    (60 x 40 x 8 plate, Ø20 hole)
- data/gt/jig-01-edit-double-hole/jig_1__1__KOR.step   (Ø20 x 8 keep-out cylinder)

Run from the repo root:

    python _to_move_to_dataset_repo/_build_jig01_edit_fixture.py
"""
from __future__ import annotations

from pathlib import Path

from build123d import Box, Cylinder, export_step

GT_DIR = Path("data/gt/jig-01-edit-double-hole")

PLATE_X = 60.0
PLATE_Y = 40.0
PLATE_Z = 8.0
NEW_HOLE_DIAMETER = 20.0  # 2x the original 10 mm
NEW_HOLE_RADIUS = NEW_HOLE_DIAMETER / 2


def build_ground_truth() -> Path:
    plate = Box(PLATE_X, PLATE_Y, PLATE_Z)
    hole = Cylinder(radius=NEW_HOLE_RADIUS, height=PLATE_Z)
    part = plate - hole
    out = GT_DIR / "ground_truth.step"
    export_step(part, str(out))
    return out


def build_kor() -> Path:
    """KOR sub-volume: Ø20 x 8 cylinder centred on the hole."""
    kor = Cylinder(radius=NEW_HOLE_RADIUS, height=PLATE_Z)
    out = GT_DIR / "jig_1__1__KOR.step"
    export_step(kor, str(out))
    return out


def main() -> None:
    GT_DIR.mkdir(parents=True, exist_ok=True)

    gt_step = build_ground_truth()
    print(f"  wrote {gt_step}")

    kor_step = build_kor()
    print(f"  wrote {kor_step}")

    print()
    print("Geometry done. Renders (input.png + GT renders/) must be made")
    print("locally, the headless renderer does not run in the Cursor sandbox.")


if __name__ == "__main__":
    main()
