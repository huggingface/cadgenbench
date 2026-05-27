"""Generate all STEP fixtures used by the test suite.

Each ``generate_*`` function is **idempotent**: it skips the export when
its target file already exists. Re-generating a fixture (e.g. after
intentionally changing its build123d definition) is therefore a
two-step manual gesture: ``rm tests/fixtures/<name>.step`` then re-run
either the affected generator or ``python generate_fixtures.py``.

Why idempotent: ``export_step`` embeds the current timestamp in the STEP
header. If a test autouse fixture invokes the generators every session,
every test run would silently rewrite the on-disk fixture (same
geometry, different timestamp) and leave the working tree dirty in
git. Idempotent generators keep the fixtures stable across runs.
"""
from pathlib import Path

from build123d import (
    Box,
    BuildPart,
    Cylinder,
    Locations,
    Mode,
    Sphere,
    export_step,
)

FIXTURES_DIR = Path(__file__).parent


def generate_box() -> None:
    """10x20x30 box.  Volume=6000, faces=6, 1 solid."""
    path = FIXTURES_DIR / "box.step"
    if path.exists():
        return
    with BuildPart() as p:
        Box(10, 20, 30)
    export_step(p.part, str(path))


def generate_open_shell() -> None:
    """Single face extracted from a box, no solid, volume=0."""
    path = FIXTURES_DIR / "open_shell.step"
    if path.exists():
        return
    top_face = Box(10, 20, 30).faces().sort_by().last
    export_step(top_face, str(path))


def generate_two_solids() -> None:
    """Two separate boxes in one file.  solid_count=2, volume=2000."""
    path = FIXTURES_DIR / "two_solids.step"
    if path.exists():
        return
    with BuildPart() as p:
        Box(10, 10, 10)
        with Locations([(25, 0, 0)]):
            Box(10, 10, 10)
    export_step(p.part, str(path))


def generate_sphere() -> None:
    """Sphere r=10.  Volume=4/3*pi*r^3, faces=1 (single NURBS face)."""
    path = FIXTURES_DIR / "sphere.step"
    if path.exists():
        return
    with BuildPart() as p:
        Sphere(10)
    export_step(p.part, str(path))


def generate_cylinder_with_hole() -> None:
    """Cylinder r=20 h=40 minus r=5 hole.  Volume=pi*(20^2-5^2)*40, faces=4."""
    path = FIXTURES_DIR / "cylinder_with_hole.step"
    if path.exists():
        return
    with BuildPart() as p:
        Cylinder(20, 40)
        Cylinder(5, 40, mode=Mode.SUBTRACT)
    export_step(p.part, str(path))


ALL_GENERATORS = [
    generate_box,
    generate_open_shell,
    generate_two_solids,
    generate_sphere,
    generate_cylinder_with_hole,
]


def main() -> None:
    for fn in ALL_GENERATORS:
        path = FIXTURES_DIR / f"{fn.__name__.removeprefix('generate_')}.step"
        before_existed = path.exists()
        fn()
        print(f"{'kept (already existed)' if before_existed else 'generated'}: {path.name}")


if __name__ == "__main__":
    main()
