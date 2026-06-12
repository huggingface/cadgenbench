# OpenSCAD

A script-based solid modeller built on constructive solid geometry (CSG). You
describe geometry as a tree of primitives combined with boolean operations and
transforms; the kernel evaluates it into a triangle mesh.

**Default units: millimeters (mm).** OpenSCAD is unitless, but the benchmark
treats every number as mm.

## How your code is run

- Write a single `scad` block describing the **final model as top-level
  geometry** (top-level object calls are unioned automatically).
- Do **not** write any export/render command — the harness compiles your script
  to `output.stl` for you (the equivalent of `openscad -o output.stl model.scad`).
- Use `echo(...)` for diagnostics; echoes appear in the run output.
- The result must be a single, watertight, manifold solid (see *Manifold
  output* below) or it fails the validity gate.

## 3D primitives

```scad
cube([x, y, z], center = true);     // box; center=false anchors at origin corner
sphere(r = 10, $fn = 64);           // sphere
cylinder(h = 20, r = 5, $fn = 64);  // cylinder
cylinder(h = 20, r1 = 5, r2 = 2);   // cone / frustum
polyhedron(points = [...], faces = [...]);  // arbitrary mesh (advanced)
```

## Boolean operations

```scad
union()        { a(); b(); }   // combine
difference()   { a(); b(); }   // subtract every child after the first from the first
intersection() { a(); b(); }   // overlap only
```

## Transforms

```scad
translate([x, y, z]) child();
rotate([rx, ry, rz]) child();        // degrees, applied X then Y then Z
rotate(a = 90, v = [0, 0, 1]) child();
scale([sx, sy, sz]) child();
mirror([1, 0, 0]) child();
multmatrix(m = [[...],[...],[...],[...]]) child();  // 4x4 affine
```

## 2D + extrusion

```scad
circle(r = 5, $fn = 64);
square([w, h], center = true);
polygon(points = [[0,0],[10,0],[10,5]]);

linear_extrude(height = 10, twist = 0, scale = 1) { circle(5); }
rotate_extrude(angle = 360, $fn = 128) { translate([10, 0]) circle(2); }  // profile must be in +X
```

## Hull, Minkowski

```scad
hull()      { a(); b(); }      // convex hull of children
minkowski() { a(); b(); }      // sweep a by b (rounding, offsets) — expensive
```

## Reuse: modules and functions

```scad
module washer(d_out, d_in, t) {
    difference() {
        cylinder(h = t, d = d_out, $fn = 96);
        translate([0, 0, -1]) cylinder(h = t + 2, d = d_in, $fn = 96);
    }
}
washer(20, 8, 3);

function leg(i) = [i * 30, 0, 0];
for (i = [0 : 3]) translate(leg(i)) cube(5);
```

## Resolution: `$fn`, `$fa`, `$fs`

- `$fn` = fixed number of fragments per full circle (e.g. `$fn = 96`). Higher =
  smoother curves but more triangles.
- `$fa` (min angle) / `$fs` (min size) auto-select fragments when `$fn` is 0.
- Set a sensible `$fn` on every curved primitive; coarse curves can read as a
  poor shape match. Avoid absurdly high values (slow, huge meshes).

## Manifold output (required)

The exported mesh must be a closed, manifold, orientation-consistent surface:

- Prefer clean booleans of solid primitives. `union()` overlapping solids
  rather than abutting them face-to-face (coincident faces can leave
  non-manifold edges).
- When subtracting a through-hole, make the cutting tool **poke out past both
  faces** (e.g. extend a drilled cylinder by ±1 mm) so it never leaves a
  zero-thickness coplanar wall.
- Avoid zero-thickness walls, self-intersections that don't resolve, and
  degenerate (zero-volume) geometry.
- A non-manifold result is reported back to you by the auto-validation; fix the
  geometry and re-emit.

## Canonical pose (recommended)

Center the bounding box at the origin and order the extents
`Lx >= Ly >= Lz` (longest along X, shortest along Z) so rigid alignment stays
stable on symmetric parts.
