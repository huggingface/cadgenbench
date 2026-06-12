# CAD Validity (Hard Gate)

Runs before every other metric on the raw candidate. Any failure sets
`is_valid = False` and forces `cad_score = 0`, with a human-readable reason.

For **STEP/BREP** candidates, `is_valid = True` requires all of:

1. **Well-formed BREP.** `BRepCheck_Analyzer.IsValid()` reports no per-face, per-edge, or per-vertex errors (self-intersecting wires, edges off their surface, and similar defects).
2. **Watertight.** Every shell is closed: no naked or free edges.
3. **Meshable as a closed orientable manifold.** Tessellation yields a triangle mesh that is manifold (every edge in at most 2 triangles), closed (every edge in exactly 2 triangles, i.e. $3F = 2E$), and orientation-consistent (each shared edge is traversed in opposite directions by its two triangles).

For **mesh** candidates (`output.stl`, `output.obj`, `output.off`,
`output.3mf`, or `output.ply`), BREP checks do not apply. The submitted mesh
itself must pass the mesh gate in item 3: manifold, closed, and
orientation-consistent.

Code: [`validity.py`](../../src/cadgenbench/common/validity.py)

## Advisory diagnostics (flagged, not gated)

The thresholds below are not part of the gate and never affect `cad_score`.
They flag geometry that passes validity but exhibits the sliver faces, loose
tolerances, or near-degenerate features characteristic of fragile exports. A
flag identifies geometry worth cleaning up; it is not a rejection.

| Diagnostic | Flag when | Rationale |
| --- | --- | --- |
| Minimum face area | below `0.001 mm²` | Healthy parts bottom out near `0.05 mm²`, whereas genuine defects fall to `1e-19`–`3e-4`. The threshold sits roughly 50× below healthy geometry and well above the degenerate range. |
| Maximum face aspect ratio (length / width) | above `1000` | Healthy faces are single- to double-digit; sliver faces range from `1e5` to `5e8`. The threshold is far above any legitimate face and well below the defects. |
| Maximum BREP tolerance | above `0.1 mm` | Healthy parts sit near `0.05 mm`; larger values indicate loose or imprecise exports. |
