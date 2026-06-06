# CAD Validity (Hard Gate)

Runs before every other metric on the raw candidate STEP. Any failure sets `is_valid = False` and forces `cad_score = 0`, with a human-readable reason. `is_valid = True` requires all of:

1. **Well-formed BREP.** `BRepCheck_Analyzer.IsValid()` reports no per-face, per-edge, or per-vertex errors (self-intersecting wires, edges off their surface, and similar defects).
2. **Watertight.** Every shell is closed: no naked or free edges.
3. **Meshable as a closed orientable manifold.** Tessellation yields a triangle mesh that is manifold (every edge in at most 2 triangles), closed (every edge in exactly 2 triangles, i.e. $3F = 2E$), and orientation-consistent (each shared edge is traversed in opposite directions by its two triangles).

Code: [`validity.py`](../../src/cadgenbench/common/validity.py)
