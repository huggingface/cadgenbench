# CAD Validity (Hard Gate)

Runs before every other metric; any failure zeroes `cad_score` so an invalid candidate can never beat a worse-but-valid one.

## What we check

`is_valid = True` iff all three of the following hold.

### 1. BREP well-formedness

`BRepCheck_Analyzer.IsValid()` (Open CASCADE) reports no per-face, per-edge, or per-vertex topology errors. Catches self-intersecting wires, edges whose curves don't lie on their underlying surfaces, and the usual classical-BREP defects. These are the failures STEP authors make most often: the part looks fine in a viewer but blows up the first time it's Boolean'd against.

### 2. Watertightness

Every shell is closed: no naked / free edges. A BREP that `BRepCheck_Analyzer` accepts but whose shells are open is not a solid; it cannot be 3D-printed, Boolean'd against, or topologically analysed. Surfaced in `topology_errors` as:

```
"BREP not watertight: at least one shell has open / naked edges"
```

The classical OCC check is lenient on open shells; downstream volume IoU, Betti, and interface IoU all assume a closed volume, so this gate is enforced separately.

### 3. Meshable as a closed orientable manifold

Tessellating the BREP with [`mesh.py`](../../src/cadgenbench/common/mesh.py) must produce a triangle mesh that is:

- **Manifold**: every edge appears in $\le 2$ triangles.
- **Closed**: every edge appears in exactly 2 triangles ($3F = 2E$ on the global mesh).
- **Orientation-consistent**: for each shared edge $(a, b)$, the two incident triangles list it in opposite orders $(a, b)$ and $(b, a)$.

Surfaced as e.g. `"mesh non-manifold: edge (220, 243) shared by 4 triangles"`. A failure here also signals a BREP defect the classical analyzer missed.

## Code pointers

- Metric: [`src/cadgenbench/common/validity.py`](../../src/cadgenbench/common/validity.py)
- Mesh-pipeline gate: [`src/cadgenbench/common/mesh.py`](../../src/cadgenbench/common/mesh.py)
- Orchestrator: [`src/cadgenbench/eval/evaluate.py`](../../src/cadgenbench/eval/evaluate.py) (`_cad_score`)
