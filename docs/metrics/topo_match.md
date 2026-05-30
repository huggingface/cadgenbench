# Topology Match

Scores whether a candidate part's 3D topology matches the ground truth's: number of solid pieces, through-handles, and internal voids.

## What we check

Three integer invariants of the solid as a 3-manifold with boundary, the **Betti numbers**:

- **$b_0$**: connected solid components (pieces of material).
- **$b_1$**: independent through-handles (e.g. through-holes, closed loops of material).
- **$b_2$**: enclosed internal voids (cavities fully wrapped by material).

Betti is coordinate- and representation-invariant: the same physical part modelled in two different CAD systems, with different feature trees and BREP face decompositions, yields identical $(b_0, b_1, b_2)$.

### Examples

| Part                                | $b_0$ | $b_1$ | $b_2$ |
| ----------------------------------- | ----- | ----- | ----- |
| Solid block                         |  1    |  0    |  0    |
| Plate with one through-hole         |  1    |  1    |  0    |
| Plate with four through-holes       |  1    |  4    |  0    |
| Hollow ball (one closed cavity)     |  1    |  0    |  1    |
| Two separate cubes                  |  2    |  0    |  0    |
| Plate with blind hole / pocket      |  1    |  0    |  0    |

Blind features are topologically trivial. A pocket that doesn't go through is just a deformation of a flat surface and leaves Betti unchanged. Such features are covered by `shape_similarity` and `interface_match`, not here.

## Compute on the mesh, not on the BREP

Direct BREP Euler-Poincaré on the cellular complex,

```
V − E + F − R = 2 (S − G)
```

where `R` is the total count of inner face-wires, depends on the modeller's choices for periodic-face seams and shared inner rings. The same shape authored two ways can give different Betti, including impossible values ($b_1 = -38$ on one real-world part for one BREP encoding we tested).

Tessellating to a closed orientable triangle mesh and computing $\chi = V - E + F$ on that mesh sidesteps the problem entirely. Every face is a topological disk, every edge is shared by exactly two triangles, and $\chi$ is the genuine Euler characteristic of the boundary surface. The metric is also independent of input format: any closed orientable manifold mesh works, STEP is just the format the benchmark ingests.

## Pipeline

For one candidate against one GT:

1. **Tessellate.** `BRepMesh_IncrementalMesh(shape, linear_deflection)` produces a per-face triangulation. The deflection is computed once from the *GT* bounding-box diagonal and applied to **both** GT and candidate so the two meshes are tessellated at the same scale:

   ```
   linear_deflection = clamp(0.001 × bbox_diagonal(GT), 0.005, 0.5)  # mm
   ```

2. **Unify edge nodes.** OCC tessellates each face independently and leaves coincident-but-distinct vertices on shared edges. We walk each topological edge and, via `BRep_Tool.PolygonOnTriangulation_s(edge, triangulation, location)`, force the per-face node indices on both sides of the edge to weld to a single global vertex. This is what makes $3F = 2E$ hold exactly.

3. **Mesh sanity gate** (three checks):
   - **Manifold**: every edge appears in $\le 2$ triangles.
   - **Closed**: every edge appears in exactly 2 triangles (equivalently $3F = 2E$).
   - **Orientation-consistent**: for each shared edge $(a, b)$, the two incident triangles list it in opposite orders $(a, b)$ and $(b, a)$.

   Any failure sets `is_valid = False`, appends a descriptive entry to `topology_errors`, and propagates through the validity gate to `cad_score = 0`.

4. **Connected components.** Union-find on triangle adjacency. Each component is a closed orientable 2-manifold.

5. **Containment ($b_0$ and $b_2$).** For each component, pick an interior probe point (centroid of a seed triangle, nudged inward by a small step relative to bbox); count how many *other* components contain it via even/odd ray casting. Even ⇒ outer shell of a solid, contributes to $b_0$. Odd ⇒ inner shell of a void, contributes to $b_2$.

6. **Apply Euler-Poincaré.** With $\chi_{\text{surface}} = V - E + F$ and the identity $\chi(\partial S) = 2\,\chi(S) = 2(b_0 - b_1 + b_2)$:

   $$
   b_1 = b_0 + b_2 - \chi_{\text{surface}} / 2
   $$

7. **Cross-check.** Compare mesh-derived $b_0$ against BREP `solid_count`. Disagreement indicates a pipeline bug, marked invalid.

## Score

Per Betti number, a fuzzy log-ratio against the GT:

$$
s_i \;=\; \exp\!\Bigl(-\bigl|\log\bigl((b_i^{\text{cand}}+1)/(b_i^{\text{gt}}+1)\bigr)\bigr|\Bigr) \;\in\; [0, 1]
$$

The aggregate is the unweighted mean over the three axes:

$$
\text{topo\_match} \;=\; \tfrac{1}{3}\bigl(s_0 + s_1 + s_2\bigr) \;\in\; [0, 1]
$$

Each $s_i$ equals $1$ iff the candidate's Betti matches the GT's on that axis, and decays smoothly as the count drifts in either direction, equivalent for non-negative integers to $(\min(b^{\text{cand}}, b^{\text{gt}})+1) / (\max(b^{\text{cand}}, b^{\text{gt}})+1)$. The $+1$ shift keeps the ratio finite when either Betti is zero (so "1 vs 0" is $1/2$, not undefined) and prevents "off by one near zero" from collapsing to a binary fail.

The score is **symmetric** in candidate / GT and the same on every axis, so a one-step drift on $b_0$, $b_1$, or $b_2$ contributes the same penalty.

Per-axis scores are persisted as `topology_metrics.per_axis_scores = {"b0": ..., "b1": ..., "b2": ...}` in `result.json` for diagnostics.

### Examples

For a few fixed GT counts, here is what the per-axis score looks like as the candidate count moves:

| $b^{\text{gt}}$ | $b^{\text{cand}}$ | $s_i$           |
| ---             | ---               | ---             |
| any             | $= b^{\text{gt}}$ | $1.000$         |
| $0$             | $1$               | $0.500$         |
| $0$             | $2$               | $\approx 0.333$ |
| $0$             | $5$               | $\approx 0.167$ |
| $4$             | $3$               | $0.800$         |
| $4$             | $5$               | $\approx 0.833$ |
| $4$             | $8$               | $\approx 0.556$ |
| $10$            | $8$               | $\approx 0.818$ |
| $10$            | $5$               | $\approx 0.545$ |

## What this metric does *not* test

- **Position of features.** A plate with one hole on the left and a plate with one hole on the right have identical Betti $(1, 1, 0)$. Feature *position* is covered by `interface_match` when sub-volume specs exist; otherwise it falls to `shape_similarity`.
- **Blind features.** Pockets, embossed text, fillets, chamfers; all topologically trivial. They affect `shape_similarity` and, where declared, `interface_match`. Topology score is invariant under them by construction.
- **Local correctness.** A torus and a thin tube around a stick both have $(1, 1, 0)$. Betti is invariant under arbitrary continuous deformation. Shape correctness is `shape_similarity`'s job.

## Code pointers

- Metric: [`src/cadgenbench/eval/topo_match.py`](../../src/cadgenbench/eval/topo_match.py)
- Mesh utilities (tessellate, unify, sanity-gate): [`src/cadgenbench/common/mesh.py`](../../src/cadgenbench/common/mesh.py)
- Validity gate: [`src/cadgenbench/common/validity.py`](../../src/cadgenbench/common/validity.py)
- Orchestrator: [`src/cadgenbench/eval/evaluate.py`](../../src/cadgenbench/eval/evaluate.py)
- Fixtures: [`tests/fixtures/topo_metric/`](../../tests/fixtures/topo_metric/)
