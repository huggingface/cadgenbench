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

## Pipeline

Betti numbers are computed once on the tessellated  mesh. The [validity gate](./cad_validity.md) has already confirmed the mesh is a watertight, manifold, orientation-consistent triangle surface. Given that mesh:

1. **Connected components.** Union-find on triangle adjacency; each component is a closed 2-manifold.
2. **Containment ($b_0$, $b_2$).** For each component, take an interior point and count how many *other* components enclose it by ray casting. Enclosed an even number of times means the outer shell of a solid and counts toward $b_0$; odd means the inner shell of a void and counts toward $b_2$.
3. **Handles ($b_1$).** From the surface Euler characteristic $\chi = V - E + F$ and the identity $\chi = 2(b_0 - b_1 + b_2)$, solve $b_1 = b_0 + b_2 - \chi/2$.

## Score

Each Betti axis gets a fuzzy log-ratio against the GT, raised to a sharpness exponent $\alpha = 2$ so a wrong count counts as a real defect rather than a near miss:

$$
s_i = \exp\left(-\alpha\left|\log\frac{b_i^{\text{cand}}+1}{b_i^{\text{gt}}+1}\right|\right) = \left(\frac{\min+1}{\max+1}\right)^{\alpha} \in [0, 1]
$$

It is $1$ when the counts match and decays smoothly otherwise; the $+1$ shift keeps it finite when a count is zero. The aggregate is the **product** over the three axes:

$$
s_0 \cdot s_1 \cdot s_2 \in [0, 1]
$$

The product, not the mean, means one wrong axis collapses the score: topology is discrete, so getting two of three invariants right is not a partial match. Per-axis scores are saved under `topology_metrics.per_axis_scores` in `result.json`.

Two examples:

- **Wrong number of pieces.** GT `(1, 0, 0)`, candidate `(2, 0, 0)`: only $b_0$ differs, so $s_0 = (2/3)^2 = 0.444$ and the topology match is `0.444`.
- **Wrong number of through-holes.** GT `(1, 2, 0)`, candidate `(1, 4, 0)`: only $b_1$ differs, so $s_1 = (3/5)^2 = 0.360$ and the topology match is `0.360`.

## What this metric does *not* test

- **Position of features.** A plate with one hole on the left and a plate with one hole on the right have identical Betti $(1, 1, 0)$. Feature *position* is covered by `interface_match` when sub-volume specs exist; otherwise it falls to `shape_similarity`.
