# Shape Similarity

Scores whether a candidate part's bulk geometry matches the ground truth, independently of topology and mating-interface specs. Two sub-metrics, each in $[0, 1]$, averaged into a single `shape_similarity_score`.

## What we check

| Sub-metric                | What it measures                                      | Best at catching                                       |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| `shape_point_cloud_f1`    | Normal-weighted surface-point Chamfer F1              | Bulk shape drift, missing or extra bulk material       |
| `shape_volume_iou`        | Volumetric overlap of the solids                      | Wrong size / scale, gross volume mismatch              |

$$
\text{shape\_similarity\_score} \;=\; \tfrac{1}{2}\bigl(\text{shape\_point\_cloud\_f1} + \text{shape\_volume\_iou}\bigr)
$$

The two sub-metrics cancel each other's blind spots. Point-cloud F1 is dominated by big flat faces and is sensitive to surface position. Volume IoU is invariant to feature position but captures size / scale error; on its own it is degenerate (two differently-shaped parts with similar bulk that overlap can score near 1), which point-cloud F1 catches. The mean of the two is the simplest signal that both agree.

F1 and IoU are preferred over a mean Chamfer distance because they are rate-based: each point either agrees within tolerance or doesn't, so a single outlier moves the score. A mean distance smooths outliers out and saturates.

## Pipeline

For one candidate against one GT:

1. **Rigid alignment.** ICP-align the candidate to the GT ([`alignment.py`](../../src/cadgenbench/eval/alignment.py)). Persists `aligned/output_aligned.step` and the RMSE; both are reused on re-runs.
2. **Tessellate.** Both shapes are meshed at a shared deflection derived from the GT bounding-box diagonal so candidate and GT live at one scale (gate-validated closed orientable manifold, see [`mesh.py`](../../src/cadgenbench/common/mesh.py)).
3. **Compute the two sub-metrics** (below).
4. **Average them** into `shape_similarity_score`.

---

### Sub-metric 1: `shape_point_cloud_f1`

Symmetric normal-weighted Chamfer F1 over surface point clouds.

50 000 points are area-weighted-sampled from each welded mesh; each sample carries the outward unit normal of its source triangle. A candidate point is a **hit** when both conditions hold:

1. Nearest-neighbour distance on the GT cloud is within $\tau_{\text{pc}} = \max(10^{-6},\ 0.01 \cdot \mathrm{diag}(\mathrm{bbox}_{\text{GT}}))$, i.e. 1 % of the GT bounding-box diagonal.
2. Outward unit normals satisfy $n_{\text{cand}} \cdot n_{\text{gt}} > 0.9$ (≈25° tolerance).

The same two-gate definition applies in the reverse direction. Then

$$
\text{precision} = \frac{\#\{p \in C : \text{hits GT}\}}{|C|}
\qquad
\text{recall} = \frac{\#\{q \in G : \text{hits candidate}\}}{|G|}
\qquad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

The normal gate rejects "right place, wrong side" matches: the back face of a thin wall, a flipped-orientation candidate, or two surfaces brushing past each other through a hole. It promotes the metric from "points are nearby" to "the same surface is nearby".

The 1 %-of-diagonal threshold scales naturally from a 10 mm rivet to a 1 m bracket; a fixed-mm tolerance does not.

### Sub-metric 2: `shape_volume_iou`

$$
\mathrm{IoU} = \frac{\mathrm{vol}(A \cap B)}{\mathrm{vol}(A \cup B)}
$$

Computed on the welded meshes via the [`manifold3d`](https://github.com/elalish/manifold) Boolean kernel. The union volume uses inclusion-exclusion: $|A \cup B| = |A| + |B| - |A \cap B|$.

Volume IoU is the one sub-metric a layperson can interpret without explanation ("what fraction of the GT solid is in the right place"). It's invariant to face decomposition, sampling seed, and view choice. On its own it's degenerate, two differently-shaped parts with similar bulk that happen to overlap can score near 1; point-cloud F1 catches that.

---

## Editing tasks: renormalization against the no-op

On an **editing** fixture the unedited input is already a valid solid
that is *almost* the GT, so this axis (a global similarity measure)
scores the no-op high. For editing fixtures the `shape_similarity_score`
is therefore renormalized against the no-op baseline
`b_shape = shape_similarity(input.step, GT)` before it enters
`cad_score`: `s_renorm = max(0, (s − b_shape) / (1 − b_shape))`, so the
no-op maps to `0` and a perfect candidate to `1`. The two sub-metrics
above are computed and reported unchanged — the renormalization happens
on their mean. See [`../metrics.md`](../metrics.md) § *Editing tasks* and
[`edit_baseline.py`](../../src/cadgenbench/eval/edit_baseline.py).

## What this metric does *not* test

- **Sub-threshold features.** Both sub-metrics use thresholds proportional to the GT bounding box (1 % of bbox diagonal). A $\varnothing\,3$ mm hole on a 200 mm bracket sits well below that and can be misplaced by ~1 mm without penalty. Small mating features are scored via [`interface_match.md`](./interface_match.md), where the verification region is sized to the feature, not the part.
- **Tolerance-grade precision.** Precision and recall saturate as the candidate gets closer than the threshold. The metric measures "looks right at the scale of the part", not "hits a 50 µm spec".

## Code pointers

- Metric implementation: [`shape_similarity.py`](../../src/cadgenbench/eval/shape_similarity.py)
  - `shape_point_cloud_f1`: `_point_cloud_f1_stats`
  - `shape_volume_iou`: `_volume_overlap_stats`
- Alignment: [`alignment.py`](../../src/cadgenbench/eval/alignment.py)
- Point sampling: [`sampling.py`](../../src/cadgenbench/eval/sampling.py)
- Orchestrator: [`evaluate.py`](../../src/cadgenbench/eval/evaluate.py)
