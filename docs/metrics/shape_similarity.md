# Shape Similarity

Scores whether a candidate's bulk geometry matches the ground truth, independently of topology and mating interfaces. It is the mean of two sub-metrics, each in $[0, 1]$:

$$
\text{shape\_similarity\_score} \;=\; \tfrac{1}{2}\bigl(\text{shape\_point\_cloud\_f1} + \text{shape\_volume\_iou}\bigr)
$$

The two cover each other's blind spots. Point-cloud F1 is sensitive to where surfaces sit but is dominated by large flat faces; volume IoU captures size and gross volume error but is near-degenerate alone (two different shapes with similar bulk can still overlap well). Both have to agree for a high score.

## `shape_point_cloud_f1`

A symmetric, normal-weighted Chamfer F1 over surface point clouds. Points are area-weighted sampled from each mesh, each carrying its triangle's outward normal. A point is a **hit** when the nearest point on the other cloud is within 0.5% of the GT bounding-box diagonal **and** the two normals agree to within 20°. Precision (candidate points that hit) and recall (GT points that hit) combine into the F1.

The distance threshold scales with part size, and the normal gate rejects "right place, wrong side" matches such as the back face of a thin wall or a flipped part.

## `shape_volume_iou`

$$
\mathrm{IoU} \;=\; \frac{\mathrm{vol}(A \cap B)}{\mathrm{vol}(A \cup B)}
$$

The fraction of the combined volume the two solids share, computed with the [`manifold3d`](https://github.com/elalish/manifold) Boolean kernel. It is the easiest component to read directly: how much of the GT solid the candidate gets in the right place.

## What this metric does not test

- **Small features.** Both sub-metrics use a tolerance proportional to part size, so a small hole or boss can be misplaced without moving the score. Those are scored by [interface match](./interface_match.md), whose region is sized to the feature.
