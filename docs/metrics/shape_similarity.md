# Shape Similarity

Scores whether a candidate's bulk geometry matches the ground truth, independently of topology and mating interfaces. It is the mean of two sub-metrics, each in $[0, 1]$:

$$
\text{shape similarity} \;=\; \tfrac{1}{2}\,\bigl(\text{point-cloud F1} + \text{volume IoU}\bigr)
$$

The two are complementary: point-cloud F1 measures surface placement, volume IoU measures occupied volume, and a candidate has to satisfy both to score well.

## Point-cloud F1

Checks that the candidate's surface lies where the ground truth's does and faces the same way. We sample points across both surfaces, each tagged with its outward normal, and call a point matched when the nearest point on the other surface is within 0.5% of the GT bounding-box diagonal **and** the two normals agree to within 20°. Precision (matched candidate points) and recall (matched GT points) combine into the F1 score (`shape_point_cloud_f1`).

## Volume IoU

The shared volume of the two solids divided by their combined volume (intersection over union), computed with the [`manifold3d`](https://github.com/elalish/manifold) Boolean kernel and reported as `shape_volume_iou`.

## What this metric does not test

- **Small features.** Both sub-metrics use a tolerance proportional to part size, so a small hole or boss can be misplaced without moving the score. Those are scored by [interface match](./interface_match.md), whose region is sized to the feature.
