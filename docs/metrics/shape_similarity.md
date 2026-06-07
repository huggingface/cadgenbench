# Shape Similarity

Scores whether a candidate's bulk geometry matches the ground truth, independently of topology and mating interfaces. It is the mean of two sub-metrics, each in $[0, 1]$:

$$
\text{shape similarity} = \frac{1}{2}\left(\text{surface distance F1} + \text{volume IoU}\right)
$$

The two are complementary: surface distance F1 measures surface placement, volume IoU measures occupied volume, and a candidate has to satisfy both to score well.

## Surface Distance F1

Checks that the candidate's surface lies where the ground truth's does and faces the same way. We sample points across both surfaces, each tagged with its outward normal, and call a point matched when the **closest point on the other mesh's surface** is within 0.5% of the GT bounding-box diagonal **and** the surface normals there agree to within 20°. Matching against the surface (not the other cloud's nearest sample) makes the score exact on a perfect match and robust to tessellation and small deformations. Precision (matched candidate points) and recall (matched GT points) combine into the F1 score (`shape_surface_distance_f1`).

## Volume IoU

The shared volume of the two solids divided by their combined volume (intersection over union), computed with the [`manifold3d`](https://github.com/elalish/manifold) Boolean kernel and reported as `shape_volume_iou`.

## What this metric does not test

- **Small features.** Both sub-metrics use a tolerance proportional to part size, so a small hole or boss can be misplaced without moving the score. Those are scored by [interface match](./interface_match.md), whose region is sized to the feature.
