# Interface Match

Scores whether a candidate part's mating interfaces match the ground truth's specification.

## What we check

A part has one or more **mating interfaces**: regions that mate with another object (bolt holes, sockets, bosses, slots, pockets).

Interfaces that must align together rigidly (for example the four bolt holes on a single mounting face, or a bolt pattern together with a boss on the same bracket) belong to one **mating group**. Independent interfaces live in separate mating groups and are scored separately, with no enforced relationship between their poses.

We score each interface **volumetrically**: the region the interface specifies must match the candidate's free space (for holes / slots / pockets) or solid (for bosses / protrusions) in shape, size, and position. Volume is the simplest single signal that captures all three.

### In scope

- Round, slotted, and hex holes
- Internal pockets and recesses
- External bosses and protrusions

### Out of scope

Flatness, parallelism, perpendicularity, position tolerance, datum reference frames, surface finish, threading. These require GD&T annotations that the metric does not ingest.

## Notation

- **`R`**: the canonical reference region of one interface (a cylindrical region for a round hole or pin, a prism for a hex boss, a stadium prism for a slot, and so on).
- **`fit_type ∈ {KOR, KIR}`**: `KOR` (keep-out region) if the candidate's solid must be absent in `R` (holes, slots, pockets); `KIR` (keep-in region) if it must be present (bosses, protrusions).
- **`group_id`**: 1-indexed integer identifying a mating group. Sub-volumes sharing the same `group_id` are pose-searched together; sub-volumes in different groups are scored independently.

## File layout per fixture

```
test_N/
├── gt.step
├── jig_<group_id>__<index>__<fit_type>.step           # one file per sub-volume
├── jig_<group_id>__<index>__<fit_type>.step
└── ...
```

Each fixture contains:

1. `gt.step`: the ground-truth part. `gt.step` itself doubles as the "correct candidate" for the metric.
2. One or more sub-volume STEP files. Each contains the solid geometry of one interface region `R`, positioned at the absolute GT-specified pose. No auxiliary files (no main jig STEP, no YAML, no parametric annotation): the STEP is the complete description.

### Filename

```
jig_<group_id>__<index>__<fit_type>.step
   │                  │         │
   │                  │         └── one of {KOR, KIR}
   │                  └── 1-indexed integer within group_id (sub-volume index)
   └── 1-indexed integer
```

All sub-volume STEP files share the ground-truth coordinate frame and are positioned at their absolute GT-specified pose.

### Examples

A bracket whose bolt pattern and boss must align together (single mating group, mixed fit types):

```
test_3/
├── gt.step
├── jig_1__1__KOR.step
├── jig_1__2__KOR.step
├── jig_1__3__KOR.step
├── jig_1__4__KOR.step
└── jig_1__5__KIR.step
```

A plate with three mechanically independent interfaces (three mating groups):

```
test_4/
├── gt.step
├── jig_1__1__KOR.step
├── jig_2__1__KOR.step
└── jig_3__1__KOR.step
```

## Metric

For each mating group independently:

1. **Rigid alignment.** Align the candidate to the GT with the production
   rotation + translation aligner
   ([`src/cadgenbench/eval/alignment.py`](../../src/cadgenbench/eval/alignment.py)).

2. **Bounded pose search.** The GT is perfectly labelled by construction; the
   candidate is inferred and still carries a small residual after rigid bulk
   alignment. The IoU should reflect feature fit, not that residual. So the
   group's GT-specified pose is perturbed by ±1° per axis and ±1 % of the GT
   bounding-box diagonal per translation axis (e.g. ±1 mm on a 100 mm part,
   ±5 mm on a 500 mm part), and 32 poses are sampled by default. All
   sub-volumes in the group move together. The zero-perturbation pose is always
   sampled, so the per-sub-volume IoU is monotone in the search budget. The
   sampler is a deterministic Sobol low-discrepancy sequence (no random
   component, no seed). Feature correspondences are ambiguous on symmetric bolt
   patterns, missing features, and mixed interfaces; bulk alignment optimizes
   whole-shape agreement rather than interface fit; and any correspondence-based
   variant would couple the metric to a BREP face structure we don't want to
   assume. A future pass could replace the Sobol scan with a local optimiser
   (Nelder-Mead / Powell) on the IoU surface itself.

3. **Per-sub-volume IoU.** For each sub-volume `R` in the group, the candidate region `C` is measured inside a verification region `bbox_R` made of `R` itself plus a thin shell of opposite-material around it:

   - `KOR` (hole): the shell is the GT material immediately around the hole. Oversize candidate holes eat into the shell; undersize holes leave the shell intact but shrink `C`. Both lower the IoU.
   - `KIR` (boss): the shell is the empty air immediately outside the boss. Oversize candidate bosses bulge into the shell; undersize ones shrink `C`. Both lower the IoU.

   The shell is built by inflating `AABB(R)` outward by `margin_M` and intersecting with either `GT.solid` (KOR) or its complement (KIR). This catches oversize and undersize equally well, and closes the "place a KOR cylinder in empty space and trivially pass" shortcut: a candidate that omits the surrounding material loses the shell from its IoU even though it nominally has no solid inside `R`.

   ```
   margin_M    = max(2.0 mm, 0.20 * longest_extent(R))
   inflated_R  = AABB(R) inflated outward by margin_M

   shell       = inflated_R ∩ GT.solid          # for fit_type=KOR
   shell       = inflated_R  \  GT.solid        # for fit_type=KIR
   bbox_R      = R ∪ shell                      # the verification region

   C_KOR       = bbox_R  \  candidate_solid     # for fit_type=KOR
   C_KIR       = bbox_R  ∩  candidate_solid     # for fit_type=KIR
   C           = C_KOR  or  C_KIR               # selected by fit_type

   IoU         = vol(R ∩ C) / vol(R ∪ C)
   ```

   During bounded pose search, `bbox_R` stays fixed at the GT pose; only `R` is perturbed. This avoids edge-of-part artifacts where a moved `bbox_R` can create artificial free space outside the GT part.

   The per-sub-volume IoU is saturated to 1.0 above a threshold of 0.99 before aggregation (rationale in the main metrics doc, [Design notes § Why mesh-based computation](../metrics.md#why-mesh-based-computation-with-iou-saturation)). The pose search returns the pose maximising the per-sub-volume IoU.

4. **Soft pass/fail ramp.** Each sub-volume's pose-searched IoU is mapped through a ramp before aggregation: $\text{IoU} \ge 0.95 \to 1.0$, $\text{IoU} \le 0.80 \to 0.0$, linear in between. A mating feature either fits within tolerance or it doesn't, so this crushes sloppy fits toward $0$ instead of letting a feature that is only ~0.85 IoU bank most of the credit. The $0.80$ floor sits comfortably above tessellation noise, so a genuinely good fit is never zeroed by accident. The raw (pre-ramp) IoU is still reported per sub-volume for diagnostics; only the aggregate score is ramped.

5. **Per-group score.** Minimum ramped score across the group's sub-volumes.

6. **Per-fixture score.** Mean of per-group scores. The mean (not the min) gives partial credit when a part has multiple *mechanically independent* mating groups: a candidate that nails one group and misses another scores above zero, but a candidate that breaks a single composite group still drops the score sharply because the inner aggregation is `min`.

## What this metric does *not* test

- **Functional fit vs. specification fit.** The metric scores geometric match to `R`. A 30 × 14 slot when the specification is 30 × 12 still accepts the same bolt, but the IoU drops because geometry deviates from spec.
- **Sliding or motion behaviour.** Only fit at the GT-specified pose (with the bounded pose-search budget) is scored.
- **Threading or fit class** (H7/g6 and similar mechanical-fit designations).

## Labeller brief

Short version of what a labeller delivers. The full canonical contract (file names, canonical pose, sanity-check tooling) lives in `AUTHORING.md` in the [`cadgenbench-data-gt`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data-gt) dataset alongside the GT artefacts; this section is the metric-specific subset for readers who jump in here first.

For each part the labeller delivers:

1. `ground_truth.step`: the ground-truth part, in canonical pose (bbox centroid at the origin, bbox extents ordered $L_x \ge L_y \ge L_z$; see `AUTHORING.md` § *Canonical pose* in [`cadgenbench-data-gt`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data-gt)).
2. For every mating group, one or more sub-volume STEP files named `jig_<group_id>__<index>__<fit_type>.step`, with:
   - `group_id` a 1-indexed integer. Sub-volumes sharing the same `group_id` must move together rigidly during pose search.
   - `index` a 1-indexed integer within that group.
   - `fit_type` exactly `KOR` (keep-out region: candidate must be empty here) or `KIR` (keep-in region: candidate must have solid material here).
3. All sub-volume STEP files share the ground-truth coordinate frame and are positioned at their absolute GT-specified pose. This is the *only* "rigid-body annotation": the pose lives inside the sub-volume STEP itself.

The internal authoring workflow is unconstrained. A common approach is to model a full mating fixture in CAD (base plate, pins, sockets, etc.) and export each interface region as its own STEP file at the end. The delivered artifacts are `ground_truth.step` plus the per-sub-volume STEPs.

## Code pointers

- Metric: [`src/cadgenbench/eval/interface_match.py`](../../src/cadgenbench/eval/interface_match.py)
- Orchestrator: [`src/cadgenbench/eval/evaluate.py`](../../src/cadgenbench/eval/evaluate.py)
- Fixtures: [`tests/fixtures/jig_metric/`](../../tests/fixtures/jig_metric/)
- Fixture generator: [`tests/fixtures/jig_metric/generate.py`](../../tests/fixtures/jig_metric/generate.py)
