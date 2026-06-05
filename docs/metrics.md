# CADGenBench: CAD Score metrics overview

How CADGenBench scores one generated CAD part (a STEP file)
against one ground-truth STEP file. This document
is the canonical reference. Each metric is summarised here and
detailed in its own deep-dive section below.

---

## TL;DR

For one candidate against one GT:

1. **Validity gate.** If the candidate STEP isn't a valid, watertight,
   meshable solid, `cad_score = 0`.
2. Otherwise, `cad_score` is a **weighted** mean of three independent
   $[0, 1]$ metrics:

```
            0                                                              if not is_valid
cad_score =
            0.4·shape_similarity + 0.4·interface + 0.2·topology_match      otherwise
```

(This is the **generation** composition. **Editing** tasks renormalize
the shape axis against the no-op input and reweight differently. See
[§ Editing tasks](#editing-tasks-no-op-renormalization) below.)

| Component | Range | What it asks | Deep dive |
| --- | --- | --- | --- |
| CAD Validity (gate) | $\{0, 1\}$ | Is the geometry valid? | [deep dive](./metrics/cad_validity.md) |
| Shape Similarity | $[0, 1]$ | Does the bulk geometry match? | [deep dive](./metrics/shape_similarity.md) |
| Topology Match | $[0, 1]$ | Same components / holes / voids? | [deep dive](./metrics/topo_match.md) |
| Interface Match | $[0, 1]$ | Does it bolt up to the same fixture? | [deep dive](./metrics/interface_match.md) |

---

## Coordinate convention & alignment

We instruct submissions to centre their models at $(0, 0, 0)$ and give
rules for orientation (longest axis, mounting frame), but in any case
rigidly align the outputs to the GT before scoring. Alignment is rotation
+ translation only, never scale. The production aligner generates identity,
PCA multi-start, and Open3D FGR candidates, refines them with Open3D
multi-scale point-to-plane ICP, then selects the final pose by
downstream-like shape agreement (bidirectional F1, capped symmetric
Chamfer, RMSE) rather than ICP residual. Trusted mesh sidecars are aligned
in memory and are not re-tessellated.

---

## Composition

```
                  ┌──────────────────┐
                  │  candidate STEP  │
                  └─────────┬────────┘
                            │
                            ▼
                  ┌──────────────────┐    fail   ┌─────────────────┐
                  │   CAD Validity   ├──────────►│  cad_score = 0  │
                  │   (hard gate)    │           └─────────────────┘
                  └─────────┬────────┘
                            │ pass
                            ▼
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Shape          │  │ Topology       │  │ Interface      │
│ Similarity     │  │ Match          │  │ Match          │
│ (mean of 2     │  │ (Betti b₀b₁b₂  │  │ (per-group     │
│  sub-metrics)  │  │  agreement)    │  │  pose-searched │
│                │  │                │  │  IoU)          │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              weighted mean → cad_score ∈ [0, 1]
```

### Three orthogonal metrics

The three score components are orthogonal by construction. Each catches a class of error the others are blind to:

- **Shape Similarity** catches "wrong bulk geometry"; blind to topology (a torus and a thin loop pass the same IoU).
- **Topology Match** catches "wrong number of holes / pieces / voids"; blind to feature position (one hole left vs one hole right is identical).
- **Interface Match** catches "wrong feature position / size against spec"; blind to overall shape (the four bolt holes fit regardless of bracket appearance).

Validity is a gate, not a term.
---

## The four metrics at a glance

### 1. CAD Validity

A three-check hard gate run on the raw candidate STEP:

1. **BREP well-formedness**: Open CASCADE's `BRepCheck_Analyzer.IsValid()` reports no per-face / per-edge / per-vertex errors.
2. **Watertightness**: every shell is closed (no naked / free edges).
3. **Meshable as a closed orientable manifold**: tessellates to a mesh that is manifold, closed ($3F = 2E$), and orientation-consistent.

Any failure ⇒ `is_valid = False` ⇒ `cad_score = 0`, with a human-readable list of failing checks.

→ See [more details below](./metrics/cad_validity.md).

### 2. Shape Similarity

Arithmetic mean of two sub-metrics, each in $[0, 1]$:

```
shape_similarity = mean(point_cloud_f1, volume_iou)
```

- **`shape_point_cloud_f1`**: normal-weighted symmetric F1 of 50 k surface points per shape; hit requires distance within 0.5 % of GT bbox diagonal *and* matched-pair normals within ≈20°.
- **`shape_volume_iou`**: $\mathrm{vol}(A \cap B) / \mathrm{vol}(A \cup B)$.

The two cancel each other's blind spots. Point-cloud F1 is dominated by big flat faces and is sensitive to surface position; volume IoU is invariant to feature position but captures occupied-volume error.

→ See [more details below](./metrics/shape_similarity.md).

### 3. Topology Match

Three integer invariants of the candidate solid (the **Betti numbers** of its boundary, i.e. coordinate-free counts of topological features), computed on the tessellated mesh rather than the BREP:

- **$b_0$**: number of connected solid components.
- **$b_1$**: number of independent through-handles (through-holes).
- **$b_2$**: number of enclosed internal voids.

Per axis, a fuzzy log-ratio against the GT:

$$
s_i = \exp\!\Bigl(-\bigl|\log\bigl((b_i^{\text{cand}}+1)/(b_i^{\text{gt}}+1)\bigr)\bigr|\Bigr) \in [0, 1]
$$

Score: $s_0 \cdot s_1 \cdot s_2 \in [0, 1]$ (the **product**, not the mean). The $+1$ shift keeps the ratio finite when either Betti is zero and gives "off by one near zero" graceful (rather than catastrophic) decay; the per-axis scores are persisted alongside the aggregate for diagnostics. The product means a single badly-wrong axis collapses the score toward $0$. Topology is discrete, so getting two of three invariants right is not a partial match.

Topologically *trivial* features (blind pockets, fillets, chamfers, embossed text) leave Betti unchanged. They are covered by [Shape Similarity](#2-shape-similarity) and [Interface Match](#4-interface-match) instead.

→ See [more details below](./metrics/topo_match.md).

### 4. Interface Match

A part has one or more **mating groups**: sets of features that must align rigidly with another object (e.g. the four bolt holes on one mounting face). Each group is specified as one or more **sub-volume STEPs** (cylinders for round holes, prisms for hex bosses, stadium-prisms for slots), each labelled either `KOR` (keep-out region: candidate must be empty here) or `KIR` (keep-in region: candidate must have material here).

Per sub-volume we compute a volumetric IoU against the candidate, with an asymmetric verification shell of opposite-material around the region (so oversize *and* undersize errors both register). Each IoU then passes through a soft pass/fail ramp ($\text{IoU} \ge 0.95 \to 1$, $\le 0.80 \to 0$, linear between) so sloppy fits go to $0$ rather than banking partial credit. Per-group score is the min over its (ramped) sub-volumes; per-fixture score is the mean over groups.

→ See [more details below](./metrics/interface_match.md).

---

## Editing tasks: no-op renormalization

Most fixtures are **generation** tasks and use the composition above
unchanged. **Editing** tasks (an `input.step` plus an edit request,
`task_type: editing`) need one adjustment.

The problem: an editing GT is a small, local modification of the
input, so the unedited input is already a valid solid that is *almost*
the GT. All three scored axes are **global** similarity measures, so
the "no-op" strategy (submit the input unchanged) scores high and often
higher than a real attempt that perturbs the unchanged bulk. Scoring
editing tasks with the raw composition would reward doing nothing.

The fix anchors the **shape-similarity axis** against the no-op:

```
b_shape   = shape_similarity(input.step, GT)          # the no-op's raw score
s_renorm  = max(0, (shape_similarity − b_shape) / (1 − b_shape))
```

`b_shape` (the no-op) maps to `0`; a perfect candidate stays at `1`;
anything at or below the no-op floors at `0`. **Topology and interface
match stay raw**. Most edits leave them unchanged (so they
contribute equally to every candidate), and where an edit *does* move
them they already discriminate, and a candidate that *breaks* them
should still be penalized.

For editing fixtures the per-fixture score is a **weighted** mean
(shape is the axis that actually resolves most edits, so it dominates),
with absent axes dropping out and the remaining weights renormalizing:

```
            0                                                   if not is_valid
cad_score =
            0.6·s_renorm + 0.3·interface + 0.1·topo_match        otherwise
```

Editing weights differ from generation (which uses 0.4 / 0.4 / 0.2):
shape dominates at 0.6 because it is the axis that actually resolves most
edits, while topology and interface are frequently non-discriminating on
a given edit (the edit rarely changes Betti numbers or mating fit) and so
are nearly "free" for the no-op, which only the shape axis is renormalized
against. They are kept small (0.3 / 0.1) so the no-op caps low: a no-op
scores at most `0.6·0 + 0.3 + 0.1 = 0.4` (and less when the edit moves
topology/interface, since the no-op then misses those too); any genuine
shape improvement clears it. The validity gate still hard-zeros as for
generation.

**`b_shape` depends only on `input.step`, `ground_truth.step`, and the
shape/alignment implementation. It is precomputed once at authoring
time pre input, and committed to the GT dataset as `<fixture>/edit_baseline.json`;
the grader reads it back and never recomputes it per submission. The
**presence** of that file is also how the grader knows a fixture is an
editing task. See the authoring doc in the GT dataset
(`AUTHORING.md`) for the precompute + the headroom gate that rejects
edits too small for the shape metric to resolve.

Implementation: [`src/cadgenbench/eval/edit_baseline.py`](../src/cadgenbench/eval/edit_baseline.py),
wired into `_cad_score` in [`evaluate.py`](../src/cadgenbench/eval/evaluate.py).
The renormalized + raw shape values are persisted under
`result.json["edit_metrics"]` for the report and debug panel.

---

## Worked examples

Three minimal examples, one per axis. Each is built so that a single
metric is what separates the candidate from the ground truth while the
other two stay quiet, which is what it means for the three to be
orthogonal.

### Example 1: Shape Similarity catches wrong bulk geometry

Two single-piece parts with no holes and no mating features, so
topology and interface have nothing to disagree about and only the
bulk geometry is in play. The ground truth is an L-bracket; the
candidate is a plain block of roughly the same footprint. They enclose
different volumes and present different surfaces, so **shape
similarity** is the axis that notices the candidate simply isn't the
right shape.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_1_shape/gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_1_shape/candidate_iso.png) |

### Example 2: Topology Match catches a wrong feature count

Topology counts the pieces, through-handles, and internal voids. Two
cases, each changing one count while the bulk shape barely moves, so
shape similarity stays high and the eye is easily fooled.

A wrong number of through-holes: two bars of the same size and outline,
where the ground truth has two through-holes and the candidate has four.
The Betti vectors are `(1, 2, 0)` and `(1, 4, 0)`, so only the
through-handle count $b_1$ differs and scores `0.60`; the topology match
is the product `1.00 · 0.60 · 1.00 = 0.60`.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_2_topology/gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_2_topology/candidate_iso.png) |

A wrong number of pieces: the ground truth is one solid bar, the
candidate came out as two disconnected blocks. Now the component count
$b_0$ differs, `(1, 0, 0)` against `(2, 0, 0)`, and scores `0.667`; the
topology match is `0.667 · 1.00 · 1.00 = 0.667`.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_2_topology/components_gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_2_topology/components_candidate_iso.png) |

The three axis scores are multiplied, not averaged, so one wrong count
pulls the whole match down and a candidate cannot bank credit for the
two easy axes. Interface match aggregates differently: it takes the
worst feature inside a mating group and averages across independent
groups, shown next.

### Example 3: Interface Match catches a misplaced mating feature

A mounting plate with two bolt holes and a central slot. The candidate
keeps the outline, the hole count, and the slot, but the slot is shifted
off its specified position, so the plate would not seat on its fixture.
The bulk shape barely moves and the feature counts do not change, so
shape similarity (`0.89`) and topology (`1.00`) both stay high. Only
interface match sees the problem and scores `0.67`: the two holes mate,
but the offset slot fails its group. The result is
`cad_score = 0.4 · 0.89 + 0.4 · 0.67 + 0.2 · 1.00 = 0.82`.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_3_interface/gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_3_interface/candidate_iso.png) |

The overlay makes it concrete. Each keep-out region the spec requires to
be empty is drawn in red. The two holes line up, but where the
candidate's material intrudes into the slot region the disagreement
lights up in yellow.

![Interface overlay](./metrics/illustrations/example_3_interface/interface_overlay.png)
