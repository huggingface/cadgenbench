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

Topology is weighted down (0.2): it is comparatively easy to score well
on these parts, so it should not carry a full third of the headline.
Absent axes drop out and the remaining weights renormalize (e.g. a
fixture with no interface spec scores `(0.4·shape + 0.2·topology) / 0.6`).

(This is the **generation** composition. **Editing** tasks renormalize
the shape axis against the no-op input and reweight differently — see
[§ Editing tasks](#editing-tasks-no-op-renormalization) below.)

| Component | Range | What it asks | Deep dive |
| --- | --- | --- | --- |
| CAD Validity (gate) | $\{0, 1\}$ | Is the geometry valid? | [deep dive](./metrics/cad_validity.md) |
| Shape Similarity | $[0, 1]$ | Does the bulk geometry match? | [deep dive](./metrics/shape_similarity.md) |
| Topology Match | $[0, 1]$ | Same components / holes / voids? | [deep dive](./metrics/topo_match.md) |
| Interface Match | $[0, 1]$ | Does it bolt up to the same fixture? | [deep dive](./metrics/interface_match.md) |

---

## What CAD Score does *not* measure

- **GD&T / PMI annotations**: surface finish, tolerance frames, datum references, threading.
- **Functional fit vs spec fit**: a 30 × 14 slot still accepts a bolt the spec wants in a 30 × 12 slot; we score geometry, not function.
- **Manufacturability**: machine choice and DfM cost are not modelled.

---

## Coordinate convention & alignment

We instruct submissions to centre their models at $(0, 0, 0)$ and give rules for orientation (longest axis, mounting frame), but in any case rigidly align the outputs to the GT before scoring: PCA on surface-point clouds with all 24 octahedral rotation candidates, then ICP refinement.

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

### Why validity is a gate, not a term

A scoring scheme that lets an invalid solid earn partial credit rewards "looks roughly right in a viewer" over "is a real 3D-printable part". Anything less than `is_valid` is hard-zeroed.

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

- **`shape_point_cloud_f1`**: normal-weighted symmetric F1 of 50 k surface points per shape; hit requires distance within 1 % of GT bbox diagonal *and* matched-pair normals within ≈25°.
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

Score: $\tfrac{1}{3}(s_0 + s_1 + s_2) \in [0, 1]$. The $+1$ shift keeps the ratio finite when either Betti is zero and gives "off by one near zero" graceful (rather than catastrophic) decay; the per-axis scores are persisted alongside the aggregate for diagnostics.

Topologically *trivial* features (blind pockets, fillets, chamfers, embossed text) leave Betti unchanged. They are covered by [Shape Similarity](#2-shape-similarity) and [Interface Match](#4-interface-match) instead.

→ See [more details below](./metrics/topo_match.md).

### 4. Interface Match

A part has one or more **mating groups**: sets of features that must align rigidly with another object (e.g. the four bolt holes on one mounting face). Each group is specified as one or more **sub-volume STEPs** (cylinders for round holes, prisms for hex bosses, stadium-prisms for slots), each labelled either `KOR` (keep-out region: candidate must be empty here) or `KIR` (keep-in region: candidate must have material here).

Per sub-volume we compute a volumetric IoU against the candidate, with an asymmetric verification shell of opposite-material around the region (so oversize *and* undersize errors both register). Per-group score is the min over its sub-volumes; per-fixture score is the mean over groups.

→ See [more details below](./metrics/interface_match.md).

---

## Editing tasks: no-op renormalization

Most fixtures are **generation** tasks and use the composition above
unchanged. **Editing** tasks (an `input.step` plus an edit request,
`task_type: editing`) need one adjustment.

The problem: an editing GT is a small, local modification of the
input, so the unedited input is already a valid solid that is *almost*
the GT. All three scored axes are **global** similarity measures, so
the "no-op" strategy (submit the input unchanged) scores high — often
higher than a real attempt that perturbs the unchanged bulk. Scoring
editing tasks with the raw composition would reward doing nothing.

The fix anchors the **shape-similarity axis** against the no-op:

```
b_shape   = shape_similarity(input.step, GT)          # the no-op's raw score
s_renorm  = max(0, (shape_similarity − b_shape) / (1 − b_shape))
```

`b_shape` (the no-op) maps to `0`; a perfect candidate stays at `1`;
anything at or below the no-op floors at `0`. **Topology and interface
match stay raw** — most edits leave them unchanged (so they
contribute equally to every candidate), and where an edit *does* move
them they already discriminate, and a candidate that *breaks* them
should still be penalized.

For editing fixtures the per-fixture score is a **weighted** mean
(shape is the axis that actually resolves most edits, so it dominates),
with absent axes dropping out and the remaining weights renormalizing:

```
            0                                                   if not is_valid
cad_score =
            0.5·s_renorm + 0.3·interface + 0.2·topo_match        otherwise
```

Editing weights differ from generation (which uses 0.4 / 0.4 / 0.2):
shape dominates at 0.5 because it is the axis that actually resolves most
edits, while topology and interface are frequently non-discriminating on
a given edit (the edit rarely changes Betti numbers or mating fit), so
topology is toned down to 0.2. A no-op therefore scores at most
`0.5·0 + 0.3 + 0.2 = 0.5` (and less when the edit moves
topology/interface, since the no-op then misses those too); any genuine
shape improvement clears it. The validity gate still hard-zeros as for
generation.

**`b_shape` is a fixture constant**, not a per-submission quantity: it
depends only on `input.step`, `ground_truth.step`, and the
shape/alignment implementation. It is precomputed once at authoring
time and committed to the GT dataset as `<fixture>/edit_baseline.json`;
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

### Example 1: Shape Similarity drives the score

This fixture has no interface specification, so the interface term drops out and the remaining weights renormalize: `cad_score = (0.4·shape_similarity + 0.2·topology_match) / 0.6`. The candidate captures the rough silhouette of the bracket but misplaces nearly every feature.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_1_shape/gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_1_shape/candidate_iso.png) |

| Component | Value | Notes |
| --- | --- | --- |
| Validity | ✅ valid, watertight, single solid | gate passes |
| `shape_point_cloud_f1` | `0.672` | bulk surface roughly right |
| `shape_volume_iou` | `0.558` | candidate is ~30 % off in occupied volume |
| **`shape_similarity_score`** | **`0.615`** | mean of the two sub-metrics |
| `topology_match` | `0.939` | $s_0 = 1.000$ ($b_0$ match), $s_1 = 0.818$ ($b_1 = 8$ vs GT $10$ → $9/11$), $s_2 = 1.000$ ($b_2$ match) |
| **`cad_score`** | **`0.723`** | $(0.4 \cdot 0.615 + 0.2 \cdot 0.939) / 0.6$ (no interface axis) |

### Example 2: Interface Match catches what Shape Similarity misses

A perfect topology match, a near-perfect shape match, and yet the candidate would not fit the GT's bolt pattern. All four mounting holes are present and individually look fine, but their *positions* are systematically off by a few millimetres.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_2_interface/gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_2_interface/candidate_iso.png) |

Visually indistinguishable. The interface-match overlay makes the problem obvious: GT sub-volumes (yellow / orange) sit next to, not on top of, the candidate's actual holes (black) in the top view.

![Interface overlay](./metrics/illustrations/example_2_interface/interface_overlay.png)

| Component | Value | Notes |
| --- | --- | --- |
| Validity | ✅ valid, watertight | gate passes |
| `shape_similarity_score` | `0.931` | pc-F1 0.944, vol-IoU 0.918 |
| `topology_match` | `1.000` | $(b_0, b_1, b_2) = (1, 4, 0)$ exact match |
| `interface_match` | **`0.0915`** | per-group min over the four `KOR` sub-volume IoUs: 0.092, 0.103, 0.098, 0.102 |
| **`cad_score`** | **`0.609`** | $0.4 \cdot 0.931 + 0.4 \cdot 0.0915 + 0.2 \cdot 1.000$ |

### Example 3: Topology Match catches what Shape Similarity misses

The candidate captures the silhouette of the fixture (cylindrical flange + central post + tall pin) well enough to score a moderate shape similarity, but the topology is wildly off: it produced half the expected through-holes and added internal voids that shouldn't exist.

| Ground truth | Candidate |
| :--: | :--: |
| ![GT iso](./metrics/illustrations/example_3_topology/gt_iso.png) | ![Candidate iso](./metrics/illustrations/example_3_topology/candidate_iso.png) |

| Component | Value | Notes |
| --- | --- | --- |
| Validity | ✅ valid, watertight | gate passes |
| `shape_similarity_score` | `0.432` | pc-F1 0.499, vol-IoU 0.364 |
| `topology_match` | **`0.626`** | $s_0 = 1.000$ ($b_0 = 1$ match), $s_1 = 0.545$ ($b_1 = 5$ vs GT $10$ → $6/11$), $s_2 = 0.333$ ($b_2 = 2$ vs GT $0$ → $1/3$) |
| **`cad_score`** | **`0.497`** | $(0.4 \cdot 0.432 + 0.2 \cdot 0.626) / 0.6$ (no interface axis) |

---

## Design notes

### Mesh-based computation (with IoU saturation)

Two parts of the pipeline operate on tessellated meshes rather than the BREP directly: the Boolean operations behind volume IoU and interface IoU (run on [`manifold3d`](https://github.com/elalish/manifold)), and the Betti-number computation behind topology match. Mesh-derived results are independent of the modeller's face decomposition: the same physical part authored two different ways gives the same numbers.

The trade-off is **tessellation residue**: a candidate that is geometrically identical to the GT but independently tessellated typically leaves a 0.1 to 1 % volume difference, which would drop IoU below 1.0 even for perfect candidates. Per-sub-volume IoU is therefore saturated to 1.0 above 0.99. Authoring-equivalent perfect candidates still score 1.0; real geometric errors drop IoU well below 0.99 and are unaffected. Betti is integer-valued and has no analogous noise.

---

## Code pointers

- Orchestrator: [`src/cadgenbench/eval/evaluate.py`](../src/cadgenbench/eval/evaluate.py) (`_cad_score`)
- Per-metric implementations:
  - [`cad_validity.py`](../src/cadgenbench/common/validity.py)
  - [`shape_similarity.py`](../src/cadgenbench/eval/shape_similarity.py)
  - [`topo_match.py`](../src/cadgenbench/eval/topo_match.py)
  - [`interface_match.py`](../src/cadgenbench/eval/interface_match.py)
- Geometry utilities: [`align.py`](../src/cadgenbench/eval/alignment.py), [`measurements.py`](../src/cadgenbench/common/measurements.py), [`sampling.py`](../src/cadgenbench/eval/sampling.py), [`mesh.py`](../src/cadgenbench/common/mesh.py)
- Headless renderer: [`src/cadgenbench/common/viewer.py`](../src/cadgenbench/common/viewer.py)
- Entry points: `cadgenbench baseline run` ([`src/cadgenbench/baseline/_cli.py`](../src/cadgenbench/baseline/_cli.py)) and `cadgenbench evaluate` ([`src/cadgenbench/eval/_cli.py`](../src/cadgenbench/eval/_cli.py))
