# Benchmark submission requirements

What an agent / model / human contributor needs to produce to be scored
by the CAD Score pipeline. Sister document to
[`authoring.md`](authoring.md), which covers what a *benchmark labeller*
needs to produce on the ground-truth side.

For the scoring math itself see [`docs/metrics.md`](../metrics.md) and
the per-metric deep dives under [`docs/metrics/`](../metrics/).

---

## What you submit

**One** STEP file per fixture, written into
`results/<run_name>/<fixture_name>/output.step` (or `output.stp`).
Nothing else is required. No description, no metadata, no sub-volumes.

The grading pipeline picks the candidate STEP up, aligns it rigidly to
the GT, and runs the four metric categories on it.

The submission contract is **task-agnostic**: a fixture with
`task_type: generation` and one with `task_type: editing` both expect
the same `output.step`. For editing fixtures the agent additionally
finds the starting STEP (typically `input.step`) already present in
its working directory, but the file it writes back is still just
`output.step`. See [`authoring.md`](authoring.md) for the fixture
schema.

## What the grader writes back

Running `cadgenbench evaluate results/<run_name>/` produces two kinds
of JSON file. Both are generator-agnostic: identical schema whether
the candidate STEP came from a baseline agent, a script, or a human.

### Per-fixture: `results/<run_name>/<fixture_name>/result.json`

```jsonc
{
  "status": "valid",                          // "valid" | "invalid" | "missing"
  "validation": {                             // absent when status == "missing"
    "is_valid": true,
    "is_watertight": true,
    "solid_count": 1,
    "shell_count": 1,
    "face_count": 7,
    "volume": 16686.73,
    "bbox": {"x": 60.0, "y": 40.0, "z": 8.0},
    "topology_errors": []
  },
  "alignment":         { "rmse": 0.0124 },
  "gt_metrics":        { "shape_similarity_score": 0.84, ... },
  "shape_diagnostics": { ... },
  "interface_metrics": { "score": 0.93, "contexts": { ... } },  // optional
  "topology_metrics":  { "score": 1.0, "candidate": {...}, "gt": {...} },
  "cad_score":         0.917
}
```

`status` is the **single source of truth** for "did this fixture
produce a scorable STEP?":

- `"valid"` , `output.step` exists and passed the validity gate.
- `"invalid"` , `output.step` exists but failed the gate. `cad_score = 0`.
- `"missing"` , no `output.step` in the work dir. `cad_score = 0`,
  and the metric blocks (`gt_metrics`, `interface_metrics`, ...) are
  absent.

### Run-level: `results/<run_name>/run_summary.json`

Aggregates every per-fixture `result.json` (and reads
`task_type` from `data/inputs/<f>/description.yaml`):

```jsonc
{
  "aggregate_score":  0.624,        // mean cad_score over ALL fixtures, includes zeros
  "validity_rate":    0.875,        // n_valid / n_fixtures
  "n_fixtures":       8,
  "n_valid":          7,
  "n_invalid":        1,
  "n_missing":        0,
  "score_by_task_type": {
    "generation": 0.601,
    "editing":    0.792
  },
  "per_task_scores": {
    "generation": {"score": 0.601, "validity_rate": 0.857, "n_fixtures": 7,
                   "n_valid": 6, "n_invalid": 1, "n_missing": 0},
    "editing":    {"score": 0.792, "validity_rate": 1.0,   "n_fixtures": 1,
                   "n_valid": 1, "n_invalid": 0, "n_missing": 0}
  },
  "per_fixture_scores": {
    "jig-01-single-hole-plate":  {"status": "valid", "cad_score": 0.832, "task_type": "generation"},
    "jig-01-edit-double-hole":   {"status": "valid", "cad_score": 0.792, "task_type": "editing"},
    "...":                       {...}
  }
}
```

`aggregate_score` is the **arithmetic mean over every fixture**, with
invalid and missing fixtures contributing zero. Validity is therefore
already baked into the headline number; the separate `validity_rate`
axis reports how many fixtures cleared the gate at all.

`score_by_task_type` lets you read the two task families
(generation, editing) without re-aggregating. Adding a new task type
to a fixture (just set `task_type:` in `description.yaml`) creates a
new bucket automatically; no schema change required.

### Baseline-only debug info (ignored by the grader and the report tools)

When the candidate was produced by the included baseline agent, the
agent writes a sibling `baseline_debug.json` per fixture with
`stopped_reason` and `total_duration_s`, plus per-turn artefacts
(`turn_N/code_N.py`, `turn_N/stdout_N.txt`, `conversation.json`).
None of these are read by `result.json`, by `run_summary.json`, or by
the `cadgenbench report` tools, so external submissions producing
only `output.step` files are unaffected.

## What "valid" means

A submission's geometry must pass the **CAD Validity** gate. Anything
that doesn't is hard-zeroed (`cad_score = 0`); the gate cannot be
side-stepped by being good on the other axes.

The gate is a three-part conjunction (see
[`docs/metrics/cad_validity.md`](../metrics/cad_validity.md) for full
details):

1. **BREP well-formedness**, `BRepCheck_Analyzer.IsValid()` reports no
   per-face / per-edge / per-vertex topology errors.
2. **Watertightness**, every shell is closed (no naked / free edges).
3. **Meshable as a closed orientable manifold**, boundary
   tessellation produces a mesh with `3F = 2E`, every edge incident to
   exactly two triangles, with opposite orientations.

Why all three: downstream topology and shape metrics compute
divergence-theorem volumes and Euler-characteristic counts on the
boundary, which are only well-defined on a closed orientable manifold.

### Self-check before submitting

To verify your output passes the gate locally:

```bash
# Sanity scripts live under _to_move_to_dataset_repo/ until the
# cadgenbench-data HF dataset repo is created (they'll move there).
python _to_move_to_dataset_repo/sanity_check_submission.py path/to/output.step
```

Exits non-zero on any validity failure and prints the specific reason
(non-watertight, non-manifold edge, etc.). Same gate the grading
pipeline runs.

## Canonical pose: recommended, not enforced

The grading pipeline always aligns your candidate to the GT before
scoring (PCA over surface point clouds, plus an optional ICP refine).
Alignment is robust on most parts but can be ambiguous on
rotationally- or mirror-symmetric shapes, where the principal axes
have multiple "valid" orderings.

You can sidestep that risk by emitting your candidate in the same
canonical pose the benchmark GT uses:

1. **Bbox centroid at the origin**, bounding-box centre at $(0, 0, 0)$.
2. **Bbox extents ordered $L_x \ge L_y \ge L_z$**, longest axis along
   $X$, mid along $Y$, shortest along $Z$.
3. **Natural mounting / reference face down**, if the part has one,
   place it on the $z = -L_z/2$ plane with its outward normal along
   $-Z$. Parts without an obvious reference face: rules 1–2 suffice.

These rules are **required for benchmark GT** (see
[`authoring.md`](authoring.md)) but only **recommended** for
candidates. Following them is the easiest way to keep alignment RMSE
low on symmetric parts.

## What gets scored

Once your candidate clears the validity gate, the pipeline computes
the CAD Score from up to four orthogonal components:

| Component | Range | Read more |
| --- | --- | --- |
| Shape Similarity | $[0, 1]$ | [`metrics/shape_similarity.md`](../metrics/shape_similarity.md) |
| Topology Match | $[0, 1]$ | [`metrics/topo_match.md`](../metrics/topo_match.md) |
| Interface Match | $[0, 1]$ | [`metrics/interface_match.md`](../metrics/interface_match.md) |

The headline `cad_score` is the unweighted mean of the components that
are applicable to the fixture (Interface Match drops out when the
fixture has no labelled sub-volumes). See [`metrics.md`](../metrics.md)
for the full composition rule.

## Frequently confused things

- **You do not provide sub-volumes.** Sub-volumes (`jig_*.step`) live
  *only* on the GT side. They describe what regions of space the
  candidate must fill or leave empty. They're created by benchmark
  labellers, scored against your single `output.step`.
- **You are not graded on STEP authoring style.** Number of BREP
  faces, feature tree depth, parametric vs explicit, none of that
  matters as long as the geometry passes validity. Two STEP files of
  the same shape score the same.
- **`output.step` must live at the fixture-dir root, exactly that
  path: `results/<run_name>/<fixture_name>/output.step`.** The grader
  reads only that file - it has no concept of "turns", "iterations",
  or any internal structure your generator may use under the fixture
  dir (those are debug artefacts and ignored). Multiple files at
  sibling paths won't be picked up.
- **`output.step` must contain a single solid (or compound of solids).**
  Anything that isn't loadable by build123d / OCC is a load failure →
  validity = False → score 0.

## Code pointers

- Gate implementation: [`src/cadgenbench/common/validity.py`](../../src/cadgenbench/common/validity.py)
- Local self-check: [`_to_move_to_dataset_repo/sanity_check_submission.py`](../../_to_move_to_dataset_repo/sanity_check_submission.py) (slated to move to the `cadgenbench-data` dataset repo)
- Orchestrator that grades you: [`src/cadgenbench/eval/evaluate.py`](../../src/cadgenbench/eval/evaluate.py)
