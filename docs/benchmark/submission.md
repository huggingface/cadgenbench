# Benchmark submission requirements

This document specifies the artifacts a submission must provide to be scored
by the CAD Score pipeline. For the scoring methodology, see
[`docs/metrics.md`](../metrics.md) and the per-metric deep dives under
[`docs/metrics/`](../metrics/).

## Submission layout

Create one directory per sample under `results/<run_name>/`, and write each
candidate to `results/<run_name>/<sample_name>/output.<ext>`. No other files
are required: no description, no metadata, no sub-volumes.

Two candidate kinds are accepted:

- **BREP / STEP** — `output.step` (or `output.stp`). Validated against the CAD
  validity gate (well-formed, watertight BREP that meshes into a closed
  manifold).
- **Triangle mesh** — `output.stl`, `output.obj`, `output.off`, `output.3mf`,
  or `output.ply`. Validated against the **mesh** validity gate (welded into a
  watertight, manifold, orientation-consistent surface) instead of the BREP
  gate, and scored without re-tessellation. This is the path for kernels that
  emit meshes rather than B-reps (e.g. OpenSCAD).

A STEP is preferred when both are present. The grader reads the candidate,
aligns it rigidly to the ground truth, and scores it. A sample directory
without any `output.*` candidate is recorded as `status: "missing"` and
contributes `cad_score = 0`.

The geometric metrics (alignment, shape similarity, topology, interface match)
run on meshes regardless of candidate kind, so STEP and mesh submissions are
scored by the same math. Only the validity gate differs: a mesh is held to the
mesh-manifold gate, a STEP to the stricter watertight-BREP gate.

The contract is task-agnostic: `generation` and `editing` samples both expect
the same `output.*` candidate. An editing sample additionally provides its
starting STEP (`input.step`) in the working directory, but the file written
back is still a single `output.*` candidate.

## Grader output

`cadgenbench evaluate results/<run_name>/` produces two kinds of JSON file.
Both use the same schema whether the candidate was produced by a baseline
agent, a script, or a human.

### Per-sample: `results/<run_name>/<sample_name>/result.json`

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
  "metric_errors":     { "shape_volume_iou": "RuntimeError: ..." },  // optional; only when a sub-metric raised
  "interface_metrics": { "score": 0.93, "contexts": { ... } },       // optional; only with jig sub-volumes
  "topology_metrics":  { "score": 1.0, "candidate": {...}, "gt": {...} },
  "edit_metrics":      { "baseline_shape_similarity": 0.71, ... },   // editing samples only
  "cad_score":         0.917
}
```

`status` is the authoritative indicator of whether a sample produced a
scorable candidate:

- `"valid"`: an accepted `output.*` candidate exists and passed the relevant
  validity gate.
- `"invalid"`: an accepted `output.*` candidate exists but failed the gate;
  `cad_score = 0`.
- `"missing"`: no accepted `output.*` candidate in the working directory;
  `cad_score = 0`,
  and the metric blocks (`gt_metrics`, `interface_metrics`, ...) are omitted.

`metric_errors` appears only when a shape sub-metric raises on a valid
candidate. These metrics are deterministic on valid CAD, so a failure is
exceptional: the affected sub-metric scores `0` (a crash never raises the
score), and the exception is recorded so that the `0` remains auditable.

### Run-level: `results/<run_name>/run_summary.json`

Aggregates every per-sample `result.json`, reading `task_type` from
`data/inputs/<sample>/description.yaml`:

```jsonc
{
  "aggregate_score":  0.624,        // run-level CAD Score (see metrics.md)
  "validity_rate":    0.875,        // n_valid / n_samples
  "n_samples":        8,
  "n_valid":          7,
  "n_invalid":        1,
  "n_missing":        0,
  "score_by_task_type": {
    "generation": 0.601,
    "editing":    0.792
  },
  "per_task_scores": {
    "generation": {"score": 0.601, "validity_rate": 0.857, "n_samples": 7,
                   "n_valid": 6, "n_invalid": 1, "n_missing": 0},
    "editing":    {"score": 0.792, "validity_rate": 1.0,   "n_samples": 1,
                   "n_valid": 1, "n_invalid": 0, "n_missing": 0}
  },
  "per_sample_scores": {
    "101":  {"status": "valid", "cad_score": 0.832, "task_type": "generation"},
    "201":  {"status": "valid", "cad_score": 0.792, "task_type": "editing"},
    "...":                       {...}
  }
}
```

`aggregate_score` is the run's headline CAD Score; see
[`docs/metrics.md`](../metrics.md) for how it is composed. `validity_rate`
reports how many samples cleared the validity gate, and `score_by_task_type`
breaks the headline down by task family. A new `task_type` in
`description.yaml` adds a bucket with no schema change.

## Validity requirements

Geometry must pass the CAD Validity gate; otherwise it scores `cad_score = 0`
regardless of its performance on the other axes. STEP/BREP candidates must be
well-formed, watertight B-reps that mesh into a closed manifold. Mesh
candidates skip BREP checks and must directly be watertight, manifold,
orientation-consistent triangle meshes; see
[`docs/metrics/cad_validity.md`](../metrics/cad_validity.md) for the exact
checks.

That document also defines advisory diagnostics (for example, sliver faces
and loose tolerances) that flag fragile geometry without affecting the score.
These are recommendations for robust B-reps (and mesh hygiene where relevant),
not requirements.

### Self-check before submitting

Run the bundled `sanity_check_submission.py` script from the
[`cadgenbench-data`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data)
dataset against your candidate:

```bash
DATA=$(python -c 'from cadgenbench.common.paths import data_inputs_dir; print(data_inputs_dir())')
python "$DATA/sanity_check_submission.py" path/to/output.step
# or, for a mesh submission:
python "$DATA/sanity_check_submission.py" path/to/output.stl
```

It applies the same gate as the grader, exits non-zero on any failure, and
prints the specific reason (non-watertight, non-manifold edge, and so on).

## Canonical pose (recommended)

The grader always aligns your candidate to the ground truth before scoring,
using rotation and translation only. Alignment is quite reliable for most parts but can remain ambiguous for rotationally or mirror-symmetric shapes, where several poses are genuinely equivalent.

To avoid that ambiguity, emit your candidate in the same canonical pose as
the ground truth:

1. Bounding-box centre at the origin $(0, 0, 0)$.
2. Bounding-box extents ordered $L_x \ge L_y \ge L_z$: longest axis along
   $X$, intermediate along $Y$, shortest along $Z$.
3. If the part has a natural mounting or reference face, place it on the
   $z = -L_z/2$ plane with its outward normal along $-Z$. Rules 1-2 suffice
   for parts without an obvious reference face.

Following the rules is not needed, but is the most reliable way to keep alignment stable on symmetric parts.

## Code pointers

- Scoring math and metric definitions: [`docs/metrics.md`](../metrics.md)
- Validity gate: [`src/cadgenbench/common/validity.py`](../../src/cadgenbench/common/validity.py)
- Grading orchestrator: [`src/cadgenbench/eval/evaluate.py`](../../src/cadgenbench/eval/evaluate.py)
- Run-level aggregation: [`src/cadgenbench/eval/run_summary.py`](../../src/cadgenbench/eval/run_summary.py)
