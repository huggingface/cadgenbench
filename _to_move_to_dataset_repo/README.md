# Dataset-curation scripts (holding folder)

These scripts validate that fixture ground-truth STEP files are well-formed.
They are **dataset-authoring tools**, not part of the public benchmark code.

They live here temporarily until the `cadgenbench-data` HF dataset repo is
created, at which point they will be moved there with `git mv`. Until then,
they remain in this holding folder so they are preserved in git history but
explicitly excluded from the `cadgenbench` package build (see
`pyproject.toml`'s `tool.setuptools.packages.find` exclusion).

## Contents

- `sanity_check_gt.py` — validates that each fixture's `ground_truth.step`
  is canonical-posed, watertight, and BREP-valid.
- `sanity_check_jig_metric.py` — runs the interface-match metric on
  each fixture's jig sub-volumes against its GT and asserts saturation.
- `sanity_check_submission.py` — validates a candidate submission directory
  conforms to the submission spec (`docs/benchmark/submission.md`).
- `sanity_check_topo.py` — validates Betti-number agreement between the
  GT BREP and its tessellation.

## Why they were moved out

These tools require the GT STEPs to run, which we may eventually keep
private. Coupling the public benchmark code to internal authoring
workflows would make a future split painful. Keeping them in this
holding folder makes the lift a one-command `git mv`.

Any metric-exercising logic that is genuinely useful as a regression
test for the metric code itself should be lifted into `tests/eval/`
rather than living here.
