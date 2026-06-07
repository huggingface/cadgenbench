# CADGenBench

[![HF Space](https://img.shields.io/badge/🤗%20Space-Leaderboard-yellow)](https://huggingface.co/spaces/HuggingAI4Engineering/cadgenbench-leaderboard)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-Submissions-yellow)](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-submissions)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

A benchmark for **AI-driven CAD generation and editing**. Given a
textual or visual description of a mechanical part, a system must
produce a valid, geometrically correct 3D model. Given an existing STEP
file and a requested edit, it must apply that edit.

The benchmark targets AI models and makes no assumption about the CAD
environment (`build123d`, Autodesk Fusion, OnShape): a submission is
just a STEP file per fixture. Each fixture declares its type
(`generation` or `editing`) in `description.yaml`; the same metrics and
`output.step` contract apply to both.

**Submit and view the leaderboard:**
[`HuggingAI4Engineering/cadgenbench-leaderboard`](https://huggingface.co/spaces/HuggingAI4Engineering/cadgenbench-leaderboard).

## What this repo contains

This GitHub repo is the **source code behind the benchmark**. It is
*not* something you need to install to participate. Three things live
here:

- **Scoring engine** (`src/cadgenbench/eval/`): the CAD Score pipeline
  the leaderboard Space runs server-side against your submitted STEP
  files.
- **Docs** (`docs/`): metric definitions and the submission contract.
- **Reference baseline** (`src/cadgenbench/baseline/`): an optional
  example generator that turns a fixture description into a submission
  (iteratively writes [`build123d`](https://github.com/gumyr/build123d)
  Python, validates the STEP, and repeats until valid).

Evaluation itself happens on the Space: ground truth is held privately
in [`cadgenbench-data-gt`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data-gt)
and the Space is the only consumer.

## How to submit

Full contract (zip layout, `meta.json` fields, validity gate, optional
canonical pose) is at
[`docs/benchmark/submission.md`](docs/benchmark/submission.md). In
short:

1. For each fixture in
   [`cadgenbench-data`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data),
   produce an `output.step`. Any tool works.
2. Zip them as `submission.zip` with one folder per fixture plus a
   small `meta.json` at the root.
3. Upload via the **Submit** tab on the
   [leaderboard Space](https://huggingface.co/spaces/HuggingAI4Engineering/cadgenbench-leaderboard).

The Space validates the zip, runs the eval, publishes a row to the
leaderboard, and writes a self-contained per-submission HTML report
that you can link to or download.

Rows publish as unvalidated; promotion to a validated tier is a
separate methodology review by the maintainer team. See
[`docs/benchmark/validation.md`](docs/benchmark/validation.md) for the
review process and accepted evidence types.

A `sanity_check_submission.py` script shipped alongside the fixtures in
`cadgenbench-data` lets you exercise the same validity gate locally
before uploading; see
[`docs/benchmark/submission.md#self-check-before-submitting`](docs/benchmark/submission.md#self-check-before-submitting).

## Metrics

The Space scores each candidate STEP against ground truth on four
axes:

| Metric | What it captures |
|---|---|
| **Validity** | Is the BREP well-formed, watertight, tessellable? Gate: failure zeroes the rest. |
| **Shape similarity** | Geometry distance (point-cloud F1, volume IoU). |
| **Interface match** | Mating-feature correctness via authored keep-in / keep-out sub-volumes. |
| **Topology match** | Betti numbers (b0, b1, b2) of the tessellated boundary. |

The **CAD Score** is a weighted combination of the applicable component
scores, gated by validity. See [`docs/metrics.md`](docs/metrics.md) for
the full specification and [`docs/metrics/`](docs/metrics/) for the
per-axis details.

## Reference baseline (optional)

The reference baseline is an iterative agent that writes `build123d`
Python in a loop until it produces a valid STEP. Use it to see what an
end-to-end run looks like, or as a starting point for your own
generator. It targets Python 3.12 and installs entirely via pip.

```bash
# 1. Python 3.12 env (venv, uv, conda, etc.)
python -m venv .venv && source .venv/bin/activate

# 2. Editable install with the baseline + dev extras
pip install -e ".[baseline,dev]"

# 3. Provider API keys for whichever model(s) you plan to run
cp .env.example .env   # then fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.

# 4. Point at the public fixture-inputs dataset on the Hub. cadgenbench
# snapshot-downloads it on first use and caches under
# ~/.cache/huggingface/hub/.
export CADGENBENCH_DATA_REPO=HuggingAI4Engineering/cadgenbench-data
```

Rendering (per-turn visual feedback to the agent) is in-process via
PyVista/VTK; no Chromium or browser install is needed. On a bare
headless Linux box VTK needs system OpenGL libs (e.g. `libgl1` /
Mesa); macOS works out of the box.

Verify:

```bash
cadgenbench --help
pytest tests/ -q
```

`cgb` is a shorter alias.

Run on one fixture, or in parallel on all of them:

```bash
# Single fixture (fixture names are the dataset's folder names, e.g. 101)
cadgenbench baseline run 101 --model openai/gpt-5.5

# All fixtures, in parallel
cadgenbench baseline run --all --parallel 4 --model openai/gpt-5.5
```

**Using a different LLM.** `--model` takes any
[LiteLLM](https://docs.litellm.ai/docs/providers) `provider/model`
string; just set the matching key in `.env`. For example:

```bash
cadgenbench baseline run 101 --model anthropic/claude-opus-4-7   # ANTHROPIC_API_KEY
cadgenbench baseline run 101 --model gemini/gemini-3.1-pro-preview  # GEMINI_API_KEY
cadgenbench baseline run 101 --model openai/gpt-5.5             # OPENAI_API_KEY
```

For reasoning models, `--reasoning-effort {minimal,low,medium,high}`
sets the thinking budget (mapped per provider by LiteLLM). See
`cadgenbench baseline run --help` for the full flag set.

Output lands at `results/<timestamp>_<model_slug>/<fixture>/output.step`.
The baseline only *generates* candidates; scoring against ground truth
happens on the leaderboard Space after you submit.

Bundle a run directory into a submission zip (top-level `meta.json` +
one `output.step` per fixture, per the submission contract):

```bash
cadgenbench baseline package results/20260602_120000_gpt-5.5 \
    --submitter "Your Name" --name "My agent v1" --agree
```

Writes `<run_dir>.zip`, ready to upload on the Space's **Submit** tab.
`agree_to_publish` is `false` until you pass `--agree`.

## Dataset

Fixtures live in two HF dataset repos:

- [`HuggingAI4Engineering/cadgenbench-data`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data):
  **public**; inputs (descriptions, optional input STEPs and renders)
  for every fixture, plus the `sanity_check_submission.py` helper.
- [`HuggingAI4Engineering/cadgenbench-data-gt`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data-gt):
  **private**; ground truth (`ground_truth.step`, optional jig
  sub-volumes, renders) and the labeller-facing `AUTHORING.md` /
  sanity-check scripts. Only the leaderboard Space reads from it.
  Keeping GT private makes the Space's eval the source of truth.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
