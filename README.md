# CADGenBench

A benchmark for **LLM-driven CAD work**: how well do today's LLMs
either (a) turn a textual or visual description of a mechanical part
into a valid, geometrically correct 3D model, or (b) take an existing
STEP file and apply a requested edit to it?

The two task types are declared per fixture in
`data/inputs/<fixture>/description.yaml` via the `task_type` field
(`generation` or `editing`). The same agent, the same metrics, and
the same `output.step` submission contract apply to both.

This repository contains:

- The **benchmark**: fixtures (`data/inputs/` + `data/gt/`), the
  scoring metrics, and the report tools.
- A **reference baseline**: an iterative LLM agent that writes
  build123d Python, validates its STEP output, and loops until done.

The dataset will eventually live in a sibling HF dataset repo
(`<org>/cadgenbench-data`); for v1 the seven fixtures live in `data/`
inside this repo.

## Installation

CADGenBench targets Python 3.12 and installs entirely via pip. Use any
environment isolation tool you like (`python -m venv`, `uv venv`,
`conda create`, `pyenv`, ...); cadgenbench has no opinion.

```bash
# 1. Create a Python 3.12 environment your favourite way, e.g.:
python -m venv .venv && source .venv/bin/activate
# or:  uv venv && source .venv/bin/activate
# or:  conda create -n cadgenbench python=3.12 && conda activate cadgenbench

# 2. Editable install with the baseline + dev extras
pip install -e ".[baseline,dev]"

# 3. One-time download of the headless Chromium that powers STEP -> PNG
# rendering (used by `cadgenbench report` and the baseline's per-turn
# visual feedback). Cached under ~/Library/Caches/ms-playwright/ on macOS
# or ~/.cache/ms-playwright/ on Linux, so this is a per-machine setup
# step, not per-env.
playwright install chromium

# 4. (only if you'll run the baseline) provider API keys
cp .env.example .env   # then fill in ANTHROPIC_API_KEY / OPENAI_API_KEY / ...
```

`build123d` (and its `cadquery-ocp` BREP kernel) installs via pip on all
major platforms — manylinux + macOS arm64 + macOS x86_64 + Windows wheels
all live on PyPI. No conda detour required.

Verify:

```bash
cadgenbench --help
pytest tests/ -q
```

All `cadgenbench` commands must be run from the repo root (the
directory that contains `data/`). `cgb` is a shorter alias for the
same entry point.

## Quick start: evaluate a candidate

The benchmark scores per-fixture STEP outputs from any generator
(LLM agent, script, manual modelling) against the ground truth.

1. For every fixture name in `data/inputs/`, produce an `output.step`
   file and place it under
   `results/<your_run_name>/<fixture_name>/output.step`.

2. Score:

   ```bash
   cadgenbench evaluate results/<your_run_name>/
   ```

   This aligns each candidate to the ground truth, computes the four
   metric axes (validity, shape similarity, interface match,
   topology match), writes a per-fixture `result.json` carrying the
   per-fixture `status` (`valid` / `invalid` / `missing`) + `cad_score`,
   and rolls everything up into a single `run_summary.json` at the run
   root with `aggregate_score`, `validity_rate`, and
   `score_by_task_type` (one entry per task type, e.g. `generation`
   and `editing`).

3. Inspect:

   ```bash
   cadgenbench report single results/<your_run_name>/
   # writes results_<timestamp>.html, self-contained, opens in any browser
   ```

4. Compare against another run:

   ```bash
   cadgenbench report compare results/run_a/ results/run_b/ \
       --label "Run A" --label "Run B"
   ```

See [docs/benchmark/submission.md](docs/benchmark/submission.md) for the
exact submission contract (validity gate, canonical pose, file naming).

## Reference baseline

The repository ships a reference baseline: an LLM agent that writes
build123d Python in a loop until it produces a valid STEP. Use it to
sanity-check the install, as a worked example end-to-end, or as a
starting point for your own generator.

```bash
# Generation: single fixture, single model
cadgenbench baseline run jig-01-single-hole-plate \
    --model anthropic/claude-opus-4-7

# Editing: same CLI, just point at an editing fixture. The agent
# finds input.step in its work dir, modifies it, exports output.step.
cadgenbench baseline run jig-01-edit-double-hole \
    --model anthropic/claude-opus-4-7

# All fixtures (mix of generation + editing), in parallel
cadgenbench baseline run --all --parallel 4 \
    --model anthropic/claude-opus-4-7

# Compare the default trio (Claude Opus 4.7, Gemini 3.1 Pro, GPT-5.5)
# on every fixture in one command. Writes compare_<timestamp>.html.
cadgenbench baseline compare-llms --all --parallel 4

# ...or pick your own LLMs explicitly:
cadgenbench baseline compare-llms --all \
    --models anthropic/claude-sonnet-4-6 openai/gpt-5.5 \
    --label "Sonnet 4.6" --label "GPT-5.5"
```

The default trio for `compare-llms` is the current flagship from each
of Anthropic, Google, and OpenAI as of May 2026
(`anthropic/claude-opus-4-7`, `gemini/gemini-3.1-pro-preview`,
`openai/gpt-5.5`). Override with `--models` to compare a different
set. Provider-specific quirks (e.g. GPT-5 family rejecting
`temperature != 1`) are handled automatically by the LLM client.

Each run lands at `results/<timestamp>_<model_slug>/`; the `evaluate`
and `report` commands above work on it the same way as on a
hand-produced run directory.

## Commands

```
cadgenbench evaluate <run_dir> [--force]
cadgenbench baseline run [fixtures...] [--all] [--model M] ...
cadgenbench baseline compare-llms [fixtures...] --models M1 M2 ... [--label L ...] [-o out.html]
cadgenbench report single <run_dir> [-o out.html]
cadgenbench report compare <run_dir>... [-o out.html] [--label L ...]
```

Each subcommand prints its full flag list with `--help`.

## Metrics

The benchmark scores a candidate STEP against a ground-truth STEP along
four orthogonal axes:

| Metric | What it captures |
|---|---|
| **Validity** | Is the BREP well-formed, watertight, tessellable? Gate. Failure zeroes the rest. |
| **Shape similarity** | Geometry distance (point-cloud F1, volume IoU, feature-edge F1 on dihedral mesh edges). |
| **Interface match** | Mating-feature correctness via authored keep-in / keep-out sub-volumes. |
| **Topology match** | Betti numbers (b0, b1, b2) of the tessellated boundary. |

The **CAD Score** is the arithmetic mean of every applicable component
score, gated by validity. See [docs/metrics.md](docs/metrics.md) for the
full specification and [docs/metrics/](docs/metrics/) for the per-axis
deep dives.

## Dataset

For v1, the fixtures (mating-jig parts plus a derived editing-task variant)
live under `data/` in this repo. A separate HF dataset repo
(`<org>/cadgenbench-data`) is planned; once it exists the file-system
loader will be swapped for `datasets.load_dataset(...)` and `data/`
will be removed from this repo's history.

See [docs/benchmark/authoring.md](docs/benchmark/authoring.md) for the
fixture schema and [docs/benchmark/submission.md](docs/benchmark/submission.md)
for what a candidate submission must look like.

## License

TBD.
