# Validation

Two tiers. Every submission auto-publishes as `unvalidated`. Maintainers
promote rows to `validated` after reviewing methodology evidence. The
leaderboard renders the two tiers as separate tables, both sorted by
`aggregate_score` descending. For the submission contract see
[`submission.md`](submission.md).

## Evidence types

Promotion requires one of:

| `validation_method` | Means |
|---|---|
| `code` | Agent source public at `agent_url`, reviewable end-to-end. |
| `traces` | Submission zip contains intermediate generation artifacts (per-turn STEPs, prompt logs, model outputs). |
| `api` | Submitter provides a callable endpoint the maintainer team can use to reproduce results. |
| `manual` | Maintainer team re-ran the agent against the public inputs and matched the scores. |

In practice we usually re-run the agent ourselves (`manual`) when it
uses an openly available CAD environment. For agents depending on a
paid-license CAD package the maintainer team doesn't hold, `manual`
isn't available; pick another evidence type.

## How to request review

Open a discussion on
[`HuggingAI4Engineering/cadgenbench-submissions`](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-submissions/discussions),
mention the `submission_id`, and link the relevant evidence. No SLA;
maintainers also review top-scoring unvalidated rows proactively.

## How promotion works

Maintainers promote from the admin panel on the leaderboard Space: pick
the submission, choose the `validation_method`, mark it validated. The
row moves from the unvalidated to the validated table on the next
refresh, and the validated table shows the accepted `validation_method`.
Demotion reverses it and clears `validation_method`. If the Space is
unavailable, a maintainer can edit the row in `results.jsonl` directly
on the dataset repo instead.

Promotion writes `validation_status: validated` and `validation_method`
onto the row. No other fields change.
