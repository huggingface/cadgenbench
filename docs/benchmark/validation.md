# Validation

Two tiers. Every submission auto-publishes as `unvalidated`. Maintainers
promote rows to `validated` after reviewing methodology evidence. Both
tiers stay visible on the leaderboard, sorted by `aggregate_score`
descending. For the submission contract see
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

Promotion writes `validation_status: validated`, `validation_method`,
and `validated_at` (UTC ISO-8601) onto the row.
