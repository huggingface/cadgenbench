#!/usr/bin/env python
"""Sanity-check the topology-match metric on one result-dir/fixture pair.

Re-runs the full :func:`cadgenbench.eval.evaluate.evaluate_result`
pipeline so the new topology-match score lands in ``result.json``, then
emits a self-contained HTML report (base64-embedded PNGs, no external
assets) for visual eyeballing.

Usage (from the repo root)::

    python _to_move_to_dataset_repo/sanity_check_topo.py \\
        results/<run_name>/jig-01-single-hole-plate \\
        --out /tmp/topo_sanity.html

The script expects ``<result_dir>/result.json`` and a candidate STEP to
already exist (produced by the baseline run). It picks up GT renders
from ``data/gt/<fixture>/renders/`` (named like the result dir),
candidate renders from ``<result_dir>/renders/``, and embeds whatever
views are available.

Open the resulting HTML in a browser; it lists:
- the candidate vs GT Betti vectors and per-axis match flags,
- the full ``cad_score`` and its component scores,
- side-by-side 4-view renders for candidate and GT.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from cadgenbench.eval.evaluate import GT_STEP_NAME, evaluate_result


VIEWS = ("iso", "front", "top", "right")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _img_data_uri(path: Path) -> str:
    """Encode a PNG as a ``data:image/png;base64,...`` URI."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _bool_pill(ok: bool) -> str:
    cls = "ok" if ok else "bad"
    text = "match" if ok else "mismatch"
    return f'<span class="pill {cls}">{text}</span>'


def _score_pill(score: float) -> str:
    """Colour a per-Betti fuzzy log-ratio score (1.0 → green, drops → red)."""
    if score >= 0.999:
        cls = "ok"
    elif score >= 0.6:
        cls = "warn"
    else:
        cls = "bad"
    return f'<span class="pill {cls}">{score:.3f}</span>'


def _render_html(
    *,
    result_dir: Path,
    gt_dir: Path,
    fixture: str,
    data: dict,
) -> str:
    validation = data.get("validation") or {}
    shape_score = (data.get("gt_metrics") or {}).get("shape_similarity_score")
    iface_score = (data.get("interface_metrics") or {}).get("score")
    topo = data.get("topology_metrics") or {}
    topo_score = topo.get("score")
    cad_score = data.get("cad_score")

    def pill(v: float | None) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else "<em>n/a</em>"

    def betti_row(label: str, b: dict | None) -> str:
        if not b:
            return f"<tr><th>{label}</th><td colspan='4'><em>n/a</em></td></tr>"
        return (
            f"<tr><th>{label}</th>"
            f"<td>{b['b0']}</td><td>{b['b1']}</td><td>{b['b2']}</td>"
            f"<td>χ={b['chi_surface']}, comp={b['n_components']}, "
            f"F={b['n_triangles']}, V={b['n_vertices']}, "
            f"defl={b['linear_deflection_mm']:.4f} mm</td></tr>"
        )

    per_axis_scores = topo.get("per_axis_scores") or {}

    # Render pairs
    cand_renders = result_dir / "renders"
    gt_renders = gt_dir
    rows = []
    for view in VIEWS:
        c = cand_renders / f"{view}.png"
        g = gt_renders / f"{view}.png"
        cand_html = (
            f'<img src="{_img_data_uri(c)}" />'
            if c.exists()
            else "<em>candidate render missing</em>"
        )
        gt_html = (
            f'<img src="{_img_data_uri(g)}" />'
            if g.exists()
            else "<em>GT render missing</em>"
        )
        rows.append(
            f"<tr><th>{view}</th>"
            f"<td>{cand_html}</td>"
            f"<td>{gt_html}</td></tr>",
        )
    render_table = "\n".join(rows)

    topo_match_table = (
        "<table class='betti'>"
        "<thead><tr><th>side</th><th>b₀</th><th>b₁</th><th>b₂</th>"
        "<th>diagnostics</th></tr></thead>"
        "<tbody>"
        + betti_row("candidate", topo.get("candidate"))
        + betti_row("ground truth", topo.get("gt"))
        + "</tbody></table>"
    )

    score_pills = "  ".join(
        f"<strong>{k}</strong> {_score_pill(float(v))}"
        for k, v in per_axis_scores.items()
    ) if per_axis_scores else "<em>not computed</em>"

    errors = validation.get("topology_errors") or []
    errors_html = (
        "<ul>" + "".join(f"<li>{e}</li>" for e in errors) + "</ul>"
        if errors
        else "<em>none</em>"
    )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Topo Sanity - {fixture}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 1100px; margin: 28px auto; padding: 0 18px;
         background: #f5f7fa; color: #1a1a1a; line-height: 1.45; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 32px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }}
  .meta {{ color: #57606a; font-family: monospace; font-size: 13px; }}
  .scores {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0 8px; }}
  .scorecard {{ flex: 1 1 200px; background: #fff; border: 1px solid #d0d7de;
                border-radius: 8px; padding: 12px 14px; }}
  .scorecard .label {{ text-transform: uppercase; font-size: 11px;
                       color: #57606a; letter-spacing: .04em; }}
  .scorecard .value {{ font-size: 28px; font-weight: 600; line-height: 1.1; }}
  .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
           font-size: 12px; font-weight: 500; }}
  .pill.ok  {{ background: #ddf4e3; color: #1a7f37; }}
  .pill.bad {{ background: #ffebe9; color: #cf222e; }}
  .pill.warn {{ background: #fff4ce; color: #9a6700; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 12px; text-align: left;
            border-bottom: 1px solid #eaeef2; vertical-align: top; }}
  th {{ background: #f6f8fa; font-weight: 600; color: #24292f; }}
  table.betti td:nth-child(2),
  table.betti td:nth-child(3),
  table.betti td:nth-child(4) {{ font-family: monospace; text-align: center; }}
  table.renders img {{ max-height: 200px; max-width: 100%;
                       background: #fff; border: 1px solid #eee;
                       border-radius: 4px; }}
  table.renders th {{ width: 60px; }}
  .matches {{ font-size: 14px; margin-top: 10px; }}
</style>
</head>
<body>

<h1>Topology sanity check</h1>
<div class="meta">
  fixture: <strong>{fixture}</strong><br />
  result_dir: {result_dir}<br />
  gt_dir: {gt_dir}
</div>

<h2>Headline scores</h2>
<div class="scores">
  <div class="scorecard">
    <div class="label">CAD score (new)</div>
    <div class="value">{pill(cad_score)}</div>
  </div>
  <div class="scorecard">
    <div class="label">Shape similarity</div>
    <div class="value">{pill(shape_score)}</div>
  </div>
  <div class="scorecard">
    <div class="label">Interface match</div>
    <div class="value">{pill(iface_score)}</div>
  </div>
  <div class="scorecard">
    <div class="label">Topo match</div>
    <div class="value">{pill(topo_score)}</div>
  </div>
</div>

<h2>Validity</h2>
<table>
  <tr><th>is_valid</th>
      <td>{_bool_pill(bool(validation.get("is_valid")))}</td></tr>
  <tr><th>is_watertight</th>
      <td>{_bool_pill(bool(validation.get("is_watertight")))}</td></tr>
  <tr><th>solid_count</th>
      <td>{validation.get("solid_count", "n/a")}</td></tr>
  <tr><th>shell_count</th>
      <td>{validation.get("shell_count", "n/a")}</td></tr>
  <tr><th>topology_errors</th>
      <td>{errors_html}</td></tr>
</table>

<h2>Betti numbers</h2>
<div class="matches">Per-axis scores: {score_pills}</div>
{topo_match_table}

<h2>Side-by-side renders</h2>
<table class="renders">
  <thead><tr><th>view</th><th>candidate</th><th>ground truth</th></tr></thead>
  <tbody>
    {render_table}
  </tbody>
</table>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate one result-dir + emit a self-contained HTML report "
            "with the new topology-match metric."
        ),
    )
    parser.add_argument(
        "result_dir", type=Path,
        help="Path to a fixture's result directory (containing result.json).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/topo_sanity.html"),
        help="Where to write the HTML report.",
    )
    parser.add_argument(
        "--skip-rescore", action="store_true",
        help=(
            "Don't re-run evaluate_result; just read the existing "
            "result.json and emit the HTML."
        ),
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not (result_dir / "result.json").exists():
        print(f"ERR: {result_dir}/result.json not found", file=sys.stderr)
        return 1

    fixture = result_dir.name
    gt_dir = (Path.cwd() / "data" / "gt" / fixture).resolve()
    if not (gt_dir / GT_STEP_NAME).exists():
        print(f"ERR: {gt_dir}/{GT_STEP_NAME} not found", file=sys.stderr)
        print(
            "Run from the repo root (the directory that contains data/).",
            file=sys.stderr,
        )
        return 1

    if not args.skip_rescore:
        print(f"Re-evaluating {fixture}...")
        evaluate_result(result_dir, gt_dir)
        print("  done. result.json updated.")

    data = json.loads((result_dir / "result.json").read_text())

    html = _render_html(
        result_dir=result_dir,
        gt_dir=gt_dir,
        fixture=fixture,
        data=data,
    )
    args.out.write_text(html)
    print(f"HTML written to: {args.out}")

    # Summary line so the user knows what to expect before opening.
    topo = data.get("topology_metrics") or {}
    cand = topo.get("candidate") or {}
    gt = topo.get("gt") or {}
    cad = data.get("cad_score")
    print(
        f"  cad_score={cad!r}  "
        f"betti(cand)=({cand.get('b0')}, {cand.get('b1')}, {cand.get('b2')})  "
        f"betti(gt)=({gt.get('b0')}, {gt.get('b1')}, {gt.get('b2')})  "
        f"topo={topo.get('score')!r}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
