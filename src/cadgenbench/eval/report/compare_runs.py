# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``cadgenbench report compare`` -- HTML comparison of N experiment runs.

Shows a summary chart and table, then per-fixture detail cards with input,
GT, and each run's outputs side by side.

Navigation: click a row to view details, j/k or arrow keys to move
between fixtures, Escape to return to the summary.

Usage::

    cadgenbench report compare results/run_a results/run_b
    cadgenbench report compare run_a run_b run_c --union
    cadgenbench report compare run_a run_b -o comparison.html
    cadgenbench report compare run_a run_b --label "Opus 4.7" --label "Sonnet 4.6"
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

import yaml

try:
    from cadgenbench.eval.shape_similarity import METRIC_DISPLAY
except Exception:
    METRIC_DISPLAY = {}  # type: ignore[assignment]

VIEWS = ["iso", "front", "top", "right", "bottom"]


def _data_gt_dir() -> Path:
    """Resolve ``data/gt/`` via the shared cadgenbench data-dir helper."""
    from cadgenbench.common.paths import data_gt_dir
    return data_gt_dir()

RUN_COLORS = [
    ("#1976d2", "#1565c0"),  # blue
    ("#e64a19", "#e65100"),  # orange
    ("#2e7d32", "#1b5e20"),  # green
    ("#7b1fa2", "#6a1b9a"),  # purple
    ("#c62828", "#b71c1c"),  # red
    ("#00838f", "#006064"),  # teal
    ("#ef6c00", "#e65100"),  # amber
    ("#4527a0", "#311b92"),  # deep purple
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _fmt(v: float | None, decimals: int = 2) -> str:
    return "-" if v is None else f"{v:.{decimals}f}"


def _run_label(run: dict, all_runs: list[dict] | None = None) -> str:
    """Derive a short human label for a run.

    Uses just the model short name when unique across runs; appends the
    timestamp when multiple runs share the same model.
    """
    params = run.get("params", {})
    config = params.get("config", {})
    model = config.get("model")
    short = model.split("/")[-1] if model else run["timestamp"]

    if all_runs:
        dupes = sum(
            1 for r in all_runs
            if (r.get("params", {}).get("config", {}).get("model") or "") == (model or "")
        )
        if dupes > 1:
            return f"{short} ({run['timestamp']})"

    return short


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _inputs_dir_for(gt_dir: Path | None) -> Path | None:
    """Sibling lookup: data/gt/<f>/ -> data/inputs/<f>/."""
    if gt_dir is None:
        return None
    cand = gt_dir.parent.parent / "inputs" / gt_dir.name
    return cand if cand.exists() else None


_STEP_SUFFIXES = (".step", ".stp")


def _load_description(gt_dir: Path) -> tuple[str, list[Path], list[Path], bool]:
    """``(text, image_files, shape_render_pngs, wants_shape)`` for the input panel.

    Editing fixtures ship the starting solid as ``input.step``; the raw
    STEP can't be shown with ``<img>``, so we surface its canonical-view
    PNGs from ``inputs/<fixture>/renders/`` (mirroring the GT renders)
    instead. See the matching helper in ``single_run.py``.
    """
    inputs_dir = _inputs_dir_for(gt_dir)
    if inputs_dir is None:
        return "", [], [], False
    desc_path = inputs_dir / "description.yaml"
    if not desc_path.exists():
        return "", [], [], False
    data = yaml.safe_load(desc_path.read_text()) or {}
    text = data.get("description", "")
    image_files: list[Path] = []
    wants_shape = False
    for name in data.get("input_files", []):
        p = inputs_dir / name
        if p.suffix.lower() in _STEP_SUFFIXES:
            wants_shape = True
            continue
        if p.exists():
            image_files.append(p)
    if not image_files and not wants_shape:
        p = inputs_dir / "input.png"
        if p.exists():
            image_files.append(p)
    shape_render_pngs: list[Path] = []
    if wants_shape:
        renders_dir = inputs_dir / "renders"
        shape_render_pngs = [
            renders_dir / f"{v}.png"
            for v in VIEWS
            if (renders_dir / f"{v}.png").exists()
        ]
    return text, image_files, shape_render_pngs, wants_shape


def _discover_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    params: dict = {}
    pp = run_dir / "params.json"
    if pp.exists():
        params = json.loads(pp.read_text())

    timestamp = params.get("timestamp", run_dir.name)

    fixtures: dict[str, dict] = {}
    for fd in sorted(run_dir.iterdir()):
        if not fd.is_dir():
            continue
        name = fd.name
        rp = fd / "result.json"
        if not rp.exists():
            continue
        result = json.loads(rp.read_text())

        gt_dir = _data_gt_dir() / name
        if not gt_dir.exists():
            gt_dir = None

        fixtures[name] = {
            "name": name,
            "result": result,
            "result_dir": fd,
            "gt_dir": gt_dir,
        }

    run_summary: dict = {}
    sp = run_dir / "run_summary.json"
    if sp.exists():
        try:
            run_summary = json.loads(sp.read_text())
        except Exception:
            pass

    return {
        "run_dir": run_dir,
        "timestamp": timestamp,
        "params": params,
        "run_summary": run_summary,
        "fixtures": fixtures,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _images_html(pngs: list[Path]) -> str:
    if not pngs:
        return ""
    parts = ['<div class="images">']
    for vp in pngs:
        uri = _data_uri(vp)
        parts.append(
            f'<div class="view"><img src="{uri}" alt="{vp.stem}" loading="lazy">'
            f"<span>{vp.stem}</span></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_gt_images(gt_dir: Path | None) -> str:
    if gt_dir is None:
        return '<span class="note">-</span>'
    renders_dir = gt_dir / "renders"
    pngs = [renders_dir / f"{v}.png" for v in VIEWS if (renders_dir / f"{v}.png").exists()]
    return _images_html(pngs) or '<span class="note">no renders</span>'


def _render_result_images(result_dir: Path | None) -> str:
    if result_dir is None:
        return '<span class="note">-</span>'
    renders_dir = result_dir / "renders"
    if not renders_dir.is_dir():
        return '<span class="note">no renders</span>'
    all_pngs = list(renders_dir.glob("*.png"))
    view_order = {v: i for i, v in enumerate(VIEWS)}
    pngs = sorted(all_pngs, key=lambda p: (view_order.get(p.stem, len(VIEWS)), p.stem))
    return _images_html(pngs) or '<span class="note">no renders</span>'


def _color(run_idx: int) -> tuple[str, str]:
    """Return (bar_color, text_color) for a run index."""
    return RUN_COLORS[run_idx % len(RUN_COLORS)]


# Headline-only metrics for summary tables. Per-component shape scores
# (point cloud F1, volume IoU, edge F1) are shown as a secondary line on
# each per-fixture run card.
SUMMARY_METRICS = [
    ("shape_similarity_score", "Shape Similarity", False),  # (key, label, lower_is_better)
]

# ---------------------------------------------------------------------------
# Helpers for ranking fixtures
# ---------------------------------------------------------------------------

def _sort_key_for_entry(entry: dict) -> float:
    """Return a float used to rank fixtures worst-first.

    Higher = worse. Uses ``1 - min(shape_similarity_score)`` across runs.
    """
    runs_data = entry["runs"]
    shape_scores = [
        (fix or {}).get("result", {}).get("gt_metrics", {}).get("shape_similarity_score")
        for fix in runs_data
    ]
    valid_scores = [v for v in shape_scores if v is not None]
    return 1.0 - min(valid_scores) if valid_scores else 0.0


def _quality_class(score: float | None) -> str:
    """Row / cell background class based on shape similarity (higher = better)."""
    if score is None:
        return "q-none"
    if score >= 0.9:
        return "q-high"
    if score >= 0.6:
        return "q-mid"
    return "q-low"


def _render_top5(entries: list[dict], n_runs: int, labels: list[str]) -> str:
    """Thumbnail comparison of the 5 worst fixtures (GT + each run iso side by side)."""
    metric_label = "Shape Gap"

    ranked = sorted(entries, key=_sort_key_for_entry, reverse=True)
    top5 = ranked[:5]

    p = ['<div class="top5">']
    p.append(
        f'<div class="top5-title">Top 5 worst cases &nbsp;'
        f'<span class="top5-metric-label">(sorted by {metric_label})</span></div>'
    )

    # Header row
    p.append('<div class="top5-header">')
    p.append('<div class="top5-cell top5-name-hdr">Fixture</div>')
    p.append('<div class="top5-cell">Ground Truth</div>')
    for ri in range(n_runs):
        _, text_col = _color(ri)
        p.append(
            f'<div class="top5-cell" style="color:{text_col};font-weight:600">'
            f'{html.escape(labels[ri])}</div>'
        )
    p.append('</div>')  # top5-header

    for idx_in_top5, entry in enumerate(top5):
        global_idx = next(
            i for i, e in enumerate(entries) if e["name"] == entry["name"]
        )
        runs_data = entry["runs"]
        gt_dir = next((fix["gt_dir"] for fix in runs_data if fix and fix.get("gt_dir")), None)

        p.append(
            f'<div class="top5-row" onclick="showDetail({global_idx})" '
            f'style="cursor:pointer" title="Click for details">'
        )

        # fixture name + sort metric value
        sort_val = _sort_key_for_entry(entry)
        sort_text = f"{sort_val:.3f}"
        p.append(
            f'<div class="top5-cell top5-name">'
            f'<strong>{html.escape(entry["name"])}</strong><br>'
            f'<span class="top5-err">{metric_label}: {sort_text}</span>'
            f'</div>'
        )

        # GT thumbnail
        gt_iso = (gt_dir / "renders" / "iso.png") if gt_dir else None
        if gt_iso and gt_iso.exists():
            uri = _data_uri(gt_iso)
            p.append(f'<div class="top5-cell"><img src="{uri}" class="top5-img" alt="GT"></div>')
        else:
            p.append('<div class="top5-cell top5-noimg">no renders</div>')

        # Per-run thumbnails
        for ri in range(n_runs):
            fix = runs_data[ri]
            bar_col, _ = _color(ri)
            if fix is None:
                p.append('<div class="top5-cell top5-noimg">-</div>')
                continue
            result_dir = fix.get("result_dir")
            iso = (result_dir / "renders" / "iso.png") if result_dir else None
            if iso and iso.exists():
                uri = _data_uri(iso)
                gt_m = fix.get("result", {}).get("gt_metrics", {})
                if gt_m.get("shape_similarity_score") is not None:
                    pill = f'Shape: {gt_m["shape_similarity_score"]:.3f}'
                else:
                    pill = ""
                p.append(
                    f'<div class="top5-cell">'
                    f'<img src="{uri}" class="top5-img" alt="{html.escape(labels[ri])}">'
                    f'<span class="top5-pill" style="background:{bar_col}">{pill}</span>'
                    f'</div>'
                )
            else:
                p.append('<div class="top5-cell top5-noimg">no renders</div>')

        p.append('</div>')  # top5-row

    p.append('</div>')  # top5
    return "\n".join(p)


def _render_summary_table(
    entries: list[dict], n_runs: int, labels: list[str],
) -> str:
    """Summary table with one column per run per metric, winner highlighted."""
    show_cad_score = any(
        (fix or {}).get("result", {}).get("cad_score") is not None
        for entry in entries
        for fix in entry["runs"]
    )
    show_interface = any(
        ((fix or {}).get("result", {}).get("interface_metrics") or {}).get("score") is not None
        for entry in entries
        for fix in entry["runs"]
    )
    show_topo = any(
        ((fix or {}).get("result", {}).get("topology_metrics") or {}).get("score") is not None
        for entry in entries
        for fix in entry["runs"]
    )

    # Column ordering: combined headline first, then shape similarity,
    # then interface, then topology.
    col_defs: list[tuple[str, str, bool, str]] = []  # (key, label, lower_is_better, source)
    if show_cad_score:
        col_defs.append(("cad_score", "CAD Score", False, "top"))
    for mk, mlabel, lib in SUMMARY_METRICS:
        col_defs.append((mk, mlabel, lib, "gt"))
    if show_interface:
        col_defs.append(("score", "Interface", False, "interface"))
    if show_topo:
        col_defs.append(("score", "Topo", False, "topology"))

    p = ['<table class="summary-table" id="summary-table">']

    group_row = "<thead><tr><th rowspan='2'>Fixture</th>"
    sub_row = "<tr>"
    col_idx = 1
    for mk, mlabel, _, source in col_defs:
        group_row += f"<th colspan='{n_runs}' class='metric-group'>{mlabel}</th>"
        for ri in range(n_runs):
            _, text_col = _color(ri)
            short = html.escape(labels[ri])
            sub_row += (
                f'<th class="sortable run-subhdr" style="color:{text_col}" '
                f'data-col="{col_idx}" title="{short}">{short}</th>'
            )
            col_idx += 1
    group_row += "</tr>"
    sub_row += "</tr></thead>"
    p.append(group_row)
    p.append(sub_row)

    p.append("<tbody>")

    sums: list[list[float]] = [[0.0] * n_runs for _ in col_defs]
    counts: list[list[int]] = [[0] * n_runs for _ in col_defs]

    for i, entry in enumerate(entries):
        runs_data = entry["runs"]

        best_shape_score = max(
            ((fix or {}).get("result", {}).get("gt_metrics", {}).get("shape_similarity_score") or 0)
            for fix in runs_data
        )
        row_bg = _quality_class(best_shape_score if any(
            (fix or {}).get("result", {}).get("gt_metrics") for fix in runs_data
        ) else None)

        cells = f'<td class="fix-name">{html.escape(entry["name"])}</td>'
        for ci, (mk, _, lower_is_better, source) in enumerate(col_defs):
            if source == "top":
                values = [
                    (fix or {}).get("result", {}).get(mk)
                    for fix in runs_data
                ]
            elif source == "interface":
                values = [
                    ((fix or {}).get("result", {}).get("interface_metrics") or {}).get(mk)
                    for fix in runs_data
                ]
            elif source == "topology":
                values = [
                    ((fix or {}).get("result", {}).get("topology_metrics") or {}).get(mk)
                    for fix in runs_data
                ]
            else:
                values = [
                    (fix or {}).get("result", {}).get("gt_metrics", {}).get(mk)
                    for fix in runs_data
                ]
            valid = [v for v in values if v is not None]
            winner_val = (min(valid) if lower_is_better else max(valid)) if valid else None

            for ri, v in enumerate(values):
                if v is not None:
                    sums[ci][ri] += v
                    counts[ci][ri] += 1
                    meta = METRIC_DISPLAY.get(mk)
                    txt = f"{format(v, meta.fmt)}{meta.suffix}" if meta else _fmt(v, 3)
                    is_winner = (winner_val is not None and abs(v - winner_val) < 1e-6)
                    cls = "cell-winner" if is_winner else ""
                    cells += f'<td class="{cls}" data-v="{v:.4f}">{txt}</td>'
                else:
                    cells += '<td class="cell-nogt" data-v="-1">-</td>'

        p.append(
            f'<tr class="data-row {row_bg}" onclick="showDetail({i})" '
            f'style="cursor:pointer">{cells}</tr>'
        )

    # Averages footer
    avg_cells = "<td class='fix-name avg-label'>Averages</td>"
    for ci, (mk, _, _, _) in enumerate(col_defs):
        for ri in range(n_runs):
            n = counts[ci][ri]
            if n > 0:
                avg = sums[ci][ri] / n
                meta = METRIC_DISPLAY.get(mk)
                txt = f"{format(avg, meta.fmt)}{meta.suffix}" if meta else _fmt(avg, 3)
                avg_cells += f'<td class="avg-cell" data-v="{avg:.4f}"><strong>{txt}</strong></td>'
            else:
                avg_cells += '<td class="cell-nogt avg-cell" data-v="-1">-</td>'
    p.append(f'<tr class="avg-row">{avg_cells}</tr>')

    p.append("</tbody></table>")
    return "\n".join(p)


def _render_input_panel(gt_dir: Path | None, runs_data: list) -> str:
    """Input section: task description text + input images."""
    del runs_data  # only used for description fallback previously; now we always read from description.yaml
    p = ['<div class="input-section">', "<h3>Input</h3>"]
    if gt_dir:
        desc_text, input_imgs, input_shape_pngs, wants_shape = _load_description(gt_dir)
        if desc_text:
            p.append(f'<p class="desc">{html.escape(desc_text)}</p>')
        for img_path in input_imgs:
            uri = _data_uri(img_path)
            if uri:
                p.append(f'<img src="{uri}" alt="input" class="input-img" loading="lazy">')
        if input_shape_pngs:
            p.append(_images_html(input_shape_pngs))
        elif wants_shape:
            p.append('<span class="note">no input renders</span>')
    p.append("</div>")
    return "\n".join(p)


def _render_gt_metrics_pills(result: dict) -> str:
    """Pills for CAD Score, Shape, Interface, Topo + a muted components line."""
    gt_m = result.get("gt_metrics") or {}
    iface_m = result.get("interface_metrics") or {}
    topo_m = result.get("topology_metrics") or {}
    cad = result.get("cad_score")
    shape = gt_m.get("shape_similarity_score")
    iface = iface_m.get("score")
    topo = topo_m.get("score")
    if all(v is None for v in (cad, shape, iface, topo)):
        return ""

    parts: list[str] = ['<div class="gt-metrics">']
    if cad is not None:
        meta = METRIC_DISPLAY.get("cad_score")
        val = f"{format(cad, meta.fmt)}{meta.suffix}" if meta else _fmt(cad, 3)
        parts.append(f'<span class="gm-pill"><b>CAD Score: {val}</b></span>')
    if shape is not None:
        meta = METRIC_DISPLAY.get("shape_similarity_score")
        val = f"{format(shape, meta.fmt)}{meta.suffix}" if meta else _fmt(shape, 3)
        parts.append(f'<span class="gm-pill">Shape: <strong>{val}</strong></span>')
    if iface is not None:
        n_ctx = len(iface_m.get("contexts") or {})
        parts.append(
            f'<span class="gm-pill">Interface: <strong>{iface:.3f}</strong>'
            f' <span class="metric-sub">(ctx {n_ctx})</span></span>'
        )
    if topo is not None:
        cand_b = topo_m.get("candidate") or {}
        gt_b = topo_m.get("gt") or {}
        cand_sig = f"({cand_b.get('b0')},{cand_b.get('b1')},{cand_b.get('b2')})"
        gt_sig = f"({gt_b.get('b0')},{gt_b.get('b1')},{gt_b.get('b2')})"
        parts.append(
            f'<span class="gm-pill">Topo: <strong>{topo:.3f}</strong>'
            f' <span class="metric-sub">{cand_sig} vs {gt_sig}</span></span>'
        )
    parts.append("</div>")

    component_keys = ("shape_point_cloud_f1", "shape_volume_iou")
    components = [(k, gt_m.get(k)) for k in component_keys if gt_m.get(k) is not None]
    if components:
        comp_parts: list[str] = []
        for k, v in components:
            meta = METRIC_DISPLAY.get(k)
            label = meta.label if meta else k
            display = f"{format(v, meta.fmt)}{meta.suffix}" if meta else _fmt(v, 3)
            comp_parts.append(f"{html.escape(label)}: {display}")
        parts.append(
            f'<div class="metrics-sub">Shape components &middot; '
            f'{" &middot; ".join(comp_parts)}</div>'
        )

    return "\n".join(parts)


def _render_status_pill(status: str) -> str:
    """Per-run status pill (valid / invalid / missing)."""
    cls = {
        "valid": "status-valid",
        "invalid": "status-invalid",
        "missing": "status-missing",
    }.get(status, "status-unknown")
    return f'<span class="status-pill {cls}">{html.escape(status)}</span>'


def _render_run_column(fix: dict | None, label: str, bar_col: str, text_col: str) -> str:
    """One run column: header, renders, geometry pills, status."""
    del bar_col
    p = ['<div class="col">']
    if fix:
        r = fix["result"]
        status = r.get("status", "?")
        p.append(
            f'<h3 style="color:{text_col};border-color:{text_col}">'
            f"{html.escape(label)} {_render_status_pill(status)}</h3>"
        )
        p.append(_render_result_images(fix.get("result_dir")))
        p.append(_render_gt_metrics_pills(r))
    else:
        p.append(f'<h3 style="color:{text_col};border-color:{text_col}">{html.escape(label)}</h3>')
        p.append('<p class="note">Not in this run</p>')
    p.append("</div>")
    return "\n".join(p)


def _render_fixture_card(
    entry: dict, idx: int, n_runs: int, labels: list[str],
) -> str:
    name = entry["name"]
    runs_data = entry["runs"]

    gt_dir = next((fix["gt_dir"] for fix in runs_data if fix and fix.get("gt_dir")), None)

    p = [
        f'<div class="fixture-card" data-idx="{idx}" style="display:none">',
        f'<h2>{html.escape(name)}</h2>',
        _render_input_panel(gt_dir, runs_data),
        '<div class="n-col">',
        '<div class="col">',
        "<h3>Ground Truth</h3>",
        _render_gt_images(gt_dir),
        "</div>",
    ]

    for ri in range(n_runs):
        bar_col, text_col = _color(ri)
        p.append(_render_run_column(runs_data[ri], labels[ri], bar_col, text_col))

    p += ["</div>", "</div>"]  # n-col, fixture-card
    return "\n".join(p)


def _render_config_diff(runs: list[dict], labels: list[str]) -> str:
    configs = [r.get("params", {}).get("config", r.get("params", {})) for r in runs]
    all_keys = sorted(
        set(k for c in configs for k in c.keys()) - {"fixtures", "timestamp"}
    )
    diff_keys = [k for k in all_keys if len({str(c.get(k)) for c in configs}) > 1]

    if not diff_keys:
        return '<div class="config-diff"><p class="note">Configs identical</p></div>'

    p = ['<div class="config-diff"><h3>Config Diff</h3>']
    p.append('<table class="diff-table"><thead><tr><th>Parameter</th>')
    for ri, label in enumerate(labels):
        _, text_col = _color(ri)
        p.append(f'<th style="color:{text_col}">{html.escape(label)}</th>')
    p.append("</tr></thead><tbody>")

    for k in diff_keys:
        vals = [c.get(k) for c in configs]
        cells = f"<td>{html.escape(k)}</td>"
        for v in vals:
            cells += f"<td>{html.escape(str(v) if v is not None else '\u2014')}</td>"
        p.append(f'<tr class="diff-changed">{cells}</tr>')

    p.append("</tbody></table></div>")
    return "\n".join(p)


def _css() -> str:
    return """\
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1700px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { margin-top: 0; }

/* Quality row colors */
.q-high { background: #e8f5e9; }
.q-mid  { background: #fff9c4; }
.q-low  { background: #ffebee; }
.q-none { background: #f5f5f5; }

/* Summary table winner / no-GT cells */
.cell-winner { background: #c8e6c9; font-weight: 700; }
.cell-nogt   { color: #bbb; }
.avg-row     { background: #eceff1; font-size: 0.9em; }
.avg-cell    { border-top: 2px solid #90a4ae; }
.avg-label   { font-weight: 700; font-style: italic; }

.run-header { background: white; border-radius: 8px; padding: 16px 20px;
              margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.run-labels { display: flex; gap: 24px; margin-top: 8px; font-size: 0.9em; flex-wrap: wrap; }

/* Per-run aggregate band (one card per run, from run_summary.json) */
.aggregate-band { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 14px; }
.agg-card { flex: 1 1 200px; background: #fafbfc; border: 2px solid #ccc;
            border-radius: 8px; padding: 10px 14px; min-width: 180px; }
.agg-label { font-size: 0.78em; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.04em; margin-bottom: 6px; }
.agg-score { font-size: 1.7em; line-height: 1.0; margin: 4px 0 8px; }
.agg-score b { color: #263238; }
.agg-score span { display: block; font-size: 0.5em; color: #78909c;
                  text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
.agg-row { display: flex; justify-content: space-between; font-size: 0.82em;
           color: #455a64; padding: 2px 0; }
.agg-row span { color: #78909c; }

/* Per-run status pill */
.status-pill { padding: 2px 9px; border-radius: 10px; font-size: 0.72em;
               font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
               margin-left: 8px; vertical-align: middle; }
.status-valid   { background: #e8f5e9; color: #2e7d32; }
.status-invalid { background: #ffebee; color: #c62828; }
.status-missing { background: #f5f5f5; color: #757575; }
.status-unknown { background: #f5f5f5; color: #999; }

.config-diff { background: white; border-radius: 8px; padding: 12px 16px;
               margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.config-diff h3 { margin-top: 0; color: #555; font-size: 0.9em; text-transform: uppercase; }
.diff-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
.diff-table th { background: #f5f5f5; padding: 4px 10px; text-align: left; }
.diff-table td { padding: 4px 10px; border-bottom: 1px solid #eee; }
.diff-changed { background: #fff9c4; }

/* Top-5 worst cases thumbnail grid */
.top5 { background: white; border-radius: 8px; padding: 16px 20px;
        margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.top5-title { font-size: 0.85em; font-weight: 700; color: #37474f;
              text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 10px; }
.top5-metric-label { font-weight: 400; color: #888; text-transform: none; }
.top5-header { display: flex; gap: 8px; padding-bottom: 6px;
               border-bottom: 2px solid #eceff1; margin-bottom: 4px;
               font-size: 0.78em; text-align: center; }
.top5-row { display: flex; gap: 8px; padding: 6px 0;
            border-bottom: 1px solid #f5f5f5; align-items: center; }
.top5-row:hover { background: #f5f5f5; }
.top5-cell { flex: 1; text-align: center; font-size: 0.8em; min-width: 0; }
.top5-name-hdr { flex: 0 0 130px; text-align: left; font-weight: 600; color: #555; }
.top5-name { flex: 0 0 130px; text-align: left; line-height: 1.4; }
.top5-err  { font-size: 0.85em; color: #c62828; }
.top5-img  { max-height: 110px; max-width: 100%; border: 1px solid #e0e0e0;
             border-radius: 4px; }
.top5-noimg { color: #bbb; font-style: italic; font-size: 0.8em; }
.top5-pill { display: block; font-size: 0.75em; color: white; border-radius: 8px;
             padding: 1px 7px; margin: 3px auto 0; width: fit-content; }

/* Summary table */
.metric-group.weight-group { background: #bf360c; }
.cell-wt { color: #e65100; }
.summary-table { width: 100%; border-collapse: collapse; background: white;
                 border-radius: 8px; overflow: hidden;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px; }
.summary-table th { background: #37474f; color: white; padding: 8px 10px;
                    text-align: center; font-size: 0.8em; text-transform: uppercase; }
.summary-table th.metric-group { border-left: 2px solid #546e7a; }
.summary-table th.run-subhdr { font-size: 0.75em; max-width: 100px;
                                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.summary-table th.sortable { cursor: pointer; }
.summary-table th.sortable:hover { background: #455a64; }
.summary-table td { padding: 6px 10px; border-bottom: 1px solid #eee;
                    font-size: 0.88em; text-align: center; }
.summary-table td.fix-name { text-align: left; font-size: 0.85em; }
.summary-table tr.data-row:hover { filter: brightness(0.96); }

/* Fixture detail card */
.nav-bar { display: flex; align-items: center; gap: 12px; padding: 12px 16px;
           background: white; border-radius: 8px; margin-bottom: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); position: sticky; top: 0;
           z-index: 100; }
.nav-bar button { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
                  background: white; cursor: pointer; font-size: 0.9em; }
.nav-bar button:hover:not(:disabled) { background: #e3f2fd; }
.nav-bar button:disabled { opacity: 0.4; cursor: default; }
#fixture-label { flex: 1; text-align: center; font-weight: 600; }
.kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px;
       padding: 1px 5px; font-size: 0.75em; font-family: monospace; color: #555; }

.n-col { display: flex; gap: 16px; margin: 12px 0; }
.n-col .col { flex: 1; min-width: 0; }
.n-col .col h3 { font-size: 0.9em; text-transform: uppercase;
                 border-bottom: 1px solid #eee; padding-bottom: 4px; }
@media (max-width: 1100px) { .n-col { flex-direction: column; } }

.fixture-card { background: white; border-radius: 8px; padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.input-section { margin-bottom: 8px; }
.input-section h3 { color: #555; font-size: 0.9em; text-transform: uppercase;
                    border-bottom: 1px solid #eee; padding-bottom: 4px; }

.gt-metrics { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
.gm-pill { background: #e3f2fd; border-radius: 12px; padding: 3px 10px;
           font-size: 0.8em; color: #1565c0; }
.metric-sub { color: #607d8b; font-weight: 400; font-size: 0.9em; }
.metrics-sub { padding: 4px 10px; font-size: 0.78em; color: #607d8b;
               background: #f5f7fa; border-radius: 6px; margin: 4px 0; }

.meta-metrics { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0; }
.mm-pill { background: #fff3e0; border-radius: 12px; padding: 3px 10px;
           font-size: 0.8em; color: #e65100; }

.interface-metrics { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0; }
.im-pill { background: #ede7f6; border-radius: 12px; padding: 3px 10px;
          font-size: 0.8em; color: #4527a0; }

.cost-summary { font-size: 0.78em; color: #888; cursor: pointer; margin-top: 6px; }
.cost-detail  { font-size: 0.8em; color: #666; padding: 6px 8px;
                background: #fafafa; border-radius: 4px; line-height: 1.8; }

.desc { background: #fafafa; padding: 10px; border-left: 3px solid #ccc;
        white-space: pre-wrap; max-height: 150px; overflow-y: auto; font-size: 0.85em; }
.note { color: #888; font-style: italic; font-size: 0.85em; }
.images { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
.view { text-align: center; }
.view img { max-height: 160px; border: 1px solid #ddd; border-radius: 4px; }
.view span { display: block; font-size: 0.65em; color: #888; margin-top: 2px; }
.input-img { max-height: 200px; max-width: 100%; border: 1px solid #ddd;
             border-radius: 4px; }

"""


JS = """\
let currentIdx = -1;
const total = document.querySelectorAll('.fixture-card').length;

function showSummary() {
  document.getElementById('summary-view').style.display = '';
  document.getElementById('detail-view').style.display = 'none';
  currentIdx = -1;
}

function showDetail(idx) {
  if (idx < 0 || idx >= total) return;
  document.getElementById('summary-view').style.display = 'none';
  document.getElementById('detail-view').style.display = '';
  document.querySelectorAll('.fixture-card').forEach(c => c.style.display = 'none');
  document.querySelectorAll('.fixture-card')[idx].style.display = '';
  currentIdx = idx;
  updateNav();
  window.scrollTo(0, 0);
}

function updateNav() {
  document.getElementById('prev-btn').disabled = (currentIdx <= 0);
  document.getElementById('next-btn').disabled = (currentIdx >= total - 1);
  const names = window._fixtureNames || [];
  document.getElementById('fixture-label').textContent =
    (currentIdx + 1) + ' / ' + total + ': ' + (names[currentIdx] || '');
}

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (currentIdx === -1) return;
  if (e.key === 'j' || e.key === 'ArrowRight') {
    e.preventDefault(); showDetail(currentIdx + 1);
  } else if (e.key === 'k' || e.key === 'ArrowLeft') {
    e.preventDefault(); showDetail(currentIdx - 1);
  } else if (e.key === 'Escape') {
    e.preventDefault(); showSummary();
  }
});

document.querySelectorAll('.sortable').forEach(function(th) {
  th.addEventListener('click', function() {
    const col = parseInt(this.dataset.col);
    const table = document.getElementById('summary-table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr.data-row'));
    const avgRow = tbody.querySelector('tr.avg-row');
    const asc = this.dataset.dir !== 'asc';
    this.dataset.dir = asc ? 'asc' : 'desc';
    rows.sort(function(a, b) {
      const va = parseFloat(a.children[col].dataset.v || 0);
      const vb = parseFloat(b.children[col].dataset.v || 0);
      return asc ? va - vb : vb - va;
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
    if (avgRow) tbody.appendChild(avgRow);
  });
});
"""


def _render_aggregate_band(runs: list[dict], labels: list[str]) -> str:
    """Per-run aggregate panel, one card per run, read from each
    run's ``run_summary.json``. Renders nothing when no run has a summary
    (e.g. comparing two pre-summary external submissions).
    """
    summaries = [r.get("run_summary") or {} for r in runs]
    if not any(summaries):
        return ""

    cards: list[str] = ['<div class="aggregate-band">']
    for ri, (summary, label) in enumerate(zip(summaries, labels)):
        _, text_col = _color(ri)
        agg = summary.get("aggregate_score")
        validity = summary.get("validity_rate")
        n = summary.get("n_fixtures") or 0
        n_valid = summary.get("n_valid", 0)
        n_invalid = summary.get("n_invalid", 0)
        n_missing = summary.get("n_missing", 0)
        by_task = summary.get("score_by_task_type") or {}

        agg_str = f"{float(agg):.3f}" if agg is not None else "-"
        validity_str = (
            f"{float(validity) * 100:.1f}% ({n_valid}/{n})"
            if validity is not None else "-"
        )

        card: list[str] = [
            f'<div class="agg-card" style="border-color:{text_col}">',
            f'<div class="agg-label" style="color:{text_col}">{html.escape(label)}</div>',
            f'<div class="agg-score"><b>{agg_str}</b><span>benchmark score</span></div>',
            f'<div class="agg-row"><span>validity</span><b>{validity_str}</b></div>',
        ]
        if n_invalid:
            card.append(f'<div class="agg-row"><span>invalid</span><b>{n_invalid}</b></div>')
        if n_missing:
            card.append(f'<div class="agg-row"><span>missing</span><b>{n_missing}</b></div>')
        for task_type, score in by_task.items():
            card.append(
                f'<div class="agg-row"><span>{html.escape(task_type)}</span>'
                f'<b>{float(score):.3f}</b></div>'
            )
        card.append("</div>")
        cards.append("\n".join(card))
    cards.append("</div>")
    return "\n".join(cards)


def generate_html(
    runs: list[dict],
    labels: list[str],
    entries: list[dict],
    mode: str,
) -> str:
    n_runs = len(runs)
    title_parts = " vs ".join(labels)
    title = f"Compare: {title_parts}"
    fixture_names_js = json.dumps([e["name"] for e in entries])

    p = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{_css()}</style>",
        "</head><body>",
    ]

    p.append('<div class="run-header">')
    p.append(f"<h1>{html.escape(title)}</h1>")
    p.append(f'<div style="font-size:0.85em;color:#888">Mode: {mode} · {len(entries)} fixtures</div>')
    p.append('<div class="run-labels">')
    for ri, label in enumerate(labels):
        _, text_col = _color(ri)
        p.append(f'<span style="color:{text_col};font-weight:600">'
                 f"Run {ri + 1}: {html.escape(label)}</span>")
    p.append("</div>")
    p.append(_render_aggregate_band(runs, labels))
    p.append("</div>")

    p.append(_render_config_diff(runs, labels))

    p.append('<div id="summary-view">')
    p.append(
        '<p style="color:#888;font-size:0.85em">'
        "Click a row for details. "
        '<span class="kbd">j</span>/<span class="kbd">k</span> to navigate, '
        '<span class="kbd">Esc</span> to return. '
        "Row color: <span style='background:#e8f5e9;padding:1px 6px'>≥0.90</span> "
        "<span style='background:#fff9c4;padding:1px 6px'>≥0.60</span> "
        "<span style='background:#ffebee;padding:1px 6px'>&lt;0.60</span> shape similarity. "
        "Winner cell highlighted green.</p>"
    )
    p.append(_render_top5(entries, n_runs, labels))
    p.append(_render_summary_table(entries, n_runs, labels))
    p.append("</div>")

    p.append('<div id="detail-view" style="display:none">')
    p.append('<div class="nav-bar">')
    p.append('<button onclick="showSummary()">\u2190 Summary</button>')
    p.append(
        '<button id="prev-btn" onclick="showDetail(currentIdx-1)">\u2190 Prev '
        '<span class="kbd">k</span></button>'
    )
    p.append('<span id="fixture-label"></span>')
    p.append(
        '<button id="next-btn" onclick="showDetail(currentIdx+1)">Next '
        '<span class="kbd">j</span> \u2192</button>'
    )
    p.append("</div>")
    for i, entry in enumerate(entries):
        p.append(_render_fixture_card(entry, i, n_runs, labels))
    p.append("</div>")

    p.append(f"<script>window._fixtureNames = {fixture_names_js};\n{JS}</script>")
    p.append("</body></html>")
    return "\n".join(p)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench report compare`` subcommand."""
    p = subparsers.add_parser(
        "compare",
        help="HTML comparison of 2+ experiment runs.",
        description="Compare experiment runs (2 or more).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  cadgenbench report compare results/run_a results/run_b\n"
            "  cadgenbench report compare run_a run_b run_c --union\n"
            "  cadgenbench report compare run_a run_b --label 'Opus 4.7' --label 'Sonnet 4.6'\n"
        ),
    )
    p.add_argument("runs", nargs="+", type=Path,
                   help="Paths to run directories (2 or more).")
    p.add_argument("--label", action="append", dest="labels", metavar="LABEL",
                   help="Custom label for a run (repeat once per run, in order).")
    p.add_argument("--union", action="store_true",
                   help="Show all fixtures (default: intersection only).")
    p.add_argument("-o", "--output", type=Path, help="Output HTML path.")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench report compare``."""
    if len(args.runs) < 2:
        print("Need at least 2 run directories to compare.")
        return 2

    runs = [_discover_run(p) for p in args.runs]
    n_runs = len(runs)

    if args.labels:
        if len(args.labels) != n_runs:
            print(f"Got {len(args.labels)} --label flags but {n_runs} runs.")
            return 2
        labels = args.labels
    else:
        labels = [_run_label(r, all_runs=runs) for r in runs]

    all_fixture_sets = [set(r["fixtures"].keys()) for r in runs]
    if args.union:
        all_names = sorted(set().union(*all_fixture_sets))
        mode = "union"
    else:
        all_names = sorted(set.intersection(*all_fixture_sets))
        mode = "intersection"

    entries = []
    for name in all_names:
        run_fixtures = [r["fixtures"].get(name) for r in runs]
        entries.append({"name": name, "runs": run_fixtures})

    if not entries:
        print("No fixtures to compare (intersection is empty; try --union)")
        return 1

    out = generate_html(runs, labels, entries, mode)
    out_path = args.output or Path(
        f"compare_{'_vs_'.join(r['timestamp'] for r in runs)}.html",
    )
    out_path.write_text(out)
    print(f"Wrote {out_path} ({len(entries)} fixtures, {out_path.stat().st_size // 1024} KB)")
    return 0
