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

"""``cadgenbench report single`` -- HTML report for one experiment run.

Shows a sortable summary table and per-fixture detail cards with input,
ground truth, output renders, metrics, and a debug panel with a per-turn
timeline/slider, full LLM responses, code, and execution results.

Navigation: click a row to view details, j/k or arrow keys to move
between fixtures, Escape to return to the summary table.

Usage::

    cadgenbench report single results/20260417_120000_sonnet-4-6
    cadgenbench report single results/<run_dir> -o report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

import yaml

VIEWS = ["iso", "front", "top", "right", "bottom"]


def _data_gt_dir() -> Path:
    """Resolve ``data/gt/`` via the shared cadgenbench data-dir helper."""
    from cadgenbench.common.paths import data_gt_dir
    return data_gt_dir()


try:
    from cadgenbench.eval.shape_similarity import METRIC_DISPLAY
except Exception:
    METRIC_DISPLAY = {}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _input_src(img_path: Path, inputs_dir: Path | None, base_url: str | None) -> str | None:
    """``<img src>`` for an input asset: a proxy URL (hosted) or base64 (local).

    When *base_url* (the fixture's input root) is given, the asset is
    referenced as ``{base_url}/<relpath>`` where ``<relpath>`` is the asset's
    path relative to *inputs_dir* (e.g. ``input.png`` or ``renders/iso.png``),
    so it streams lazily through the Space's input proxy instead of bloating
    the HTML. Falls back to base64 inlining otherwise (the portable local
    report)."""
    if base_url and inputs_dir is not None:
        try:
            rel = img_path.relative_to(inputs_dir).as_posix()
        except ValueError:
            rel = img_path.name
        return f"{base_url}/{rel}"
    return _data_uri(img_path)


def _fmt_metric(key: str, value: float) -> str:
    meta = METRIC_DISPLAY.get(key)
    if meta:
        return f"{format(value, meta.fmt)}{meta.suffix}"
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _inputs_dir_for(gt_dir: Path | None) -> Path | None:
    """Locate a fixture's inputs directory.

    Resolves two layouts, in order:

    1. **Combined sibling** (``gt_dir.parent.parent / "inputs" /
       <name>``): a local ``data/gt`` + ``data/inputs`` tree, used by
       local dev and the unit tests. Checked first so it resolves
       without touching the Hub (and hermetically in tests).
    2. **Canonical resolver**
       (:func:`cadgenbench.common.paths.data_inputs_dir`): the two-repo
       Hub layout (inputs and ground truth in separate dataset repos,
       the production Space/Jobs path), where the sibling above does not
       exist.

    Returns ``None`` when inputs can't be resolved; the input column then
    degrades gracefully (no crash), matching prior behavior.
    """
    if gt_dir is None:
        return None
    sibling = gt_dir.parent.parent / "inputs" / gt_dir.name
    if sibling.exists():
        return sibling
    try:
        from cadgenbench.common.paths import data_inputs_dir
        cand = data_inputs_dir() / gt_dir.name
    except Exception:
        return None
    return cand if cand.exists() else None


_STEP_SUFFIXES = (".step", ".stp")


def _load_description(gt_dir: Path) -> tuple[str, list[Path], list[Path], bool]:
    """Resolve the input column's text + media for a fixture.

    Returns ``(text, image_files, shape_render_pngs, wants_shape)``:

    - ``image_files``: input images to embed directly (e.g. the
      generation-task drawing ``input.png``).
    - ``shape_render_pngs``: canonical-view PNGs of an editing task's
      starting shape. Editing fixtures ship the starting solid as
      ``input.step``; the raw STEP can't be shown with ``<img>`` (it
      renders as a broken image), so we display its pre-rendered views
      from ``inputs/<fixture>/renders/`` exactly like the GT column.
    - ``wants_shape``: the fixture declared a STEP input. Lets the
      caller render an explicit "no input renders" note when the
      render PNGs weren't shipped, instead of a silent blank.
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
            # Shape input: rendered separately below, never embedded raw.
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


def discover_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()

    params: dict = {}
    params_path = run_dir / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())

    timestamp = params.get("timestamp", run_dir.name)

    fixtures: list[dict] = []
    for fixture_dir in sorted(run_dir.iterdir()):
        if not fixture_dir.is_dir():
            continue
        name = fixture_dir.name
        rp = fixture_dir / "result.json"
        if not rp.exists():
            continue

        result = json.loads(rp.read_text())

        gt_dir = _data_gt_dir() / name
        if not gt_dir.exists():
            gt_dir = None

        fixtures.append({
            "name": name,
            "result": result,
            "result_dir": fixture_dir,
            "gt_dir": gt_dir,
        })

    run_summary: dict = {}
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        try:
            run_summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    return {
        "run_dir": run_dir,
        "timestamp": timestamp,
        "params": params,
        "run_summary": run_summary,
        "fixtures": fixtures,
    }


# Canonical display order for GT metrics (matches compare_results.py).
# Headline-only: shape_similarity_score is shown next to cad_score and
# interface in summary tables. Per-component scores (point cloud F1, volume
# IoU, edge F1) live in the per-fixture detail card's secondary line.
SUMMARY_METRICS = [
    ("shape_similarity_score", "Shape Similarity", False),  # (key, label, lower_is_better)
]


def _quality_class(score: float | None) -> str:
    if score is None:
        return "q-none"
    if score >= 0.9:
        return "q-high"
    if score >= 0.6:
        return "q-mid"
    return "q-low"


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _legend_html(items: list[tuple[str, str]]) -> str:
    """Build a compact color-chip legend (``[(css_color, label), ...]``).

    Shared by the interface-overlay and edit-diff headings so both read the
    same way; each chip's color matches the corresponding render color exactly.
    """
    parts = ['<span class="legend">']
    for color, label in items:
        parts.append(
            f'<span class="legend-chip" style="background:{color}"></span>'
            f"{html.escape(label)}"
        )
    parts.append("</span>")
    return "".join(parts)


# Legend color chips, kept in lockstep with the render palettes so the report
# explains exactly what the viewer sees. Interface overlay:
# cadgenbench.eval.interface_match_viz (PART/KOR/KIR/DISAGREEMENT); edit diff:
# cadgenbench.common.viewer (DIFF_GHOST/ADDED/REMOVED).
_IFACE_LEGEND = [
    ("#2e73db", "your part"),
    ("#e64d4d", "keep-out (must stay empty)"),
    ("#33b34d", "keep-in (must be filled)"),
    ("#ffd900", "disagreement"),
]
_EDIT_DIFF_LEGEND = [
    ("#bdc4d1", "your output"),
    ("#2173f5", "extra material vs GT"),
    ("#e62929", "missing material vs GT"),
]


def _images_html(pngs: list[Path], *, base_url: str | None = None) -> str:
    """Render a row of view thumbnails.

    By default each image is inlined as a base64 data URI, which keeps the
    report a single self-contained file (the artifact a submitter produces
    locally with ``cadgenbench report single``). When *base_url* is given the
    images are referenced as ``{base_url}/{filename}`` instead; the hosted
    leaderboard passes the public render-bucket URL so the large WebP/PNG bytes
    live in object storage rather than bloating the HTML. ``base_url`` only
    changes how ``<img src>`` is written; it grants no write access and the
    local file is still used to know which views exist and in what order.
    """
    if not pngs:
        return ""
    parts = ['<div class="images">']
    for vp in pngs:
        src = f"{base_url}/{vp.name}" if base_url else _data_uri(vp)
        parts.append(
            f'<div class="view"><img src="{src}" alt="{vp.stem}" loading="lazy">'
            f"<span>{vp.stem}</span></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _render_gt_images(gt_dir: Path | None, *, base_url: str | None = None) -> str:
    """GT views for a fixture.

    Inlined as base64 by default (the portable local submitter report). When
    *base_url* is given (the hosted report) the views are referenced as
    ``{base_url}/renders/<view>.png`` instead — *base_url* is the fixture's GT
    root, e.g. the Space's token-holding GT proxy ``/gt/<fixture>``, so the
    private GT bytes are streamed lazily through the Space rather than baked
    into the HTML. The local PNGs are still enumerated either way to know which
    views exist and in what order.
    """
    if not gt_dir:
        return '<p class="note">GT source not found</p>'
    renders_dir = gt_dir / "renders"
    pngs = [renders_dir / f"{v}.png" for v in VIEWS if (renders_dir / f"{v}.png").exists()]
    renders_base = f"{base_url}/renders" if base_url else None
    return _images_html(pngs, base_url=renders_base) or '<p class="note">No GT renders</p>'


def _render_output_images(result_dir: Path, *, base_url: str | None = None) -> str:
    renders_dir = result_dir / "renders"
    if not renders_dir.is_dir():
        return '<p class="note">No output renders</p>'
    all_pngs = list(renders_dir.glob("*.png"))
    view_order = {v: i for i, v in enumerate(VIEWS)}
    pngs = sorted(all_pngs, key=lambda p: (view_order.get(p.stem, len(VIEWS)), p.stem))
    return _images_html(pngs, base_url=base_url) or '<p class="note">No output renders</p>'


def _render_edit_diff(result_dir: Path, *, base_url: str | None = None) -> str:
    """Embed the editing-task diff turntable (``renders/edit_diff.webp``).

    The ghost-body turntable lights up only the material that differs from GT
    (blue = added by the output, red = present in GT but missing), which makes
    a small or internal edit legible where the plain aligned output looks
    identical to the ground truth. Inlined as base64 by default; when *base_url*
    is given (hosted report) it is referenced from the public render bucket
    instead, which is what keeps the WebP out of the HTML. No fallback: when the
    WebP is absent (an invalid candidate that never rendered, or a fixture
    evaluated before the diff existed) the column shows an explicit note rather
    than reverting to the static views.
    """
    webp = result_dir / "renders" / "edit_diff.webp"
    if not webp.exists():
        return '<p class="note">No edit-diff render</p>'
    src = f"{base_url}/edit_diff.webp" if base_url else _data_uri(webp)
    return f'<img src="{src}" alt="edit diff" class="edit-diff-img" loading="lazy">'


# ---------------------------------------------------------------------------
# Fixture card
# ---------------------------------------------------------------------------

def _render_fixture_card(
    fix: dict,
    idx: int,
    *,
    render_base_url: str | None = None,
    gt_base_url: str | None = None,
    input_base_url: str | None = None,
) -> str:
    result = fix["result"]
    gt_dir = fix["gt_dir"]
    result_dir = fix["result_dir"]
    # Per-fixture base URLs for the hosted report (display only). When set, the
    # corresponding images are referenced by URL + lazy-loaded instead of
    # base64-inlined, so the heavy bytes stay out of the HTML and only the
    # fixture the reader actually opens fetches them:
    #   - candidate renders + interface overlay: public render bucket
    #     (``{render_base_url}/<fixture>/<file>``).
    #   - GT views + ground-truth PDF: the Space's token-holding GT proxy
    #     (``{gt_base_url}/<fixture>/...``), because GT is private.
    #   - input drawings / starting-shape renders: the Space's input proxy
    #     (``{input_base_url}/<fixture>/...``).
    # All ``None`` for a local submitter report, which inlines base64 so the
    # file stays self-contained and portable.
    fixture_base = f"{render_base_url}/{fix['name']}" if render_base_url else None
    gt_base = f"{gt_base_url}/{fix['name']}" if gt_base_url else None
    input_base = f"{input_base_url}/{fix['name']}" if input_base_url else None
    gt_m = result.get("gt_metrics", {})
    cad_score = result.get("cad_score")
    headline = cad_score if cad_score is not None else gt_m.get("shape_similarity_score")

    p = [f'<div class="fixture-card" data-idx="{idx}" style="display:none">']

    quality_cls = _quality_class(headline if gt_m or cad_score is not None else None)
    status = result.get("status", "?")
    status_cls = {
        "valid": "status-valid",
        "invalid": "status-invalid",
        "missing": "status-missing",
    }.get(status, "status-unknown")
    p.append(
        f'<h2 class="card-title {quality_cls}">{html.escape(fix["name"])} '
        f'<span class="tag {status_cls}">{html.escape(status)}</span>'
        f"</h2>"
    )

    # Headline metrics (top of card, CAD Score / Shape / Interface / Topo).
    iface_m = result.get("interface_metrics", {})
    iface_score = iface_m.get("score")
    shape_score = gt_m.get("shape_similarity_score")
    topo_m = result.get("topology_metrics") or {}
    topo_score = topo_m.get("score")
    # Editing fixtures: the shape axis is renormalized against the no-op
    # input baseline and cad_score is a weighted (not equal) mean. When
    # present, surface the renormalized shape value + the no-op baseline.
    edit_m = result.get("edit_metrics") or {}
    is_editing = bool(edit_m)
    shape_renorm = edit_m.get("shape_similarity_renormalized")
    shape_baseline = edit_m.get("baseline_shape_similarity")
    if any(v is not None for v in (cad_score, shape_score, iface_score, topo_score)):
        n_components = sum(
            1 for v in (shape_score, iface_score, topo_score) if v is not None
        )
        p.append('<div class="headline-metrics">')
        if cad_score is not None:
            cad_sub = (
                "editing &middot; weighted 0.5 / 0.3 / 0.2 "
                "(shape renormalized vs no-op)"
                if is_editing
                else "weighted 0.4 / 0.4 / 0.2 over "
                f"{n_components} available component(s)"
            )
            p.append(
                f'<div class="headline-pill headline-cad">'
                f'<span class="headline-label">CAD Score</span>'
                f'<span class="headline-value">'
                f"{_fmt_metric('cad_score', cad_score)}</span>"
                f'<span class="headline-sub">{cad_sub}</span>'
                f'</div>'
            )
        if shape_score is not None:
            if is_editing and shape_renorm is not None:
                shape_value = _fmt_metric("shape_similarity_score", shape_renorm)
                shape_sub = (
                    f'<span class="headline-sub">renormalized &middot; '
                    f"no-op b={float(shape_baseline):.3f} &middot; "
                    f"raw {float(shape_score):.3f}</span>"
                )
            else:
                shape_value = _fmt_metric("shape_similarity_score", shape_score)
                shape_sub = ""
            p.append(
                f'<div class="headline-pill headline-shape">'
                f'<span class="headline-label">Shape Similarity</span>'
                f'<span class="headline-value">{shape_value}</span>'
                f"{shape_sub}"
                f'</div>'
            )
        if iface_score is not None:
            n_ctx = len(iface_m.get("contexts", {}))
            p.append(
                f'<div class="headline-pill headline-iface">'
                f'<span class="headline-label">Interface match</span>'
                f'<span class="headline-value">{float(iface_score):.3f}</span>'
                f'<span class="headline-sub">{n_ctx} context(s)</span>'
                f'</div>'
            )
        if topo_score is not None:
            cand_b = topo_m.get("candidate") or {}
            gt_b = topo_m.get("gt") or {}
            p.append(
                f'<div class="headline-pill headline-topo">'
                f'<span class="headline-label">Topo match</span>'
                f'<span class="headline-value">{float(topo_score):.3f}</span>'
                f'<span class="headline-sub">'
                f'cand ({cand_b.get("b0")},{cand_b.get("b1")},{cand_b.get("b2")}) '
                f'vs gt ({gt_b.get("b0")},{gt_b.get("b1")},{gt_b.get("b2")})'
                f'</span>'
                f'</div>'
            )
        p.append("</div>")

    # Shape component breakdown (compact, muted), right under the headline.
    component_keys = ("shape_point_cloud_f1", "shape_volume_iou")
    components = [(k, gt_m.get(k)) for k in component_keys if gt_m.get(k) is not None]
    if components:
        parts: list[str] = []
        for k, v in components:
            meta = METRIC_DISPLAY.get(k)
            label = meta.label if meta else k
            parts.append(f"{html.escape(label)}: {_fmt_metric(k, v)}")
        components_label = (
            "Shape components (pre-renorm)" if is_editing else "Shape components"
        )
        p.append(
            f'<div class="metrics-sub">{components_label} &middot; '
            f'{" &middot; ".join(parts)}</div>'
        )

    # Three-column: Input | GT | Output
    p.append('<div class="three-col">')

    p.append('<div class="col">')
    p.append("<h3>Input</h3>")
    if gt_dir:
        desc_text, input_imgs, input_shape_pngs, wants_shape = _load_description(gt_dir)
        if desc_text:
            p.append(f'<p class="desc">{html.escape(desc_text)}</p>')
        inputs_dir = _inputs_dir_for(gt_dir)
        for img_path in input_imgs:
            src = _input_src(img_path, inputs_dir, input_base)
            if src:
                p.append(f'<img src="{src}" alt="input" class="input-img" loading="lazy">')
        # Editing tasks: show the starting shape's canonical views
        # (same grid as GT/Output). Falls back to a note if the render
        # PNGs weren't shipped with the input fixture.
        if input_shape_pngs:
            shape_base = f"{input_base}/renders" if input_base else None
            p.append(_images_html(input_shape_pngs, base_url=shape_base))
        elif wants_shape:
            p.append('<p class="note">No input renders</p>')
        gt_pdf = gt_dir / "ground_truth.pdf"
        if gt_pdf.exists():
            src = f"{gt_base}/ground_truth.pdf" if gt_base else _data_uri(gt_pdf)
            if src:
                p.append(f'<iframe src="{src}" class="pdf-embed" loading="lazy"></iframe>')
    p.append("</div>")

    if is_editing:
        # Editing fixtures: the GT and aligned-output 4-view grids are visually
        # near-identical for a small or internal edit, so replace both with the
        # single ghost-diff turntable that isolates what actually changed. GT
        # column is dropped entirely; the diff already carries the GT reference.
        p.append('<div class="col">')
        p.append(
            "<h3>Output vs ground truth (edit diff) "
            f"{_legend_html(_EDIT_DIFF_LEGEND)}</h3>"
        )
        p.append(_render_edit_diff(result_dir, base_url=fixture_base))
        p.append("</div>")
    else:
        p.append('<div class="col">')
        p.append("<h3>Ground Truth</h3>")
        p.append(_render_gt_images(gt_dir, base_url=gt_base))
        p.append("</div>")

        p.append('<div class="col">')
        p.append("<h3>Output (aligned)</h3>")
        p.append(_render_output_images(result_dir, base_url=fixture_base))
        p.append("</div>")

    p.append("</div>")  # three-col

    # Interface overlay (only when fixture has sub-volumes; yellow = disagreement).
    # The overlay is a per-submission artifact, so on the hosted report it is
    # referenced from the public render bucket (the eval job uploads it next to
    # the turntable renders) rather than base64-inlined.
    overlay = result_dir / "interface_overlay.png"
    if overlay.exists():
        src = (
            f"{fixture_base}/interface_overlay.png"
            if fixture_base
            else _data_uri(overlay)
        )
        if src:
            p.append('<div class="iface-overlay">')
            p.append(
                "<h3>Interface overlay "
                f"{_legend_html(_IFACE_LEGEND)}</h3>"
            )
            p.append(
                f'<img src="{src}" alt="interface overlay" '
                f'class="iface-overlay-img" loading="lazy">'
            )
            p.append("</div>")

    # Mesh validation, prominent pills (always run independently by pipeline)
    val = result.get("validation")
    if val and "error" not in val:
        def _bool_pill(label: str, value: bool) -> str:
            cls = "val-ok" if value else "val-fail"
            mark = "\u2713" if value else "\u2717"
            return f'<span class="val-pill {cls}">{mark} {html.escape(label)}</span>'

        p.append('<div class="metrics-bar">')
        p.append(_bool_pill("valid", val.get("is_valid", False)))
        p.append(_bool_pill("watertight", val.get("is_watertight", False)))
        p.append(f'<span class="val-pill val-info">solids: {val.get("solid_count", "?")}</span>')
        p.append(f'<span class="val-pill val-info">faces: {val.get("face_count", "?")}</span>')
        bb = val.get("bbox", {})
        if bb:
            p.append(
                f'<span class="val-pill val-info">'
                f'bbox: {bb.get("x","?")}×{bb.get("y","?")}×{bb.get("z","?")} mm'
                f'</span>'
            )
        errs = val.get("topology_errors", [])
        if errs:
            p.append(f'<span class="val-pill val-fail">errors: {html.escape(", ".join(errs[:3]))}</span>')
        p.append("</div>")
    elif val and "error" in val:
        p.append(
            f'<div class="metrics-bar"><span class="val-pill val-fail">'
            f'validation error: {html.escape(val["error"])}</span></div>'
        )

    p.append("</div>")  # fixture-card
    return "\n".join(p)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _render_summary_table(fixtures: list[dict]) -> str:
    has_cad_score = any(f["result"].get("cad_score") is not None for f in fixtures)
    present_metrics = [
        (mk, mlabel) for mk, mlabel, _ in SUMMARY_METRICS
        if any(f["result"].get("gt_metrics", {}).get(mk) is not None for f in fixtures)
    ]
    has_interface = any(
        (f["result"].get("interface_metrics") or {}).get("score") is not None
        for f in fixtures
    )
    has_topo = any(
        (f["result"].get("topology_metrics") or {}).get("score") is not None
        for f in fixtures
    )
    col_idx = 2
    header = (
        "<thead><tr>"
        "<th>Sample</th>"
        "<th>Status</th>"
    )
    if has_cad_score:
        header += f'<th class="sortable" data-col="{col_idx}">CAD Score</th>'
        col_idx += 1
    for _, mlabel in present_metrics:
        header += f'<th class="sortable" data-col="{col_idx}">{html.escape(mlabel)}</th>'
        col_idx += 1
    if has_interface:
        header += f'<th class="sortable" data-col="{col_idx}">Interface</th>'
        col_idx += 1
    if has_topo:
        header += f'<th class="sortable" data-col="{col_idx}">Topo</th>'
        col_idx += 1
    header += "</tr></thead><tbody>"

    p = ['<table class="summary-table" id="summary-table">', header]

    for i, fix in enumerate(fixtures):
        result = fix["result"]
        gt_m = result.get("gt_metrics", {})
        iface_m = result.get("interface_metrics") or {}
        topo_m = result.get("topology_metrics") or {}
        cad = result.get("cad_score")
        headline = cad if cad is not None else gt_m.get("shape_similarity_score")
        row_cls = _quality_class(headline if gt_m or cad is not None else None)
        status = result.get("status", "?")
        status_cls = {
            "valid": "status-valid",
            "invalid": "status-invalid",
            "missing": "status-missing",
        }.get(status, "status-unknown")

        row = (
            f'<tr class="{row_cls}" onclick="showDetail({i})" style="cursor:pointer">'
            f"<td>{html.escape(fix['name'])}</td>"
            f'<td><span class="status-pill {status_cls}">{html.escape(status)}</span></td>'
        )
        if has_cad_score:
            v = cad
            display = _fmt_metric("cad_score", v) if v is not None else "-"
            row += f'<td data-v="{v if v is not None else -1}"><b>{display}</b></td>'
        for mk, _ in present_metrics:
            v = gt_m.get(mk)
            display = _fmt_metric(mk, v) if v is not None else "-"
            row += f'<td data-v="{v if v is not None else -1}">{display}</td>'
        if has_interface:
            v = iface_m.get("score")
            display = f"{v:.3f}" if v is not None else "-"
            row += f'<td data-v="{v if v is not None else -1}">{display}</td>'
        if has_topo:
            v = topo_m.get("score")
            display = f"{v:.3f}" if v is not None else "-"
            row += f'<td data-v="{v if v is not None else -1}">{display}</td>'
        row += "</tr>"
        p.append(row)

    p.append("</tbody></table>")
    return "\n".join(p)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """\
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1600px; margin: 0 auto; padding: 20px; background: #f8f9fa; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { margin-top: 0; }
.tag { font-size: 0.6em; color: #666; font-weight: normal; font-family: monospace;
       margin-left: 6px; }

/* Quality classes */
.q-high { background: #e8f5e9; }
.q-mid  { background: #fff9c4; }
.q-low  { background: #ffebee; }
.q-none { background: #f5f5f5; }

.run-header { background: white; border-radius: 8px; padding: 16px 20px;
              margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.run-header-top { display: flex; align-items: center; justify-content: space-between;
                  gap: 16px; flex-wrap: wrap; }
.run-header-top h1 { border-bottom: none; padding-bottom: 0; margin: 0; }
.download-zip { background: #37474f; color: #fff; text-decoration: none;
                padding: 8px 16px; border-radius: 6px; font-size: 0.9em;
                font-weight: 600; white-space: nowrap; flex-shrink: 0; }
.download-zip:hover { background: #455a64; }
.run-meta { color: #666; font-size: 0.9em; margin-top: 4px; }
.run-meta span { margin-right: 16px; }
.run-stats { margin-top: 8px; font-size: 0.95em; }
.run-stats span { margin-right: 20px; font-weight: 500; }

.summary-table { width: 100%; border-collapse: collapse; background: white;
                 border-radius: 8px; overflow: hidden;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.summary-table th { background: #37474f; color: white; padding: 10px 12px;
                    text-align: left; font-size: 0.85em; text-transform: uppercase;
                    letter-spacing: 0.05em; }
.summary-table th.sortable { cursor: pointer; }
.summary-table th.sortable:hover { background: #455a64; }
.summary-table td { padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.9em; }
.summary-table tr:hover { filter: brightness(0.97); }

.nav-bar { display: flex; align-items: center; gap: 12px; padding: 12px 16px;
           background: white; border-radius: 8px; margin-bottom: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); position: sticky; top: 0; z-index: 100; }
.nav-bar button { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
                  background: white; cursor: pointer; font-size: 0.9em; }
.nav-bar button:hover:not(:disabled) { background: #e3f2fd; }
.nav-bar button:disabled { opacity: 0.4; cursor: default; }
#fixture-label { flex: 1; text-align: center; font-weight: 600; }
.kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px;
       padding: 1px 5px; font-size: 0.75em; font-family: monospace; color: #555; }

.three-col { display: flex; gap: 20px; margin: 16px 0; }
.three-col .col { flex: 1; min-width: 0; }
.three-col .col h3 { color: #555; font-size: 0.9em; text-transform: uppercase;
                     border-bottom: 1px solid #eee; padding-bottom: 4px; }
@media (max-width: 1000px) { .three-col { flex-direction: column; } }

.fixture-card { background: white; border-radius: 8px; padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.card-title { padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; }

.desc { background: #fafafa; padding: 10px; border-left: 3px solid #ccc;
        white-space: pre-wrap; max-height: 200px; overflow-y: auto;
        font-size: 0.9em; margin: 4px 0; }
.note { color: #888; font-style: italic; font-size: 0.9em; }
.images { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.view { text-align: center; }
.view img { max-height: 180px; border: 1px solid #ddd; border-radius: 4px; }
.view span { display: block; font-size: 0.7em; color: #888; margin-top: 2px; }
.input-img { max-height: 250px; max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.edit-diff-img { display: block; max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }
.pdf-embed { width: 100%; height: 400px; border: 1px solid #ddd;
             border-radius: 4px; margin-top: 8px; }

.metrics-bar { display: flex; gap: 16px; flex-wrap: wrap; padding: 10px 14px;
               background: #e3f2fd; border-radius: 6px; margin: 8px 0;
               font-size: 0.9em; align-items: center; }
.gt-metric  { font-weight: 500; color: #1565c0; }
.metric-sub { color: #607d8b; font-weight: 400; font-size: 0.95em; }
.metrics-sub { padding: 6px 14px; font-size: 0.82em; color: #607d8b;
               background: #f5f7fa; border-radius: 6px; margin: 4px 0 8px; }

/* Headline metrics row, top of fixture card */
.headline-metrics { display: flex; gap: 12px; margin: 8px 0 12px;
                    flex-wrap: wrap; }
.headline-pill { flex: 1 1 200px; min-width: 200px;
                 background: #ffffff; border: 1px solid #cfd8dc;
                 border-radius: 10px; padding: 12px 16px;
                 display: flex; flex-direction: column; gap: 2px;
                 box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.headline-label { font-size: 0.78em; text-transform: uppercase;
                  letter-spacing: 0.04em; color: #607d8b; font-weight: 600; }
.headline-value { font-size: 1.8em; font-weight: 700; color: #1a1a1a;
                  line-height: 1.1; }
.headline-sub { font-size: 0.78em; color: #90a4ae; font-style: italic; }
.headline-cad { border-left: 4px solid #37474f; }
.headline-cad .headline-value { color: #37474f; }
.headline-shape { border-left: 4px solid #1565c0; }
.headline-shape .headline-value { color: #1565c0; }
.headline-iface { border-left: 4px solid #4527a0; }
.headline-iface .headline-value { color: #4527a0; }
.headline-topo { border-left: 4px solid #006d77; }
.headline-topo .headline-value { color: #006d77; }
.iface-overlay { margin: 16px 0; }
.iface-overlay h3 { color: #4527a0; font-size: 0.9em; text-transform: uppercase;
                    border-bottom: 1px solid #eee; padding-bottom: 4px;
                    display: flex; align-items: baseline; gap: 10px; }
.iface-overlay-legend { color: #888; font-size: 0.78em; font-weight: 400;
                        text-transform: none; }
/* Color-chip legend (interface overlay + edit diff). Chip colors mirror the
   render palettes; see _IFACE_LEGEND / _EDIT_DIFF_LEGEND. */
.legend { color: #6b7785; font-size: 0.78em; font-weight: 400;
          text-transform: none; letter-spacing: normal; line-height: 1.6; }
.legend-chip { display: inline-block; width: 11px; height: 11px;
               border-radius: 3px; vertical-align: middle;
               margin: 0 5px 0 14px; border: 1px solid rgba(0,0,0,0.18); }
.iface-overlay-img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
                     display: block; }
.meta-bar   { background: #fff8e1; }
.wt-metric  { font-weight: 500; color: #e65100; }
.iface-bar  { background: #ede7f6; }
.iface-metric { font-weight: 500; color: #4527a0; }

.val-pill { padding: 2px 10px; border-radius: 12px; font-size: 0.88em; font-weight: 500; }
.val-ok   { background: #e8f5e9; color: #2e7d32; }
.val-fail { background: #ffebee; color: #c62828; }
.val-info { background: #f5f5f5; color: #555; }

/* ── Per-fixture status pill (valid / invalid / missing) ─────── */
.status-pill { padding: 2px 10px; border-radius: 12px; font-size: 0.82em;
               font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.status-valid   { background: #e8f5e9; color: #2e7d32; }
.status-invalid { background: #ffebee; color: #c62828; }
.status-missing { background: #f5f5f5; color: #757575; }
.status-unknown { background: #f5f5f5; color: #999; }

/* Similarity Q&A */
.qa-list { list-style: none; margin: 0; padding: 4px 12px 8px; }
.qa-item { padding: 5px 0; border-bottom: 1px solid #f0f0f0; font-size: 0.82em;
           line-height: 1.5; }
.qa-item:last-child { border-bottom: none; }
.qa-pass { color: #2e7d32; font-weight: 600; }
.qa-fail { color: #c62828; font-weight: 600; }
.qa-reasoning { color: #555; }
.method-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
                font-size: 0.78em; font-weight: 600; margin-right: 4px; }
.method-visual { background: #e3f2fd; color: #1565c0; }
.method-code   { background: #f3e5f5; color: #6a1b9a; }
.code-snippet-toggle { font-size: 0.78em; color: #888; cursor: pointer;
                        text-decoration: underline; }
"""

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

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
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = this.dataset.dir !== 'asc';
    this.dataset.dir = asc ? 'asc' : 'desc';
    rows.sort(function(a, b) {
      const va = parseFloat(a.children[col].dataset.v || 0);
      const vb = parseFloat(b.children[col].dataset.v || 0);
      return asc ? va - vb : vb - va;
    });
    rows.forEach(function(r) { tbody.appendChild(r); });
  });
});

// Deep-link: opening the report at `#fixture=<name>` (or `#idx=<n>`)
// jumps straight to that fixture's detail card instead of the summary
// view. The leaderboard gallery links thumbnails this way so a click
// lands on the right fixture. Inert (stays on the summary view) when
// there is no hash or the name doesn't match a fixture.
function openHashTarget() {
  const hash = (window.location.hash || '').replace(/^#/, '');
  if (!hash) return;
  const params = new URLSearchParams(hash);
  const names = window._fixtureNames || [];
  let idx = -1;
  if (params.has('fixture')) {
    idx = names.indexOf(params.get('fixture'));
  } else if (params.has('idx')) {
    idx = parseInt(params.get('idx'), 10);
  }
  if (idx >= 0 && idx < total) showDetail(idx);
}
openHashTarget();
window.addEventListener('hashchange', openHashTarget);
"""


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _render_run_summary_header(summary: dict, n_fixtures_fallback: int) -> str:
    """Render the aggregate-score panel from ``run_summary.json``.

    Gracefully degrades to a single "n fixtures" line when no summary
    file is present (e.g. the run was evaluated before run_summary.json
    existed).
    """
    if not summary:
        return (
            f'<div class="run-stats">'
            f'<span>{n_fixtures_fallback} samples</span>'
            f'</div>'
        )

    n = summary.get("n_fixtures", n_fixtures_fallback)
    n_valid = summary.get("n_valid", 0)
    n_invalid = summary.get("n_invalid", 0)
    n_missing = summary.get("n_missing", 0)
    aggregate = summary.get("aggregate_score")
    validity_rate = summary.get("validity_rate")
    by_task = summary.get("score_by_task_type") or {}

    parts: list[str] = ['<div class="run-stats">']
    if aggregate is not None:
        parts.append(
            f'<span>Benchmark score: <b>{float(aggregate):.3f}</b></span>'
        )
    if validity_rate is not None:
        parts.append(
            f'<span>Validity rate: <b>{float(validity_rate) * 100:.1f}%</b> '
            f'({n_valid}/{n})</span>'
        )
    if n_invalid:
        parts.append(f'<span>Invalid: <b>{n_invalid}</b></span>')
    if n_missing:
        parts.append(f'<span>Missing: <b>{n_missing}</b></span>')
    parts.append(f'<span>{n} samples</span>')
    parts.append("</div>")

    if by_task:
        task_parts: list[str] = ['<div class="run-stats">']
        for task_type, score in by_task.items():
            task_parts.append(
                f'<span>{html.escape(task_type)}: '
                f'<b>{float(score):.3f}</b></span>'
            )
        task_parts.append("</div>")
        parts.append("\n".join(task_parts))

    return "\n".join(parts)


def generate_html(
    run: dict,
    *,
    render_base_url: str | None = None,
    gt_base_url: str | None = None,
    input_base_url: str | None = None,
    download_url: str | None = None,
) -> str:
    """Build the single-run report HTML.

    Args:
        run: The discovered run (see :func:`discover_run`).
        render_base_url: Optional public base URL for the candidate renders and
            interface overlay. When ``None`` (a submitter running ``cadgenbench
            report single`` locally), these are inlined as base64. When set (the
            hosted leaderboard), they are referenced as
            ``{render_base_url}/<fixture>/<file>`` from the public render bucket.
        gt_base_url: Optional base URL for the ground-truth views + PDF. When
            ``None`` they are base64-inlined. When set (hosted report) they are
            referenced as ``{gt_base_url}/<fixture>/...`` — GT is private, so
            this points at the Space's token-holding GT proxy.
        input_base_url: Optional base URL for input drawings / starting-shape
            renders. ``None`` inlines base64; set (hosted report) references
            them as ``{input_base_url}/<fixture>/...`` via the Space input proxy.
        download_url: Optional URL of the submission's STEP zip. When set, a
            "Download submission ZIP" button is rendered in the run header
            (mirrors the leaderboard gallery's per-submission download). ``None``
            (a local run with no published artifact) omits the button.

    The three ``*_base_url`` knobs are display-only: they only change
    ``<img src>`` and grant no storage write access. Set together on the hosted
    report so every heavy asset is a lazy-loaded link and the HTML stays small;
    left ``None`` for the local report so it remains one self-contained,
    portable file.
    """
    fixtures = run["fixtures"]
    timestamp = run["timestamp"]
    summary = run.get("run_summary") or {}

    fixture_names_js = json.dumps([f["name"] for f in fixtures])

    title = f"CADGenBench / {timestamp}"
    p = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>Results: {html.escape(title)}</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
    ]

    # Run header, agnostic to who produced the candidates. The title row is a
    # flex container so an optional "Download submission ZIP" button sits at the
    # top-right (mirrors the gallery's per-submission download).
    p.append('<div class="run-header">')
    p.append('<div class="run-header-top">')
    p.append(f"<h1>Results: {html.escape(title)}</h1>")
    if download_url:
        href = html.escape(str(download_url), quote=True)
        p.append(
            f'<a class="download-zip" href="{href}" download '
            f'rel="noopener">&#11015; Download submission ZIP</a>'
        )
    p.append("</div>")
    p.append(_render_run_summary_header(summary, len(fixtures)))
    p.append("</div>")

    # Summary view
    p.append('<div id="summary-view">')
    p.append(
        '<p style="color:#888;font-size:0.85em">'
        "Click a row to view details. "
        '<span class="kbd">j</span>/<span class="kbd">k</span> '
        "to navigate, "
        '<span class="kbd">Esc</span> to return. '
        "Row color: <span style='background:#e8f5e9;padding:1px 6px'>&ge;0.90</span> "
        "<span style='background:#fff9c4;padding:1px 6px'>&ge;0.60</span> "
        "<span style='background:#ffebee;padding:1px 6px'>&lt;0.60</span> shape similarity.</p>"
    )
    p.append(_render_summary_table(fixtures))
    p.append("</div>")

    # Detail view
    p.append('<div id="detail-view" style="display:none">')
    p.append('<div class="nav-bar">')
    p.append('<button onclick="showSummary()">&#8592; Summary</button>')
    p.append(
        '<button id="prev-btn" onclick="showDetail(currentIdx-1)">&#8592; Prev '
        '<span class="kbd">k</span></button>'
    )
    p.append('<span id="fixture-label"></span>')
    p.append(
        '<button id="next-btn" onclick="showDetail(currentIdx+1)">Next '
        '<span class="kbd">j</span> &#8594;</button>'
    )
    p.append("</div>")
    for i, fix in enumerate(fixtures):
        p.append(_render_fixture_card(
            fix, i,
            render_base_url=render_base_url,
            gt_base_url=gt_base_url,
            input_base_url=input_base_url,
        ))
    p.append("</div>")

    p.append(f"<script>window._fixtureNames = {fixture_names_js};\n{JS}</script>")
    p.append("</body></html>")
    return "\n".join(p)


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``cadgenbench report single`` subcommand."""
    p = subparsers.add_parser(
        "single",
        help="HTML report for one experiment run.",
        description="Generate interactive HTML for a single result directory.",
    )
    p.add_argument("run_dir", type=Path, help="Path to a result run directory.")
    p.add_argument("-o", "--output", type=Path, help="Output HTML path.")
    p.add_argument(
        "--render-base-url",
        default=None,
        help=(
            "Optional public base URL for the candidate renders + interface "
            "overlay. Omit (default) to inline as base64 for a self-contained "
            "report; set it (hosted leaderboard) to reference them as "
            "<base>/<fixture>/<file>."
        ),
    )
    p.add_argument(
        "--gt-base-url",
        default=None,
        help=(
            "Optional base URL for the ground-truth views + PDF (e.g. the "
            "Space GT proxy '/gt'). Omit to inline base64; set to reference "
            "as <base>/<fixture>/renders/<view>.png."
        ),
    )
    p.add_argument(
        "--input-base-url",
        default=None,
        help=(
            "Optional base URL for input drawings / starting-shape renders "
            "(e.g. the Space input proxy '/task-input'). Omit to inline "
            "base64; set to reference as <base>/<fixture>/<relpath>."
        ),
    )
    p.add_argument(
        "--download-url",
        default=None,
        help=(
            "Optional URL of the submission's STEP zip. When set, a 'Download "
            "submission ZIP' button is shown in the run header (hosted report); "
            "omit for a local report with no published artifact."
        ),
    )
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Execute ``cadgenbench report single``."""
    run_data = discover_run(args.run_dir)

    if not run_data["fixtures"]:
        print(f"No samples found in {args.run_dir}")
        return 1

    html_out = generate_html(
        run_data,
        render_base_url=args.render_base_url,
        gt_base_url=args.gt_base_url,
        input_base_url=args.input_base_url,
        download_url=args.download_url,
    )
    out_path = args.output or Path(f"results_{run_data['timestamp']}.html")
    out_path.write_text(html_out)
    print(
        f"Wrote {out_path} ({len(run_data['fixtures'])} samples, "
        f"{out_path.stat().st_size // 1024} KB)"
    )
    return 0
