#!/usr/bin/env python
"""Sanity-check a benchmark ground-truth directory.

Walks one or all fixtures under ``data/gt/`` (relative to the repo root)
and asserts every rule from ``docs/benchmark/authoring.md`` §
*Sanity checks*. Exits non-zero on any failure; prints a per-fixture
table summarising what passed / failed.

Usage::

    python _to_move_to_dataset_repo/sanity_check_gt.py --all
    python _to_move_to_dataset_repo/sanity_check_gt.py <fixture-name>
    python _to_move_to_dataset_repo/sanity_check_gt.py --fixture-path data/gt/jig-01-single-hole-plate

Each rule is captured as a :class:`Check` and reported as PASS / FAIL
with a short reason. Failures don't abort other checks within the same
fixture; we surface all problems at once for efficient triage.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cadgenbench.common.validity import analyze_step
from cadgenbench.common.mesh import (
    deflection_for_bbox,
    tessellate_and_validate,
)
from cadgenbench.common.measurements import _compute_bbox
from cadgenbench.eval.interface_match import (
    SubVolume,
    discover_sub_volumes,
)
from cadgenbench.eval.topo_match import compute_betti_for_step
from cadgenbench.eval.booleans import (
    intersect,
    manifold_volume,
    mesh_to_manifold,
)

GT_STEP_NAME = "ground_truth.step"

# Tolerance defaults, all bbox-relative so they scale with part size.
POSE_REL_TOL = 0.001        # |centroid|/diag and |L_x-L_y|/diag tolerance
KOR_KIR_EPS = 0.01          # max volume bleed for KOR / KIR fit-type consistency
CLIP_OFFSET_REL = 0.05      # sub-volume AABB allowed outside GT AABB
INFLATED_MARGIN_FRACTION = 0.20
INFLATED_MARGIN_FLOOR = 2.0  # mm, matches interface_match._build_sub_volume_cache


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One sanity check on one fixture."""

    rule: str
    ok: bool
    detail: str  # one-line human-readable reason, never empty


@dataclass
class FixtureReport:
    fixture: str
    checks: list[Check]

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def _check_validity(gt_step: Path) -> Check:
    try:
        a = analyze_step(gt_step)
    except Exception as exc:
        return Check("validity", False, f"analyze_step raised: {exc}")
    v = a.validation
    if v.is_valid:
        return Check(
            "validity",
            True,
            f"is_valid=True (watertight={v.is_watertight}, solids="
            f"{a.measurements.solid_count})",
        )
    reason = v.topology_errors[0] if v.topology_errors else "is_valid=False"
    return Check("validity", False, reason)


def _check_canonical_pose(gt_step: Path) -> tuple[Check, Check]:
    """Centroid-at-origin + axis ordering checks (rules 1, 2 of the spec)."""
    try:
        a = analyze_step(gt_step)
    except Exception as exc:
        msg = f"analyze_step raised: {exc}"
        return (
            Check("canonical_pose.centroid", False, msg),
            Check("canonical_pose.axes", False, msg),
        )

    bb = a.measurements.bounding_box
    diag = bb.diagonal
    tol = max(0.001, POSE_REL_TOL * diag)

    cx = (bb.x_min + bb.x_max) / 2
    cy = (bb.y_min + bb.y_max) / 2
    cz = (bb.z_min + bb.z_max) / 2
    max_off = max(abs(cx), abs(cy), abs(cz))
    centroid_ok = max_off <= tol
    centroid = Check(
        "canonical_pose.centroid",
        centroid_ok,
        f"|centroid|_inf={max_off:.4f} mm, tol={tol:.4f} mm "
        f"(centroid=({cx:.3f}, {cy:.3f}, {cz:.3f}))",
    )

    L = sorted([bb.size_x, bb.size_y, bb.size_z], reverse=True)
    expected_L_sorted = (bb.size_x, bb.size_y, bb.size_z)
    # Allow tolerance-scale slack for near-equal extents.
    axes_ok = (
        bb.size_x + tol >= bb.size_y
        and bb.size_y + tol >= bb.size_z
    )
    axes = Check(
        "canonical_pose.axes",
        axes_ok,
        f"L=({bb.size_x:.3f}, {bb.size_y:.3f}, {bb.size_z:.3f}); "
        f"need Lx≥Ly≥Lz; sorted={L}",
    )
    return centroid, axes


def _check_gt_self_score(
    gt_step: Path, sub_volumes: list[SubVolume],
) -> Check | None:
    """End-to-end smoke: ``interface_score(GT, GT)`` must be ~1.0.

    Mathematically this must always be 1.0 when rules 6/7 (KOR/KIR
    consistency) pass: Boolean idempotence and the per-sub-volume
    saturation in :mod:`cadgenbench.eval.interface_match` guarantee
    it. The check is a belt-and-braces tripwire against future
    refactors that break the end-to-end pipeline plumbing without
    breaking the per-sub-volume math, and runs the full
    Manifold-Boolean pipeline once per fixture (~2 s).

    Returns ``None`` when the fixture has no sub-volumes (no
    interface metric to score).
    """
    if not sub_volumes:
        return None

    from cadgenbench.eval.interface_match import best_iou_in_context

    by_context: dict[int, list[SubVolume]] = {}
    for sv in sub_volumes:
        by_context.setdefault(sv.context_id, []).append(sv)

    context_scores: list[float] = []
    for ctx_id in sorted(by_context):
        try:
            per_sv = best_iou_in_context(
                gt_step,
                by_context[ctx_id],
                gt_step,
                n_samples=1,  # zero pose only; identity is exact
            )
        except Exception as exc:
            return Check(
                "subvolume.gt_self_score",
                False,
                f"best_iou_in_context raised for ctx={ctx_id}: {exc}",
            )
        context_scores.append(min(per_sv.values()))
    overall = sum(context_scores) / len(context_scores)

    tol = 1e-6
    if overall >= 1.0 - tol:
        return Check(
            "subvolume.gt_self_score",
            True,
            f"interface_score(GT, GT) = {overall:.6f} (≥ 1 − {tol})",
        )
    return Check(
        "subvolume.gt_self_score",
        False,
        f"interface_score(GT, GT) = {overall:.6f} (< 1 − {tol}; "
        f"pipeline bug — rules 6/7 should have caught this upstream)",
    )


def _check_topology_consistency(gt_step: Path) -> Check:
    """Mesh-derived b_0 must equal BREP solid_count."""
    try:
        a = analyze_step(gt_step)
        betti = compute_betti_for_step(gt_step)
    except Exception as exc:
        return Check("topology.b0_solid_match", False, f"raised: {exc}")
    if betti.b0 == a.measurements.solid_count:
        return Check(
            "topology.b0_solid_match",
            True,
            f"b0={betti.b0} == solid_count={a.measurements.solid_count} "
            f"(b1={betti.b1}, b2={betti.b2})",
        )
    return Check(
        "topology.b0_solid_match",
        False,
        f"mesh b0={betti.b0} != BREP solid_count={a.measurements.solid_count}",
    )


def _check_subvolumes(
    fixture_dir: Path, gt_step: Path, sub_volumes: list[SubVolume],
) -> list[Check]:
    """Rules 5–9 (sub-volume validity, KOR/KIR consistency, disjoint AABBs,
    plausible pose). Returns one Check per (rule, sub-volume) pair so all
    failures surface at once.
    """
    checks: list[Check] = []
    if not sub_volumes:
        return checks

    try:
        gt_analysis = analyze_step(gt_step)
        gt_defl = deflection_for_bbox(gt_analysis.measurements.bounding_box.diagonal)
        gt_mesh = tessellate_and_validate(gt_step, gt_defl)
        gt_manifold = mesh_to_manifold(gt_mesh)
        gt_bb = gt_analysis.measurements.bounding_box
        clip_offset = CLIP_OFFSET_REL * gt_bb.diagonal
    except Exception as exc:
        checks.append(
            Check(
                "subvolume.setup",
                False,
                f"failed to prepare GT for sub-volume checks: {exc}",
            ),
        )
        return checks

    # Pre-compute per-sub-volume info: validity, manifold, bbox.
    info: dict[str, dict] = {}
    for sv in sub_volumes:
        rec: dict = {"sv": sv}
        try:
            sv_v = analyze_step(sv.path)
        except Exception as exc:
            checks.append(
                Check(
                    f"subvolume.validity[{sv.name}]",
                    False,
                    f"analyze_step raised: {exc}",
                ),
            )
            continue
        if not sv_v.validation.is_valid:
            checks.append(
                Check(
                    f"subvolume.validity[{sv.name}]",
                    False,
                    sv_v.validation.topology_errors[0]
                    if sv_v.validation.topology_errors
                    else "is_valid=False",
                ),
            )
            continue
        checks.append(
            Check(
                f"subvolume.validity[{sv.name}]",
                True,
                "is_valid=True",
            ),
        )

        try:
            sv_mesh = tessellate_and_validate(sv.path, gt_defl)
            sv_manifold = mesh_to_manifold(sv_mesh)
        except Exception as exc:
            checks.append(
                Check(
                    f"subvolume.mesh[{sv.name}]",
                    False,
                    f"tessellate / manifold failed: {exc}",
                ),
            )
            continue

        rec["mesh"] = sv_mesh
        rec["manifold"] = sv_manifold
        rec["vol"] = manifold_volume(sv_manifold)
        rec["bb_min"] = sv_mesh.vertices.min(axis=0)
        rec["bb_max"] = sv_mesh.vertices.max(axis=0)

        # Inflated AABB matching interface_match._build_sub_volume_cache.
        bb_size = rec["bb_max"] - rec["bb_min"]
        margin = max(INFLATED_MARGIN_FLOOR, INFLATED_MARGIN_FRACTION * bb_size.max())
        rec["inflated_min"] = rec["bb_min"] - margin
        rec["inflated_max"] = rec["bb_max"] + margin

        # Rule 6/7: KOR / KIR fit-type consistency via Manifold IoU vs GT.
        inter = intersect(sv_manifold, gt_manifold)
        vol_inter = manifold_volume(inter)
        ratio = vol_inter / rec["vol"] if rec["vol"] > 0 else 0.0
        if sv.fit_type == "KOR":
            ok = ratio <= KOR_KIR_EPS
            checks.append(
                Check(
                    f"subvolume.fit_type[{sv.name}]",
                    ok,
                    f"KOR: vol(R∩GT)/vol(R) = {ratio:.4f}, "
                    f"expected ≤ {KOR_KIR_EPS} (region must be empty in GT)",
                ),
            )
        else:  # KIR
            ok = ratio >= 1.0 - KOR_KIR_EPS
            checks.append(
                Check(
                    f"subvolume.fit_type[{sv.name}]",
                    ok,
                    f"KIR: vol(R∩GT)/vol(R) = {ratio:.4f}, "
                    f"expected ≥ {1.0 - KOR_KIR_EPS} (region must be filled in GT)",
                ),
            )

        # Rule 9: plausible pose. KIR sub-volumes must sit inside the GT
        # AABB (a KIR outside GT would mean "candidate has material
        # outside the GT footprint" — almost always a labelling error).
        # KOR sub-volumes are intrinsically allowed to extend past the
        # GT AABB — their job is "the candidate must be empty here,
        # including just outside the part" (e.g. mounting-plane KORs
        # below a bolt pattern to enforce a flat assembly face).
        gt_lo = np.array([gt_bb.x_min, gt_bb.y_min, gt_bb.z_min])
        gt_hi = np.array([gt_bb.x_max, gt_bb.y_max, gt_bb.z_max])
        outside = (
            (rec["bb_min"] < gt_lo - clip_offset).any()
            or (rec["bb_max"] > gt_hi + clip_offset).any()
        )
        if sv.fit_type == "KIR" and outside:
            checks.append(
                Check(
                    f"subvolume.pose[{sv.name}]",
                    False,
                    f"KIR AABB extends outside GT AABB + "
                    f"{clip_offset:.3f} mm clip (sub bb="
                    f"[{rec['bb_min']}, {rec['bb_max']}])",
                ),
            )
        else:
            detail = (
                f"AABB within GT AABB + clip ({clip_offset:.3f} mm)"
                if not outside
                else f"KOR extends past GT AABB by up to "
                     f"{float(max(
                         (gt_lo - rec['bb_min']).max(),
                         (rec['bb_max'] - gt_hi).max(),
                     )):.3f} mm — allowed for KOR"
            )
            checks.append(
                Check(f"subvolume.pose[{sv.name}]", True, detail),
            )

        info[sv.name] = rec

    # Rule 8: pairwise disjoint inflated AABBs within each context_id —
    # **only when the fit_types differ**. Same-fit_type overlap is
    # legal (counter-bore halves both KOR; a slab KOR enclosing smaller
    # KOR features; etc.) because they place compatible constraints on
    # the same region. Opposite fit_types overlapping is physically
    # impossible (one says "empty here", the other "filled here") and
    # remains a hard failure.
    by_ctx: dict[int, list[tuple[str, str]]] = {}
    for sv in sub_volumes:
        if sv.name in info:
            by_ctx.setdefault(sv.context_id, []).append((sv.name, sv.fit_type))
    for ctx_id, entries in by_ctx.items():
        if len(entries) < 2:
            continue
        for (a_name, a_fit), (b_name, b_fit) in itertools.combinations(entries, 2):
            ra, rb = info[a_name], info[b_name]
            overlap = bool(
                (ra["inflated_min"] <= rb["inflated_max"]).all()
                and (rb["inflated_min"] <= ra["inflated_max"]).all()
            )
            same_fit = a_fit == b_fit
            ok = (not overlap) or same_fit
            if overlap:
                detail = (
                    f"inflated AABBs overlap — same fit_type ({a_fit}), "
                    f"allowed"
                    if same_fit
                    else f"inflated AABBs overlap with opposite fit_types "
                         f"({a_fit} vs {b_fit}) — physically impossible"
                )
            else:
                detail = "inflated AABBs disjoint"
            checks.append(
                Check(
                    f"subvolume.disjoint_bbox_R[ctx={ctx_id}]({a_name},{b_name})",
                    ok,
                    detail,
                ),
            )

    return checks


# ---------------------------------------------------------------------------
# Per-fixture orchestrator
# ---------------------------------------------------------------------------


def check_fixture(fixture_dir: Path) -> FixtureReport:
    gt_step = fixture_dir / GT_STEP_NAME
    name = fixture_dir.name

    checks: list[Check] = []
    if not gt_step.exists():
        checks.append(
            Check(
                "files.ground_truth",
                False,
                f"{GT_STEP_NAME} not found in {fixture_dir}",
            ),
        )
        return FixtureReport(fixture=name, checks=checks)

    checks.append(Check("files.ground_truth", True, f"{GT_STEP_NAME} present"))
    checks.append(_check_validity(gt_step))
    centroid_chk, axes_chk = _check_canonical_pose(gt_step)
    checks.append(centroid_chk)
    checks.append(axes_chk)
    checks.append(_check_topology_consistency(gt_step))

    sub_volumes = discover_sub_volumes(fixture_dir)
    if sub_volumes:
        checks.extend(_check_subvolumes(fixture_dir, gt_step, sub_volumes))
        self_score = _check_gt_self_score(gt_step, sub_volumes)
        if self_score is not None:
            checks.append(self_score)

    return FixtureReport(fixture=name, checks=checks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _data_gt_dir() -> Path:
    """``data/gt/`` relative to the current working directory.

    The same convention the rest of cadgenbench uses: commands run from
    the repo root, where ``data/`` lives.
    """
    return Path.cwd() / "data" / "gt"


def _discover_fixtures() -> list[Path]:
    gt_root = _data_gt_dir()
    if not gt_root.is_dir():
        import sys
        print(f"ERROR: not a directory: {gt_root}", file=sys.stderr)
        print(
            "Run from the repo root (the directory that contains data/).",
            file=sys.stderr,
        )
        sys.exit(2)
    return sorted(d for d in gt_root.iterdir() if d.is_dir())


def _print_report(report: FixtureReport, *, verbose: bool) -> None:
    status = "PASS" if report.all_ok else "FAIL"
    print(f"\n[{status}] {report.fixture}")
    for chk in report.checks:
        if chk.ok and not verbose:
            continue
        tag = "ok " if chk.ok else "FAIL"
        print(f"  {tag}  {chk.rule:<40}  {chk.detail}")


def main() -> int:
    import sys
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check benchmark ground-truth directories against the "
            "rules in docs/benchmark/authoring.md."
        ),
    )
    parser.add_argument(
        "fixture", nargs="?", metavar="FIXTURE",
        help="Fixture name (a directory under data/gt/). "
             "Mutually exclusive with --all / --fixture-path.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Check every fixture under data/gt/.",
    )
    parser.add_argument(
        "--fixture-path", type=Path, metavar="PATH", default=None,
        help="Check one fixture by direct directory path.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print PASS lines too (default: only FAIL lines per fixture).",
    )
    args = parser.parse_args()

    selectors = [bool(args.all), bool(args.fixture), args.fixture_path is not None]
    if sum(selectors) != 1:
        print(
            "ERROR: pick exactly one of: --all, FIXTURE positional, --fixture-path PATH",
            file=sys.stderr,
        )
        return 2

    if args.all:
        fixture_dirs = _discover_fixtures()
    elif args.fixture:
        fixture_dirs = [_data_gt_dir() / args.fixture]
    else:
        fixture_dirs = [args.fixture_path.resolve()]

    if not fixture_dirs:
        print("ERROR: no fixtures resolved", file=sys.stderr)
        return 2

    all_pass = True
    total_fail = 0
    for fdir in fixture_dirs:
        if not fdir.is_dir():
            print(f"\n[FAIL] {fdir.name}: not a directory: {fdir}")
            all_pass = False
            total_fail += 1
            continue
        report = check_fixture(fdir)
        _print_report(report, verbose=args.verbose)
        if not report.all_ok:
            all_pass = False
            total_fail += sum(1 for c in report.checks if not c.ok)

    print()
    if all_pass:
        print(f"All {len(fixture_dirs)} fixture(s) passed.")
        return 0
    print(f"FAIL: {total_fail} check failure(s) across {len(fixture_dirs)} fixture(s).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
