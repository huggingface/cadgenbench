#!/usr/bin/env python
"""Local self-check for a benchmark submission STEP.

Runs the same CAD-validity gate that the grading pipeline uses: BREP
well-formedness + watertightness + meshable-as-closed-manifold. Exits
non-zero on any failure with the specific reason; exits 0 silently on
a clean submission.

Usage::

    python _to_move_to_dataset_repo/sanity_check_submission.py path/to/output.step

The gate is the deciding factor for whether ``cad_score = 0``. See
``docs/benchmark/submission.md`` for the full contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cadgenbench.common.validity import analyze_step
from cadgenbench.common.mesh import deflection_for_bbox


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CAD-validity gate on one candidate STEP. Same gate "
            "the grading pipeline uses."
        ),
    )
    parser.add_argument("step", type=Path, help="Path to the candidate STEP file.")
    parser.add_argument(
        "--quiet", action="store_true",
        help="On pass, exit silently. On fail, still print the reason.",
    )
    args = parser.parse_args()

    if not args.step.exists():
        print(f"ERROR: file not found: {args.step}", file=sys.stderr)
        return 2

    try:
        result = analyze_step(args.step)
    except Exception as exc:
        print(f"FAIL  STEP load failed: {exc}", file=sys.stderr)
        return 1

    val = result.validation
    m = result.measurements

    if val.is_valid:
        if not args.quiet:
            defl = deflection_for_bbox(m.bounding_box.diagonal)
            print(
                f"PASS  {args.step.name}: is_valid=True watertight=True\n"
                f"      solids={m.solid_count} shells={m.shell_count} "
                f"faces={m.face_count}\n"
                f"      volume={m.volume:.2f}  bbox="
                f"{m.bounding_box.size_x:.2f}×{m.bounding_box.size_y:.2f}×"
                f"{m.bounding_box.size_z:.2f}  defl_used={defl:.4f} mm",
            )
        return 0

    print(
        f"FAIL  {args.step.name}: is_valid=False  watertight={val.is_watertight}",
        file=sys.stderr,
    )
    for err in val.topology_errors[:10]:
        print(f"      - {err}", file=sys.stderr)
    if len(val.topology_errors) > 10:
        print(f"      ... and {len(val.topology_errors) - 10} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
