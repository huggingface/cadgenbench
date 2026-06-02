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

"""CAD validity, one of the v1 metric categories.

Answers three questions about a STEP file:

1. **Is the BREP well-formed and watertight?** (:class:`ValidationResult`)
   ``is_valid`` is True iff **all** of the following hold:

   - ``BRepCheck_Analyzer.IsValid()`` reports no per-face / per-edge /
     per-vertex topology errors over the whole shape.
   - Every shell is closed (``_is_watertight``), i.e. each oriented face
     skin meets its neighbours edge-to-edge with no naked / free edges.
   - Geometry quality (``_collect_geometry_quality_errors``): no
     near-degenerate edge (length >= ``MIN_EDGE_LENGTH_MM``), no tiny
     face (area >= ``MIN_FACE_AREA_MM2``), and bounded BREP tolerance
     (<= ``MAX_TOLERANCE_MM``). Fixed floors stated in the submitter's
     own geometry terms, rejected up front instead of failing the mesh
     gate downstream.

   A non-watertight BREP cannot be 3D-printed, Boolean'd against, or
   topologically analysed for handles / cavities, so it is rejected at
   the gate. The original cause is recorded in ``topology_errors``
   (e.g. ``"BREP not watertight: open shells / naked edges"``).

2. **Can it be tessellated into a clean closed manifold mesh?**, required
   so that downstream Betti / topology-match math has a well-defined
   :math:`\\chi` of the boundary. The mesh-pipeline gate lives in
   :mod:`cadgenbench.common.mesh` and surfaces any failure as a
   ``topology_errors`` entry (e.g. ``"mesh non-manifold: edge (220, 243)
   shared by 4 triangles"``) that propagates into ``is_valid = False``.
3. **What does the geometry look like?** (:class:`Measurements`, sourced
   from :mod:`cadgenbench.common.measurements`), bounding box, volume,
   topology counts.

These come back together in :class:`ValidityResult` so callers never have
to load or mesh the STEP twice. Companion modules in this package:

- :mod:`cadgenbench.eval.shape_similarity`, point-cloud / volume /
  edge-F1 metrics comparing candidate to ground truth.
- :mod:`cadgenbench.eval.interface_match`, keep-in / keep-out region
  matching against authored sub-volumes (jig metric).
- :mod:`cadgenbench.eval.topo_match`, Betti-number agreement
  (:math:`b_0`, :math:`b_1`, :math:`b_2`) computed on the tessellated
  boundary. See `docs/metrics/topo_match.md` for the specification.
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from cadgenbench.common.measurements import BBox, Measurements, _measure_wrapped

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Topology / manifold validity of a BREP and its tessellated boundary.

    Fields:
        is_valid: True iff (a) ``BRepCheck_Analyzer.IsValid()`` reports
            no errors over the whole shape, AND (b) every shell is
            closed (``is_watertight``), AND (c) the geometry clears the
            quality floors (no near-degenerate edge, tiny face, or
            inflated tolerance; see ``_collect_geometry_quality_errors``),
            AND (d) the boundary tessellation is a clean closed orientable
            manifold (every edge in exactly two triangles with opposite
            orientations). Required by the existing zero-cascade in
            :func:`cadgenbench.eval.evaluate._cad_score`.
        is_watertight: every shell is closed AND no BREP topology errors.
            A pure BREP signal, independent of the mesh pipeline. It
            is *also* an input to ``is_valid``: a non-watertight shape
            is never valid.
        topology_errors: de-duplicated human-readable strings combining
            BREP errors (``"Face: BRepCheck_SelfIntersectingWire"``),
            the watertight gate (``"BREP not watertight: open shells /
            naked edges"``), the geometry-quality gate (``"3 edge(s)
            shorter than 0.001 mm (shortest 1.099e-05 mm)"``), and
            mesh-pipeline errors
            (``"mesh non-manifold: edge (220, 243) shared by 4 triangles"``).
            Designed to be displayed verbatim when ``is_valid`` is False
            so a human or LLM can diagnose the failure mode without
            re-running the analyzer.
    """

    is_valid: bool
    is_watertight: bool
    topology_errors: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class ValidityResult:
    """Validity + measurements together, the primary return type.

    Returned by :func:`analyze_step`. Callers that need only one half can
    use :func:`validate_step` (validity only) or
    :func:`cadgenbench.common.measurements.measure_step` (measurements only).
    """

    validation: ValidationResult
    measurements: Measurements


# Backwards-compatible alias for the previous name. Will be removed once
# external callers (none known) migrate.
AnalysisResult = ValidityResult


# ---------------------------------------------------------------------------
# Geometry-quality policy
# ---------------------------------------------------------------------------

# Fixed, part-size-independent geometry-quality floors. They live in the
# submitter's own terms (edge length, face area, BREP tolerance) so a part
# can be designed and checked against them directly, not against the mesher's
# output. These are the single source of truth; the labeler and submission
# docs quote them.
MIN_EDGE_LENGTH_MM = 0.001  # reject any non-degenerate edge shorter than this
MIN_FACE_AREA_MM2 = 0.001  # reject any face smaller than this
MAX_TOLERANCE_MM = 0.1  # reject if any edge / vertex tolerance exceeds this


# ---------------------------------------------------------------------------
# Mesh-safeguard policy
# ---------------------------------------------------------------------------
#
# Three bounds on the (otherwise unbounded) cost of meshing a part, applied
# identically to ground truth and submissions. They exist so a single
# pathological / runaway STEP cannot hang the grader or exhaust memory; every
# threshold is deliberately *generous* so it never touches a legitimate part.
#
#   1. File-size pre-filter   — cheapest; rejects before the STEP is even
#      parsed, so an absurd upload can't OOM the loader.
#   2. Triangle-count ceiling — checked on the produced mesh, before the
#      downstream boolean / topology stage consumes it.
#
# Both of the above are **deterministic** (same verdict on every machine), so
# tripping one marks the part invalid → ``cad_score`` 0, exactly like any other
# validity failure, with the reason recorded in ``topology_errors``.
#
#   3. Per-mesh process-kill timeout — OCC tessellation is a native call that
#      will not honour a Python signal mid-flight, so the only reliable bound
#      is to run each mesh in a child process and *kill* it on overrun. This
#      wall-clock bound is machine-dependent, so it is NOT a clean scoring
#      gate on its own: a timeout is retried once on a fresh worker, and only a
#      second timeout marks the part invalid (saving the offending STEP for
#      debugging). It is the runaway-work backstop the two deterministic
#      ceilings cannot be.
#
# Ground truth is asserted to clear all three (a GT that trips one is an
# authoring bug, surfaced as a loud exception rather than a silent zero).
MAX_TRIANGLES = int(os.environ.get("CADGENBENCH_MAX_TRIANGLES", 1_000_000))
MAX_STEP_FILE_BYTES = int(
    os.environ.get("CADGENBENCH_MAX_STEP_FILE_BYTES", 50_000_000),
)
# Wall-clock seconds per individual mesh. <= 0 disables process isolation and
# meshes in-process (useful for debugging / constrained CI); the deterministic
# ceilings still apply.
MESH_TIMEOUT_S = float(os.environ.get("CADGENBENCH_MESH_TIMEOUT_S", 180.0))
# Number of meshing attempts before a timing-out part is declared invalid. The
# first timeout is treated as possibly machine-transient and retried once on a
# fresh worker; the value is intentionally not configurable as a *gate* knob.
_MESH_TIMEOUT_ATTEMPTS = 2


class MeshTimeoutError(RuntimeError):
    """Raised when a single mesh exceeds the per-mesh wall-clock timeout.

    Distinct from :class:`cadgenbench.common.mesh.MeshSanityError` (a
    deterministic geometry defect): a timeout is a machine-dependent
    backstop, so it is only fatal after the configured number of attempts.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_step(
    step_path: str | Path,
    *,
    is_ground_truth: bool = False,
) -> ValidationResult:
    """Validate a STEP file and return :class:`ValidationResult` only.

    Runs the **full** gate: file-size pre-filter + ``BRepCheck`` +
    watertight + mesh-gate (triangle ceiling + per-mesh timeout).
    Equivalent to ``analyze_step(step_path).validation`` but skips
    returning the measurements; use :func:`analyze_step` when you also
    need them and don't want to parse the STEP twice.

    ``is_ground_truth``: when True a safeguard violation (oversized file,
    triangle ceiling, repeated timeout) is raised as a loud authoring
    error instead of returning ``is_valid=False`` — GT must clear every
    ceiling.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the STEP file cannot be loaded, or (for ground
            truth) a safeguard ceiling is violated.
    """
    step_path = Path(step_path)
    fs_error = _file_size_error(step_path)
    if fs_error is not None:
        _raise_if_ground_truth(fs_error, step_path, is_ground_truth)
        return ValidationResult(
            is_valid=False, is_watertight=False, topology_errors=(fs_error,),
        )
    wrapped = _load_step_wrapped(step_path)
    return _validate_wrapped(
        wrapped, step_path=step_path, is_ground_truth=is_ground_truth,
    )


def analyze_step(
    step_path: str | Path,
    *,
    mesh_cache: dict[float, object] | None = None,
    is_ground_truth: bool = False,
) -> ValidityResult:
    """Load a STEP file once and return both validity and measurements.

    Order: file-size pre-filter (before the STEP is parsed), then
    measurements (cheap), then validation reuses the bbox diagonal to
    compute its tessellation deflection without re-walking the shape.

    ``mesh_cache``: optional ``{deflection: Mesh}`` dict. When supplied and
    the shape reaches the mesh gate, the tessellated boundary mesh is stored
    here so callers can reuse it (e.g. for rendering) instead of
    tessellating the same part a second time.

    ``is_ground_truth``: see :func:`validate_step`.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the STEP file cannot be loaded, or (for ground
            truth) a safeguard ceiling is violated.
    """
    step_path = Path(step_path)
    fs_error = _file_size_error(step_path)
    if fs_error is not None:
        _raise_if_ground_truth(fs_error, step_path, is_ground_truth)
        return _oversized_validity_result(fs_error)
    wrapped = _load_step_wrapped(step_path)
    measurements = _measure_wrapped(wrapped)
    validation = _validate_wrapped(
        wrapped, bbox_diagonal=measurements.bounding_box.diagonal,
        mesh_cache=mesh_cache,
        step_path=step_path,
        is_ground_truth=is_ground_truth,
    )
    return ValidityResult(validation=validation, measurements=measurements)


def parse_step(step_path: str | Path) -> None:
    """Cheap check that a file can be loaded as STEP geometry.

    Runs only the STEP reader (``build123d.import_step``), no BRepCheck,
    no watertight test, no mesh tessellation. Used by upstream callers
    (e.g. the leaderboard's submit handler) that want to reject
    non-STEP / corrupted uploads at request time but leave the full
    validity gate to a downstream evaluator.     A file that parses here
    can still be reported as invalid by :func:`analyze_step`; that's
    the per-fixture validity signal, distinct from "is this even STEP".

    Also enforces the file-size pre-filter so an oversized upload is
    rejected at request time with a clear reason rather than reaching the
    (much more expensive) loader / mesher.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file is not loadable STEP geometry, or
            exceeds the file-size ceiling.
    """
    step_path = Path(step_path)
    fs_error = _file_size_error(step_path)
    if fs_error is not None:
        raise RuntimeError(fs_error)
    _load_step_wrapped(step_path)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_step_wrapped(step_path: str | Path):  # type: ignore[no-untyped-def]
    """Load a STEP file and return the raw OCC ``TopoDS_Shape``."""
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    from build123d import import_step

    try:
        shape = import_step(str(step_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load STEP file: {step_path}") from exc

    if shape is None or not shape.wrapped:
        raise RuntimeError(f"STEP file produced no geometry: {step_path}")

    return shape.wrapped


def _validate_wrapped(
    wrapped,  # type: ignore[no-untyped-def]
    *,
    bbox_diagonal: float | None = None,
    mesh_cache: dict[float, object] | None = None,
    step_path: str | Path | None = None,
    is_ground_truth: bool = False,
) -> ValidationResult:
    """Validate a pre-loaded OCC ``TopoDS_Shape``.

    ``step_path`` is the STEP this shape was loaded from. When provided
    the mesh gate runs each tessellation in a killable child process so a
    runaway native call can be bounded by :data:`MESH_TIMEOUT_S`; when
    ``None`` (e.g. a shape built in-memory) it meshes in-process and only
    the deterministic ceilings apply. ``is_ground_truth`` escalates any
    safeguard violation to a loud exception (see :func:`validate_step`).

    Runs the full three-part validity gate:

    1. ``BRepCheck_Analyzer.IsValid()`` (per-face / per-edge / per-vertex
       topology).
    2. ``_is_watertight``, every shell closed AND no BRepCheck errors.
    3. Mesh-pipeline gate, tessellate the boundary and confirm the
       result is a clean closed orientable manifold (delegated to
       :func:`cadgenbench.common.mesh.validate_mesh`). Only run when (1)
       and (2) already passed, since meshing an invalid BREP wastes
       time and produces misleading errors.

    ``bbox_diagonal`` is the GT- or self-bbox diagonal used to derive
    the tessellation deflection via
    :func:`cadgenbench.common.mesh.deflection_for_bbox`. When ``None``
    (the :func:`validate_step` caller) it is computed locally from the
    wrapped shape.
    """
    from OCP.BRepCheck import BRepCheck_Analyzer

    analyzer = BRepCheck_Analyzer(wrapped)
    brep_ok = bool(analyzer.IsValid())
    topology_errors = _collect_errors(analyzer, wrapped) if not brep_ok else []

    # Belt-and-braces: require every shell to be closed *and* BRepCheck to
    # report no topology errors. Either alone is not quite enough, a closed
    # shell with invalid curves isn't usefully "watertight" for downstream
    # use.
    is_watertight = _is_watertight(wrapped) and not topology_errors

    # A non-watertight BREP is not a valid solid for our purposes (cannot
    # be 3D-printed, Boolean'd, or topologically analysed). Surface the
    # reason verbatim so the failure mode is debuggable from result.json.
    if brep_ok and not is_watertight:
        topology_errors.append(
            "BREP not watertight: at least one shell has open / naked "
            "edges (failed _is_watertight)",
        )

    # Sliver-face gate. A face thinner than the tessellation deflection
    # cannot be meshed into a watertight boundary (OCC drops or degrades
    # its triangulation), so it would otherwise fail downstream as a
    # confusing mesh-gate crash. Catch it here, on the BREP, with a clear
    # per-face reason. Runs only on an otherwise-valid watertight BREP so
    # the deflection is meaningful and we are not piling onto a shape
    # that is already rejected.
    mesh_ok = True
    if brep_ok and is_watertight:
        quality_errors = _collect_geometry_quality_errors(wrapped)
        topology_errors.extend(quality_errors)
        if quality_errors:
            mesh_ok = False
        elif is_ground_truth:
            # Trusted, pre-verified part (ground truth / jig sub-volume; it
            # already passed the authoring gate in sanity_check_gt.py). Skip
            # the mesh gate so analysis never tessellates the shape here. Its
            # mesh is produced exactly once, on demand, at the deflection the
            # metric actually needs (e.g. a sub-volume at its parent GT's
            # coarser deflection). Tessellating it here at a *different* (own)
            # deflection would be redundant re-verification AND leave an OCC
            # triangulation that BRepMesh then refuses to coarsen, silently
            # defeating the deflection override.
            mesh_ok = True
        else:
            deflection = _tessellation_deflection(wrapped, bbox_diagonal)
            mesh_ok = _run_mesh_gate(
                wrapped,
                deflection,
                topology_errors,
                mesh_cache=mesh_cache,
                step_path=step_path,
                is_ground_truth=is_ground_truth,
            )

    is_valid = brep_ok and is_watertight and mesh_ok

    return ValidationResult(
        is_valid=is_valid,
        is_watertight=is_watertight,
        topology_errors=tuple(topology_errors),
    )


def _tessellation_deflection(
    wrapped,  # type: ignore[no-untyped-def]
    bbox_diagonal: float | None,
) -> float:
    """Deflection the mesh gate (and sliver gate) tessellate at.

    Mirrors :func:`cadgenbench.common.mesh.deflection_for_bbox`, computing
    the bbox locally when the caller did not supply it so both gates use
    one consistent value.
    """
    from cadgenbench.common.mesh import deflection_for_bbox

    if bbox_diagonal is None:
        from cadgenbench.common.measurements import _compute_bbox

        bbox_diagonal = float(_compute_bbox(wrapped).diagonal)
    return deflection_for_bbox(bbox_diagonal)


def _run_mesh_gate(
    wrapped,  # type: ignore[no-untyped-def]
    deflection: float,
    topology_errors: list[str],
    *,
    mesh_cache: dict[float, object] | None = None,
    step_path: str | Path | None = None,
    is_ground_truth: bool = False,
) -> bool:
    """Run the mesh-pipeline gate and append any failure to *topology_errors*.

    Returns True iff the tessellated boundary is a clean closed
    orientable manifold (so the topology-match metric's
    :math:`\\chi`-based math is well defined) *and* it clears the mesh
    safeguards (triangle ceiling + per-mesh timeout).
    """
    from cadgenbench.common.mesh import MeshSanityError

    try:
        # Single robust meshing path (escalating deflection), now wrapped in
        # the safeguard layer (timeout + triangle ceiling). The part is valid
        # if any ladder rung yields a closed manifold within the ceilings. The
        # resulting mesh is cached for reuse by the metric accessors so GT,
        # candidate, validity, and the cache all share one code path / mesh.
        mesh = safeguarded_tessellate(
            step_path,
            deflection,
            wrapped=wrapped,
            is_ground_truth=is_ground_truth,
        )
        if mesh_cache is not None:
            mesh_cache[float(deflection)] = mesh
    except MeshSanityError as exc:
        # Geometry defect or triangle-ceiling breach. For GT this is an
        # authoring bug, so escalate; for a submission it is a score signal.
        if is_ground_truth:
            raise
        # MeshSanityError messages already start with the failure mode
        # ("mesh non-manifold: ...", "mesh not closed: ...", "mesh exceeds
        # triangle ceiling: ...", ...), matching the example in the
        # ValidationResult docstring, append as-is, no prefix.
        topology_errors.append(str(exc))
        return False
    except MeshTimeoutError as exc:
        # Repeated timeout. ``safeguarded_tessellate`` already re-raised for
        # GT, so reaching here means a submission; record it as a score
        # signal (the offending STEP was saved for debugging).
        topology_errors.append(str(exc))
        return False
    return True


# ---------------------------------------------------------------------------
# Mesh safeguards: file-size pre-filter, triangle ceiling, per-mesh timeout
# ---------------------------------------------------------------------------


def _file_size_error(step_path: Path) -> str | None:
    """Return a reason string if *step_path* exceeds the file-size ceiling.

    Returns ``None`` when the file is within budget or its size cannot be
    read (a missing file is left for the loader to report as
    ``FileNotFoundError``, preserving existing behaviour).
    """
    try:
        size = step_path.stat().st_size
    except OSError:
        return None
    if size > MAX_STEP_FILE_BYTES:
        return (
            f"STEP file is {size} bytes, exceeding the "
            f"{MAX_STEP_FILE_BYTES}-byte ceiling "
            f"(~{MAX_STEP_FILE_BYTES // 1_000_000} MB)"
        )
    return None


def _raise_if_ground_truth(reason: str, step_path: Path, is_ground_truth: bool) -> None:
    """Escalate a safeguard violation to a loud error for ground truth.

    GT is asserted to clear every ceiling, so a violation is an authoring
    bug we never want to mask as a silent ``cad_score`` 0.
    """
    if is_ground_truth:
        raise RuntimeError(
            f"GROUND TRUTH violates a mesh safeguard ({reason}): {step_path}",
        )


def _oversized_validity_result(reason: str) -> ValidityResult:
    """A fully-formed invalid :class:`ValidityResult` for an unparsed file.

    The file is deliberately *not* loaded (that is the point of the cheap
    pre-filter), so measurements are zeroed and only the reason is carried.
    """
    return ValidityResult(
        validation=ValidationResult(
            is_valid=False, is_watertight=False, topology_errors=(reason,),
        ),
        measurements=Measurements(
            solid_count=0,
            shell_count=0,
            face_count=0,
            volume=0.0,
            bounding_box=BBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )


def _check_triangle_ceiling(mesh) -> None:  # type: ignore[no-untyped-def]
    """Raise :class:`MeshSanityError` if *mesh* exceeds the triangle ceiling.

    Checked on the produced mesh, before the downstream boolean / topology
    stage consumes it. Deterministic, so it is a clean validity signal.
    """
    if mesh.n_triangles > MAX_TRIANGLES:
        from cadgenbench.common.mesh import MeshSanityError

        raise MeshSanityError(
            f"mesh exceeds triangle ceiling: {mesh.n_triangles} triangles "
            f"> {MAX_TRIANGLES} (deflection {mesh.linear_deflection_mm} mm)",
        )


def safeguarded_tessellate(
    step_path: str | Path | None,
    deflection: float,
    *,
    wrapped=None,  # type: ignore[no-untyped-def]
    ladder: tuple[int, ...] | None = None,
    is_ground_truth: bool = False,
):  # -> Mesh
    """Robustly tessellate one part under the mesh safeguards.

    The single chokepoint the validity gate and the metric mesh accessor
    both route through, so neither tessellates without the triangle
    ceiling and (when a path is available) the per-mesh process-kill
    timeout applied.

    When *step_path* is given and :data:`MESH_TIMEOUT_S` > 0 the
    tessellation runs in a reusable, killable child process: a single
    overrun is retried once on a fresh worker, and a second overrun raises
    :class:`MeshTimeoutError` (or, for ground truth, a loud
    ``RuntimeError``) after saving the offending STEP for debugging. When
    no path is available (an in-memory shape) it meshes in-process from
    *wrapped*; the triangle ceiling still applies.

    Raises:
        MeshSanityError: geometry defect or triangle-ceiling breach.
        MeshTimeoutError: repeated timeout on a submission part.
        RuntimeError: a ground-truth part that breaches any safeguard.
    """
    from cadgenbench.common.mesh import DEFLECTION_LADDER, robust_tessellate_shape

    if ladder is None:
        ladder = DEFLECTION_LADDER

    # In-process path: no STEP file to hand a child process, or isolation
    # disabled. Mesh directly from the loaded shape; the ceiling still gates.
    if step_path is None or MESH_TIMEOUT_S <= 0:
        if wrapped is None:
            wrapped = _load_step_wrapped(step_path)
        mesh = robust_tessellate_shape(wrapped, deflection, ladder=ladder)
        _check_triangle_ceiling(mesh)
        return mesh

    step_path = Path(step_path)
    last_timeout: MeshTimeoutError | None = None
    for attempt in range(1, _MESH_TIMEOUT_ATTEMPTS + 1):
        try:
            return _mesh_in_subprocess(step_path, deflection, ladder)
        except MeshTimeoutError as exc:
            last_timeout = exc
            logger.warning(
                "Mesh attempt %d/%d timed out for %s: %s",
                attempt, _MESH_TIMEOUT_ATTEMPTS, step_path, exc,
            )

    # Every attempt timed out. Save the offending STEP for debugging and
    # turn the (now repeated, less machine-transient) timeout into a verdict.
    saved = _save_timeout_step(step_path)
    saved_note = f"; saved offending STEP to {saved}" if saved else ""
    reason = (
        f"mesh timeout: tessellation exceeded {MESH_TIMEOUT_S:g}s on "
        f"{_MESH_TIMEOUT_ATTEMPTS} attempts ({last_timeout}){saved_note}"
    )
    _raise_if_ground_truth(reason, step_path, is_ground_truth)
    raise MeshTimeoutError(reason)


# A single reusable, killable mesh worker per process. Spawned lazily on the
# first timed mesh; torn down and recreated whenever a mesh overruns (the only
# reliable way to abort a stuck native OCC call). Each importing process (the
# parent, or an eval ProcessPool worker) gets its own.
_MESH_POOL = None


def _get_mesh_pool():  # type: ignore[no-untyped-def]
    global _MESH_POOL
    if _MESH_POOL is None:
        # ``spawn`` matches the eval CLI's pool: a fresh interpreter per
        # worker with no inherited (and possibly half-initialised native)
        # state, and it is safe to nest inside the eval ProcessPoolExecutor.
        _MESH_POOL = mp.get_context("spawn").Pool(processes=1)
    return _MESH_POOL


def _reset_mesh_pool() -> None:
    """Kill the mesh worker (aborting any stuck native call) and drop it."""
    global _MESH_POOL
    if _MESH_POOL is not None:
        try:
            _MESH_POOL.terminate()
            _MESH_POOL.join()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        _MESH_POOL = None


atexit.register(_reset_mesh_pool)


def _mesh_worker(
    step_path_str: str,
    deflection: float,
    angular: float,
    ladder: tuple[int, ...],
):  # -> Mesh   (runs in the child process)
    """Child-process entry: load + robustly mesh one STEP, enforce the ceiling.

    Loads the shape itself (an OCC ``TopoDS_Shape`` cannot cross a process
    boundary) and returns the validated :class:`cadgenbench.common.mesh.Mesh`
    (plain numpy arrays, picklable). Raises :class:`MeshSanityError` on a
    geometry defect or ceiling breach; the parent re-raises it.
    """
    from cadgenbench.common.mesh import robust_tessellate_shape

    wrapped = _load_step_wrapped(step_path_str)
    mesh = robust_tessellate_shape(
        wrapped, deflection, angular_deflection_rad=angular, ladder=ladder,
    )
    _check_triangle_ceiling(mesh)
    return mesh


def _mesh_in_subprocess(
    step_path: Path,
    deflection: float,
    ladder: tuple[int, ...],
    *,
    angular: float = 0.5,
):  # -> Mesh
    """Mesh *step_path* in the killable worker, bounded by :data:`MESH_TIMEOUT_S`.

    Reuses the persistent worker (import cost paid once per process); on
    overrun it kills + drops the worker and raises :class:`MeshTimeoutError`
    so the next attempt starts on a fresh one.
    """
    pool = _get_mesh_pool()
    async_result = pool.apply_async(
        _mesh_worker,
        (str(step_path), float(deflection), float(angular), tuple(ladder)),
    )
    try:
        return async_result.get(timeout=MESH_TIMEOUT_S)
    except mp.TimeoutError:
        # The worker is stuck in a native call that ignores signals; the only
        # reliable abort is to terminate the process and start fresh.
        _reset_mesh_pool()
        raise MeshTimeoutError(
            f"exceeded {MESH_TIMEOUT_S:g}s wall-clock",
        ) from None


def _save_timeout_step(step_path: Path) -> Path | None:
    """Copy a repeatedly-timing-out STEP aside for debugging; return the path.

    Best-effort: a failure to save must never mask the underlying timeout
    verdict. Destination is ``CADGENBENCH_TIMEOUT_DEBUG_DIR`` (default a
    ``cadgenbench_mesh_timeouts`` dir under the system temp dir).
    """
    try:
        debug_dir = Path(
            os.environ.get(
                "CADGENBENCH_TIMEOUT_DEBUG_DIR",
                Path(tempfile.gettempdir()) / "cadgenbench_mesh_timeouts",
            ),
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        dst = debug_dir / (
            f"{step_path.stem}-{os.getpid()}-{int(time.time())}{step_path.suffix}"
        )
        shutil.copy2(step_path, dst)
        return dst
    except Exception:  # noqa: BLE001 - never let debug-save mask the verdict
        logger.warning("Failed to save timing-out STEP %s", step_path, exc_info=True)
        return None


def _collect_geometry_quality_errors(
    wrapped,  # type: ignore[no-untyped-def]
) -> list[str]:
    """Reject BREPs with degenerate or low-quality geometry up front.

    Three fixed, part-size-independent floors that a submitter or author
    can design to and verify in their own CAD terms, instead of against the
    mesher's output:

    - every non-degenerate edge at least ``MIN_EDGE_LENGTH_MM`` long,
    - every face at least ``MIN_FACE_AREA_MM2`` in area,
    - every edge / vertex tolerance at most ``MAX_TOLERANCE_MM``.

    A near-degenerate edge is the artifact that drove the example_3 mesh
    failure (an 11 nm edge), so it is the load-bearing check; the area and
    tolerance floors are sanity bounds. Returns one summary string per
    violated floor (count plus worst offender); empty when the geometry is
    clean.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    errors: list[str] = []

    edges = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_EDGE, edges)
    short_edges: list[float] = []
    max_tol = 0.0
    for ei in range(1, edges.Size() + 1):
        edge = TopoDS.Edge_s(edges.FindKey(ei))
        max_tol = max(max_tol, float(BRep_Tool.Tolerance_s(edge)))
        if BRep_Tool.Degenerated_s(edge):
            continue
        props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, props)
        length = float(props.Mass())
        if length < MIN_EDGE_LENGTH_MM:
            short_edges.append(length)
    if short_edges:
        errors.append(
            f"{len(short_edges)} edge(s) shorter than {MIN_EDGE_LENGTH_MM} mm "
            f"(shortest {min(short_edges):.3e} mm)",
        )

    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_FACE, faces)
    small_faces: list[float] = []
    for fi in range(1, faces.Size() + 1):
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(TopoDS.Face_s(faces.FindKey(fi)), props)
        area = float(props.Mass())
        if area < MIN_FACE_AREA_MM2:
            small_faces.append(area)
    if small_faces:
        errors.append(
            f"{len(small_faces)} face(s) smaller than {MIN_FACE_AREA_MM2} mm^2 "
            f"(smallest {min(small_faces):.3e} mm^2)",
        )

    vertices = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(wrapped, TopAbs_VERTEX, vertices)
    for vi in range(1, vertices.Size() + 1):
        max_tol = max(
            max_tol, float(BRep_Tool.Tolerance_s(TopoDS.Vertex_s(vertices.FindKey(vi)))),
        )
    if max_tol > MAX_TOLERANCE_MM:
        errors.append(
            f"BREP tolerance {max_tol:.3e} mm exceeds the maximum "
            f"{MAX_TOLERANCE_MM} mm",
        )

    return errors


def _is_watertight(shape) -> bool:  # type: ignore[no-untyped-def]
    """True iff *shape* has >= 1 shell and every shell is closed.

    A shell is closed when its oriented faces form a watertight skin, every
    edge is shared by exactly two faces (with opposite orientations). Uses
    ``BRepCheck_Shell.Closed``, which is the same predicate OCC uses
    internally to validate solids and which correctly handles periodic
    seams (e.g. a sphere's single-face shell).
    """
    from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Shell
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    any_shell = False
    while explorer.More():
        shell = TopoDS.Shell_s(explorer.Value())
        if BRepCheck_Shell(shell).Closed() != BRepCheck_NoError:
            return False
        any_shell = True
        explorer.Next()
    return any_shell


def _collect_errors(analyzer, shape) -> list[str]:  # type: ignore[no-untyped-def]
    """Walk sub-shapes and collect human-readable BRepCheck error strings.

    Uses ``TopTools_IndexedMapOfShape`` to visit each sub-shape exactly once;
    a plain ``TopExp_Explorer`` would yield shared edges / vertices multiple
    times (once per parent context) and produce duplicate errors.
    """
    from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Status
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    shape_type_names = {
        TopAbs_FACE: "Face",
        TopAbs_EDGE: "Edge",
        TopAbs_VERTEX: "Vertex",
    }

    errors: list[str] = []
    for sub_type, type_name in shape_type_names.items():
        sub_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, sub_type, sub_map)
        for i in range(1, sub_map.Size() + 1):
            sub = sub_map.FindKey(i)
            check_result = analyzer.Result(sub)
            if check_result is None:
                continue
            for status in check_result.Status():
                if status != BRepCheck_NoError:
                    errors.append(f"{type_name}: {BRepCheck_Status(status).name}")
    return errors
