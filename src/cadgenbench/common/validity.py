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

from dataclasses import dataclass, field
from pathlib import Path

from cadgenbench.common.measurements import Measurements, _measure_wrapped


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
# Public API
# ---------------------------------------------------------------------------


def validate_step(step_path: str | Path) -> ValidationResult:
    """Validate a STEP file and return :class:`ValidationResult` only.

    Runs the **full** three-part gate: ``BRepCheck`` + watertight +
    mesh-gate. Equivalent to ``analyze_step(step_path).validation`` but
    skips returning the measurements; use :func:`analyze_step` when you
    also need them and don't want to parse the STEP twice.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the STEP file cannot be loaded.
    """
    wrapped = _load_step_wrapped(step_path)
    return _validate_wrapped(wrapped)


def analyze_step(step_path: str | Path) -> ValidityResult:
    """Load a STEP file once and return both validity and measurements.

    Order: measurements first (cheap), then validation reuses the bbox
    diagonal to compute its tessellation deflection without re-walking
    the shape.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the STEP file cannot be loaded.
    """
    wrapped = _load_step_wrapped(step_path)
    measurements = _measure_wrapped(wrapped)
    validation = _validate_wrapped(
        wrapped, bbox_diagonal=measurements.bounding_box.diagonal,
    )
    return ValidityResult(validation=validation, measurements=measurements)


def parse_step(step_path: str | Path) -> None:
    """Cheap check that a file can be loaded as STEP geometry.

    Runs only the STEP reader (``build123d.import_step``), no BRepCheck,
    no watertight test, no mesh tessellation. Used by upstream callers
    (e.g. the leaderboard's submit handler) that want to reject
    non-STEP / corrupted uploads at request time but leave the full
    validity gate to a downstream evaluator. A file that parses here
    can still be reported as invalid by :func:`analyze_step`; that's
    the per-fixture validity signal, distinct from "is this even STEP".

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file is not loadable STEP geometry.
    """
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
) -> ValidationResult:
    """Validate a pre-loaded OCC ``TopoDS_Shape``.

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
        if not quality_errors:
            deflection = _tessellation_deflection(wrapped, bbox_diagonal)
            mesh_ok = _run_mesh_gate(wrapped, deflection, topology_errors)
        else:
            mesh_ok = False

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
) -> bool:
    """Run the mesh-pipeline gate and append any failure to *topology_errors*.

    Returns True iff the tessellated boundary is a clean closed
    orientable manifold (so the topology-match metric's
    :math:`\\chi`-based math is well defined).
    """
    from cadgenbench.common.mesh import (
        MeshSanityError,
        tessellate_shape,
        validate_mesh,
    )

    try:
        mesh = tessellate_shape(wrapped, deflection)
        validate_mesh(mesh)
    except MeshSanityError as exc:
        # MeshSanityError messages already start with the failure mode
        # ("mesh non-manifold: ...", "mesh not closed: ...",
        # "mesh orientation inconsistent: ..."), matching the example in
        # the ValidationResult docstring, append as-is, no prefix.
        topology_errors.append(str(exc))
        return False
    return True


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
