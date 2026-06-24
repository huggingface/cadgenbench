"""Lazy per-STEP geometry artifacts shared by validation and metrics.

This module intentionally keeps caches in-process. Callers that run many
workers can still put serialized mesh arrays next to the dataset later; each
worker can hydrate those into its own Python / Manifold objects.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cadgenbench.common.mesh import Mesh
from cadgenbench.common.measurements import BBox, Measurements, _measure_wrapped
from cadgenbench.common.validity import (
    ValidationResult,
    ValidityResult,
    _file_size_error,
    _load_step_wrapped,
    _oversized_validity_result,
    _raise_if_ground_truth,
    _validate_wrapped,
    safeguarded_tessellate,
)


def sidecar_path_for(step_path: Path | str) -> Path:
    """Path of the trusted-mesh sidecar for *step_path* (``<stem>.mesh.npz``).

    A sidecar next to a STEP marks it "trusted, valid by construction": its
    presence makes :class:`StepArtifacts` load the supplied mesh and skip
    both validation and tessellation (see :meth:`StepArtifacts._sidecar_path`).
    """
    p = Path(step_path)
    return p.with_name(p.stem + ".mesh.npz")


def write_mesh_sidecar(step_path: Path | str, mesh: Mesh) -> None:
    """Write *mesh* as the trusted-mesh sidecar next to *step_path*.

    Used when a STEP's mesh is already known (e.g. a rigidly aligned mesh
    transformed from an already-tessellated source) so downstream
    :class:`StepArtifacts` consumers reuse it instead of re-tessellating.
    """
    path = sidecar_path_for(step_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as fh:
            np.savez(
                fh,
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                triangles=np.asarray(mesh.triangles, dtype=np.int64),
                linear_deflection_mm=np.asarray(mesh.linear_deflection_mm),
            )
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


@dataclass
class StepArtifacts:
    """Lazy artifacts for one STEP file.

    The object is deliberately small and process-local. OCC shapes and
    ``manifold3d.Manifold`` instances are not treated as stable serialized
    dataset artifacts; callers can serialize the mesh arrays separately if
    they need cross-worker reuse.
    """

    step_path: Path | str
    mesh_cache_dir: Path | str | None = None
    deflection_override: float | None = None
    is_ground_truth: bool = False
    _wrapped: object | None = field(default=None, init=False, repr=False)
    _analysis: ValidityResult | None = field(default=None, init=False, repr=False)
    _meshes: dict[float, object] = field(default_factory=dict, init=False, repr=False)
    _manifolds: dict[float, object] = field(default_factory=dict, init=False, repr=False)
    _bettis: dict[float, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.step_path = Path(self.step_path)
        if self.mesh_cache_dir is None:
            self.mesh_cache_dir = os.environ.get("CADGENBENCH_MESH_CACHE_DIR")
        if self.mesh_cache_dir is not None:
            self.mesh_cache_dir = Path(self.mesh_cache_dir)

    @property
    def wrapped(self):  # type: ignore[no-untyped-def]
        """Loaded OCC ``TopoDS_Shape``, parsed once per artifact."""
        if self._wrapped is None:
            self._wrapped = _load_step_wrapped(self.step_path)
        return self._wrapped

    @property
    def analysis(self) -> ValidityResult:
        """Validity + measurements, computed once from the loaded shape."""
        if self._analysis is None:
            # A supplied-mesh sidecar means "trusted, valid by construction":
            # skip validity. Measurements still come from the BREP (no mesh).
            if self._sidecar_path() is not None:
                measurements = _measure_wrapped(self.wrapped)
                self._analysis = ValidityResult(
                    validation=ValidationResult(
                        is_valid=True,
                        is_watertight=True,
                        topology_errors=(),
                    ),
                    measurements=measurements,
                )
                return self._analysis
            cached = self._load_analysis_cache()
            if cached is not None:
                self._analysis = cached
            else:
                # File-size pre-filter, before the STEP is parsed: refuse an
                # oversized part rather than paying to load + measure it.
                fs_error = _file_size_error(Path(self.step_path))
                if fs_error is not None:
                    _raise_if_ground_truth(
                        fs_error, Path(self.step_path), self.is_ground_truth,
                    )
                    self._analysis = _oversized_validity_result(fs_error)
                    return self._analysis
                measurements = _measure_wrapped(self.wrapped)
                validation = _validate_wrapped(
                    self.wrapped,
                    bbox_diagonal=measurements.bounding_box.diagonal,
                    # Mesh the validity gate at this part's one deflection so
                    # the cached mesh is exactly what mesh()/manifold()/betti()
                    # later read — never a second tessellation at another scale.
                    deflection=self.deflection_override,
                    mesh_cache=self._meshes,
                    step_path=Path(self.step_path),
                    is_ground_truth=self.is_ground_truth,
                )
                self._analysis = ValidityResult(
                    validation=validation,
                    measurements=measurements,
                )
                self._store_analysis_cache(self._analysis)
        return self._analysis

    def deflection(self) -> float:
        """Tessellation deflection for this part.

        ``deflection_override`` (set at construction) wins when present —
        used to mesh small sub-volumes at their parent GT's scale so a
        cross-solid Boolean shares one tessellation scale on both
        operands. Otherwise it is derived from this part's own bbox.
        """
        if self.deflection_override is not None:
            return float(self.deflection_override)
        from cadgenbench.common.mesh import deflection_for_bbox

        return deflection_for_bbox(self.analysis.measurements.bounding_box.diagonal)

    def mesh(self):
        """The part's one validated mesh at its deflection, produced once + cached.

        Each part is tessellated at exactly one deflection (its own, or the
        ``deflection_override`` for a sub-volume) and cached, so every caller
        reads the same mesh and nothing is ever re-meshed at a second
        resolution. Meshing goes through :func:`safeguarded_tessellate`
        (per-mesh timeout + triangle ceiling); a cache hit does no work.
        """
        from cadgenbench.common.mesh import MeshSanityError

        # Invalid parts have no mesh; fail fast rather than tessellate.
        validation = self.analysis.validation
        if not validation.is_valid:
            reason = (
                validation.topology_errors[0]
                if validation.topology_errors
                else "is_valid=False"
            )
            raise MeshSanityError(f"{self.step_path.name}: not a valid mesh ({reason})")

        deflection = self.deflection()
        # A trusted supplied mesh is the reference: use it, never re-mesh.
        if deflection not in self._meshes:
            sidecar = self._load_sidecar_mesh()
            if sidecar is not None:
                self._meshes[deflection] = sidecar
                return sidecar
        if deflection in self._meshes:
            self._store_mesh_cache(deflection, self._meshes[deflection])
        else:
            cached = self._load_mesh_cache(deflection)
            if cached is not None:
                self._meshes[deflection] = cached
                return cached
            # Tessellate once at this deflection, then cache.
            mesh = safeguarded_tessellate(
                self.step_path,
                deflection,
                wrapped=self.wrapped,
                is_ground_truth=self.is_ground_truth,
            )
            self._meshes[deflection] = mesh
            self._store_mesh_cache(deflection, mesh)
        return self._meshes[deflection]

    def manifold(self):
        """``manifold3d.Manifold`` for the part's cached validated mesh."""
        from cadgenbench.eval.booleans import mesh_to_manifold

        deflection = self.deflection()
        if deflection not in self._manifolds:
            self._manifolds[deflection] = mesh_to_manifold(self.mesh())
        return self._manifolds[deflection]

    def betti(self):
        """Betti numbers for the part's cached validated mesh."""
        from cadgenbench.eval.topo_match import compute_betti_from_mesh

        deflection = self.deflection()
        if deflection not in self._bettis:
            self._bettis[deflection] = compute_betti_from_mesh(self.mesh())
        return self._bettis[deflection]

    def _sidecar_path(self) -> Path | None:
        """Path to the trusted supplied-mesh sidecar, if one exists.

        The sidecar lives next to the STEP with the same stem and a
        ``.mesh.npz`` suffix (e.g. ``ground_truth.step`` ->
        ``ground_truth.mesh.npz``). Its presence == "trusted, skip checks".
        """
        cand = sidecar_path_for(self.step_path)
        return cand if cand.exists() else None

    @property
    def has_sidecar(self) -> bool:
        """True iff this part carries a trusted supplied-mesh sidecar.

        Callers use this to keep the part on the trusted-mesh path
        (transform the cached mesh) rather than re-tessellating its STEP.
        """
        return self._sidecar_path() is not None

    def _load_sidecar_mesh(self) -> Mesh | None:
        """Load the supplied-mesh sidecar as a :class:`Mesh`, or ``None``."""
        path = self._sidecar_path()
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            triangles = np.asarray(data["triangles"], dtype=np.int64)
            deflection = float(data["linear_deflection_mm"])
        return Mesh(
            vertices=vertices,
            triangles=triangles,
            linear_deflection_mm=deflection,
        )

    def _cache_stem(self) -> str | None:
        if self.mesh_cache_dir is None:
            return None
        stat = self.step_path.stat()
        name = "".join(
            c if c.isalnum() or c in "._-" else "_"
            for c in self.step_path.stem
        )
        return f"{name}-{stat.st_size}-{stat.st_mtime_ns}"

    def _mesh_cache_path(self, linear_deflection_mm: float) -> Path | None:
        stem = self._cache_stem()
        if stem is None:
            return None
        defl = f"{linear_deflection_mm:.12g}".replace(".", "p")
        key = f"{stem}-{defl}.npz"
        return Path(self.mesh_cache_dir) / key

    def _analysis_cache_path(self) -> Path | None:
        stem = self._cache_stem()
        if stem is None:
            return None
        return Path(self.mesh_cache_dir) / f"{stem}-analysis.json"

    def _load_analysis_cache(self) -> ValidityResult | None:
        path = self._analysis_cache_path()
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            bb = data["measurements"]["bounding_box"]
            measurements = Measurements(
                solid_count=int(data["measurements"]["solid_count"]),
                shell_count=int(data["measurements"]["shell_count"]),
                face_count=int(data["measurements"]["face_count"]),
                volume=float(data["measurements"]["volume"]),
                bounding_box=BBox(
                    x_min=float(bb["x_min"]),
                    x_max=float(bb["x_max"]),
                    y_min=float(bb["y_min"]),
                    y_max=float(bb["y_max"]),
                    z_min=float(bb["z_min"]),
                    z_max=float(bb["z_max"]),
                ),
            )
            validation = ValidationResult(
                is_valid=bool(data["validation"]["is_valid"]),
                is_watertight=bool(data["validation"]["is_watertight"]),
                topology_errors=tuple(data["validation"]["topology_errors"]),
            )
            return ValidityResult(validation=validation, measurements=measurements)
        except Exception:
            return None

    def _store_analysis_cache(self, analysis: ValidityResult) -> None:
        path = self._analysis_cache_path()
        if path is None or path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        bb = analysis.measurements.bounding_box
        data = {
            "validation": {
                "is_valid": analysis.validation.is_valid,
                "is_watertight": analysis.validation.is_watertight,
                "topology_errors": list(analysis.validation.topology_errors),
            },
            "measurements": {
                "solid_count": analysis.measurements.solid_count,
                "shell_count": analysis.measurements.shell_count,
                "face_count": analysis.measurements.face_count,
                "volume": analysis.measurements.volume,
                "bounding_box": {
                    "x_min": bb.x_min,
                    "x_max": bb.x_max,
                    "y_min": bb.y_min,
                    "y_max": bb.y_max,
                    "z_min": bb.z_min,
                    "z_max": bb.z_max,
                },
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(data, sort_keys=True))
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)

    def _load_mesh_cache(self, linear_deflection_mm: float) -> Mesh | None:
        path = self._mesh_cache_path(linear_deflection_mm)
        if path is None or not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                vertices = np.asarray(data["vertices"], dtype=np.float64)
                triangles = np.asarray(data["triangles"], dtype=np.int64)
                deflection = float(data["linear_deflection_mm"])
            # The cache file is keyed (in its name) by the requested
            # deflection; the stored mesh's own ``linear_deflection_mm`` may
            # be finer (the robust tessellator escalated). Trust the filename
            # key and return the stored mesh with its true deflection.
            return Mesh(
                vertices=vertices,
                triangles=triangles,
                linear_deflection_mm=deflection,
            )
        except Exception:
            # Corrupt or stale cache entries should never affect scoring.
            return None

    def _store_mesh_cache(self, linear_deflection_mm: float, mesh: object) -> None:
        path = self._mesh_cache_path(linear_deflection_mm)
        if path is None or path.exists() or not isinstance(mesh, Mesh):
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("wb") as fh:
                np.savez(
                    fh,
                    vertices=mesh.vertices,
                    triangles=mesh.triangles,
                    linear_deflection_mm=np.asarray(mesh.linear_deflection_mm),
                )
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
