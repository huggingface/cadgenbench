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

"""Interface match metric category.

Scores whether a candidate part's interfaces (holes, pockets, bosses)
match the ground truth's specification. Full spec lives at
[`docs/metrics/interface_match.md`](../../docs/metrics/interface_match.md).

Public API (in evaluation order):

- :class:`SubVolume`            : one canonical reference region.
- :func:`discover_sub_volumes`  : find a fixture's sub-volumes.
- :func:`iou_at_pose`           : IoU for one sub-volume at the exact
                                  GT pose (no search).
- :func:`best_iou_in_context`   : deterministic bounded pose search
                                  per ``context_id``; returns
                                  ``{sv.name: max IoU}``.
- :func:`interface_score_iou`   : pose-searched IoU per sub-volume
                                  across the whole fixture.
- :func:`interface_score`       : single-number aggregated score.
- :func:`disagreement_volume`,
  :func:`score_candidate`       : cheap binary-style discriminator
                                  used by the visualiser; signed
                                  disagreement volume rather than IoU.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from cadgenbench.common.artifacts import StepArtifacts


# ---------------------------------------------------------------------------
# Fit type
# ---------------------------------------------------------------------------

# Valid ``fit_type`` values. ``KOR`` = keep-out region: the candidate's
# solid must be absent in R (holes / slots / pockets). ``KIR`` = keep-in
# region: the candidate's solid must be present in R (bosses / protrusions).
FIT_TYPES = frozenset({"KOR", "KIR"})


# ---------------------------------------------------------------------------
# Sub-volume discovery
# ---------------------------------------------------------------------------

# Filename convention: jig_<context_id>__<index>__<fit_type>.step
_SUBVOL_RE = re.compile(r"^jig_(\d+)__(\d+)__(KOR|KIR)\.step$")


@dataclass(frozen=True)
class SubVolume:
    """One canonical reference region for a single interface.

    Built from a STEP filename of the form
    ``jig_<context_id>__<index>__<fit_type>.step``. See
    :func:`discover_sub_volumes`.
    """

    path: Path
    context_id: int
    index: int
    fit_type: str  # "KOR" (keep-out region) or "KIR" (keep-in region)

    @property
    def name(self) -> str:
        """Stable label used in score dictionaries: ``"index__fit_type"``."""
        return f"{self.index}__{self.fit_type}"


def discover_sub_volumes(fixture_dir: str | Path) -> list[SubVolume]:
    """Return all sub-volume STEP files in *fixture_dir*, sorted by filename.

    Files not matching the canonical naming pattern are ignored.
    """
    fixture_dir = Path(fixture_dir)
    out: list[SubVolume] = []
    for p in sorted(fixture_dir.glob("jig_*__*__*.step")):
        m = _SUBVOL_RE.match(p.name)
        if not m:
            continue
        out.append(
            SubVolume(
                path=p,
                context_id=int(m.group(1)),
                index=int(m.group(2)),
                fit_type=m.group(3),
            ),
        )
    return out


# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

# Below this volume (mm^3) treat a disagreement as numerical noise.
DEFAULT_DISAGREEMENT_EPSILON = 1.0

# Per-sub-volume IoU pass threshold.
DEFAULT_IOU_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Disagreement scoring (cheap binary-style discriminator)
# ---------------------------------------------------------------------------


def disagreement_volume(part_step: str | Path, sv: SubVolume) -> float:
    """Return the disagreement volume (mm^3) between a candidate part and
    one sub-volume.

    Disagreement is the volume of the part of `R` where the candidate
    disagrees with the spec:

    - For ``fit_type == "KOR"`` (keep-out region):  ``vol(R ∩ candidate_solid)``
      (candidate has solid material where it should be empty).
    - For ``fit_type == "KIR"`` (keep-in region): ``vol(R \\ candidate_solid)``
      (candidate is missing material that should be present).

    A correct candidate returns 0. The function is the simplest
    discrimination signal usable before the full IoU + pose-search
    metric is in place.
    """
    from build123d import import_step

    part = import_step(str(part_step))
    R = import_step(str(sv.path))

    if sv.fit_type == "KOR":
        result = R & part
    elif sv.fit_type == "KIR":
        result = R - part
    else:  # pragma: no cover - construction-time guarantee
        raise ValueError(
            f"Unknown fit_type {sv.fit_type!r} on {sv.path.name}"
        )

    if result is None:
        return 0.0

    # build123d returns a single Shape when the Boolean produces one solid
    # and a ShapeList when it splits into multiple disconnected solids.
    if hasattr(result, "wrapped") and result.wrapped is not None:
        return float(result.volume)
    children = [s for s in result if hasattr(s, "wrapped") and s.wrapped is not None]
    if not children:
        return 0.0
    return sum(float(s.volume) for s in children)


def score_candidate(
    part_step: str | Path,
    fixture_dir: str | Path,
) -> dict[str, float]:
    """Score one candidate against every sub-volume in *fixture_dir*.

    Returns ``{SubVolume.name: disagreement_mm3}`` for every sub-volume
    discovered. The mapping is suitable for the GT self-test (every
    value should be ≈ 0) and for ranking broken candidates.

    Raises :class:`FileNotFoundError` if *fixture_dir* has no sub-volumes.
    """
    fixture_dir = Path(fixture_dir)
    sub_volumes = discover_sub_volumes(fixture_dir)
    if not sub_volumes:
        raise FileNotFoundError(
            f"No jig_<id>__<index>__<fit>.step files in {fixture_dir}",
        )
    return {sv.name: disagreement_volume(part_step, sv) for sv in sub_volumes}


# ---------------------------------------------------------------------------
# IoU scoring
# ---------------------------------------------------------------------------
#
# Two layers:
#
# 1. Pose-independent precomputation per sub-volume: tessellate R, build
#    inflated_R AABB, intersect/subtract with GT for the verification
#    shell, union R + shell into bbox_R, cache vol(R). Captured as
#    :class:`_SubVolumeCache`.
# 2. Pose-dependent evaluation: transform R rigidly by a pose offset
#    (:func:`manifold3d.Manifold.transform`), keep bbox_R fixed at the GT
#    pose, compute the candidate region C inside fixed bbox_R, return
#    IoU.
#
# Both ``iou_at_pose`` (single shot, GT pose) and ``best_iou_in_context``
# (bounded deterministic search) build on this split.

# Pose: (tx, ty, tz, rx_deg, ry_deg, rz_deg). Translation in mm, rotation
# in degrees XYZ Euler about the world origin. The candidate is assumed
# globally aligned (see :mod:`cadgenbench.eval.alignment`), so the world
# origin is approximately the part centroid and rotations are local.
_ZERO_POSE: tuple[float, float, float, float, float, float] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
)

# Bounded-search budget: ±1% of the GT bounding-box diagonal per
# translation axis, ±1° per rotation axis. The translation window is
# proportional to part scale (small parts get a tight window in mm,
# large parts a looser one) so the budget tracks the residual
# misalignment ICP leaves behind, which itself scales with the diagonal.
# Callers that need an absolute window in mm can pass an explicit
# ``max_translation_mm`` and bypass the per-fixture derivation.
TRANSLATION_FRACTION_OF_BBOX = 0.01
DEFAULT_MAX_ROTATION_DEG = 1.0
DEFAULT_N_SAMPLES = 32

# IoU >= this saturates to 1.0 per sub-volume, absorbing sub-millimetre
# tessellation residue so an authoring-equivalent submission scores 1.0
# cleanly. Real geometric errors drop IoU well below 0.99 and are
# unaffected.
SATURATION_THRESHOLD = 0.99

# Inflated-AABB margin around R: floor + fraction-of-longest-extent.
_INFLATED_MARGIN_FLOOR_MM = 2.0
_INFLATED_MARGIN_FRACTION = 0.20


@dataclass(frozen=True)
class _SubVolumeCache:
    """Pose-independent precomputation for one sub-volume.

    Built once per (sub-volume, GT) pair and reused across candidate
    poses. Holds ``manifold3d.Manifold`` objects, which are in-process
    only; cross-process callers must pass STEP paths and rebuild locally.
    """

    sv: SubVolume
    R: object          # manifold3d.Manifold of R at the GT pose
    bbox_R: object     # manifold3d.Manifold of R ∪ shell
    vol_R: float

    @property
    def name(self) -> str:
        return self.sv.name

    @property
    def fit_type(self) -> str:
        return self.sv.fit_type


@dataclass(frozen=True)
class _CandidateRegionCache:
    """Candidate-side region C for one sub-volume, fixed across pose search."""

    sub_volume: _SubVolumeCache
    C: object
    vol_C: float

    @property
    def name(self) -> str:
        return self.sub_volume.name


@dataclass
class InterfaceMatchArtifacts:
    """GT-side interface artifacts reused across candidates in one worker."""

    gt_step: Path
    sub_volumes: list[SubVolume]
    gt_artifacts: StepArtifacts | None = None
    linear_deflection_mm: float | None = None
    _sub_volume_artifacts: dict[Path, StepArtifacts] | None = None
    _sub_volume_caches: dict[Path, _SubVolumeCache] | None = None

    def __post_init__(self) -> None:
        self.gt_step = Path(self.gt_step)
        if self.gt_artifacts is None:
            self.gt_artifacts = StepArtifacts(self.gt_step)
        if self._sub_volume_artifacts is None:
            self._sub_volume_artifacts = {}
        if self._sub_volume_caches is None:
            self._sub_volume_caches = {}

    @classmethod
    def for_fixture(
        cls,
        fixture_dir: str | Path,
        *,
        gt_step: str | Path | None = None,
    ) -> "InterfaceMatchArtifacts":
        fixture_dir = Path(fixture_dir)
        return cls(
            gt_step=Path(gt_step) if gt_step is not None else fixture_dir / "gt.step",
            sub_volumes=discover_sub_volumes(fixture_dir),
        )

    @property
    def deflection_mm(self) -> float:
        from cadgenbench.common.mesh import deflection_for_bbox

        if self.linear_deflection_mm is None:
            assert self.gt_artifacts is not None
            self.linear_deflection_mm = deflection_for_bbox(
                self.gt_artifacts.analysis.measurements.bounding_box.diagonal,
            )
        return float(self.linear_deflection_mm)

    @property
    def gt_manifold(self):
        assert self.gt_artifacts is not None
        return self.gt_artifacts.manifold(self.deflection_mm)

    def sub_volume_artifact(self, sv: SubVolume) -> StepArtifacts:
        assert self._sub_volume_artifacts is not None
        key = Path(sv.path)
        if key not in self._sub_volume_artifacts:
            self._sub_volume_artifacts[key] = StepArtifacts(key)
        return self._sub_volume_artifacts[key]

    def cache_for(self, sv: SubVolume) -> _SubVolumeCache:
        assert self._sub_volume_caches is not None
        key = Path(sv.path)
        if key not in self._sub_volume_caches:
            self._sub_volume_caches[key] = _build_sub_volume_cache(
                sv,
                self.gt_manifold,
                self.deflection_mm,
                sv_artifacts=self.sub_volume_artifact(sv),
            )
        return self._sub_volume_caches[key]


def _build_sub_volume_cache(
    sv: SubVolume,
    gt_manifold,
    deflection_mm,
    *,
    sv_artifacts: StepArtifacts | None = None,
):
    """Build the pose-independent cache for one sub-volume.

    *gt_manifold* is a pre-tessellated ``manifold3d.Manifold`` of the
    ground-truth solid (the caller deduplicates the tessellation across
    sub-volumes of one fixture). *deflection_mm* is the chord-error
    deflection used to tessellate this sub-volume, derived from the GT
    bbox by the caller so candidate and GT share one scale.
    """
    import manifold3d as m3d

    from cadgenbench.eval.booleans import (
        intersect,
        manifold_volume,
        mesh_to_manifold,
        subtract,
        union,
    )
    from cadgenbench.common.mesh import tessellate_and_validate

    R_mesh = (
        sv_artifacts.mesh(deflection_mm)
        if sv_artifacts is not None
        else tessellate_and_validate(sv.path, deflection_mm)
    )
    R = mesh_to_manifold(R_mesh)

    # Inflated AABB built via manifold3d's cube primitive.
    lo = R_mesh.vertices.min(axis=0)
    hi = R_mesh.vertices.max(axis=0)
    longest = float((hi - lo).max())
    margin = max(_INFLATED_MARGIN_FLOOR_MM, _INFLATED_MARGIN_FRACTION * longest)
    inflated_lo = lo - margin
    inflated_hi = hi + margin
    inflated = m3d.Manifold.cube(
        (inflated_hi - inflated_lo).tolist(), center=False,
    ).translate(inflated_lo.tolist())

    if sv.fit_type == "KOR":
        shell = intersect(inflated, gt_manifold)
    elif sv.fit_type == "KIR":
        shell = subtract(inflated, gt_manifold)
    else:  # pragma: no cover - construction-time guarantee
        raise ValueError(f"Unknown fit_type {sv.fit_type!r}")

    bbox_R = union(R, shell)
    return _SubVolumeCache(sv=sv, R=R, bbox_R=bbox_R, vol_R=manifold_volume(R))


def _iou_with_cache(
    cache: _SubVolumeCache,
    candidate_manifold,
    pose: tuple[float, ...],
) -> float:
    """IoU for ``cache`` vs candidate (Manifold) at ``pose``.

    Rigidly transforms R by *pose* (XYZ Euler in degrees + translation
    in mm) while keeping ``bbox_R`` fixed at the GT pose. Computes the
    candidate region C inside fixed ``bbox_R`` per fit_type and returns
    ``vol(R ∩ C) / vol(R ∪ C)``, saturated to 1.0 above
    :data:`SATURATION_THRESHOLD`.
    """
    from cadgenbench.eval.booleans import (
        apply_pose,
        intersect,
        manifold_volume,
        subtract,
    )

    R_p = cache.R if pose == _ZERO_POSE else apply_pose(cache.R, pose)

    if cache.fit_type == "KOR":
        C = subtract(cache.bbox_R, candidate_manifold)
    else:  # KIR
        C = intersect(cache.bbox_R, candidate_manifold)

    vol_R = cache.vol_R
    vol_C = manifold_volume(C)
    if vol_C <= 0:
        return 0.0
    vol_inter = manifold_volume(intersect(R_p, C))
    vol_union = vol_R + vol_C - vol_inter
    if vol_union <= 0:
        return 0.0
    iou = vol_inter / vol_union
    return 1.0 if iou >= SATURATION_THRESHOLD else iou


def _candidate_region_for_cache(
    cache: _SubVolumeCache,
    candidate_manifold,
) -> _CandidateRegionCache:
    """Compute the pose-independent candidate region C for one sub-volume."""
    from cadgenbench.eval.booleans import (
        intersect,
        manifold_volume,
        subtract,
    )

    if cache.fit_type == "KOR":
        C = subtract(cache.bbox_R, candidate_manifold)
    else:  # KIR
        C = intersect(cache.bbox_R, candidate_manifold)
    return _CandidateRegionCache(
        sub_volume=cache,
        C=C,
        vol_C=manifold_volume(C),
    )


def _iou_with_candidate_region(
    candidate_region: _CandidateRegionCache,
    pose: tuple[float, ...],
) -> float:
    """IoU using precomputed candidate region C for one pose."""
    from cadgenbench.eval.booleans import (
        apply_pose,
        intersect,
        manifold_volume,
    )

    cache = candidate_region.sub_volume
    if candidate_region.vol_C <= 0:
        return 0.0

    R_p = cache.R if pose == _ZERO_POSE else apply_pose(cache.R, pose)
    vol_inter = manifold_volume(intersect(R_p, candidate_region.C))
    vol_union = cache.vol_R + candidate_region.vol_C - vol_inter
    if vol_union <= 0:
        return 0.0
    iou = vol_inter / vol_union
    return 1.0 if iou >= SATURATION_THRESHOLD else iou


def _tessellate_for_iou(step_path: Path, deflection_mm: float):
    """Tessellate *step_path* and return its ``manifold3d.Manifold``."""
    from cadgenbench.eval.booleans import mesh_to_manifold
    from cadgenbench.common.mesh import tessellate_and_validate

    mesh = tessellate_and_validate(step_path, deflection_mm)
    return mesh_to_manifold(mesh)


def iou_at_pose(
    part_step: str | Path,
    sv: SubVolume,
    gt_step: str | Path,
) -> float:
    """IoU(R, C) for one sub-volume at the exact GT-specified pose.

    The reference region ``R`` is augmented by a *verification shell*
    of opposite material (plate material around a hole; empty air
    around a boss). Together they form the bounding region
    ``bbox_R``:

    - ``fit_type == "KOR"`` (keep-out region)  →  ``bbox_R = R ∪ (inflated_R ∩ GT.solid)``
      (hole plus the plate material immediately around it).
    - ``fit_type == "KIR"`` (keep-in region)   →  ``bbox_R = R ∪ (inflated_R \\ GT.solid)``
      (boss plus the empty air immediately around it on the outward side).

    Where ``inflated_R`` is ``AABB(R)`` extended outward by
    ``margin_M = max(2.0 mm, 0.20 × longest_extent(R))``.

    The candidate region ``C`` is then:

    - ``fit_type == "KOR"``  →  ``C = bbox_R \\ candidate_solid``
    - ``fit_type == "KIR"``  →  ``C = bbox_R ∩ candidate_solid``

    Returns ``IoU = vol(R ∩ C) / vol(R ∪ C)`` saturated to 1.0 at
    :data:`SATURATION_THRESHOLD`. A perfectly-fitting candidate scores
    1.0; an oversize or undersize feature scores below the
    :data:`DEFAULT_IOU_THRESHOLD` (0.95).

    No pose search is applied here -- evaluation is at exactly the
    GT-specified pose. The full v1 metric layers
    :func:`best_iou_in_context` on top of this primitive.
    """
    artifacts = InterfaceMatchArtifacts(gt_step=Path(gt_step), sub_volumes=[sv])
    defl = artifacts.deflection_mm
    candidate_manifold = StepArtifacts(part_step).manifold(defl)
    cache = artifacts.cache_for(sv)
    return _iou_with_cache(cache, candidate_manifold, _ZERO_POSE)


# ---------------------------------------------------------------------------
# Bounded pose search (per context_id)
# ---------------------------------------------------------------------------


def _generate_poses(
    n_samples: int,
    max_translation_mm: float,
    max_rotation_deg: float,
) -> list[tuple[float, float, float, float, float, float]]:
    """Return deterministic *n_samples* poses including the zero pose.

    The zero pose is always first so the search result is monotone:
    pose-search IoU ≥ GT-pose IoU. Remaining samples come from a
    deterministic Sobol low-discrepancy sequence mapped to
    ``[-max, +max]`` per axis.
    """
    from scipy.stats import qmc

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    poses: list[tuple[float, float, float, float, float, float]] = [_ZERO_POSE]
    if n_samples == 1:
        return poses

    n_needed = n_samples - 1
    m = math.ceil(math.log2(n_needed + 1))
    sobol = qmc.Sobol(d=6, scramble=False)
    unit_points = sobol.random_base2(m=m)[1 : n_needed + 1]

    for u in unit_points:
        poses.append((
            (float(u[0]) * 2.0 - 1.0) * max_translation_mm,
            (float(u[1]) * 2.0 - 1.0) * max_translation_mm,
            (float(u[2]) * 2.0 - 1.0) * max_translation_mm,
            (float(u[3]) * 2.0 - 1.0) * max_rotation_deg,
            (float(u[4]) * 2.0 - 1.0) * max_rotation_deg,
            (float(u[5]) * 2.0 - 1.0) * max_rotation_deg,
        ))
    return poses


def best_iou_in_context(
    part_step: str | Path,
    sub_volumes: list[SubVolume],
    gt_step: str | Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    workers: int = 1,
    max_translation_mm: float | None = None,
    max_rotation_deg: float = DEFAULT_MAX_ROTATION_DEG,
    candidate_artifacts: StepArtifacts | None = None,
    interface_artifacts: InterfaceMatchArtifacts | None = None,
) -> dict[str, float]:
    """Deterministic bounded pose search for one rigid mating context.

    Samples *n_samples* poses (the first is always the zero pose, so the
    result is at least the GT-pose IoU). For each pose, all *sub_volumes*
    are transformed rigidly together and the IoU is evaluated against
    *part_step*. Returns ``{sv.name: max IoU}`` with each IoU saturated
    to 1.0 above :data:`SATURATION_THRESHOLD`.

    Args:
        part_step: Candidate STEP file. Assumed globally aligned to the
            GT frame (see :mod:`cadgenbench.eval.alignment`).
        sub_volumes: Sub-volumes that move together as one rigid body.
            Typically the output of filtering :func:`discover_sub_volumes`
            by ``context_id``.
        gt_step: Ground-truth STEP, used to build the verification
            shell around each sub-volume.
        n_samples: Pose count, including the zero pose.
        workers: Number of threads used for pose/sub-volume scoring. Keep at
            1 when running inside already-parallel worker pools to avoid
            oversubscription.
        max_translation_mm: Half-width of the per-axis translation
            sampling interval, in mm. When ``None`` (the default) the
            window is derived from the GT bounding-box diagonal as
            ``TRANSLATION_FRACTION_OF_BBOX * diag``.
        max_rotation_deg: Half-width of the per-axis rotation sampling
            interval, in degrees XYZ Euler about the world origin.
    """
    if not sub_volumes:
        raise ValueError("best_iou_in_context: sub_volumes must not be empty")

    interface_artifacts = interface_artifacts or InterfaceMatchArtifacts(
        gt_step=Path(gt_step),
        sub_volumes=sub_volumes,
    )
    gt_diag = float(
        interface_artifacts.gt_artifacts.analysis.measurements.bounding_box.diagonal,
    )
    if max_translation_mm is None:
        max_translation_mm = TRANSLATION_FRACTION_OF_BBOX * gt_diag

    poses = _generate_poses(
        n_samples=n_samples,
        max_translation_mm=max_translation_mm,
        max_rotation_deg=max_rotation_deg,
    )

    defl = interface_artifacts.deflection_mm
    candidate_artifacts = candidate_artifacts or StepArtifacts(part_step)
    candidate_manifold = candidate_artifacts.manifold(defl)
    caches = [interface_artifacts.cache_for(sv) for sv in sub_volumes]
    candidate_regions = [
        _candidate_region_for_cache(c, candidate_manifold)
        for c in caches
    ]

    per_sv = {c.name: 0.0 for c in caches}
    tasks = [(pose, c) for pose in poses for c in candidate_regions]
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(int(workers), len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = pool.map(
                lambda item: (
                    item[1].name,
                    _iou_with_candidate_region(item[1], item[0]),
                ),
                tasks,
            )
            for name, iou in results:
                if iou > per_sv[name]:
                    per_sv[name] = iou
    else:
        for pose, c in tasks:
            iou = _iou_with_candidate_region(c, pose)
            if iou > per_sv[c.name]:
                per_sv[c.name] = iou
    return per_sv


def interface_score_iou(
    part_step: str | Path,
    fixture_dir: str | Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    workers: int = 1,
    max_translation_mm: float | None = None,
    max_rotation_deg: float = DEFAULT_MAX_ROTATION_DEG,
) -> dict[str, float]:
    """Pose-searched IoU per sub-volume for one candidate against one fixture.

    Each ``context_id`` is searched independently via
    :func:`best_iou_in_context`. Returns ``{SubVolume.name: max IoU}`` for
    every sub-volume in *fixture_dir*. Diagnostic output; use
    :func:`interface_score` for a single aggregated number.
    """
    fixture_dir = Path(fixture_dir)
    sub_volumes = discover_sub_volumes(fixture_dir)
    if not sub_volumes:
        raise FileNotFoundError(
            f"No jig_<id>__<index>__<fit>.step files in {fixture_dir}",
        )
    gt_step = fixture_dir / "gt.step"
    if not gt_step.exists():
        raise FileNotFoundError(f"gt.step missing in {fixture_dir}")

    by_context: dict[int, list[SubVolume]] = {}
    for sv in sub_volumes:
        by_context.setdefault(sv.context_id, []).append(sv)

    artifacts = InterfaceMatchArtifacts(gt_step=gt_step, sub_volumes=sub_volumes)
    candidate_artifacts = StepArtifacts(part_step)
    out: dict[str, float] = {}
    for ctx_svs in by_context.values():
        out.update(best_iou_in_context(
            part_step,
            ctx_svs,
            gt_step,
            n_samples=n_samples,
            workers=workers,
            max_translation_mm=max_translation_mm,
            max_rotation_deg=max_rotation_deg,
            candidate_artifacts=candidate_artifacts,
            interface_artifacts=artifacts,
        ))
    return out


def interface_score(
    part_step: str | Path,
    fixture_dir: str | Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    workers: int = 1,
    max_translation_mm: float | None = None,
    max_rotation_deg: float = DEFAULT_MAX_ROTATION_DEG,
) -> float:
    """Single-number interface-match score in [0, 1] for one candidate.

    Aggregation:

    - **Within a context** (sub-volumes sharing the same ``context_id``):
      take the ``min`` IoU. A composite interface fails if any
      sub-feature fails -- a bracket with a broken boss does not
      mate, regardless of how well its bolt holes match.
    - **Across contexts**: take the ``mean`` of per-context scores.
      Independent interfaces contribute proportionally, so a part
      with one wrong interface among many still earns partial
      credit.

    For a fixture with a single context this collapses to the
    plain min across its pose-searched sub-volumes. For a fixture
    with N independent single-interface contexts this is the mean
    of the N max-IoUs.

    All keyword arguments are forwarded to
    :func:`best_iou_in_context` per context.
    """
    fixture_dir = Path(fixture_dir)
    sub_volumes = discover_sub_volumes(fixture_dir)
    if not sub_volumes:
        raise FileNotFoundError(
            f"No jig_<id>__<index>__<fit>.step files in {fixture_dir}",
        )
    gt_step = fixture_dir / "gt.step"
    if not gt_step.exists():
        raise FileNotFoundError(f"gt.step missing in {fixture_dir}")

    by_context: dict[int, list[SubVolume]] = {}
    for sv in sub_volumes:
        by_context.setdefault(sv.context_id, []).append(sv)

    artifacts = InterfaceMatchArtifacts(gt_step=gt_step, sub_volumes=sub_volumes)
    candidate_artifacts = StepArtifacts(part_step)
    per_context: list[float] = []
    for ctx_svs in by_context.values():
        per_sv = best_iou_in_context(
            part_step,
            ctx_svs,
            gt_step,
            n_samples=n_samples,
            workers=workers,
            max_translation_mm=max_translation_mm,
            max_rotation_deg=max_rotation_deg,
            candidate_artifacts=candidate_artifacts,
            interface_artifacts=artifacts,
        )
        per_context.append(min(per_sv.values()))
    return sum(per_context) / len(per_context)


def _shape_volume(result) -> float:
    """Sum the volumes of a build123d Boolean result.

    Handles both single-Shape and ShapeList results that arise when a
    Boolean produces one or several disconnected solids.
    """
    if result is None:
        return 0.0
    if hasattr(result, "wrapped") and result.wrapped is not None:
        return float(result.volume)
    children = [s for s in result if hasattr(s, "wrapped") and s.wrapped is not None]
    return sum(float(s.volume) for s in children)


def _safe_volume(result) -> float:
    """Volume of a Boolean result, treating null/empty / OCC failures as 0.

    OCC occasionally returns ``TopoDS_Shape`` instances whose underlying
    shape is null (e.g. when a Boolean subtract empties the result).
    Accessing ``.volume`` on those raises; callers want a numeric 0.
    """
    try:
        return _shape_volume(result)
    except Exception:
        return 0.0


def _safe_bool(left, right, op: str):
    """Run a build123d Boolean op, returning None on failure.

    ``op`` is one of ``"intersect"``, ``"subtract"``, ``"fuse"``.
    """
    try:
        if op == "intersect":
            return left & right
        if op == "subtract":
            return left - right
        if op == "fuse":
            return left + right
        raise ValueError(f"Unknown boolean op: {op!r}")
    except Exception:
        return None

