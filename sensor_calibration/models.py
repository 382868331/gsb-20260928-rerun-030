from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three coordinates")
    return array.reshape(3)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    target_id: str
    sensor_a_position: np.ndarray
    sensor_b_position: np.ndarray
    timestamp: str | None = None
    quality: str = "good"
    position_sigma: float | None = None
    # Raw per-stream clock readings for a *moving* target.  They are kept on
    # the original clocks on purpose: alignment never rewrites these values.
    # Convention used everywhere in this package: a physical event at A-clock
    # time t_a is stamped by sensor B as t_b = t_a - time_offset.  A positive
    # offset therefore means clock B runs ahead of clock A.
    t_a: float | None = None
    t_b: float | None = None
    time_sigma: float | None = None
    # Noise-free B sample used to build the true piecewise-linear trajectory
    # for temporal re-interpolation.  Real acquisitions leave this as None;
    # only synthetic checks populate it.  Matching at a shifted time moves
    # along the *clean* chord, while the noisy coordinates are what the rigid
    # residuals are checked against.
    sensor_b_position_clean: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_a_position", _vector3(self.sensor_a_position, "sensor_a_position"))
        object.__setattr__(self, "sensor_b_position", _vector3(self.sensor_b_position, "sensor_b_position"))
        if self.sensor_b_position_clean is not None:
            object.__setattr__(
                self,
                "sensor_b_position_clean",
                _vector3(self.sensor_b_position_clean, "sensor_b_position_clean"),
            )
        if self.quality not in {"good", "bad"}:
            raise ValueError("quality must be 'good' or 'bad'")
        if self.position_sigma is not None and self.position_sigma <= 0:
            raise ValueError("position_sigma must be positive")
        if self.time_sigma is not None and self.time_sigma <= 0:
            raise ValueError("time_sigma must be positive")
        if (self.t_a is None) != (self.t_b is None):
            raise ValueError("t_a and t_b must be provided together")


@dataclass(frozen=True)
class ObservationBatch:
    batch_id: str
    sensor_a_id: str
    sensor_b_id: str
    observations: tuple[Observation, ...] = field(default_factory=tuple)
    time_unit: str = "second"

    _TIME_UNIT_TO_SECOND = {
        "nanosecond": 1e-9,
        "microsecond": 1e-6,
        "millisecond": 1e-3,
        "second": 1.0,
        "minute": 60.0,
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"observation ids in batch {self.batch_id} must be unique")
        if self.time_unit not in self._TIME_UNIT_TO_SECOND:
            raise ValueError(f"unsupported time_unit {self.time_unit!r}")

    @property
    def seconds_per_unit(self) -> float:
        return self._TIME_UNIT_TO_SECOND[self.time_unit]

    def is_timed(self) -> bool:
        return bool(self.observations) and all(item.t_a is not None for item in self.observations)


@dataclass(frozen=True)
class CalibrationInput:
    dataset_id: str
    batches: tuple[ObservationBatch, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "batches", tuple(self.batches))
        batch_ids = [batch.batch_id for batch in self.batches]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("batch ids must be unique")
        sensor_pairs = {(batch.sensor_a_id, batch.sensor_b_id) for batch in self.batches}
        if len(sensor_pairs) > 1:
            raise ValueError("all batches must observe the same ordered sensor pair")


@dataclass(frozen=True)
class CoordinateConvention:
    convention_id: str
    frame_a: str
    frame_b: str
    length_unit: str
    handedness: str = "right"
    description: str = ""

    def __post_init__(self) -> None:
        if self.handedness not in {"left", "right"}:
            raise ValueError("handedness must be 'left' or 'right'")


@dataclass(frozen=True)
class CalibrationVersion:
    method_name: str
    method_version: str
    software_version: str = "0.1.0"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class RigidTransform:
    rotation: np.ndarray
    translation: np.ndarray
    from_frame: str
    to_frame: str

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 matrix")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", _vector3(self.translation, "translation"))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def rotation_angle_deg(self) -> float:
        trace_value = np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(trace_value)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "rotation_angle_deg": self.rotation_angle_deg(),
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
        }


@dataclass(frozen=True)
class CalibrationResult:
    success: bool
    transform: RigidTransform | None
    coordinate_convention: CoordinateConvention
    calibration_version: CalibrationVersion
    used_observations: tuple[tuple[str, str], ...]
    excluded_observations: tuple[tuple[str, str], ...]
    rms_residual: float | None = None
    max_residual: float | None = None
    measurement_sigma: float | None = None
    residuals: Mapping[tuple[str, str], float] = field(default_factory=dict)
    singular_values: np.ndarray | None = None
    reasons: tuple[str, ...] = ()
    status: str = "undetermined"
    distribution: str | None = None
    warnings: tuple[str, ...] = ()
    sampling_recommendations: tuple[str, ...] = ()
    influence: Mapping[tuple[str, str], "ObservationInfluence"] = field(default_factory=dict)
    influential_observations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "used_observations", tuple(self.used_observations))
        object.__setattr__(self, "excluded_observations", tuple(self.excluded_observations))
        object.__setattr__(self, "residuals", dict(self.residuals))
        object.__setattr__(self, "influence", dict(self.influence))
        if self.singular_values is not None:
            object.__setattr__(self, "singular_values", np.asarray(self.singular_values, dtype=float))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "sampling_recommendations", tuple(self.sampling_recommendations))
        object.__setattr__(
            self,
            "influential_observations",
            tuple((batch_id, observation_id) for batch_id, observation_id in self.influential_observations),
        )
        if self.status not in {"undetermined", "unstable", "valid"}:
            raise ValueError("status must be 'undetermined', 'unstable' or 'valid'")

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "distribution": self.distribution,
            "transform": None if self.transform is None else self.transform.as_dict(),
            "coordinate_convention_id": self.coordinate_convention.convention_id,
            "method": f"{self.calibration_version.method_name}@{self.calibration_version.method_version}",
            "used_observations": [list(item) for item in self.used_observations],
            "excluded_observations": [list(item) for item in self.excluded_observations],
            "rms_residual": self.rms_residual,
            "max_residual": self.max_residual,
            "measurement_sigma": self.measurement_sigma,
            "residuals": {"/".join(key): value for key, value in self.residuals.items()},
            "singular_values": None if self.singular_values is None else self.singular_values.tolist(),
            "warnings": list(self.warnings),
            "sampling_recommendations": list(self.sampling_recommendations),
            "influential_observations": [list(item) for item in self.influential_observations],
            "influence": {
                "/".join(key): {
                    "residual": value.residual,
                    "rotation_change": value.rotation_change,
                    "translation_change": value.translation_change,
                    "score": value.score,
                }
                for key, value in self.influence.items()
            },
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ObservationInfluence:
    """Leave-one-out sensitivity of the fitted transform to one observation."""

    residual: float
    rotation_change: float | None
    translation_change: float | None
    score: float | None


@dataclass(frozen=True)
class SubsetComparison:
    """One calibrated observation subset compared with the full-selection fit."""

    label: str
    result: CalibrationResult
    additional_excluded_observations: tuple[tuple[str, str], ...]
    rotation_change: float | None
    translation_change: float | None
    judgment_basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchCompatibility:
    """How one batch relates to the joint multi-batch solution."""

    batch_id: str
    relation: str
    used_observation_count: int
    standalone_status: str
    standalone_distribution: str | None
    rotation_change: float | None = None
    translation_change: float | None = None
    joint_rms_residual_on_batch: float | None = None
    residual_sigma_ratio: float | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.relation not in {
            "supports_joint_solution",
            "conflicts_with_joint_solution",
            "indeterminate_alone_covered_by_joint",
            "insufficient_observations",
        }:
            raise ValueError(f"unknown batch relation {self.relation!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "relation": self.relation,
            "used_observation_count": self.used_observation_count,
            "standalone_status": self.standalone_status,
            "standalone_distribution": self.standalone_distribution,
            "rotation_change": self.rotation_change,
            "translation_change": self.translation_change,
            "joint_rms_residual_on_batch": self.joint_rms_residual_on_batch,
            "residual_sigma_ratio": self.residual_sigma_ratio,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class JointCalibrationResult:
    """Joint fit across batches plus per-batch and traceability details."""

    joint: CalibrationResult
    per_batch: Mapping[str, CalibrationResult]
    compatibility: tuple[BatchCompatibility, ...]
    observation_weights: Mapping[tuple[str, str], float]
    duplicate_groups: tuple[tuple[str, ...], ...] = ()
    excluded_batches: tuple[str, ...] = ()
    identifiability_notes: tuple[str, ...] = ()
    effective_batch_count: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_batch", dict(self.per_batch))
        object.__setattr__(self, "compatibility", tuple(self.compatibility))
        object.__setattr__(self, "observation_weights", dict(self.observation_weights))
        object.__setattr__(
            self, "duplicate_groups", tuple(tuple(group) for group in self.duplicate_groups)
        )
        object.__setattr__(self, "excluded_batches", tuple(self.excluded_batches))
        object.__setattr__(self, "identifiability_notes", tuple(self.identifiability_notes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint": self.joint.as_dict(),
            "per_batch": {batch_id: result.as_dict() for batch_id, result in self.per_batch.items()},
            "compatibility": [item.as_dict() for item in self.compatibility],
            "observation_weights": {
                "/".join(key): value for key, value in self.observation_weights.items()
            },
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "excluded_batches": list(self.excluded_batches),
            "identifiability_notes": list(self.identifiability_notes),
            "effective_batch_count": self.effective_batch_count,
        }


@dataclass(frozen=True)
class JointScenario:
    """A named what-if selection: whole batches and/or single correspondences
    temporarily excluded without modifying the source acquisitions."""

    excluded_batches: frozenset[str] = frozenset()
    excluded_observations: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_batches", frozenset(self.excluded_batches))
        object.__setattr__(self, "excluded_observations", frozenset(self.excluded_observations))


@dataclass(frozen=True)
class JointScenarioComparison:
    """A saved comparison scenario evaluated against the full joint fit."""

    label: str
    scenario: JointScenario
    result: JointCalibrationResult
    rotation_change: float | None
    translation_change: float | None
    judgment_basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "judgment_basis", tuple(self.judgment_basis))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "scenario": {
                "excluded_batches": sorted(self.scenario.excluded_batches),
                "excluded_observations": [
                    list(item) for item in sorted(self.scenario.excluded_observations)
                ],
            },
            "result": self.result.as_dict(),
            "rotation_change": self.rotation_change,
            "translation_change": self.translation_change,
            "judgment_basis": list(self.judgment_basis),
        }


@dataclass(frozen=True)
class TimeInterval:
    """A contiguous usable interval, expressed in one clock's raw units."""

    start: float
    end: float
    clock: str
    observation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end < self.start:
            object.__setattr__(self, "end", self.start)
            object.__setattr__(self, "start", self.end)
        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "clock": self.clock,
            "observation_ids": list(self.observation_ids),
        }


@dataclass(frozen=True)
class BatchTimeSelection:
    """Which observations of one batch actually participate in the check and
    which raw-clock intervals back them.  Intervals are stored in the batch's
    own time_unit so the raw acquisition remains auditable."""

    batch_id: str
    time_unit: str
    used_a_observations: tuple[str, ...]
    used_intervals_a_clock: tuple[TimeInterval, ...]
    used_intervals_b_clock: tuple[TimeInterval, ...]
    gaps_b_clock: tuple[tuple[float, float], ...] = ()
    missing_windows_b_clock: tuple[tuple[float, float], ...] = ()
    reset_overlaps_b_clock: tuple[tuple[float, float], ...] = ()
    omitted_a_observations: tuple[str, ...] = ()
    omitted_reasons: Mapping[str, str] = field(default_factory=dict)
    feasible_offset_range: tuple[float, float] | None = None
    paired_count: int = 0
    ambiguous_pairings: tuple["PairingCandidate", ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "used_a_observations", tuple(self.used_a_observations))
        object.__setattr__(self, "omitted_a_observations", tuple(self.omitted_a_observations))
        object.__setattr__(self, "omitted_reasons", dict(self.omitted_reasons))
        object.__setattr__(
            self, "used_intervals_a_clock", tuple(self.used_intervals_a_clock)
        )
        object.__setattr__(
            self, "used_intervals_b_clock", tuple(self.used_intervals_b_clock)
        )
        object.__setattr__(self, "gaps_b_clock", tuple(tuple(gap) for gap in self.gaps_b_clock))
        object.__setattr__(
            self,
            "missing_windows_b_clock",
            tuple(tuple(window) for window in self.missing_windows_b_clock),
        )
        object.__setattr__(
            self,
            "reset_overlaps_b_clock",
            tuple(tuple(window) for window in self.reset_overlaps_b_clock),
        )
        object.__setattr__(self, "ambiguous_pairings", tuple(self.ambiguous_pairings))

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "time_unit": self.time_unit,
            "used_a_observations": list(self.used_a_observations),
            "used_intervals_a_clock": [item.as_dict() for item in self.used_intervals_a_clock],
            "used_intervals_b_clock": [item.as_dict() for item in self.used_intervals_b_clock],
            "gaps_b_clock": [list(gap) for gap in self.gaps_b_clock],
            "missing_windows_b_clock": [list(window) for window in self.missing_windows_b_clock],
            "reset_overlaps_b_clock": [list(window) for window in self.reset_overlaps_b_clock],
            "omitted_a_observations": list(self.omitted_a_observations),
            "omitted_reasons": dict(self.omitted_reasons),
            "feasible_offset_range_seconds": (
                None if self.feasible_offset_range is None else list(self.feasible_offset_range)
            ),
            "paired_count": self.paired_count,
            "ambiguous_pairings": [item.as_dict() for item in self.ambiguous_pairings],
        }


@dataclass(frozen=True)
class PairingCandidate:
    """One diagnostic B segment that can supply a rejected pairing."""

    observation_id: str
    segment_index: int
    b_time_start: float
    b_time_end: float
    selected: bool = False
    reason: str = "candidate_not_confirmed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "segment_index": self.segment_index,
            "b_time_start": self.b_time_start,
            "b_time_end": self.b_time_end,
            "selected_by_numerical_search": self.selected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TimeOffsetEstimate:
    """Estimated fixed clock offset and what the data actually identifies.

    Sign convention: t_b = t_a - offset_seconds.
    status:
      identifiable                  - curvature/acceleration or non-parallel
                                     batch motions pin the offset;
      confounded_with_motion       - straight constant motion: a whole family
                                     of (offset, translation) pairs fits;
      unobservable_target_static    - target does not move on this data, so the
                                     offset leaves no trace in residuals;
      not_estimated                - mode does not estimate an offset.
    """

    offset_seconds: float | None
    status: str
    sigma_seconds: float | None = None
    valid: bool = True
    candidate_offset_seconds: float | None = None
    candidate_sigma_seconds: float | None = None
    feasible_offset_range_seconds: tuple[float, float] | None = None
    ambiguity_direction: np.ndarray | None = None
    equivalent_translation_per_second: np.ndarray | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {
            "identifiable",
            "confounded_with_motion",
            "unobservable_target_static",
            "not_estimated",
            "fixed_offset_provided",
            "ambiguous_pairing_sources",
        }:
            raise ValueError(f"unknown time offset status {self.status!r}")
        if self.feasible_offset_range_seconds is not None:
            object.__setattr__(
                self,
                "feasible_offset_range_seconds",
                tuple(self.feasible_offset_range_seconds),
            )
        if self.ambiguity_direction is not None:
            object.__setattr__(
                self, "ambiguity_direction",
                np.asarray(self.ambiguity_direction, dtype=float).reshape(3),
            )
        if self.equivalent_translation_per_second is not None:
            object.__setattr__(
                self,
                "equivalent_translation_per_second",
                np.asarray(self.equivalent_translation_per_second, dtype=float).reshape(3),
            )
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset_seconds": self.offset_seconds,
            "status": self.status,
            "sigma_seconds": self.sigma_seconds,
            "valid": self.valid,
            "candidate_offset_seconds": self.candidate_offset_seconds,
            "candidate_sigma_seconds": self.candidate_sigma_seconds,
            "feasible_offset_range_seconds": (
                None
                if self.feasible_offset_range_seconds is None
                else list(self.feasible_offset_range_seconds)
            ),
            "ambiguity_direction": (
                None if self.ambiguity_direction is None else self.ambiguity_direction.tolist()
            ),
            "equivalent_translation_per_second": (
                None
                if self.equivalent_translation_per_second is None
                else self.equivalent_translation_per_second.tolist()
            ),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SpatiotemporalResult:
    """Joint rigid-relation / fixed clock-offset calibration result."""

    success: bool
    mode: str
    transform: RigidTransform | None
    time_offset: TimeOffsetEstimate
    coordinate_convention: CoordinateConvention
    calibration_version: CalibrationVersion
    selection: tuple[BatchTimeSelection, ...]
    used_observations: tuple[tuple[str, str], ...]
    excluded_observations: tuple[tuple[str, str], ...]
    rms_residual: float | None = None
    measurement_sigma: float | None = None
    spatial_distribution: str | None = None
    spatial_status: str = "undetermined"
    spatial_valid: bool = False
    identifiability_notes: tuple[str, ...] = ()
    duplicate_groups: tuple[tuple[str, ...], ...] = ()
    effective_batch_count: float = 0.0
    reasons: tuple[str, ...] = ()
    candidate_transform: RigidTransform | None = None
    candidate_rms_residual: float | None = None
    candidate_used_observations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection", tuple(self.selection))
        object.__setattr__(self, "used_observations", tuple(self.used_observations))
        object.__setattr__(self, "excluded_observations", tuple(self.excluded_observations))
        object.__setattr__(self, "identifiability_notes", tuple(self.identifiability_notes))
        object.__setattr__(self, "duplicate_groups", tuple(tuple(g) for g in self.duplicate_groups))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(
            self,
            "candidate_used_observations",
            tuple(self.candidate_used_observations),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "mode": self.mode,
            "transform": None if self.transform is None else self.transform.as_dict(),
            "time_offset": self.time_offset.as_dict(),
            "coordinate_convention_id": self.coordinate_convention.convention_id,
            "method": f"{self.calibration_version.method_name}@{self.calibration_version.method_version}",
            "selection": [item.as_dict() for item in self.selection],
            "used_observations": [list(item) for item in self.used_observations],
            "excluded_observations": [list(item) for item in self.excluded_observations],
            "rms_residual": self.rms_residual,
            "measurement_sigma": self.measurement_sigma,
            "spatial_distribution": self.spatial_distribution,
            "spatial_status": self.spatial_status,
            "spatial_valid": self.spatial_valid,
            "identifiability_notes": list(self.identifiability_notes),
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "effective_batch_count": self.effective_batch_count,
            "reasons": list(self.reasons),
            "candidate_transform": (
                None if self.candidate_transform is None else self.candidate_transform.as_dict()
            ),
            "candidate_rms_residual": self.candidate_rms_residual,
            "candidate_used_observations": [
                list(item) for item in self.candidate_used_observations
            ],
        }


@dataclass(frozen=True)
class SpatiotemporalScenario:
    """A what-if selection for spatiotemporal comparison: exclude whole
    batches or only part of a batch's time span (raw A-clock units)."""

    excluded_batches: frozenset[str] = frozenset()
    excluded_observations: frozenset[tuple[str, str]] = frozenset()
    excluded_a_windows: Mapping[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_batches", frozenset(self.excluded_batches))
        object.__setattr__(self, "excluded_observations", frozenset(self.excluded_observations))
        windows = {
            batch_id: tuple(tuple(window) for window in windows)
            for batch_id, windows in self.excluded_a_windows.items()
        }
        object.__setattr__(self, "excluded_a_windows", windows)


@dataclass(frozen=True)
class SpatiotemporalComparison:
    """One compared way of treating time (fixed offset / joint estimate /
    correspondable-only), or one exclusion scenario."""

    label: str
    mode: str
    result: SpatiotemporalResult
    offset_change_seconds: float | None
    rotation_change: float | None
    translation_change: float | None
    judgment_basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "judgment_basis", tuple(self.judgment_basis))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": self.mode,
            "result": self.result.as_dict(),
            "offset_change_seconds": self.offset_change_seconds,
            "rotation_change": self.rotation_change,
            "translation_change": self.translation_change,
            "judgment_basis": list(self.judgment_basis),
        }
