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

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_a_position", _vector3(self.sensor_a_position, "sensor_a_position"))
        object.__setattr__(self, "sensor_b_position", _vector3(self.sensor_b_position, "sensor_b_position"))
        if self.quality not in {"good", "bad"}:
            raise ValueError("quality must be 'good' or 'bad'")
        if self.position_sigma is not None and self.position_sigma <= 0:
            raise ValueError("position_sigma must be positive")


@dataclass(frozen=True)
class ObservationBatch:
    batch_id: str
    sensor_a_id: str
    sensor_b_id: str
    observations: tuple[Observation, ...] = field(default_factory=tuple)
    # Multiplier converting this batch's recorded length unit into the
    # coordinate convention unit (e.g. 0.001 for millimetres vs metres).
    length_scale_to_convention: float = 1.0
    # Free-form label of the source unit; the numeric scale is what is used.
    length_unit: str = "convention"
    # Observations sharing an acquisition id are repeated copies of the same
    # physical acquisition; they must not count as independent evidence.
    acquisition_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"observation ids in batch {self.batch_id} must be unique")
        if self.length_scale_to_convention <= 0:
            raise ValueError("length_scale_to_convention must be positive")


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
    # Effective measurement noise in convention units after unit conversion.
    effective_measurement_sigma: float | None = None
    # Per-batch independent fits and their agreement with the joint fit.
    batch_evaluations: Mapping[str, "BatchEvaluation"] = field(default_factory=dict)
    # Exhaustive ledger of what was done with every recorded observation.
    observation_decisions: Mapping[tuple[str, str], "ObservationDecision"] = field(default_factory=dict)
    # Repeated-acquisition groups and how much independent evidence they carry.
    acquisition_evidence: Mapping[str, "AcquisitionEvidence"] = field(default_factory=dict)

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
        object.__setattr__(self, "batch_evaluations", dict(self.batch_evaluations))
        object.__setattr__(self, "observation_decisions", dict(self.observation_decisions))
        object.__setattr__(self, "acquisition_evidence", dict(self.acquisition_evidence))
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
            "effective_measurement_sigma": self.effective_measurement_sigma,
            "batch_evaluations": {
                key: value.as_dict() for key, value in self.batch_evaluations.items()
            },
            "observation_decisions": {
                "/".join(key): value.as_dict() for key, value in self.observation_decisions.items()
            },
            "acquisition_evidence": {
                key: value.as_dict() for key, value in self.acquisition_evidence.items()
            },
        }


@dataclass(frozen=True)
class ObservationInfluence:
    """Leave-one-out sensitivity of the fitted transform to one observation."""

    residual: float
    rotation_change: float | None
    translation_change: float | None
    score: float | None


@dataclass(frozen=True)
class ObservationDecision:
    """Traceable adoption decision for one recorded observation."""

    batch_id: str
    observation_id: str
    acquisition_id: str
    adopted: bool
    decision: str  # used | excluded_marked_bad | excluded_manually | excluded_batch
    # Effective independent-evidence weight after duplicate collapsing (0..1).
    evidence_weight: float
    # Multiplicity of identical recorded observations sharing the acquisition.
    duplicate_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "observation_id": self.observation_id,
            "acquisition_id": self.acquisition_id,
            "adopted": self.adopted,
            "decision": self.decision,
            "evidence_weight": self.evidence_weight,
            "duplicate_count": self.duplicate_count,
        }


@dataclass(frozen=True)
class AcquisitionEvidence:
    """How a (possibly repeated) physical acquisition entered the joint fit."""

    acquisition_id: str
    member_references: tuple[tuple[str, str], ...]
    adopted_member_references: tuple[tuple[str, str], ...]
    multiplicity: int
    independent_weight: float
    repeated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "acquisition_id": self.acquisition_id,
            "member_references": [list(item) for item in self.member_references],
            "adopted_member_references": [list(item) for item in self.adopted_member_references],
            "multiplicity": self.multiplicity,
            "independent_weight": self.independent_weight,
            "repeated": self.repeated,
        }


@dataclass(frozen=True)
class BatchEvaluation:
    """Independent per-batch calibration result compared with the joint fit."""

    batch_id: str
    observation_count: int
    used_observation_count: int
    status: str  # undetermined | unstable | valid | excluded
    distribution: str | None
    rms_residual: float | None
    # Mean and spread of joint-fit residuals within this batch (convention units).
    joint_residual_mean: float | None
    joint_residual_rms: float | None
    # Norm of mean joint-fit residual vector: a rigid bias not explainable by
    # the common transform.
    joint_bias_norm: float | None
    # Bias significance relative to the batch's own converted uncertainty.
    bias_sigma_ratio: float | None
    rotation_change: float | None
    translation_change: float | None
    supports_joint_transform: bool | None
    excluded: bool = False
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "observation_count": self.observation_count,
            "used_observation_count": self.used_observation_count,
            "status": self.status,
            "distribution": self.distribution,
            "rms_residual": self.rms_residual,
            "joint_residual_mean": self.joint_residual_mean,
            "joint_residual_rms": self.joint_residual_rms,
            "joint_bias_norm": self.joint_bias_norm,
            "bias_sigma_ratio": self.bias_sigma_ratio,
            "rotation_change": self.rotation_change,
            "translation_change": self.translation_change,
            "supports_joint_transform": self.supports_joint_transform,
            "excluded": self.excluded,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ComparisonPlan:
    """A saved comparison scheme: exclusions plus named observation subsets."""

    plan_id: str
    dataset_id: str
    convention_id: str
    method: str
    excluded_batches: tuple[str, ...] = ()
    excluded_observations: tuple[tuple[str, str], ...] = ()
    subsets: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    config_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_batches", tuple(self.excluded_batches))
        object.__setattr__(self, "excluded_observations", tuple(self.excluded_observations))
        object.__setattr__(
            self,
            "subsets",
            {label: tuple(refs) for label, refs in self.subsets.items()},
        )
        object.__setattr__(self, "config_parameters", dict(self.config_parameters))

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "dataset_id": self.dataset_id,
            "convention_id": self.convention_id,
            "method": self.method,
            "excluded_batches": list(self.excluded_batches),
            "excluded_observations": [list(item) for item in self.excluded_observations],
            "subsets": {
                label: [list(item) for item in refs]
                for label, refs in self.subsets.items()
            },
            "config_parameters": dict(self.config_parameters),
        }


@dataclass(frozen=True)
class SubsetComparison:
    """One calibrated observation subset compared with the full-selection fit."""

    label: str
    result: CalibrationResult
    additional_excluded_observations: tuple[tuple[str, str], ...]
    rotation_change: float | None
    translation_change: float | None
    judgment_basis: tuple[str, ...] = ()
