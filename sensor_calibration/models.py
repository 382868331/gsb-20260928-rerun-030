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

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError(f"observation ids in batch {self.batch_id} must be unique")


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
    status: str = "valid"
    rms_residual: float | None = None
    max_residual: float | None = None
    residuals: Mapping[tuple[str, str], float] = field(default_factory=dict)
    singular_values: np.ndarray | None = None
    geometric_rank: int | None = None
    effective_rank: int | None = None
    motion_scale: float | None = None
    noise_scale: float | None = None
    condition_number: float | None = None
    influential_observations: tuple[tuple[str, str], ...] = ()
    influence_scores: Mapping[tuple[str, str], float] = field(default_factory=dict)
    warning_observations: tuple[tuple[str, str], ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "used_observations", tuple(self.used_observations))
        object.__setattr__(self, "excluded_observations", tuple(self.excluded_observations))
        object.__setattr__(self, "residuals", dict(self.residuals))
        object.__setattr__(self, "influential_observations", tuple(self.influential_observations))
        object.__setattr__(self, "influence_scores", dict(self.influence_scores))
        object.__setattr__(self, "warning_observations", tuple(self.warning_observations))
        if self.singular_values is not None:
            object.__setattr__(self, "singular_values", np.asarray(self.singular_values, dtype=float))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        if self.status not in {"valid", "unstable", "undetermined"}:
            raise ValueError("status must be 'valid', 'unstable' or 'undetermined'")
        if self.status == "undetermined" and self.success:
            raise ValueError("undetermined results cannot report success")
        if self.status != "undetermined" and not self.success:
            raise ValueError("only undetermined results may report failure")

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "transform": None if self.transform is None else self.transform.as_dict(),
            "coordinate_convention_id": self.coordinate_convention.convention_id,
            "method": f"{self.calibration_version.method_name}@{self.calibration_version.method_version}",
            "used_observations": [list(item) for item in self.used_observations],
            "excluded_observations": [list(item) for item in self.excluded_observations],
            "rms_residual": self.rms_residual,
            "max_residual": self.max_residual,
            "singular_values": None if self.singular_values is None else self.singular_values.tolist(),
            "geometric_rank": self.geometric_rank,
            "effective_rank": self.effective_rank,
            "motion_scale": self.motion_scale,
            "noise_scale": self.noise_scale,
            "condition_number": self.condition_number,
            "influential_observations": [list(item) for item in self.influential_observations],
            "influence_scores": {
                f"{reference[0]}|{reference[1]}": score
                for reference, score in self.influence_scores.items()
            },
            "warning_observations": [list(item) for item in self.warning_observations],
            "reasons": list(self.reasons),
        }
