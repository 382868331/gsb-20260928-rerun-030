from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np


def _vector3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three coordinates")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite coordinate")
    return tuple(float(x) for x in array)


def stable_hash(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-composable mapping."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CoordinateConvention:
    convention_id: str
    frame_a: str
    frame_b: str
    length_unit: str
    handedness: str = "right"
    description: str = ""

    def canonical(self) -> dict[str, Any]:
        return {
            "convention_id": self.convention_id,
            "frame_a": self.frame_a,
            "frame_b": self.frame_b,
            "length_unit": self.length_unit,
            "handedness": self.handedness,
            "description": self.description,
        }

    @property
    def hash(self) -> str:
        return stable_hash(self.canonical())


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    target_id: str
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]
    timestamp: str = ""
    sigma: float = 0.0
    is_bad: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must not be empty")
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        object.__setattr__(self, "point_a", _vector3(self.point_a, "point_a"))
        object.__setattr__(self, "point_b", _vector3(self.point_b, "point_b"))

    def canonical(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "target_id": self.target_id,
            "point_a": list(self.point_a),
            "point_b": list(self.point_b),
            "timestamp": self.timestamp,
            "sigma": self.sigma,
            "is_bad": self.is_bad,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    batch_id: str
    sensor_a_id: str
    sensor_b_id: str
    convention_id: str
    observations: tuple[Observation, ...]
    collected_at: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must not be empty")
        ids = [obs.observation_id for obs in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observation IDs must be unique within a batch")
        object.__setattr__(self, "observations", tuple(self.observations))

    def canonical(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "sensor_a_id": self.sensor_a_id,
            "sensor_b_id": self.sensor_b_id,
            "convention_id": self.convention_id,
            "collected_at": self.collected_at,
            "description": self.description,
            "observations": [obs.canonical() for obs in self.observations],
        }

    @property
    def hash(self) -> str:
        return stable_hash(self.canonical())


@dataclass(frozen=True, slots=True)
class CalibrationVersion:
    """Immutable provenance record for one calibration calculation."""

    version_id: str
    batch_id: str
    batch_hash: str
    convention_hash: str
    included_observation_ids: tuple[str, ...]
    excluded_observation_ids: tuple[str, ...]
    method: str = "weighted-kabsch-svd"
    parent_version_id: str | None = None
    created_at: str = ""
    notes: str = ""

    @classmethod
    def create(
        cls,
        version_id: str,
        batch: ObservationBatch,
        convention: CoordinateConvention,
        included_observation_ids: Sequence[str],
        excluded_observation_ids: Sequence[str],
        parent_version_id: str | None = None,
        notes: str = "",
        created_at: str | None = None,
    ) -> "CalibrationVersion":
        if convention.convention_id != batch.convention_id:
            raise ValueError("batch and coordinate convention IDs do not match")
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        return cls(
            version_id=version_id,
            batch_id=batch.batch_id,
            batch_hash=batch.hash,
            convention_hash=convention.hash,
            included_observation_ids=tuple(included_observation_ids),
            excluded_observation_ids=tuple(excluded_observation_ids),
            parent_version_id=parent_version_id,
            created_at=timestamp,
            notes=notes,
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "batch_id": self.batch_id,
            "batch_hash": self.batch_hash,
            "convention_hash": self.convention_hash,
            "included_observation_ids": list(self.included_observation_ids),
            "excluded_observation_ids": list(self.excluded_observation_ids),
            "method": self.method,
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @property
    def hash(self) -> str:
        return stable_hash(self.canonical())
