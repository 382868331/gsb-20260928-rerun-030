from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .models import CalibrationVersion, CoordinateConvention, Observation, ObservationBatch
from .transform import PoseTransform


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    version: CalibrationVersion
    determined: bool
    transform: PoseTransform | None
    rank: int
    singular_values: tuple[float, float, float]
    rms_error: float | None
    max_error: float | None
    per_observation_rms: tuple[float, ...]
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sufficient(self) -> bool:
        return self.determined


def _arrays(
    observations: Sequence[Observation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_a = np.asarray([obs.point_a for obs in observations], dtype=float)
    points_b = np.asarray([obs.point_b for obs in observations], dtype=float)
    sigmas = np.asarray([max(obs.sigma, 1e-12) for obs in observations], dtype=float)
    return points_a, points_b, sigmas


def _weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(points * weights[:, None], axis=0) / np.sum(weights)


def _rank_from_singular_values(values: np.ndarray, noise_scale: float) -> int:
    if values.size == 0 or not math.isfinite(noise_scale) or noise_scale <= 0:
        return int(np.sum(values > 1e-10))
    threshold = 3.0 * noise_scale
    return int(np.sum(values > threshold))


def estimate_transform(
    observations: Sequence[Observation],
    noise_scale: float = 0.0,
) -> tuple[PoseTransform | None, int, np.ndarray, tuple[str, ...]]:
    """Estimate ``B = R A + t`` using the weighted Kabsch/SVD solution."""

    if not observations:
        return None, 0, np.zeros(3), ("no observations were included",)

    points_a, points_b, sigmas = _arrays(observations)
    if len(points_a) < 3:
        singular_values = np.linalg.svd(points_a - points_a.mean(axis=0), compute_uv=False)
        rank = _rank_from_singular_values(singular_values, noise_scale)
        return None, rank, singular_values, (
            "at least three paired observations are required",
        )

    inverse_variance = 1.0 / np.square(sigmas)
    weights = inverse_variance / np.sum(inverse_variance)
    centroid_a = _weighted_centroid(points_a, weights)
    centroid_b = _weighted_centroid(points_b, weights)
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b

    singular_values = np.linalg.svd(centered_a, compute_uv=False)
    rank = _rank_from_singular_values(singular_values, noise_scale)
    if rank < 3:
        return None, rank, singular_values, (
            "target motion must span three independent directions for a unique 3-D pose",
        )

    covariance = (centered_a * weights[:, None]).T @ centered_b
    u, _, vt = np.linalg.svd(covariance)
    determinant_correction = np.eye(3)
    determinant_correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ determinant_correction @ u.T
    translation = centroid_b - rotation @ centroid_a
    return PoseTransform(rotation=rotation, translation=translation), rank, singular_values, ()


def calibrate(
    version_id: str,
    batch: ObservationBatch,
    convention: CoordinateConvention,
    parent_version_id: str | None = None,
    noise_scale: float = 0.0,
    extra_excluded_ids: Sequence[str] | None = None,
    notes: str = "",
) -> CalibrationResult:
    """Recalculate while leaving the source observation batch unchanged."""

    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")

    by_id = {obs.observation_id: obs for obs in batch.observations}
    requested_exclusions = set(extra_excluded_ids or ())
    unknown = requested_exclusions.difference(by_id)
    if unknown:
        raise KeyError(f"unknown observation IDs: {sorted(unknown)}")

    excluded = [obs for obs in batch.observations if obs.is_bad or obs.observation_id in requested_exclusions]
    included = [obs for obs in batch.observations if obs not in excluded]
    version = CalibrationVersion.create(
        version_id=version_id,
        batch=batch,
        convention=convention,
        included_observation_ids=[obs.observation_id for obs in included],
        excluded_observation_ids=[obs.observation_id for obs in excluded],
        parent_version_id=parent_version_id,
        notes=notes,
    )

    transform, rank, singular_values, reasons = estimate_transform(included, noise_scale)
    rms_error = max_error = None
    per_observation_rms: tuple[float, ...] = ()
    determined = transform is not None
    if transform is not None:
        errors = transform.apply(np.asarray([obs.point_a for obs in included])) - np.asarray(
            [obs.point_b for obs in included]
        )
        distances = np.linalg.norm(errors, axis=1)
        per_observation_rms = tuple(float(x) for x in distances)
        rms_error = float(np.sqrt(np.mean(np.square(distances))))
        max_error = float(np.max(distances))

    return CalibrationResult(
        version=version,
        determined=determined,
        transform=transform,
        rank=rank,
        singular_values=tuple(float(x) for x in singular_values),
        rms_error=rms_error,
        max_error=max_error,
        per_observation_rms=per_observation_rms,
        reasons=tuple(reasons),
    )
