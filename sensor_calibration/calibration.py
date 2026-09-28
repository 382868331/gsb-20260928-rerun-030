from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import (
    CalibrationInput,
    CalibrationResult,
    CalibrationVersion,
    CoordinateConvention,
    RigidTransform,
)


@dataclass(frozen=True)
class CalibrationConfig:
    min_points: int = 3
    condition_limit: float = 1e8
    motion_sigma_factor: float = 8.0
    exclude_marked_bad: bool = True
    additional_excluded_observations: frozenset[tuple[str, str]] = field(default_factory=frozenset)


def calibrate(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    config = config or CalibrationConfig()
    selected: list[tuple[str, str, np.ndarray, np.ndarray, float | None]] = []
    excluded: list[tuple[str, str]] = []

    for batch in calibration_input.batches:
        for observation in batch.observations:
            reference = (batch.batch_id, observation.observation_id)
            explicitly_excluded = reference in config.additional_excluded_observations
            marked_bad = config.exclude_marked_bad and observation.quality == "bad"
            if explicitly_excluded or marked_bad:
                excluded.append(reference)
            else:
                selected.append((
                    batch.batch_id,
                    observation.observation_id,
                    observation.sensor_a_position,
                    observation.sensor_b_position,
                    observation.position_sigma,
                ))

    used_references = tuple((batch_id, observation_id) for batch_id, observation_id, *_ in selected)
    if len(selected) < config.min_points:
        return CalibrationResult(
            success=False,
            transform=None,
            coordinate_convention=convention,
            calibration_version=version,
            used_observations=used_references,
            excluded_observations=tuple(excluded),
            reasons=(f"at_least_{config.min_points}_non_excluded_observations_required",),
        )

    points_a = np.vstack([item[2] for item in selected])
    points_b = np.vstack([item[3] for item in selected])
    centroid_a = points_a.mean(axis=0)
    centroid_b = points_b.mean(axis=0)
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b

    covariance = centered_a.T @ centered_b / len(selected)
    u, singular_values, vt = np.linalg.svd(covariance)
    reasons: list[str] = []

    if singular_values[1] <= np.finfo(float).eps * max(singular_values[0], 1.0):
        reasons.append("target_positions_do_not_span_three_dimensions")

    nonzero = singular_values[singular_values > 0]
    condition_number = float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf")
    if not np.isfinite(condition_number) or condition_number > config.condition_limit:
        reasons.append("target_motion_is_nearly_degenerate")

    sigmas = [item[4] for item in selected if item[4] is not None]
    if sigmas:
        uncertainty_scale = max(sigmas) / np.sqrt(len(selected))
        if singular_values[-1] < config.motion_sigma_factor * uncertainty_scale:
            reasons.append("target_motion_is_small_relative_to_measurement_uncertainty")

    if reasons:
        return CalibrationResult(
            success=False,
            transform=None,
            coordinate_convention=convention,
            calibration_version=version,
            used_observations=used_references,
            excluded_observations=tuple(excluded),
            singular_values=singular_values,
            reasons=tuple(reasons),
        )

    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    if np.linalg.det(rotation) <= 0:
        return CalibrationResult(
            success=False,
            transform=None,
            coordinate_convention=convention,
            calibration_version=version,
            used_observations=used_references,
            excluded_observations=tuple(excluded),
            singular_values=singular_values,
            reasons=("only_a_reflection_can_fit_the_correspondences",),
        )

    translation = centroid_b - rotation @ centroid_a
    transform = RigidTransform(rotation, translation, convention.frame_a, convention.frame_b)
    residual_values = np.linalg.norm(transform.apply(points_a) - points_b, axis=1)
    residuals = {
        (batch_id, observation_id): float(value)
        for (batch_id, observation_id, *_), value in zip(selected, residual_values)
    }

    return CalibrationResult(
        success=True,
        transform=transform,
        coordinate_convention=convention,
        calibration_version=version,
        used_observations=used_references,
        excluded_observations=tuple(excluded),
        rms_residual=float(np.sqrt(np.mean(residual_values**2))),
        max_residual=float(np.max(residual_values)),
        residuals=residuals,
        singular_values=singular_values,
    )
