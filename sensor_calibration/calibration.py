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
    # A centered direction counts as resolved when its RMS spread exceeds
    # motion_sigma_factor * sigma / sqrt(n).
    motion_sigma_factor: float = 8.0
    # Geometry is treated as exact (point/line/plane) below this relative floor.
    geometric_rank_tol: float = 1e-10
    # A successful fit with residual/LOO diagnostics above these factors is
    # reported as "unstable" instead of being silently trusted.
    instability_residual_factor: float = 5.0
    influence_rotation_limit_deg: float = 1.0
    influence_translation_fraction: float = 0.05
    outlier_residual_factor: float = 4.0
    leave_one_out: bool = True
    exclude_marked_bad: bool = True
    additional_excluded_observations: frozenset[tuple[str, str]] = field(default_factory=frozenset)


@dataclass(frozen=True)
class _Sample:
    reference: tuple[str, str]
    point_a: np.ndarray
    point_b: np.ndarray
    sigma: float | None


def _collect_samples(
    calibration_input: CalibrationInput,
    config: CalibrationConfig,
) -> tuple[list[_Sample], list[tuple[str, str]]]:
    selected: list[_Sample] = []
    excluded: list[tuple[str, str]] = []
    for batch in calibration_input.batches:
        for observation in batch.observations:
            reference = (batch.batch_id, observation.observation_id)
            explicitly_excluded = reference in config.additional_excluded_observations
            marked_bad = config.exclude_marked_bad and observation.quality == "bad"
            if explicitly_excluded or marked_bad:
                excluded.append(reference)
            else:
                selected.append(
                    _Sample(
                        reference=reference,
                        point_a=observation.sensor_a_position,
                        point_b=observation.sensor_b_position,
                        sigma=observation.position_sigma,
                    )
                )
    return selected, excluded


def _kabsch(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    centroid_a = points_a.mean(axis=0)
    centroid_b = points_b.mean(axis=0)
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b
    covariance = centered_a.T @ centered_b / len(points_a)
    u, _singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    if np.linalg.det(rotation) <= 0:
        return None
    translation = centroid_b - rotation @ centroid_a
    return rotation, translation, centroid_a, centroid_b


def _rotation_delta_deg(reference_rotation: np.ndarray, candidate_rotation: np.ndarray) -> float:
    relative = reference_rotation @ candidate_rotation.T
    trace_value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_value)))


def _undetermined(
    *,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    samples: list[_Sample],
    excluded: list[tuple[str, str]],
    reasons: tuple[str, ...],
    singular_values: np.ndarray | None = None,
    geometric_rank: int | None = None,
    effective_rank: int | None = None,
    motion_scale: float | None = None,
    noise_scale: float | None = None,
    condition_number: float | None = None,
) -> CalibrationResult:
    return CalibrationResult(
        success=False,
        transform=None,
        coordinate_convention=convention,
        calibration_version=version,
        used_observations=tuple(sample.reference for sample in samples),
        excluded_observations=tuple(excluded),
        status="undetermined",
        singular_values=singular_values,
        geometric_rank=geometric_rank,
        effective_rank=effective_rank,
        motion_scale=motion_scale,
        noise_scale=noise_scale,
        condition_number=condition_number,
        reasons=reasons,
    )


def calibrate(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    config = config or CalibrationConfig()
    samples, excluded = _collect_samples(calibration_input, config)
    used_references = tuple(sample.reference for sample in samples)

    if len(samples) < config.min_points:
        return CalibrationResult(
            success=False,
            transform=None,
            coordinate_convention=convention,
            calibration_version=version,
            used_observations=used_references,
            excluded_observations=tuple(excluded),
            status="undetermined",
            reasons=(f"at_least_{config.min_points}_non_excluded_observations_required",),
        )

    points_a = np.vstack([sample.point_a for sample in samples])
    points_b = np.vstack([sample.point_b for sample in samples])
    n_points = len(samples)

    centroid_a = points_a.mean(axis=0)
    centered_a = points_a - centroid_a
    _, singular_values, _ = np.linalg.svd(centered_a, full_matrices=False)
    motion_scale = float(singular_values[0] / np.sqrt(n_points))

    sigmas = np.array([sample.sigma for sample in samples if sample.sigma is not None], dtype=float)
    noise_scale = float(np.max(sigmas)) if len(sigmas) else 0.0
    noise_floor = (
        config.motion_sigma_factor * noise_scale / np.sqrt(n_points) if noise_scale > 0 else 0.0
    )
    machine_floor = config.geometric_rank_tol * float(singular_values[0]) if singular_values[0] > 0 else 0.0
    resolution_floor = max(noise_floor, machine_floor)

    geometric_rank = int(np.count_nonzero(singular_values > machine_floor))
    effective_rank = int(np.count_nonzero(singular_values > resolution_floor))
    resolved_singular = singular_values[singular_values > resolution_floor]
    condition_number = (
        float(resolved_singular[0] / resolved_singular[-1]) if len(resolved_singular) else float("inf")
    )

    common_fields = {
        "convention": convention,
        "version": version,
        "samples": samples,
        "excluded": excluded,
        "singular_values": singular_values,
        "geometric_rank": geometric_rank,
        "effective_rank": effective_rank,
        "motion_scale": motion_scale,
        "noise_scale": noise_scale if noise_scale > 0 else None,
        "condition_number": condition_number if np.isfinite(condition_number) else None,
    }

    if geometric_rank == 0 or effective_rank == 0:
        reason = (
            "target_positions_collapse_to_a_point"
            if geometric_rank == 0
            else "target_motion_is_small_relative_to_measurement_uncertainty"
        )
        return _undetermined(**common_fields, reasons=(reason,))

    # Non-collinear planar targets (effective rank 2) DO constrain a unique
    # rigid transform: the in-plane span fixes R and t, and the Kabsch reflection
    # guard selects the rotation that keeps the missing plane direction proper.
    # Only a true/near line (one resolved direction) is unidentifiable in 3D.
    if effective_rank < 2:
        return _undetermined(
            **common_fields,
            reasons=("target_positions_are_collinear_or_nearly_collinear",),
        )

    fit = _kabsch(points_a, points_b)
    if fit is None:
        return _undetermined(
            **common_fields,
            reasons=("only_a_reflection_can_fit_the_correspondences",),
        )
    rotation, translation, _centroid_a, _centroid_b = fit

    transform = RigidTransform(rotation, translation, convention.frame_a, convention.frame_b)
    residual_values = np.linalg.norm(transform.apply(points_a) - points_b, axis=1)
    residuals = {
        sample.reference: float(value)
        for sample, value in zip(samples, residual_values)
    }
    rms_residual = float(np.sqrt(np.mean(residual_values**2)))
    max_residual = float(np.max(residual_values))

    reasons: list[str] = []
    if geometric_rank == 3 and effective_rank == 2:
        reasons.append("out_of_plane_motion_is_below_measurement_resolution")
    if geometric_rank == 2:
        # Exact planar geometry is a valid, common calibration setup.
        pass
    if len(resolved_singular) >= 2 and condition_number > 1.0 / config.geometric_rank_tol:
        reasons.append("resolved_target_span_is_poorly_conditioned")

    # Residual based diagnostics: mismatched correspondences inflate residuals
    # beyond the claimed measurement uncertainty.
    # Robust residual scale (MAD of residual magnitudes): one wrong
    # correspondence biases the fit and inflates many residuals, but a robust
    # scale keeps the warning focused on the genuinely offending observation.
    residual_median = float(np.median(residual_values))
    mad_scale = 1.4826 * float(np.median(np.abs(residual_values - residual_median)))
    robust_noise = max(
        noise_scale if noise_scale > 0 else 0.0,
        mad_scale,
        rms_residual / np.sqrt(max(n_points, 1)),
        np.finfo(float).eps,
    )
    warning_observations = tuple(
        sample.reference
        for sample, value in zip(samples, residual_values)
        if value > config.outlier_residual_factor * robust_noise
    )
    if warning_observations:
        reasons.append("some_residuals_exceed_measurement_uncertainty")
    if rms_residual > config.instability_residual_factor * robust_noise:
        reasons.append("fit_residuals_are_large_relative_to_measurement_uncertainty")

    # Leave-one-out influence: which single observation moves the answer most?
    influence_scores: dict[tuple[str, str], float] = {}
    influential: list[tuple[str, str]] = []
    if config.leave_one_out and n_points >= max(config.min_points + 1, 4):
        translation_scale = max(np.linalg.norm(translation), motion_scale, 1.0)
        for index, sample in enumerate(samples):
            keep = np.ones(n_points, dtype=bool)
            keep[index] = False
            reduced_fit = _kabsch(points_a[keep], points_b[keep])
            if reduced_fit is None:
                influence_scores[sample.reference] = float("inf")
                influential.append(sample.reference)
                continue
            reduced_rotation, reduced_translation, _, _ = reduced_fit
            rotation_delta = _rotation_delta_deg(rotation, reduced_rotation)
            translation_delta = np.linalg.norm(translation - reduced_translation) / translation_scale
            score = max(
                rotation_delta / config.influence_rotation_limit_deg,
                translation_delta / config.influence_translation_fraction,
            )
            influence_scores[sample.reference] = float(score)
            if score > 1.0:
                influential.append(sample.reference)
        if influential:
            reasons.append("leave_one_out_fit_is_sensitive_to_individual_observations")
    elif config.leave_one_out and n_points < 4:
        reasons.append("too_few_observations_for_leave_one_out_influence_check")

    destabilizing = {
        "out_of_plane_motion_is_below_measurement_resolution",
        "resolved_target_span_is_poorly_conditioned",
        "fit_residuals_are_large_relative_to_measurement_uncertainty",
        "leave_one_out_fit_is_sensitive_to_individual_observations",
    }
    status = "unstable" if any(reason in destabilizing for reason in reasons) else "valid"

    return CalibrationResult(
        success=True,
        transform=transform,
        coordinate_convention=convention,
        calibration_version=version,
        used_observations=used_references,
        excluded_observations=tuple(excluded),
        status=status,
        rms_residual=rms_residual,
        max_residual=max_residual,
        residuals=residuals,
        singular_values=singular_values,
        geometric_rank=geometric_rank,
        effective_rank=effective_rank,
        motion_scale=motion_scale,
        noise_scale=noise_scale if noise_scale > 0 else None,
        condition_number=condition_number if np.isfinite(condition_number) else None,
        influential_observations=tuple(influential),
        influence_scores=influence_scores,
        warning_observations=warning_observations,
        reasons=tuple(reasons),
    )


def calibrate_subsets(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    removed_sets: dict[str, frozenset[tuple[str, str]]],
    config: CalibrationConfig | None = None,
) -> dict[str, CalibrationResult]:
    """Recalibrate after manually removing observation references.

    Source batches are never modified; each named subset gets its own config.
    The empty-string key can be used for the baseline with no manual removals.
    """
    base_config = config or CalibrationConfig()
    results: dict[str, CalibrationResult] = {}
    for label, removed_references in removed_sets.items():
        subset_config = CalibrationConfig(
            min_points=base_config.min_points,
            motion_sigma_factor=base_config.motion_sigma_factor,
            geometric_rank_tol=base_config.geometric_rank_tol,
            instability_residual_factor=base_config.instability_residual_factor,
            influence_rotation_limit_deg=base_config.influence_rotation_limit_deg,
            influence_translation_fraction=base_config.influence_translation_fraction,
            outlier_residual_factor=base_config.outlier_residual_factor,
            leave_one_out=base_config.leave_one_out,
            exclude_marked_bad=base_config.exclude_marked_bad,
            additional_excluded_observations=frozenset(
                set(base_config.additional_excluded_observations) | set(removed_references)
            ),
        )
        results[label] = calibrate(calibration_input, convention, version, subset_config)
    return results
