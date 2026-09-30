from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import (
    CalibrationInput,
    CalibrationResult,
    CalibrationVersion,
    CoordinateConvention,
    ObservationInfluence,
    RigidTransform,
    SubsetComparison,
)


@dataclass(frozen=True)
class CalibrationConfig:
    min_points: int = 3
    # Rank thresholds are relative to the largest singular value of the
    # point scatter matrix; they identify collinear / point-like geometry
    # without rejecting a properly spread planar (rank 2) configuration.
    rank_tolerance: float = 1e-10
    near_degenerate_ratio: float = 1e-4
    condition_limit: float = 1e8
    motion_sigma_factor: float = 8.0
    residual_sigma_factor: float = 4.0
    influence_score_limit: float = 0.5
    influence_relative_factor: float = 2.0
    exclude_marked_bad: bool = True
    compute_influence: bool = True
    additional_excluded_observations: frozenset[tuple[str, str]] = field(default_factory=frozenset)


_SelectedObservation = tuple[str, str, np.ndarray, np.ndarray, float | None]


def _select_observations(
    calibration_input: CalibrationInput,
    config: CalibrationConfig,
) -> tuple[list[_SelectedObservation], list[tuple[str, str]], tuple[tuple[str, str], ...]]:
    selected: list[_SelectedObservation] = []
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
    return selected, excluded, tuple((item[0], item[1]) for item in selected)


def _fit_transform(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> tuple[RigidTransform | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid_a = points_a.mean(axis=0)
    centroid_b = points_b.mean(axis=0)
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b
    scale_a = np.linalg.svd(centered_a, compute_uv=False)
    scale_b = np.linalg.svd(centered_b, compute_uv=False)
    covariance = centered_a.T @ centered_b / len(points_a)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    rigid = None
    if np.linalg.det(rotation) > 0:
        translation = centroid_b - rotation @ centroid_a
        rigid = RigidTransform(rotation, translation, "a", "b")
    return rigid, scale_a, scale_b, singular_values, u, vt


def _sampling_recommendations(
    *,
    distribution: str,
    warnings: tuple[str, ...],
) -> tuple[str, ...]:
    if distribution == "point":
        return ("add_targets_spanning_two_independent_directions_to_form_a_plane",)
    if distribution == "line":
        return ("add_targets_displaced_from_the_dominant_line_to_form_a_plane",)
    if distribution == "plane" and "planar_configuration_is_nearly_collinear" in warnings:
        return ("add_targets_along_a_second_independent_direction_within_the_plane",)
    if distribution == "volume" and "three_dimensional_configuration_is_nearly_planar" in warnings:
        return ("add_targets_displaced_from_the_fitted_plane_to_strengthen_third_dimension",)
    return ()


def calibrate(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    config = config or CalibrationConfig()
    selected, excluded, used_references = _select_observations(calibration_input, config)
    common_result_fields = dict(
        coordinate_convention=convention,
        calibration_version=version,
        used_observations=used_references,
        excluded_observations=tuple(excluded),
    )

    if len(selected) < config.min_points:
        return CalibrationResult(
            success=False,
            transform=None,
            reasons=(f"at_least_{config.min_points}_non_excluded_observations_required",),
            **common_result_fields,
        )

    points_a = np.vstack([item[2] for item in selected])
    points_b = np.vstack([item[3] for item in selected])
    sigmas = np.array([item[4] for item in selected if item[4] is not None], dtype=float)
    measurement_sigma = float(np.max(sigmas)) if len(sigmas) else 0.0
    centroid_sigma = measurement_sigma / np.sqrt(len(selected))

    rigid, scale_a, scale_b, singular_values, _, _ = _fit_transform(points_a, points_b)
    if rigid is None:
        return CalibrationResult(
            success=False,
            transform=None,
            singular_values=singular_values,
            measurement_sigma=measurement_sigma,
            reasons=("only_a_reflection_can_fit_the_correspondences",),
            **common_result_fields,
        )

    rigid = RigidTransform(rigid.rotation, rigid.translation, convention.frame_a, convention.frame_b)
    residual_values = np.linalg.norm(rigid.apply(points_a) - points_b, axis=1)
    residuals = {
        (batch_id, observation_id): float(value)
        for (batch_id, observation_id, *_), value in zip(selected, residual_values)
    }
    rms_residual = float(np.sqrt(np.mean(residual_values**2)))
    max_residual = float(np.max(residual_values))

    reference_scale = max(float(scale_a[0]), float(scale_b[0]))
    rank_floor = config.rank_tolerance * max(reference_scale, np.finfo(float).tiny)
    if measurement_sigma > 0:
        rank_floor = max(rank_floor, config.motion_sigma_factor * measurement_sigma)
    active_a = scale_a > rank_floor
    active_b = scale_b > rank_floor
    rank_a = int(np.count_nonzero(active_a))
    rank_b = int(np.count_nonzero(active_b))
    effective_rank = min(rank_a, rank_b)
    distribution = {0: "point", 1: "line", 2: "plane", 3: "volume"}[effective_rank]

    reasons: list[str] = []
    warnings: list[str] = []
    motion_too_small = reference_scale > 0 and measurement_sigma > 0 and (
        reference_scale < config.motion_sigma_factor * measurement_sigma
    )
    if motion_too_small:
        reasons.append("target_motion_is_small_relative_to_measurement_uncertainty")
    if rank_a == 0 or rank_b == 0:
        reasons.append("target_positions_concentrate_at_one_point")
    elif rank_a != rank_b:
        reasons.append("sensor_point_clouds_have_inconsistent_geometry")
    elif rank_a == 1:
        reasons.append("target_positions_are_collinear_rotation_about_line_is_unobservable")
    if reasons:
        rank_warnings = tuple(warnings)
        sampling_recommendations = _sampling_recommendations(
            distribution=distribution,
            warnings=rank_warnings,
        )
        return CalibrationResult(
            success=False,
            transform=None,
            singular_values=singular_values,
            reasons=tuple(reasons),
            distribution=distribution,
            measurement_sigma=measurement_sigma,
            warnings=rank_warnings,
            sampling_recommendations=sampling_recommendations,
            **common_result_fields,
        )

    # A genuine plane has a zero third singular value by construction.  Its
    # conditioning is therefore measured using only the two observable
    # in-plane directions; rank-three covariance conditioning must not be
    # used to reject rank-two geometry.
    active_covariance_values = singular_values[:effective_rank]
    condition_number = (
        float(active_covariance_values[0] / max(active_covariance_values[-1], np.finfo(float).tiny))
        if len(active_covariance_values)
        else float("inf")
    )
    if not np.isfinite(condition_number) or condition_number > config.condition_limit:
        warnings.append("point_configuration_is_nearly_degenerate")

    plane_ratio = min(
        float(scale_a[1] / max(scale_a[0], np.finfo(float).tiny)),
        float(scale_b[1] / max(scale_b[0], np.finfo(float).tiny)),
    )
    line_ratio = min(
        float(scale_a[2] / max(scale_a[1], np.finfo(float).tiny)),
        float(scale_b[2] / max(scale_b[1], np.finfo(float).tiny)),
    )
    if effective_rank == 2:
        if plane_ratio < config.near_degenerate_ratio:
            warnings.append("planar_configuration_is_nearly_collinear")
    elif line_ratio < config.near_degenerate_ratio:
        warnings.append("three_dimensional_configuration_is_nearly_planar")

    if measurement_sigma > 0:
        if rms_residual > config.residual_sigma_factor * measurement_sigma:
            warnings.append("residuals_are_large_relative_to_reported_measurement_uncertainty")
        expected_max = measurement_sigma * config.residual_sigma_factor
        if max_residual > expected_max:
            warnings.append("at_least_one_residual_is_large_enough_to_be_an_outlier")

    influence: dict[tuple[str, str], ObservationInfluence] = {}
    if config.compute_influence and len(selected) > config.min_points:
        for index, item in enumerate(selected):
            reference = (item[0], item[1])
            remaining_a = np.delete(points_a, index, axis=0)
            remaining_b = np.delete(points_b, index, axis=0)
            leave_one_out, _, _, _, _, _ = _fit_transform(remaining_a, remaining_b)
            rotation_change: float | None = None
            translation_change: float | None = None
            score: float | None = None
            if leave_one_out is not None:
                rotation_change = float(
                    np.linalg.norm(leave_one_out.rotation - rigid.rotation, ord="fro")
                )
                translation_change = float(np.linalg.norm(leave_one_out.translation - rigid.translation))
                fit_scale = centroid_sigma if centroid_sigma > 0 else reference_scale / np.sqrt(len(selected))
                normalized_rotation = rotation_change * reference_scale / fit_scale
                normalized_translation = translation_change / fit_scale
                score = float(np.hypot(normalized_rotation, normalized_translation))
            influence[reference] = ObservationInfluence(
                residual=residuals[reference],
                rotation_change=rotation_change,
                translation_change=translation_change,
                score=score,
            )
    finite_scores = np.array([item.score for item in influence.values() if item.score is not None])
    influential: tuple[tuple[str, str], ...] = ()
    if len(finite_scores) >= 3:
        median_score = float(np.median(finite_scores))
        threshold = max(
            config.influence_score_limit,
            config.influence_relative_factor * median_score,
        )
        sigma_for_residual = measurement_sigma if measurement_sigma > 0 else None
        influential = tuple(
            reference
            for reference, item in sorted(
                influence.items(), key=lambda entry: entry[1].score or 0.0, reverse=True
            )
            if item.score is not None
            and item.score > threshold
            and (sigma_for_residual is None or item.residual > config.residual_sigma_factor * sigma_for_residual)
        )
    if influential:
        warnings.append("one_or_more_observations_dominate_the_estimate")

    sampling_recommendations = _sampling_recommendations(
        distribution=distribution,
        warnings=tuple(warnings),
    )
    status = "unstable" if warnings else "valid"
    return CalibrationResult(
        success=True,
        transform=rigid,
        rms_residual=rms_residual,
        max_residual=max_residual,
        measurement_sigma=measurement_sigma,
        residuals=residuals,
        singular_values=singular_values,
        reasons=(),
        status=status,
        distribution=distribution,
        warnings=tuple(warnings),
        sampling_recommendations=sampling_recommendations,
        influence=influence,
        influential_observations=influential,
        **common_result_fields,
    )


def _subset_judgment_basis(
    full_result: CalibrationResult,
    subset_result: CalibrationResult,
    extra_excluded: tuple[tuple[str, str], ...],
    rotation_change: float | None,
    translation_change: float | None,
    residual_factor: float,
) -> tuple[str, ...]:
    def value_text(value: float | None) -> str:
        return "none" if value is None else f"{value:.12g}"

    basis = [
        f"full_batch_status:{full_result.status}",
        f"subset_status:{subset_result.status}",
        f"additional_excluded_observations:{len(extra_excluded)}",
        f"full_batch_rms_residual:{value_text(full_result.rms_residual)}",
        f"subset_rms_residual:{value_text(subset_result.rms_residual)}",
    ]
    if full_result.rms_residual is not None and subset_result.rms_residual is not None:
        if subset_result.rms_residual < full_result.rms_residual:
            basis.append("exclusion_reduces_rms_residual")
        else:
            basis.append("exclusion_does_not_reduce_rms_residual")
    for reference in extra_excluded:
        if reference in full_result.influential_observations:
            basis.append(f"excluded_observation_was_influential:{reference[0]}/{reference[1]}")
        else:
            basis.append(f"excluded_observation_was_not_influential:{reference[0]}/{reference[1]}")
    remaining_full_warnings = set(full_result.warnings)
    if full_result.status == "unstable" and subset_result.status == "valid":
        basis.append("exclusion_resolves_full_batch_warnings")
        remaining_full_warnings.clear()
    elif full_result.status == "unstable" and subset_result.status == "unstable":
        remaining_full_warnings.intersection_update(subset_result.warnings)
        if not remaining_full_warnings and subset_result.warnings:
            basis.append("exclusion_changes_but_does_not_resolve_warnings")
    if remaining_full_warnings:
        basis.extend(f"full_batch_warning:{warning}" for warning in sorted(remaining_full_warnings))
    if subset_result.warnings:
        basis.extend(f"subset_warning:{warning}" for warning in subset_result.warnings)
    if rotation_change is not None:
        basis.append(f"rotation_frobenius_change:{rotation_change:.12g}")
    if translation_change is not None:
        basis.append(f"translation_norm_change:{translation_change:.12g}")
    measurement_sigma = subset_result.measurement_sigma or full_result.measurement_sigma or 0.0
    if translation_change is not None and measurement_sigma > 0:
        if translation_change > residual_factor * measurement_sigma:
            basis.append("transform_change_exceeds_reported_measurement_uncertainty")
        else:
            basis.append("transform_change_is_within_reported_measurement_uncertainty")
    return tuple(basis)


def compare_observation_subsets(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    subsets: dict[str, frozenset[tuple[str, str]]] | dict[str, set[tuple[str, str]]],
    config: CalibrationConfig | None = None,
) -> tuple[CalibrationResult, tuple[SubsetComparison, ...]]:
    """Calibrate the full selection, then named manually excluded subsets.

    The source batches are never modified; each subset is expressed as
    additional exclusion references on its own configuration.
    """
    base_config = config or CalibrationConfig()
    full_result = calibrate(calibration_input, convention, version, base_config)
    comparisons: list[SubsetComparison] = []

    for label, extra_excluded in subsets.items():
        extra_tuple = tuple(sorted(extra_excluded))
        subset_config = CalibrationConfig(
            min_points=base_config.min_points,
            rank_tolerance=base_config.rank_tolerance,
            near_degenerate_ratio=base_config.near_degenerate_ratio,
            condition_limit=base_config.condition_limit,
            motion_sigma_factor=base_config.motion_sigma_factor,
            residual_sigma_factor=base_config.residual_sigma_factor,
            influence_score_limit=base_config.influence_score_limit,
            influence_relative_factor=base_config.influence_relative_factor,
            exclude_marked_bad=base_config.exclude_marked_bad,
            compute_influence=base_config.compute_influence,
            additional_excluded_observations=frozenset(
                set(base_config.additional_excluded_observations) | set(extra_excluded)
            ),
        )
        subset_result = calibrate(calibration_input, convention, version, subset_config)
        rotation_change = None
        translation_change = None
        if full_result.transform is not None and subset_result.transform is not None:
            rotation_change = float(
                np.linalg.norm(subset_result.transform.rotation - full_result.transform.rotation, ord="fro")
            )
            translation_change = float(
                np.linalg.norm(subset_result.transform.translation - full_result.transform.translation)
            )
        judgment_basis = _subset_judgment_basis(
            full_result,
            subset_result,
            extra_tuple,
            rotation_change,
            translation_change,
            base_config.residual_sigma_factor,
        )
        comparisons.append(
            SubsetComparison(
                label=label,
                result=subset_result,
                additional_excluded_observations=extra_tuple,
                rotation_change=rotation_change,
                translation_change=translation_change,
                judgment_basis=judgment_basis,
            )
        )

    return full_result, tuple(comparisons)
