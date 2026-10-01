from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import (
    AcquisitionEvidence,
    BatchEvaluation,
    CalibrationInput,
    CalibrationResult,
    CalibrationVersion,
    CoordinateConvention,
    ObservationDecision,
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
    excluded_batches: frozenset[str] = field(default_factory=frozenset)
    batch_bias_sigma_factor: float = 4.0
    batch_transform_sigma_factor: float = 4.0
    evaluate_batches: bool = True
    evaluate_batches: bool = True


def _select_observations(
    calibration_input: CalibrationInput,
    config: CalibrationConfig,
) -> tuple[list[dict], list[tuple[str, str, str, str]], list[dict], list[dict]]:
    selected: list[dict] = []
    excluded: list[tuple[str, str]] = []
    records: list[dict] = []

    for batch in calibration_input.batches:
        for observation in batch.observations:
            reference = (batch.batch_id, observation.observation_id)
            explicitly_excluded = reference in config.additional_excluded_observations
            marked_bad = config.exclude_marked_bad and observation.quality == "bad"
            batch_excluded = batch.batch_id in config.excluded_batches
            scale = batch.length_scale_to_convention
            sigma = (
                observation.position_sigma * scale
                if observation.position_sigma is not None
                else None
            )
            acquisition_id = (
                f"{batch.acquisition_id}:{observation.target_id}"
                if batch.acquisition_id is not None
                else f"{batch.batch_id}:{observation.target_id}"
            )
            record = {
                "reference": reference,
                "batch_id": batch.batch_id,
                "observation_id": observation.observation_id,
                "target_id": observation.target_id,
                "acquisition_id": acquisition_id,
                "point_a": observation.sensor_a_position * scale,
                "point_b": observation.sensor_b_position * scale,
                "sigma": sigma,
                "raw_sigma": observation.position_sigma,
                "batch_excluded": batch_excluded,
                "explicitly_excluded": explicitly_excluded,
                "marked_bad": marked_bad,
            }
            records.append(record)
            if batch_excluded:
                excluded.append((batch.batch_id, observation.observation_id, "excluded_batch", acquisition_id))
            elif marked_bad:
                excluded.append((batch.batch_id, observation.observation_id, "excluded_marked_bad", acquisition_id))
            elif explicitly_excluded:
                excluded.append((batch.batch_id, observation.observation_id, "excluded_manually", acquisition_id))
            else:
                selected.append(record)
    return selected, excluded, records, []


def _fit_transform(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> tuple[
    RigidTransform | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    return _fit_transform_weighted(points_a, points_b, np.ones(len(points_a)))


def _weighted_scatter_singular_values(points: np.ndarray, weights: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    centered = points - centroid
    scatter = (centered * weights[:, None]).T @ centered / np.sum(weights)
    eigenvalues = np.linalg.eigvalsh(scatter)
    return np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]


def _fit_transform_weighted(
    points_a: np.ndarray,
    points_b: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    RigidTransform | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    total_weight = float(np.sum(weights))
    centroid_a = (weights[:, None] * points_a).sum(axis=0) / total_weight
    centroid_b = (weights[:, None] * points_b).sum(axis=0) / total_weight
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b
    scale_a = _weighted_scatter_singular_values(points_a, weights, centroid_a)
    scale_b = _weighted_scatter_singular_values(points_b, weights, centroid_b)
    covariance = (centered_a * weights[:, None]).T @ centered_b / total_weight
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    rigid = None
    if np.linalg.det(rotation) > 0:
        translation = centroid_b - rotation @ centroid_a
        rigid = RigidTransform(rotation, translation, "a", "b")
    return rigid, scale_a, scale_b, singular_values, u, vt, centroid_a, centroid_b


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


def _batch_geometry(
    used_records: list[dict],
    config: CalibrationConfig,
) -> tuple[str, str | None, float]:
    """Independent rank assessment of one batch (used even when joint fails)."""
    if len(used_records) < config.min_points:
        return "undetermined", None, 0.0
    batch_a = np.vstack([item["point_a"] for item in used_records])
    batch_b = np.vstack([item["point_b"] for item in used_records])
    alone_sigmas = np.array(
        [item["sigma"] if item["sigma"] is not None else np.nan for item in used_records],
        dtype=float,
    )
    alone_known = alone_sigmas[~np.isnan(alone_sigmas)]
    if len(alone_known):
        alone_weights = 1.0 / np.maximum(
            np.where(np.isnan(alone_sigmas), float(np.max(alone_known)), alone_sigmas),
            np.finfo(float).tiny,
        ) ** 2
    else:
        alone_weights = np.ones(len(used_records))
    alone_weights = alone_weights / np.mean(alone_weights)
    _, alone_scale_a, alone_scale_b, _, _, _, _, _ = _fit_transform_weighted(
        batch_a, batch_b, alone_weights
    )
    alone_floor = config.rank_tolerance
    if len(alone_known):
        alone_floor = max(alone_floor, config.motion_sigma_factor * float(np.max(alone_known)))
    alone_rank = min(
        int(np.count_nonzero(alone_scale_a > alone_floor)),
        int(np.count_nonzero(alone_scale_b > alone_floor)),
    )
    distribution = {0: "point", 1: "line", 2: "plane", 3: "volume"}[alone_rank]
    status = "valid" if alone_rank >= 2 else "undetermined"
    return status, distribution, float(np.max(alone_known)) if len(alone_known) else 0.0


def calibrate(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    config = config or CalibrationConfig()
    selected, excluded, records, _unused = _select_observations(calibration_input, config)
    used_references = tuple(item["reference"] for item in selected)
    common_result_fields = dict(
        coordinate_convention=convention,
        calibration_version=version,
        used_observations=used_references,
        excluded_observations=tuple(item[:2] for item in excluded),
    )

    excluded_references = {item[:2] for item in excluded}

    def build_decisions(
        evidence_map: dict[str, float] | None = None,
        duplicate_counts: dict[str, int] | None = None,
    ) -> dict[tuple[str, str], ObservationDecision]:
        decisions: dict[tuple[str, str], ObservationDecision] = {}
        for record in records:
            reference = record["reference"]
            if reference in excluded_references:
                matching = [item for item in excluded if item[:2] == reference]
                decision = matching[0][2]
                adopted = False
            else:
                decision = "used"
                adopted = True
            acquisition_id = record["acquisition_id"]
            decisions[reference] = ObservationDecision(
                batch_id=reference[0],
                observation_id=reference[1],
                acquisition_id=acquisition_id,
                adopted=adopted,
                decision=decision,
                evidence_weight=(evidence_map or {}).get(acquisition_id, 1.0) if adopted else 0.0,
                duplicate_count=(duplicate_counts or {}).get(acquisition_id, 1),
            )
        return decisions

    def build_acquisition_evidence() -> tuple[dict[str, AcquisitionEvidence], dict[str, float], dict[str, int]]:
        groups: dict[str, list[dict]] = {}
        for record in records:
            groups.setdefault(record["acquisition_id"], []).append(record)
        evidence: dict[str, AcquisitionEvidence] = {}
        independent_weights: dict[str, float] = {}
        duplicate_counts: dict[str, int] = {}
        for acquisition_id, members in sorted(groups.items()):
            multiplicity = len(members)
            adopted_members = [item for item in members if item["reference"] not in excluded_references]
            # Repeated copies of one acquisition share one unit of independent
            # evidence: split that unit across the recorded copies instead of
            # letting the acquisition vote multiplicity times.
            independent_weight = 1.0 / multiplicity if multiplicity > 1 else 1.0
            independent_weights[acquisition_id] = independent_weight
            duplicate_counts[acquisition_id] = multiplicity
            evidence[acquisition_id] = AcquisitionEvidence(
                acquisition_id=acquisition_id,
                member_references=tuple(item["reference"] for item in members),
                adopted_member_references=tuple(item["reference"] for item in adopted_members),
                multiplicity=multiplicity,
                independent_weight=independent_weight,
                repeated=multiplicity > 1,
            )
        return evidence, independent_weights, duplicate_counts

    if len(selected) < config.min_points:
        acquisition_evidence, independent_weights, duplicate_counts = build_acquisition_evidence()
        early_batch_evaluations: dict[str, BatchEvaluation] = {}
        if config.evaluate_batches:
            for batch in calibration_input.batches:
                batch_records_early = [item for item in records if item["batch_id"] == batch.batch_id]
                early_batch_evaluations[batch.batch_id] = BatchEvaluation(
                    batch_id=batch.batch_id,
                    observation_count=len(batch_records_early),
                    used_observation_count=(
                        0
                        if batch.batch_id in config.excluded_batches
                        else sum(1 for item in selected if item["batch_id"] == batch.batch_id)
                    ),
                    status="excluded" if batch.batch_id in config.excluded_batches else "undetermined",
                    distribution=None,
                    rms_residual=None,
                    joint_residual_mean=None,
                    joint_residual_rms=None,
                    joint_bias_norm=None,
                    bias_sigma_ratio=None,
                    rotation_change=None,
                    translation_change=None,
                    supports_joint_transform=None,
                    excluded=batch.batch_id in config.excluded_batches,
                    reasons=(
                        ("batch_was_excluded_from_the_joint_fit",)
                        if batch.batch_id in config.excluded_batches
                        else ("joint_fit_has_too_few_non_excluded_observations",)
                    ),
                )
        return CalibrationResult(
            success=False,
            transform=None,
            reasons=(f"at_least_{config.min_points}_non_excluded_observations_required",),
            observation_decisions=build_decisions(),
            batch_evaluations=early_batch_evaluations,
            acquisition_evidence=acquisition_evidence,
            **common_result_fields,
        )

    acquisition_evidence, independent_weights, duplicate_counts = build_acquisition_evidence()
    points_a = np.vstack([item["point_a"] for item in selected])
    points_b = np.vstack([item["point_b"] for item in selected])
    raw_sigmas = np.array(
        [item["sigma"] if item["sigma"] is not None else np.nan for item in selected],
        dtype=float,
    )
    known_sigmas = raw_sigmas[~np.isnan(raw_sigmas)]
    # The reported noise floor used for rank / residual checks: batches with
    # different precision are combined by inverse-variance weights below, but
    # the conservative largest converted sigma keeps the identifiability test
    # honest for the least precise contributing acquisition.
    measurement_sigma = float(np.max(known_sigmas)) if len(known_sigmas) else 0.0

    evidence_weights = np.array(
        [independent_weights[item["acquisition_id"]] for item in selected],
        dtype=float,
    )
    if len(known_sigmas):
        precision = np.where(
            np.isnan(raw_sigmas),
            1.0 / max(np.max(known_sigmas) ** 2, np.finfo(float).tiny),
            1.0 / np.maximum(raw_sigmas, np.finfo(float).tiny) ** 2,
        )
        fit_weights = evidence_weights * precision
    else:
        fit_weights = evidence_weights
    fit_weights = fit_weights / np.mean(fit_weights)

    effective_total_weight = float(np.sum(fit_weights))
    effective_point_count = float(np.sum(evidence_weights))
    if measurement_sigma > 0:
        weighted_variance = float(
            np.sum(fit_weights * np.nan_to_num(raw_sigmas, nan=measurement_sigma) ** 2)
            / np.sum(fit_weights)
        )
        centroid_sigma = np.sqrt(weighted_variance / max(effective_total_weight, np.finfo(float).tiny))
    else:
        centroid_sigma = 0.0

    rigid, scale_a, scale_b, singular_values, _, _, _, _ = _fit_transform_weighted(
        points_a, points_b, fit_weights
    )
    if rigid is None:
        return CalibrationResult(
            success=False,
            transform=None,
            singular_values=singular_values,
            measurement_sigma=measurement_sigma,
            reasons=("only_a_reflection_can_fit_the_correspondences",),
            effective_measurement_sigma=measurement_sigma,
            observation_decisions=build_decisions(independent_weights, duplicate_counts),
            acquisition_evidence=acquisition_evidence,
            **common_result_fields,
        )

    rigid = RigidTransform(rigid.rotation, rigid.translation, convention.frame_a, convention.frame_b)
    residual_vectors = rigid.apply(points_a) - points_b
    residual_values = np.linalg.norm(residual_vectors, axis=1)
    residuals = {
        item["reference"]: float(value)
        for item, value in zip(selected, residual_values)
    }
    rms_residual = float(
        np.sqrt(np.sum(fit_weights * residual_values**2) / np.sum(fit_weights))
    )
    max_residual = float(np.max(residual_values))

    reference_scale = max(float(scale_a[0]), float(scale_b[0]))
    rank_floor = config.rank_tolerance * max(reference_scale, np.finfo(float).tiny)
    if measurement_sigma > 0:
        # Geometry detectability floor: a scatter direction counts only if
        # its extent exceeds the reported noise of the observations spanning
        # it.  With N independent measurements spanning a direction the
        # centroid-level floor is smaller; use the per-observation scale so a
        # single collinear batch cannot be "rescued" merely by adding points
        # along the same line, while genuinely independent directions from
        # other batches still lift the merged rank.
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
        degenerate_batch_evaluations: dict[str, BatchEvaluation] = {}
        if config.evaluate_batches:
            for batch in calibration_input.batches:
                batch_used = [item for item in selected if item["batch_id"] == batch.batch_id]
                batch_all = [item for item in records if item["batch_id"] == batch.batch_id]
                if batch.batch_id in config.excluded_batches:
                    alone_status, alone_distribution = "excluded", None
                    alone_reasons = ("batch_was_excluded_from_the_joint_fit",)
                else:
                    alone_status, alone_distribution, _ = _batch_geometry(batch_used, config)
                    alone_reasons = (
                        ("batch_geometry_is_individually_degenerate_but_other_batches_can_supply_missing_directions",)
                        if alone_status == "undetermined" and effective_rank >= 2
                        else ("batch_geometry_is_individually_degenerate",)
                    )
                degenerate_batch_evaluations[batch.batch_id] = BatchEvaluation(
                    batch_id=batch.batch_id,
                    observation_count=len(batch_all),
                    used_observation_count=len(batch_used),
                    status=alone_status,
                    distribution=alone_distribution,
                    rms_residual=None,
                    joint_residual_mean=None,
                    joint_residual_rms=None,
                    joint_bias_norm=None,
                    bias_sigma_ratio=None,
                    rotation_change=None,
                    translation_change=None,
                    supports_joint_transform=None,
                    excluded=batch.batch_id in config.excluded_batches,
                    reasons=alone_reasons,
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
            effective_measurement_sigma=measurement_sigma,
            observation_decisions=build_decisions(independent_weights, duplicate_counts),
            acquisition_evidence=acquisition_evidence,
            batch_evaluations=degenerate_batch_evaluations,
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
            reference = item["reference"]
            remaining_a = np.delete(points_a, index, axis=0)
            remaining_b = np.delete(points_b, index, axis=0)
            remaining_weights = np.delete(fit_weights, index)
            leave_one_out, _, _, _, _, _, _, _ = _fit_transform_weighted(
                remaining_a, remaining_b, remaining_weights
            )
            rotation_change: float | None = None
            translation_change: float | None = None
            score: float | None = None
            if leave_one_out is not None:
                rotation_change = float(
                    np.linalg.norm(leave_one_out.rotation - rigid.rotation, ord="fro")
                )
                translation_change = float(np.linalg.norm(leave_one_out.translation - rigid.translation))
                fit_scale = (
                    centroid_sigma
                    if centroid_sigma > 0
                    else reference_scale / np.sqrt(max(effective_point_count, 1.0))
                )
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

    repeated_acquisitions = {
        acquisition_id: evidence
        for acquisition_id, evidence in acquisition_evidence.items()
        if evidence.repeated
    }
    if repeated_acquisitions:
        warnings.append("repeated_acquisition_copies_are_counted_once")

    batch_evaluations: dict[str, BatchEvaluation] = {}
    if config.evaluate_batches:
        batch_records: dict[str, list[dict]] = {}
        for record in records:
            batch_records.setdefault(record["batch_id"], []).append(record)
        selected_by_batch: dict[str, list[dict]] = {}
        for item in selected:
            selected_by_batch.setdefault(item["batch_id"], []).append(item)
        for batch in calibration_input.batches:
            batch_id = batch.batch_id
            all_records = batch_records[batch_id]
            used_records = selected_by_batch.get(batch_id, [])
            if batch_id in config.excluded_batches:
                batch_evaluations[batch_id] = BatchEvaluation(
                    batch_id=batch_id,
                    observation_count=len(all_records),
                    used_observation_count=0,
                    status="excluded",
                    distribution=None,
                    rms_residual=None,
                    joint_residual_mean=None,
                    joint_residual_rms=None,
                    joint_bias_norm=None,
                    bias_sigma_ratio=None,
                    rotation_change=None,
                    translation_change=None,
                    supports_joint_transform=None,
                    excluded=True,
                    reasons=("batch_was_excluded_from_the_joint_fit",),
                )
                continue

            joint_indices = [index for index, item in enumerate(selected) if item["batch_id"] == batch_id]
            joint_vectors = residual_vectors[joint_indices]
            joint_norms = residual_values[joint_indices]
            mean_vector = joint_vectors.mean(axis=0)
            bias_norm = float(np.linalg.norm(mean_vector))
            batch_rms = float(np.sqrt(np.mean(joint_norms**2)))
            batch_sigmas = np.array(
                [item["sigma"] for item in used_records if item["sigma"] is not None],
                dtype=float,
            )
            batch_sigma = float(np.max(batch_sigmas)) if len(batch_sigmas) else measurement_sigma or 0.0
            bias_uncertainty = batch_sigma / np.sqrt(max(len(joint_indices), 1))
            bias_ratio = float(bias_norm / bias_uncertainty) if bias_uncertainty > 0 else None

            alone_status = "undetermined"
            alone_distribution: str | None = None
            alone_rms: float | None = None
            rotation_change: float | None = None
            translation_change: float | None = None
            alone_reasons: tuple[str, ...] = ()
            supports: bool | None = None

            if len(used_records) >= config.min_points:
                batch_a = np.vstack([item["point_a"] for item in used_records])
                batch_b = np.vstack([item["point_b"] for item in used_records])
                alone_sigmas = np.array(
                    [item["sigma"] if item["sigma"] is not None else np.nan for item in used_records],
                    dtype=float,
                )
                alone_known = alone_sigmas[~np.isnan(alone_sigmas)]
                if len(alone_known):
                    alone_floor_sigma = float(np.max(alone_known))
                    alone_weights = 1.0 / np.maximum(
                        np.where(np.isnan(alone_sigmas), alone_floor_sigma, alone_sigmas),
                        np.finfo(float).tiny,
                    ) ** 2
                else:
                    alone_weights = np.ones(len(used_records))
                alone_weights = alone_weights / np.mean(alone_weights)
                alone_fit, alone_scale_a, alone_scale_b, _, _, _, _, _ = _fit_transform_weighted(
                    batch_a, batch_b, alone_weights
                )
                alone_floor = config.rank_tolerance
                if len(alone_known):
                    alone_floor = max(alone_floor, config.motion_sigma_factor * float(np.max(alone_known)))
                alone_rank_a = int(np.count_nonzero(alone_scale_a > alone_floor))
                alone_rank_b = int(np.count_nonzero(alone_scale_b > alone_floor))
                alone_rank = min(alone_rank_a, alone_rank_b)
                alone_distribution = {0: "point", 1: "line", 2: "plane", 3: "volume"}[alone_rank]
                if alone_fit is not None and alone_rank >= 2:
                    alone_status = "valid"
                    alone_residuals = np.linalg.norm(
                        alone_fit.apply(batch_a) - batch_b, axis=1
                    )
                    alone_rms = float(
                        np.sqrt(np.sum(alone_weights * alone_residuals**2) / np.sum(alone_weights))
                    )
                    rotation_change = float(
                        np.linalg.norm(alone_fit.rotation - rigid.rotation, ord="fro")
                    )
                    translation_change = float(
                        np.linalg.norm(alone_fit.translation - rigid.translation)
                    )
                    if batch_sigma > 0:
                        supports = (
                            (bias_ratio is None or bias_ratio <= config.batch_bias_sigma_factor)
                            and translation_change <= config.batch_transform_sigma_factor * batch_sigma
                        )
                else:
                    alone_status = "undetermined"
                    alone_reasons = (
                        "batch_geometry_is_individually_degenerate_but_other_batches_can_supply_missing_directions"
                        if effective_rank > alone_rank
                        else "batch_geometry_is_individually_degenerate",
                    )
                    supports = None
            else:
                alone_reasons = ("batch_has_too_few_non_excluded_observations_for_an_independent_fit",)

            if bias_ratio is not None and bias_ratio > config.batch_bias_sigma_factor:
                warnings.append(
                    f"batch:{batch_id}:residual_bias_cannot_be_explained_by_the_common_transform"
                )
            if supports is False and batch_sigma > 0:
                warnings.append(f"batch:{batch_id}:independent_fit_disagrees_with_the_joint_transform")

            batch_evaluations[batch_id] = BatchEvaluation(
                batch_id=batch_id,
                observation_count=len(all_records),
                used_observation_count=len(used_records),
                status=alone_status,
                distribution=alone_distribution,
                rms_residual=alone_rms,
                joint_residual_mean=float(joint_norms.mean()),
                joint_residual_rms=batch_rms,
                joint_bias_norm=bias_norm,
                bias_sigma_ratio=bias_ratio,
                rotation_change=rotation_change,
                translation_change=translation_change,
                supports_joint_transform=supports,
                excluded=False,
                reasons=alone_reasons,
            )

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
        effective_measurement_sigma=measurement_sigma,
        batch_evaluations=batch_evaluations,
        observation_decisions=build_decisions(independent_weights, duplicate_counts),
        acquisition_evidence=acquisition_evidence,
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
            excluded_batches=base_config.excluded_batches,
            batch_bias_sigma_factor=base_config.batch_bias_sigma_factor,
            batch_transform_sigma_factor=base_config.batch_transform_sigma_factor,
            evaluate_batches=base_config.evaluate_batches,
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
