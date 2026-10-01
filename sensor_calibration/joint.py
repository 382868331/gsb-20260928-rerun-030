from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .calibration import CalibrationConfig, _calibrate_core, _select_observations
from .models import (
    BatchCompatibility,
    CalibrationInput,
    CalibrationResult,
    CalibrationVersion,
    CoordinateConvention,
    JointCalibrationResult,
    JointScenario,
    JointScenarioComparison,
    ObservationBatch,
)


@dataclass(frozen=True)
class JointCalibrationConfig:
    """Joint multi-batch calibration options.

    Every distinct acquisition contributes one equal share of the total
    weight, so a batch with more points never dominates merely by count;
    within a batch, observations are weighted by 1/sigma^2.  Batches whose
    observations are bit-identical to another batch are detected as
    duplicates of the same acquisition and share one acquisition weight, so
    re-adding the same data cannot fabricate extra evidence.
    """

    base: CalibrationConfig = field(default_factory=CalibrationConfig)
    excluded_batches: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_batches", frozenset(self.excluded_batches))


def _batch_fingerprint(batch: ObservationBatch) -> str:
    digest = hashlib.sha256()
    digest.update(batch.sensor_a_id.encode())
    digest.update(batch.sensor_b_id.encode())
    for observation in sorted(batch.observations, key=lambda item: item.observation_id):
        digest.update(observation.observation_id.encode())
        digest.update(np.ascontiguousarray(observation.sensor_a_position).tobytes())
        digest.update(np.ascontiguousarray(observation.sensor_b_position).tobytes())
    return digest.hexdigest()


def _duplicate_groups(batches: tuple[ObservationBatch, ...]) -> tuple[tuple[str, ...], ...]:
    by_fingerprint: dict[str, list[str]] = {}
    for batch in batches:
        by_fingerprint.setdefault(_batch_fingerprint(batch), []).append(batch.batch_id)
    return tuple(
        tuple(sorted(ids)) for ids in by_fingerprint.values() if len(ids) > 1
    )


def _joint_weights(
    selected: list[tuple[str, str, np.ndarray, np.ndarray, float | None]],
    batches: tuple[ObservationBatch, ...],
) -> np.ndarray:
    group_sizes: dict[str, int] = {}
    fingerprint_by_batch = {batch.batch_id: _batch_fingerprint(batch) for batch in batches}
    for fingerprint in fingerprint_by_batch.values():
        group_sizes[fingerprint] = group_sizes.get(fingerprint, 0) + 1

    batch_members: dict[str, list[int]] = {}
    for index, item in enumerate(selected):
        batch_members.setdefault(item[0], []).append(index)

    weights = np.zeros(len(selected), dtype=float)
    for batch_id, indices in batch_members.items():
        share = 1.0 / group_sizes[fingerprint_by_batch[batch_id]]
        sigmas = np.array([
            selected[i][4] if selected[i][4] is not None else 1.0 for i in indices
        ])
        inverse_variance = 1.0 / np.square(sigmas)
        within = inverse_variance / float(np.sum(inverse_variance))
        for position, index in enumerate(indices):
            weights[index] = share * within[position]
    return weights


def calibrate_joint(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    joint_config: JointCalibrationConfig | None = None,
) -> JointCalibrationResult:
    joint_config = joint_config or JointCalibrationConfig()
    base = joint_config.base

    excluded_batches = tuple(sorted(joint_config.excluded_batches))
    # Batch order must not affect the joint solution.
    active_batches = tuple(
        sorted(
            (batch for batch in calibration_input.batches if batch.batch_id not in joint_config.excluded_batches),
            key=lambda batch: batch.batch_id,
        )
    )
    active_input = CalibrationInput(calibration_input.dataset_id, active_batches)
    selected, excluded, used_references = _select_observations(active_input, base)

    weights = _joint_weights(selected, active_batches) if selected else None
    joint_result = _calibrate_core(
        selected, excluded, used_references, convention, version, base, weights
    )

    duplicate_groups = _duplicate_groups(active_batches)
    observation_weights = {
        reference: float(weight)
        for reference, weight in zip(used_references, weights if weights is not None else ())
    }

    per_batch: dict[str, CalibrationResult] = {}
    for batch in active_batches:
        single_input = CalibrationInput(calibration_input.dataset_id, (batch,))
        single_selected, single_excluded, single_used = _select_observations(single_input, base)
        per_batch[batch.batch_id] = _calibrate_core(
            single_selected, single_excluded, single_used, convention, version, base
        )

    consensus = _batch_consensus(per_batch, selected)
    compatibility = tuple(
        _assess_batch(batch, per_batch[batch.batch_id], joint_result, base, consensus)
        for batch in active_batches
    )

    notes: list[str] = []
    joint_rank_sufficient = joint_result.distribution in {"plane", "volume"} and joint_result.success
    for item in compatibility:
        if (
            item.standalone_distribution in {"point", "line"}
            and joint_rank_sufficient
        ):
            notes.append(
                f"batch '{item.batch_id}' is {item.standalone_distribution} alone; "
                "other batches supply the missing direction information in the joint fit"
            )
    for group in duplicate_groups:
        notes.append(
            "batches " + ", ".join(f"'{batch_id}'" for batch_id in group)
            + " are identical acquisitions; they share one acquisition weight"
        )
    for batch_id in excluded_batches:
        notes.append(f"batch '{batch_id}' excluded from the joint fit by configuration")
    if joint_result.distribution is not None:
        notes.append(f"merged geometry distribution: {joint_result.distribution}")

    effective_batch_count = float(
        len({_batch_fingerprint(batch) for batch in active_batches})
    )

    return JointCalibrationResult(
        joint=joint_result,
        per_batch=per_batch,
        compatibility=compatibility,
        observation_weights=observation_weights,
        duplicate_groups=duplicate_groups,
        excluded_batches=excluded_batches,
        identifiability_notes=tuple(notes),
        effective_batch_count=effective_batch_count,
    )


def _batch_consensus(
    per_batch: dict[str, CalibrationResult],
    selected: list[tuple[str, str, np.ndarray, np.ndarray, float | None]],
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Consensus of the standalone batch transforms.

    The joint fit is a compromise, so a batch that disagrees with it is not
    necessarily the outlier.  Conflicts are therefore judged against the
    componentwise median of the standalone translations and the projected
    mean rotation of the batches that can determine a transform alone.
    """
    transforms = [
        result.transform for result in per_batch.values() if result.transform is not None
    ]
    if not transforms:
        return None
    median_translation = np.median(
        np.vstack([transform.translation for transform in transforms]), axis=0
    )
    mean_rotation = np.mean([transform.rotation for transform in transforms], axis=0)
    u, _, vt = np.linalg.svd(mean_rotation)
    consensus_rotation = u @ vt
    if np.linalg.det(consensus_rotation) < 0:
        u[:, -1] *= -1
        consensus_rotation = u @ vt
    if selected:
        points_a = np.vstack([item[2] for item in selected])
        scatter = np.linalg.svd(points_a - points_a.mean(axis=0), compute_uv=False)
        reference_scale = float(scatter[0]) if len(scatter) else 0.0
    else:
        reference_scale = 0.0
    return median_translation, consensus_rotation, reference_scale


def _assess_batch(
    batch: ObservationBatch,
    standalone: CalibrationResult,
    joint: CalibrationResult,
    base: CalibrationConfig,
    consensus: tuple[np.ndarray, np.ndarray, float] | None,
) -> BatchCompatibility:
    used = [
        observation
        for observation in batch.observations
        if (batch.batch_id, observation.observation_id) in set(joint.used_observations)
    ]
    notes: list[str] = []
    if not used:
        return BatchCompatibility(
            batch_id=batch.batch_id,
            relation="insufficient_observations",
            used_observation_count=0,
            standalone_status=standalone.status,
            standalone_distribution=standalone.distribution,
            notes=("no observations from this batch are used in the joint fit",),
        )

    rotation_change = None
    translation_change = None
    inconsistency = None
    if standalone.transform is not None and consensus is not None:
        median_translation, consensus_rotation, reference_scale = consensus
        rotation_change = float(
            np.linalg.norm(standalone.transform.rotation - consensus_rotation, ord="fro")
        )
        translation_change = float(
            np.linalg.norm(standalone.transform.translation - median_translation)
        )
        inconsistency = translation_change + reference_scale * rotation_change

    sigmas = [observation.position_sigma for observation in used if observation.position_sigma]
    sigma_for_batch = (
        max(max(sigmas), joint.measurement_sigma or 0.0)
        if sigmas
        else (joint.measurement_sigma or None)
    )

    joint_rms = None
    residual_sigma_ratio = None
    if joint.transform is not None:
        points_a = np.vstack([observation.sensor_a_position for observation in used])
        points_b = np.vstack([observation.sensor_b_position for observation in used])
        residual_values = np.linalg.norm(joint.transform.apply(points_a) - points_b, axis=1)
        joint_rms = float(np.sqrt(np.mean(residual_values**2)))
        if sigma_for_batch:
            residual_sigma_ratio = joint_rms / sigma_for_batch

    if standalone.status == "undetermined" and joint.success:
        relation = "indeterminate_alone_covered_by_joint"
        notes.append("batch alone cannot determine a full transform; the joint fit can")
    elif (
        inconsistency is not None
        and sigma_for_batch is not None
        and inconsistency > base.residual_sigma_factor * sigma_for_batch
    ):
        relation = "conflicts_with_joint_solution"
        notes.append(
            "standalone transform of this batch disagrees with the consensus of "
            "the other batches beyond its reported measurement uncertainty"
        )
    else:
        relation = "supports_joint_solution"
    return BatchCompatibility(
        batch_id=batch.batch_id,
        relation=relation,
        used_observation_count=len(used),
        standalone_status=standalone.status,
        standalone_distribution=standalone.distribution,
        rotation_change=rotation_change,
        translation_change=translation_change,
        joint_rms_residual_on_batch=joint_rms,
        residual_sigma_ratio=residual_sigma_ratio,
        notes=tuple(notes),
    )


def _joint_judgment_basis(
    full: JointCalibrationResult,
    scenario_result: JointCalibrationResult,
    scenario: JointScenario,
    rotation_change: float | None,
    translation_change: float | None,
    residual_sigma_factor: float,
) -> tuple[str, ...]:
    def value_text(value: float | None) -> str:
        return "none" if value is None else f"{value:.12g}"

    basis = [
        f"full_joint_status:{full.joint.status}",
        f"scenario_joint_status:{scenario_result.joint.status}",
        f"full_joint_distribution:{full.joint.distribution}",
        f"scenario_joint_distribution:{scenario_result.joint.distribution}",
        f"excluded_batches:{len(scenario.excluded_batches)}",
        f"excluded_observations:{len(scenario.excluded_observations)}",
        f"full_effective_batch_count:{full.effective_batch_count:.12g}",
        f"scenario_effective_batch_count:{scenario_result.effective_batch_count:.12g}",
    ]
    if full.joint.success and not scenario_result.joint.success:
        basis.append("scenario_loses_joint_identifiability")
    if full.joint.distribution != scenario_result.joint.distribution:
        basis.append("merged_geometry_distribution_changes")
    if rotation_change is not None:
        basis.append(f"rotation_frobenius_change:{rotation_change:.12g}")
    if translation_change is not None:
        basis.append(f"translation_norm_change:{translation_change:.12g}")
    measurement_sigma = scenario_result.joint.measurement_sigma or full.joint.measurement_sigma or 0.0
    if translation_change is not None and measurement_sigma > 0:
        if translation_change > residual_sigma_factor * measurement_sigma:
            basis.append("transform_change_exceeds_reported_measurement_uncertainty")
        else:
            basis.append("transform_change_is_within_reported_measurement_uncertainty")
    for note in scenario_result.identifiability_notes:
        basis.append(f"scenario_note:{note}")
    return tuple(basis)


def compare_joint_scenarios(
    calibration_input: CalibrationInput,
    convention: CoordinateConvention,
    version: CalibrationVersion,
    scenarios: dict[str, JointScenario],
    joint_config: JointCalibrationConfig | None = None,
) -> tuple[JointCalibrationResult, tuple[JointScenarioComparison, ...]]:
    """Evaluate the full joint fit plus named exclusion scenarios.

    Source batches are never modified; every scenario is expressed as its
    own exclusion configuration, and each returned result keeps the
    per-observation (batch_id, observation_id) usage decisions.
    """
    joint_config = joint_config or JointCalibrationConfig()
    full = calibrate_joint(calibration_input, convention, version, joint_config)
    comparisons: list[JointScenarioComparison] = []
    for label, scenario in scenarios.items():
        scenario_config = JointCalibrationConfig(
            base=CalibrationConfig(
                min_points=joint_config.base.min_points,
                rank_tolerance=joint_config.base.rank_tolerance,
                near_degenerate_ratio=joint_config.base.near_degenerate_ratio,
                condition_limit=joint_config.base.condition_limit,
                motion_sigma_factor=joint_config.base.motion_sigma_factor,
                residual_sigma_factor=joint_config.base.residual_sigma_factor,
                influence_score_limit=joint_config.base.influence_score_limit,
                influence_relative_factor=joint_config.base.influence_relative_factor,
                exclude_marked_bad=joint_config.base.exclude_marked_bad,
                compute_influence=joint_config.base.compute_influence,
                additional_excluded_observations=frozenset(
                    set(joint_config.base.additional_excluded_observations)
                    | set(scenario.excluded_observations)
                ),
            ),
            excluded_batches=frozenset(
                set(joint_config.excluded_batches) | set(scenario.excluded_batches)
            ),
        )
        scenario_result = calibrate_joint(calibration_input, convention, version, scenario_config)
        rotation_change = None
        translation_change = None
        if full.joint.transform is not None and scenario_result.joint.transform is not None:
            rotation_change = float(
                np.linalg.norm(
                    scenario_result.joint.transform.rotation - full.joint.transform.rotation,
                    ord="fro",
                )
            )
            translation_change = float(
                np.linalg.norm(
                    scenario_result.joint.transform.translation - full.joint.transform.translation
                )
            )
        comparisons.append(
            JointScenarioComparison(
                label=label,
                scenario=scenario,
                result=scenario_result,
                rotation_change=rotation_change,
                translation_change=translation_change,
                judgment_basis=_joint_judgment_basis(
                    full,
                    scenario_result,
                    scenario,
                    rotation_change,
                    translation_change,
                    joint_config.base.residual_sigma_factor,
                ),
            )
        )
    return full, tuple(comparisons)
