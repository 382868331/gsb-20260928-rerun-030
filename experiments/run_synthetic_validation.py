from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sensor_calibration.calibration import (
    CalibrationConfig,
    calibrate,
    compare_observation_subsets,
)
from sensor_calibration.joint import calibrate_joint, compare_joint_scenarios
from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    JointScenario,
    Observation,
    ObservationBatch,
    SpatiotemporalScenario,
)
from sensor_calibration.synthetic import make_noisy_batch, rotation_about_axis
from sensor_calibration.spatiotemporal import SpatiotemporalConfig, calibrate_spatiotemporal, compare_spatiotemporal_scenarios


VERSION = CalibrationVersion(
    method_name="paired-point-kabsch",
    method_version="1.1.0",
    parameters={
        "distance_metric": "euclidean",
        "rotation_parameterization": "rotation_matrix",
        "geometry_ranks": "point_or_line_undetermined_plane_or_volume_estimable",
    },
)
CONVENTION = CoordinateConvention(
    convention_id="metric-right-handed-v1",
    frame_a="virtual_sensor_a",
    frame_b="virtual_sensor_b",
    length_unit="meter",
    description="Same target observations expressed independently in each sensor frame",
)


def compare_transform(result, expected_rotation, expected_translation):
    assert result.success, result.reasons
    assert result.transform is not None
    rotation_error = np.linalg.norm(result.transform.rotation - expected_rotation, ord="fro")
    translation_error = np.linalg.norm(result.transform.translation - expected_translation)
    return rotation_error, translation_error


def make_reset_moving_batch(expected_rotation, expected_translation):
    def trajectory(time_seconds):
        return np.array([
            np.cos(0.9 * time_seconds) + 0.2 * time_seconds,
            np.sin(1.3 * time_seconds),
            0.4 * np.sin(0.7 * time_seconds) + 0.1 * time_seconds,
        ])

    rng = np.random.default_rng(901)
    observations = []
    for index, time_seconds in enumerate(np.linspace(0.0, 6.0, 60)):
        position_a = trajectory(float(time_seconds))
        clean_b = expected_rotation @ position_a + expected_translation
        if time_seconds < 3.0:
            time_b = time_seconds - 0.30
        else:
            time_b = 1.0 + (time_seconds - 3.0) - 0.30
        observations.append(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=position_a,
                sensor_b_position=clean_b + rng.normal(0.0, 1e-4, 3),
                position_sigma=1e-4,
                t_a=float(time_seconds),
                t_b=float(time_b),
                sensor_b_position_clean=clean_b,
            )
        )
    return ObservationBatch("reset-motion", "virtual-sensor-a", "virtual-sensor-b", tuple(observations))


def main() -> int:
    expected_rotation = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 18.0)
    expected_translation = np.array([0.35, -0.12, 0.08])
    well_spread = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.02, 0.01],
        [0.05, 1.0, -0.03],
        [0.02, -0.04, 1.0],
        [1.05, 0.95, 0.08],
        [0.9, -0.05, 0.95],
        [-0.08, 0.92, 0.9],
        [0.75, 0.72, 0.8],
    ])

    normal_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread,
        noise_sigma=0.003,
        seed=101,
    )
    normal_result = calibrate(
        CalibrationInput("normal-synthetic-dataset", (normal_batch,)), CONVENTION, VERSION
    )
    normal_rotation_error, normal_translation_error = compare_transform(
        normal_result, expected_rotation, expected_translation
    )
    assert normal_result.status == "valid"
    assert normal_result.distribution == "volume"
    assert normal_rotation_error < 0.01
    assert normal_translation_error < 0.01
    assert (normal_result.rms_residual or 0.0) < 0.01

    planar_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.05, 0.0],
        [0.1, 1.0, 0.0],
        [0.9, 0.85, 0.0],
        [-0.2, 0.6, 0.0],
        [0.5, -0.25, 0.0],
    ])
    planar_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=planar_positions,
        noise_sigma=0.002,
        seed=104,
    )
    planar_result = calibrate(
        CalibrationInput("planar-synthetic-dataset", (planar_batch,)), CONVENTION, VERSION
    )
    planar_rotation_error, planar_translation_error = compare_transform(
        planar_result, expected_rotation, expected_translation
    )
    assert planar_result.status == "valid", planar_result.warnings
    assert planar_result.distribution == "plane"
    assert planar_rotation_error < 0.01
    assert planar_translation_error < 0.01

    near_line_positions = np.column_stack([
        np.linspace(0.0, 1.0, 7),
        np.array([0.0, 0.00002, -0.000015, 0.00003, -0.00002, 0.000025, -0.00001]),
        np.zeros(7),
    ])
    near_line_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=near_line_positions,
        noise_sigma=1e-7,
        seed=105,
    )
    near_line_result = calibrate(
        CalibrationInput("near-collinear-dataset", (near_line_batch,)), CONVENTION, VERSION
    )
    near_line_rotation_error, near_line_translation_error = compare_transform(
        near_line_result, expected_rotation, expected_translation
    )
    assert near_line_result.success
    assert near_line_result.status == "unstable"
    assert "planar_configuration_is_nearly_collinear" in near_line_result.warnings
    assert near_line_rotation_error < 0.02
    assert near_line_translation_error < 0.01

    outlier_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread,
        noise_sigma=0.003,
        seed=102,
        bad_indices=(3, 6),
        outlier_sigma=0.5,
    )
    outlier_input = CalibrationInput("outlier-synthetic-dataset", (outlier_batch,))
    polluted_result = calibrate(
        outlier_input, CONVENTION, VERSION, CalibrationConfig(exclude_marked_bad=False)
    )
    polluted_rotation_error, polluted_translation_error = compare_transform(
        polluted_result, expected_rotation, expected_translation
    )
    assert polluted_result.status == "unstable"
    assert polluted_rotation_error > 0.02 or polluted_translation_error > 0.02
    top_influential = polluted_result.influential_observations
    assert (outlier_batch.batch_id, "obs-003") in top_influential
    assert (outlier_batch.batch_id, "obs-006") in top_influential

    full_result, subset_comparisons = compare_observation_subsets(
        outlier_input,
        CONVENTION,
        VERSION,
        {
            "drop-first-bad": {(outlier_batch.batch_id, "obs-003")},
            "drop-both-bad": {
                (outlier_batch.batch_id, "obs-003"),
                (outlier_batch.batch_id, "obs-006"),
            },
        },
        CalibrationConfig(exclude_marked_bad=False),
    )
    comparison_by_label = {item.label: item for item in subset_comparisons}
    drop_both = comparison_by_label["drop-both-bad"]
    fixed_rotation_error, fixed_translation_error = compare_transform(
        drop_both.result, expected_rotation, expected_translation
    )
    assert drop_both.result.status == "valid"
    assert fixed_rotation_error < 0.01
    assert fixed_translation_error < 0.01
    assert len(outlier_batch.observations) == len(normal_batch.observations)
    assert full_result.used_observations == tuple(
        (outlier_batch.batch_id, f"obs-{index:03d}") for index in range(8)
    )

    recalculated_result = calibrate(outlier_input, CONVENTION, VERSION)
    fixed_marked_rotation_error, fixed_marked_translation_error = compare_transform(
        recalculated_result, expected_rotation, expected_translation
    )
    assert fixed_marked_rotation_error < 0.01
    assert fixed_marked_translation_error < 0.01
    assert recalculated_result.excluded_observations == (
        (outlier_batch.batch_id, "obs-003"),
        (outlier_batch.batch_id, "obs-006"),
    )

    collinear_positions = np.outer(np.linspace(0.0, 1.0, 8), np.array([1.0, 0.2, 0.0]))
    collinear_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=collinear_positions,
        noise_sigma=0.003,
        seed=106,
    )
    collinear_result = calibrate(
        CalibrationInput("collinear-dataset", (collinear_batch,)), CONVENTION, VERSION
    )
    assert not collinear_result.success
    assert collinear_result.status == "undetermined"
    assert collinear_result.transform is None
    assert (
        "target_positions_are_collinear_rotation_about_line_is_unobservable"
        in collinear_result.reasons
    )

    point_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=np.zeros((6, 3)),
        noise_sigma=0.003,
        seed=107,
    )
    point_result = calibrate(
        CalibrationInput("point-dataset", (point_batch,)), CONVENTION, VERSION
    )
    assert not point_result.success
    assert point_result.status == "undetermined"
    assert "target_positions_concentrate_at_one_point" in point_result.reasons

    base_direction = np.array([1.0, 0.2, 0.0])
    base_direction /= np.linalg.norm(base_direction)
    short_positions = np.outer(np.linspace(0.0, 0.002, 8), base_direction)
    insufficient_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=short_positions,
        noise_sigma=0.003,
        seed=103,
    )
    insufficient_result = calibrate(
        CalibrationInput("insufficient-motion-dataset", (insufficient_batch,)), CONVENTION, VERSION
    )
    assert not insufficient_result.success
    assert insufficient_result.transform is None
    assert "target_motion_is_small_relative_to_measurement_uncertainty" in insufficient_result.reasons

    # 7. Joint multi-batch evaluation: a collinear batch is undetermined
    # alone, but the joint fit borrows direction information from a
    # well-spread batch; a biased batch is flagged as conflicting; a
    # duplicated acquisition shares one weight instead of adding evidence.
    joint_collinear = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=collinear_positions,
        noise_sigma=0.001,
        seed=201,
    )
    joint_volume = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread,
        noise_sigma=0.001,
        seed=202,
    )
    biased_positions = well_spread + 0.0
    biased_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation + np.array([0.05, 0.0, 0.0]),
        target_positions_a=biased_positions,
        noise_sigma=0.001,
        seed=203,
    )
    second_volume = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread + np.array([0.3, -0.2, 0.1]),
        noise_sigma=0.001,
        seed=204,
    )
    joint_input = CalibrationInput(
        "joint-dataset", (joint_collinear, joint_volume, second_volume, biased_batch)
    )
    joint_full, joint_comparisons = compare_joint_scenarios(
        joint_input,
        CONVENTION,
        VERSION,
        {
            "exclude-biased-batch": JointScenario(
                excluded_batches=frozenset({biased_batch.batch_id})
            ),
            "exclude-collinear-batch": JointScenario(
                excluded_batches=frozenset({joint_collinear.batch_id})
            ),
        },
    )
    assert joint_full.joint.success
    joint_relations = {item.batch_id: item.relation for item in joint_full.compatibility}
    assert joint_relations[joint_collinear.batch_id] == "indeterminate_alone_covered_by_joint"
    assert joint_relations[biased_batch.batch_id] == "conflicts_with_joint_solution"
    assert joint_relations[joint_volume.batch_id] == "supports_joint_solution"
    assert joint_relations[second_volume.batch_id] == "supports_joint_solution"

    duplicate_batch = ObservationBatch(
        "joint-volume-copy",
        joint_volume.sensor_a_id,
        joint_volume.sensor_b_id,
        joint_volume.observations,
    )
    duplicated = calibrate_joint(
        CalibrationInput("duplicated-dataset", (joint_volume, duplicate_batch)),
        CONVENTION,
        VERSION,
    )
    assert duplicated.effective_batch_count == 1.0
    single = calibrate_joint(
        CalibrationInput("single-dataset", (joint_volume,)), CONVENTION, VERSION
    )
    assert single.joint.transform is not None and duplicated.joint.transform is not None
    np.testing.assert_allclose(
        duplicated.joint.transform.translation, single.joint.transform.translation, atol=1e-12
    )

    temporal_rotation = rotation_about_axis(np.array([0.2, 1.0, 0.3]), 35.0)
    temporal_translation = np.array([0.4, -0.2, 0.7])
    temporal_curve = lambda time_seconds: np.array([
        np.cos(0.9 * time_seconds) + 0.2 * time_seconds,
        np.sin(1.3 * time_seconds),
        0.4 * np.sin(0.7 * time_seconds) + 0.1 * time_seconds,
    ])
    reset_motion = make_reset_moving_batch(temporal_rotation, temporal_translation)
    temporal_input = CalibrationInput("multi-segment-temporal", (reset_motion,))
    old_temporal = calibrate_spatiotemporal(
        temporal_input,
        CONVENTION,
        VERSION,
        SpatiotemporalConfig(search_grid=41, refinement_rounds=1),
    )
    assert not old_temporal.success
    assert "clock_reset_overlap_multiple_pairing_sources" in old_temporal.reasons
    segmented_full, segment_scenarios = compare_spatiotemporal_scenarios(
        temporal_input,
        CONVENTION,
        VERSION,
        {},
        SpatiotemporalConfig(mode="segment_offsets", search_grid=41, refinement_rounds=1),
    )
    assert segmented_full.success, segmented_full.reasons
    assert segmented_full.transform is not None
    assert np.linalg.norm(segmented_full.transform.rotation - temporal_rotation) < 0.002
    assert np.linalg.norm(segmented_full.transform.translation - temporal_translation) < 0.002
    assert len(segmented_full.segment_offsets) == 2
    assert abs(segmented_full.segment_offsets[0].offset_seconds - 0.30) < 0.02
    assert abs(segmented_full.segment_offsets[1].offset_seconds - 2.30) < 0.02

    report = {
        "normal": {
            **normal_result.as_dict(),
            "rotation_frobenius_error": normal_rotation_error,
            "translation_error": normal_translation_error,
        },
        "planar": {
            **planar_result.as_dict(),
            "rotation_frobenius_error": planar_rotation_error,
            "translation_error": planar_translation_error,
        },
        "near_collinear_unstable": {
            **near_line_result.as_dict(),
            "rotation_frobenius_error": near_line_rotation_error,
            "translation_error": near_line_translation_error,
        },
        "outliers_full_batch": {
            **polluted_result.as_dict(),
            "rotation_frobenius_error": polluted_rotation_error,
            "translation_error": polluted_translation_error,
        },
        "outlier_subset_comparisons": [
            {
                "label": item.label,
                "additional_excluded_observations": [list(ref) for ref in item.additional_excluded_observations],
                "status": item.result.status,
                "rotation_change": item.rotation_change,
                "translation_change": item.translation_change,
                "judgment_basis": list(item.judgment_basis),
                "result": item.result.as_dict(),
            }
            for item in subset_comparisons
        ],
        "collinear_undetermined": collinear_result.as_dict(),
        "point_undetermined": point_result.as_dict(),
        "insufficient_motion": insufficient_result.as_dict(),
        "joint_multi_batch": joint_full.as_dict(),
        "joint_scenario_comparisons": [item.as_dict() for item in joint_comparisons],
        "joint_duplicate_acquisition": duplicated.as_dict(),
        "temporal_old_global_offset_rejected": old_temporal.as_dict(),
        "temporal_segment_offsets": segmented_full.as_dict(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
