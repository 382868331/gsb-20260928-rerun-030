import json

import numpy as np

from sensor_calibration import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
    SpatiotemporalConfig,
    calibrate_spatiotemporal,
    compare_spatiotemporal_approaches,
)
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis


CONVENTION = CoordinateConvention("c", "a", "b", "meter")
VERSION = CalibrationVersion("test", "1")
ROTATION = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 30.0)
TRANSLATION = np.array([0.2, -0.3, 0.5])


def _input(batch):
    return CalibrationInput("dataset", (batch,))


def _curved_batch(batch_id="curved", true_offset_seconds=0.35, **kwargs):
    return make_moving_batch(
        batch_id=batch_id,
        trajectory_a=lambda t: np.array([t, 0.08 * t * t, 0.0]),
        rotation=ROTATION,
        translation=TRANSLATION,
        sample_times=np.linspace(0.0, 5.0, 10),
        true_offset_seconds=true_offset_seconds,
        noise_sigma=1e-5,
        seed=3,
        **kwargs,
    )


def _reset_batch():
    observations = []
    for index, physical_time in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
        b_clock = physical_time if physical_time < 4.0 else physical_time - 4.0
        position = np.array([physical_time, physical_time * physical_time, 0.0])
        observations.append(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=position,
                sensor_b_position=position,
                position_sigma=1e-6,
                t_a=physical_time,
                t_b=b_clock,
                sensor_b_position_clean=position,
            )
        )
    return ObservationBatch("reset", "a", "b", tuple(observations))


def test_normal_nonlinear_trajectory_remains_precise():
    result = calibrate_spatiotemporal(_input(_curved_batch()), CONVENTION, VERSION)

    assert result.success
    assert result.time_offset.valid
    assert result.spatial_valid
    assert result.time_offset.status == "identifiable"
    assert result.transform is not None
    assert abs(result.time_offset.offset_seconds - 0.35) < 2e-3
    np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=1e-4)
    np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=2e-3)
    assert result.rms_residual < 1e-3


def test_clock_reset_candidates_are_diagnostic_only_and_auditable():
    result = calibrate_spatiotemporal(_input(_reset_batch()), CONVENTION, VERSION)
    serialized = result.as_dict()
    reopened = json.loads(json.dumps(serialized))

    assert not result.success
    assert result.transform is None
    assert result.used_observations == ()
    assert result.time_offset.offset_seconds is None
    assert result.time_offset.sigma_seconds is None
    assert not result.time_offset.valid
    assert result.time_offset.status == "ambiguous_pairing_sources"
    assert "clock_reset_overlap_creates_multiple_pairing_sources" in result.reasons
    assert result.candidate_transform is not None
    assert result.candidate_rms_residual is not None
    assert result.candidate_used_observations
    selection = result.selection[0]
    assert selection.reset_overlaps_b_clock == ((0.0, 3.0),)
    assert selection.ambiguous_pairings
    assert all(
        candidate.reason == "clock_reset_overlap_multiple_pairing_sources"
            or candidate.selected
        for candidate in selection.ambiguous_pairings
    )
    assert serialized["transform"] is None
    assert serialized["candidate_transform"] is not None
    assert serialized["selection"][0]["ambiguous_pairings"]
    assert reopened["success"] is False
    assert reopened["transform"] is None
    assert reopened["candidate_used_observations"]
    assert reopened["selection"][0]["reset_overlaps_b_clock"] == [[0.0, 3.0]]
    assert reopened["selection"][0]["ambiguous_pairings"]
    assert reopened["selection"][0]["omitted_reasons"]


def test_stationary_and_constant_velocity_do_not_look_like_offset_estimates():
    static = make_moving_batch(
        batch_id="static",
        trajectory_a=lambda t: np.array([1.0, 2.0, 0.0]),
        rotation=ROTATION,
        translation=TRANSLATION,
        sample_times=np.linspace(0.0, 4.0, 8),
        noise_sigma=1e-5,
        seed=1,
    )
    straight = make_moving_batch(
        batch_id="straight",
        trajectory_a=lambda t: np.array([t, 0.5 * t, 0.0]),
        rotation=ROTATION,
        translation=TRANSLATION,
        sample_times=np.linspace(0.0, 4.0, 8),
        noise_sigma=1e-5,
        seed=2,
    )

    static_result = calibrate_spatiotemporal(_input(static), CONVENTION, VERSION)
    straight_result = calibrate_spatiotemporal(_input(straight), CONVENTION, VERSION)

    assert not static_result.success
    assert static_result.time_offset.status == "unobservable_target_static"
    assert not static_result.time_offset.valid
    assert static_result.time_offset.offset_seconds is None
    assert static_result.time_offset.candidate_offset_seconds is not None
    assert "offset_not_identifiable_target_is_stationary" in static_result.reasons

    assert not straight_result.success
    assert straight_result.time_offset.status == "confounded_with_motion"
    assert not straight_result.time_offset.valid
    assert straight_result.time_offset.offset_seconds is None
    assert straight_result.time_offset.sigma_seconds is None
    assert straight_result.time_offset.feasible_offset_range_seconds is not None
    assert "offset_not_identifiable_straight_constant_velocity_motion" in straight_result.reasons


def test_missing_window_is_reported_by_correspondable_only_public_entry():
    batch = make_moving_batch(
        batch_id="missing",
        trajectory_a=lambda t: np.array([t, 0.2 * t * t, 0.0]),
        rotation=ROTATION,
        translation=TRANSLATION,
        sample_times=np.linspace(0.0, 4.0, 7),
        true_offset_seconds=0.0,
        noise_sigma=1e-5,
        seed=4,
        b_gap_units=((0.75, 3.25),),
    )
    _joint, comparisons = compare_spatiotemporal_approaches(
        _input(batch), CONVENTION, VERSION, fixed_offset_seconds=1.0,
        config=SpatiotemporalConfig(gap_factor=2.0),
    )
    overlap = next(item for item in comparisons if item.label == "fixed_offset@1s").result

    assert not overlap.success
    assert overlap.transform is None
    assert overlap.used_observations == ()
    selection = overlap.selection[0]
    assert selection.missing_windows_b_clock == ((0.6666666666666666, 3.333333333333333),)
    assert any(
        reason == "missing_b_window_blocks_pairing_at_chosen_offset"
        for reason in selection.omitted_reasons.values()
    )
    assert "chosen_offset_requires_interpolating_across_missing_b_window" in overlap.reasons
