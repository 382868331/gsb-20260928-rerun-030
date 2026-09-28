import numpy as np

from sensor_calibration import (
    ObservationBatch,
    PoseTransform,
    calibrate,
)  # fmt: skip
from sensor_calibration.models import CoordinateConvention
from sensor_calibration.synthetic import make_observations, rotation_axis_angle


CONVENTION = CoordinateConvention(
    convention_id="coord-v1",
    frame_a="a",
    frame_b="b",
    length_unit="meter",
)

GROUND_TRUTH = PoseTransform(
    rotation=rotation_axis_angle(np.array([0.2, -0.4, 0.9]), 0.37),
    translation=np.array([0.4, -0.8, 1.2]),
)

POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.8, 0.3, 0.2],
        [0.2, 0.9, 0.7],
    ]
)


def batch(observations, batch_id: str) -> ObservationBatch:
    return ObservationBatch(
        batch_id=batch_id,
        sensor_a_id="a-device",
        sensor_b_id="b-device",
        convention_id=CONVENTION.convention_id,
        observations=tuple(observations),
    )


def test_normal_noisy_sample_is_determined():
    source = batch(make_observations(POINTS, GROUND_TRUTH, 0.002, seed=1), "normal")
    result = calibrate("v1", source, CONVENTION, noise_scale=0.002)

    assert result.determined
    assert result.rank == 3
    assert np.allclose(result.transform.rotation, GROUND_TRUTH.rotation, atol=0.02)
    assert np.allclose(result.transform.translation, GROUND_TRUTH.translation, atol=0.02)
    assert result.rms_error < 0.01


def test_marked_bad_observations_are_excluded_without_mutating_batch():
    observations = make_observations(
        POINTS,
        GROUND_TRUTH,
        0.002,
        seed=2,
        bad_indices=(1, 4),
        outlier_sigma=3.0,
    )
    source = batch(observations, "outliers")
    source_hash = source.hash

    result = calibrate("v2", source, CONVENTION, parent_version_id="v1", noise_scale=0.002)

    assert result.version.excluded_observation_ids == ("obs-001", "obs-004")
    assert result.version.parent_version_id == "v1"
    assert source.hash == source_hash
    assert np.allclose(result.transform.rotation, GROUND_TRUTH.rotation, atol=0.02)
    assert np.allclose(result.transform.translation, GROUND_TRUTH.translation, atol=0.02)


def test_stationary_target_is_reported_as_underdetermined():
    stationary = np.repeat(np.array([[0.5, 0.5, 0.5]]), len(POINTS), axis=0)
    source = batch(make_observations(stationary, GROUND_TRUTH, 0.002, seed=3), "stationary")
    result = calibrate("v3", source, CONVENTION, noise_scale=0.002)

    assert not result.determined
    assert result.transform is None
    assert result.rank < 3
    assert result.reasons


def test_planar_and_collinear_samples_are_reported_as_underdetermined():
    planar = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.4, 0.0],
            [0.2, 0.9, 0.0],
            [0.6, 0.6, 0.0],
        ]
    )
    collinear = np.linspace([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 6)

    for sample in (planar, collinear):
        source = batch(make_observations(sample, GROUND_TRUTH, 0.0, seed=4), "degenerate")
        result = calibrate("v4", source, CONVENTION)
        assert not result.determined
        assert result.transform is None
        assert result.rank < 3


def test_calibration_version_records_input_hashes():
    source = batch(make_observations(POINTS, GROUND_TRUTH, 0.001, seed=5), "provenance")
    result = calibrate("v5", source, CONVENTION, noise_scale=0.001, notes="unit test")

    assert result.version.batch_id == "provenance"
    assert result.version.batch_hash == source.hash
    assert result.version.convention_hash == CONVENTION.hash
    assert len(result.version.included_observation_ids) == len(POINTS)
