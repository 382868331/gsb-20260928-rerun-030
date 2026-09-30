from __future__ import annotations

import unittest

import numpy as np

from sensor_calibration.calibration import CalibrationConfig, calibrate, calibrate_subsets
from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
)
from sensor_calibration.synthetic import make_noisy_batch, rotation_about_axis


VERSION = CalibrationVersion("paired-point-kabsch", "1.0.0")
CONVENTION = CoordinateConvention("convention-v1", "a", "b", "meter")
ROTATION = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 18.0)
TRANSLATION = np.array([0.35, -0.12, 0.08])
POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.8, 0.7, 0.8],
])


class CalibrationTests(unittest.TestCase):
    def test_normal_noisy_case_recovers_transform(self):
        batch = make_noisy_batch(
            rotation=ROTATION,
            translation=TRANSLATION,
            target_positions_a=POINTS,
            noise_sigma=0.001,
            seed=7,
        )
        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertTrue(result.success)
        self.assertEqual(result.status, "valid")
        self.assertIsNotNone(result.transform)
        assert result.transform is not None
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=0.005)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=0.005)
        self.assertLess(result.rms_residual, 0.005)
        self.assertEqual(len(result.used_observations), 8)

    def test_marked_bad_observations_are_excluded_without_mutating_batch(self):
        batch = make_noisy_batch(
            rotation=ROTATION,
            translation=TRANSLATION,
            target_positions_a=POINTS,
            noise_sigma=0.001,
            seed=8,
            bad_indices=(2, 4),
            outlier_sigma=0.7,
        )
        calibration_input = CalibrationInput("dataset", (batch,))

        polluted = calibrate(
            calibration_input,
            CONVENTION,
            VERSION,
            CalibrationConfig(exclude_marked_bad=False),
        )
        repaired = calibrate(calibration_input, CONVENTION, VERSION)

        self.assertTrue(polluted.success)
        self.assertTrue(repaired.success)
        self.assertGreater(np.linalg.norm(polluted.transform.translation - TRANSLATION), 0.05)
        np.testing.assert_allclose(repaired.transform.rotation, ROTATION, atol=0.01)
        np.testing.assert_allclose(repaired.transform.translation, TRANSLATION, atol=0.01)
        self.assertEqual(len(repaired.used_observations), 6)
        self.assertEqual(
            repaired.excluded_observations,
            ((batch.batch_id, "obs-002"), (batch.batch_id, "obs-004")),
        )
        self.assertEqual(len(batch.observations), 8)
        self.assertEqual(batch.observations[2].quality, "bad")

    def test_additional_exclusion_does_not_require_editing_source_batch(self):
        observations = []
        for index, point in enumerate(POINTS):
            observations.append(
                Observation(
                    observation_id=f"obs-{index}",
                    target_id=f"target-{index}",
                    sensor_a_position=point,
                    sensor_b_position=ROTATION @ point + TRANSLATION,
                    quality="good",
                )
            )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", tuple(observations))
        calibration_input = CalibrationInput("dataset", (batch,))
        config = CalibrationConfig(
            additional_excluded_observations=frozenset({("batch", "obs-2")})
        )

        result = calibrate(calibration_input, CONVENTION, VERSION, config)

        self.assertTrue(result.success)
        self.assertEqual(result.excluded_observations, (("batch", "obs-2"),))
        self.assertEqual(batch.observations[2].quality, "good")

    def test_planar_points_spanning_a_plane_recover_a_verifiable_transform(self):
        planar_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.4, 0.8, 0.0],
            [0.9, 0.2, 0.0],
        ])
        batch = make_noisy_batch(
            rotation=ROTATION,
            translation=TRANSLATION,
            target_positions_a=planar_points,
            noise_sigma=0.001,
            seed=11,
        )

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.geometric_rank, 2)
        self.assertEqual(result.effective_rank, 2)
        assert result.transform is not None
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=0.005)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=0.005)

    def test_noiseless_planar_quadrilateral_recovers_exact_transform(self):
        planar_points = np.array([
            [0.5, -0.5, 0.2],
            [1.5, -0.5, -0.8],
            [0.5, 1.5, 0.2],
            [1.5, 1.5, -0.8],
        ])
        observations = tuple(
            Observation(
                observation_id=f"obs-{index}",
                target_id=f"target-{index}",
                sensor_a_position=point,
                sensor_b_position=ROTATION @ point + TRANSLATION,
            )
            for index, point in enumerate(planar_points)
        )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", observations)

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.geometric_rank, 2)
        assert result.transform is not None
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=1e-10)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=1e-10)
        self.assertLess(result.rms_residual, 1e-12)

    def test_collinear_points_are_still_rejected(self):
        positions = np.outer(np.linspace(-1.0, 1.0, 6), np.array([0.3, 0.7, 0.1]))
        observations = tuple(
            Observation(
                observation_id=f"obs-{index}",
                target_id=f"target-{index}",
                sensor_a_position=point,
                sensor_b_position=ROTATION @ point + TRANSLATION,
            )
            for index, point in enumerate(positions)
        )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", observations)

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertFalse(result.success)
        self.assertEqual(result.status, "undetermined")
        self.assertIsNone(result.transform)
        self.assertEqual(result.geometric_rank, 1)
        self.assertIn(
            "target_positions_are_collinear_or_nearly_collinear",
            result.reasons,
        )

    def test_collapsed_points_are_rejected(self):
        point = np.array([0.4, -0.2, 0.9])
        observations = tuple(
            Observation(
                observation_id=f"obs-{index}",
                target_id=f"target-{index}",
                sensor_a_position=point,
                sensor_b_position=ROTATION @ point + TRANSLATION,
            )
            for index in range(4)
        )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", observations)

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertFalse(result.success)
        self.assertEqual(result.status, "undetermined")
        self.assertEqual(result.geometric_rank, 0)
        self.assertIn("target_positions_collapse_to_a_point", result.reasons)

    def test_nearly_collinear_noisy_motion_is_reported_as_undetermined(self):
        positions = np.outer(np.linspace(0.0, 0.001, 6), np.array([1.0, 0.0, 0.0]))
        batch = make_noisy_batch(
            rotation=ROTATION,
            translation=TRANSLATION,
            target_positions_a=positions,
            noise_sigma=0.003,
            seed=9,
        )

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertFalse(result.success)
        self.assertEqual(result.status, "undetermined")
        self.assertIsNone(result.transform)
        self.assertIn("target_motion_is_small_relative_to_measurement_uncertainty", result.reasons)

    def test_single_wrong_correspondence_is_flagged_as_unstable_and_influential(self):
        batch = make_noisy_batch(
            rotation=ROTATION,
            translation=TRANSLATION,
            target_positions_a=POINTS,
            noise_sigma=0.001,
            seed=12,
        )
        bad_observation = Observation(
            observation_id=batch.observations[3].observation_id,
            target_id=batch.observations[3].target_id,
            sensor_a_position=batch.observations[3].sensor_a_position,
            sensor_b_position=batch.observations[3].sensor_b_position + np.array([0.6, -0.4, 0.5]),
            quality="good",
            position_sigma=0.001,
        )
        observations = (
            batch.observations[:3] + (bad_observation,) + batch.observations[4:]
        )
        polluted_batch = ObservationBatch(batch.batch_id, batch.sensor_a_id, batch.sensor_b_id, observations)

        result = calibrate(CalibrationInput("dataset", (polluted_batch,)), CONVENTION, VERSION)

        self.assertTrue(result.success)
        self.assertEqual(result.status, "unstable")
        self.assertIn((polluted_batch.batch_id, "obs-003"), result.warning_observations)
        self.assertTrue(result.influential_observations)
        self.assertIn(
            (polluted_batch.batch_id, "obs-003"),
            result.influential_observations,
        )

    def test_subset_comparison_keeps_original_batch_and_baseline(self):
        batch = make_noisy_batch(
            rotation=ROTATION,
            translation=TRANSLATION,
            target_positions_a=POINTS,
            noise_sigma=0.001,
            seed=13,
        )
        bad_observation = Observation(
            observation_id=batch.observations[5].observation_id,
            target_id=batch.observations[5].target_id,
            sensor_a_position=batch.observations[5].sensor_a_position,
            sensor_b_position=batch.observations[5].sensor_b_position + np.array([0.5, 0.5, -0.4]),
            quality="good",
            position_sigma=0.001,
        )
        observations = (
            batch.observations[:5] + (bad_observation,) + batch.observations[6:]
        )
        polluted_batch = ObservationBatch(batch.batch_id, batch.sensor_a_id, batch.sensor_b_id, observations)
        calibration_input = CalibrationInput("dataset", (polluted_batch,))
        removed = frozenset({(polluted_batch.batch_id, "obs-005")})

        results = calibrate_subsets(
            calibration_input,
            CONVENTION,
            VERSION,
            {"baseline": frozenset(), "without_obs_005": removed},
        )

        self.assertEqual(results["baseline"].status, "unstable")
        self.assertEqual(results["without_obs_005"].status, "valid")
        self.assertEqual(
            results["without_obs_005"].excluded_observations,
            ((polluted_batch.batch_id, "obs-005"),),
        )
        self.assertEqual(len(results["baseline"].used_observations), 8)
        self.assertEqual(len(results["without_obs_005"].used_observations), 7)
        self.assertEqual(len(polluted_batch.observations), 8)
        self.assertEqual(polluted_batch.observations[5].quality, "good")
        np.testing.assert_allclose(
            results["without_obs_005"].transform.rotation, ROTATION, atol=0.01
        )

    def test_too_few_observations_is_reported_as_undetermined(self):
        observations = tuple(
            Observation(
                observation_id=f"obs-{index}",
                target_id=f"target-{index}",
                sensor_a_position=point,
                sensor_b_position=ROTATION @ point + TRANSLATION,
                position_sigma=0.001,
            )
            for index, point in enumerate(POINTS[:2])
        )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", observations)

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertFalse(result.success)
        self.assertIsNone(result.transform)
        self.assertEqual(result.reasons, ("at_least_3_non_excluded_observations_required",))


if __name__ == "__main__":
    unittest.main()
