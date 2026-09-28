from __future__ import annotations

import unittest

import numpy as np

from sensor_calibration.calibration import CalibrationConfig, calibrate
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

    def test_insufficient_motion_is_reported_as_undetermined(self):
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
        self.assertIsNone(result.transform)
        self.assertIn("target_positions_do_not_span_three_dimensions", result.reasons)
        self.assertIn("target_motion_is_small_relative_to_measurement_uncertainty", result.reasons)

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
