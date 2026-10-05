from __future__ import annotations

import unittest

import numpy as np

from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
)
from sensor_calibration.spatiotemporal import (
    SpatiotemporalConfig,
    calibrate_spatiotemporal,
)
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis


VERSION = CalibrationVersion("spatiotemporal", "1.0.0")
CONVENTION = CoordinateConvention("convention-v1", "frame-a", "frame-b", "meter")
ROTATION = rotation_about_axis(np.array([0.2, 1.0, 0.3]), 35.0)
TRANSLATION = np.array([0.4, -0.2, 0.7])
TRUE_OFFSET = 0.3
SIGMA = 1e-4


def curved_trajectory(t: float) -> np.ndarray:
    return np.array([
        np.cos(0.9 * t) + 0.2 * t,
        np.sin(1.3 * t),
        0.4 * np.sin(0.7 * t) + 0.1 * t,
    ])


def make_reset_batch(reset_to: float = 1.0, reset_at: float = 3.0) -> ObservationBatch:
    rng = np.random.default_rng(0)
    observations = []
    for index, t in enumerate(np.linspace(0.0, 6.0, 60)):
        position_a = curved_trajectory(float(t))
        clean_b = ROTATION @ position_a + TRANSLATION
        noisy_b = clean_b + rng.normal(0.0, SIGMA, 3)
        if t < reset_at:
            t_b = t - TRUE_OFFSET
        else:
            t_b = reset_to + (t - reset_at) - TRUE_OFFSET
        observations.append(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=position_a,
                sensor_b_position=noisy_b,
                position_sigma=SIGMA,
                t_a=float(t),
                t_b=float(t_b),
                sensor_b_position_clean=clean_b,
            )
        )
    return ObservationBatch("reset-batch", "sensor-a", "sensor-b", tuple(observations))


def calibrate(batch: ObservationBatch):
    return calibrate_spatiotemporal(
        CalibrationInput("dataset", (batch,)),
        CONVENTION,
        VERSION,
        SpatiotemporalConfig(),
    )


class CurvedTrajectoryTest(unittest.TestCase):
    def test_offset_and_transform_are_confirmed(self):
        batch = make_moving_batch(
            batch_id="curved",
            trajectory_a=curved_trajectory,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 6.0, 60),
            true_offset_seconds=TRUE_OFFSET,
            noise_sigma=SIGMA,
            seed=1,
        )
        result = calibrate(batch)
        self.assertTrue(result.success)
        self.assertEqual(result.time_offset.status, "identifiable")
        self.assertAlmostEqual(result.time_offset.offset_seconds, TRUE_OFFSET, places=2)
        self.assertIsNotNone(result.transform)
        self.assertIsNone(result.candidate_transform)
        self.assertEqual(result.reasons, ())


class StaticTrajectoryTest(unittest.TestCase):
    def test_static_target_keeps_specific_failure_reason(self):
        batch = make_moving_batch(
            batch_id="static",
            trajectory_a=lambda t: np.array([1.0, 2.0, 3.0]),
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 6.0, 60),
            true_offset_seconds=TRUE_OFFSET,
            noise_sigma=SIGMA,
            seed=1,
        )
        result = calibrate(batch)
        self.assertFalse(result.success)
        self.assertEqual(result.time_offset.status, "unobservable_target_static")
        self.assertIsNone(result.time_offset.offset_seconds)
        self.assertIn("offset_unobservable_target_static", result.reasons)


class ConstantVelocityTest(unittest.TestCase):
    def test_straight_constant_motion_is_not_a_confirmed_estimate(self):
        velocity = np.array([0.5, -0.2, 0.3])
        batch = make_moving_batch(
            batch_id="constant-velocity",
            trajectory_a=lambda t: np.array([1.0, 2.0, 3.0]) + t * velocity,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 6.0, 60),
            true_offset_seconds=TRUE_OFFSET,
            noise_sigma=SIGMA,
            seed=1,
        )
        result = calibrate(batch)
        self.assertFalse(result.success)
        self.assertEqual(result.time_offset.status, "confounded_with_motion")
        self.assertIsNone(result.time_offset.offset_seconds)
        self.assertIsNone(result.time_offset.sigma_seconds)
        self.assertIsNotNone(result.time_offset.candidate_offset_seconds)
        self.assertIn("offset_confounded_with_straight_constant_velocity", result.reasons)


class MissingWindowTest(unittest.TestCase):
    def test_gap_is_recorded_and_never_bridged(self):
        batch = make_moving_batch(
            batch_id="gap",
            trajectory_a=curved_trajectory,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 6.0, 60),
            true_offset_seconds=TRUE_OFFSET,
            noise_sigma=SIGMA,
            seed=1,
            b_gap_units=((2.0, 4.0),),
        )
        result = calibrate(batch)
        self.assertTrue(result.success)
        gaps = result.selection[0].gaps_b_clock
        self.assertEqual(len(gaps), 1)
        self.assertEqual(result.as_dict()["selection"][0]["gaps_b_clock"], [list(gaps[0])])


class ClockResetTest(unittest.TestCase):
    def test_reset_overlap_rejects_candidate_and_keeps_evidence(self):
        result = calibrate(make_reset_batch())
        self.assertFalse(result.success)
        self.assertIn("clock_reset_overlap_multiple_pairing_sources", result.reasons)
        self.assertNotEqual(result.time_offset.status, "identifiable")
        self.assertIsNone(result.time_offset.offset_seconds)
        self.assertIsNone(result.time_offset.sigma_seconds)
        # The fitted values stay available as clearly separated diagnostics.
        self.assertIsNotNone(result.time_offset.candidate_offset_seconds)
        self.assertIsNone(result.transform)
        self.assertIsNotNone(result.candidate_transform)
        # Ambiguous observations stay traceable after serialization.
        selection = result.as_dict()["selection"][0]
        self.assertTrue(selection["ambiguous_b_observations"])
        self.assertTrue(result.as_dict()["reasons"])

    def test_full_overlap_leaves_no_pairing_source(self):
        result = calibrate(make_reset_batch(reset_to=0.0))
        self.assertFalse(result.success)
        self.assertIn("clock_reset_overlap_multiple_pairing_sources", result.reasons)
        self.assertNotEqual(result.time_offset.status, "identifiable")
        self.assertIsNone(result.time_offset.offset_seconds)


if __name__ == "__main__":
    unittest.main()
