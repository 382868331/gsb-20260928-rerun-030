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
    return ObservationBatch(
        "reset-batch", "virtual-sensor-a", "virtual-sensor-b", tuple(observations)
    )


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


class PerSegmentOffsetTest(unittest.TestCase):
    @staticmethod
    def _curved_batches():
        first = make_moving_batch(
            batch_id="curved-a",
            trajectory_a=curved_trajectory,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 4.0, 45),
            true_offset_seconds=0.3,
            noise_sigma=SIGMA,
            seed=11,
        )
        second = make_moving_batch(
            batch_id="curved-b",
            trajectory_a=lambda t: np.array([
                0.7 * np.cos(0.8 * t),
                1.2 * t + 0.3 * np.sin(1.1 * t),
                0.5 * np.sin(0.9 * t),
            ]),
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(7.0, 11.0, 45),
            true_offset_seconds=-0.2,
            noise_sigma=SIGMA,
            seed=12,
        )
        return first, second

    def test_shared_transform_with_independent_segment_offsets(self):
        first, second = self._curved_batches()
        result = calibrate_spatiotemporal(
            CalibrationInput("segments", (first, second)),
            CONVENTION,
            VERSION,
            SpatiotemporalConfig(mode="per_batch_offset"),
        )
        self.assertTrue(result.success)
        self.assertIsNotNone(result.transform)
        self.assertAlmostEqual(
            result.segment_time_offsets["curved-a"].offset_seconds, 0.3, places=1
        )
        self.assertAlmostEqual(
            result.segment_time_offsets["curved-b"].offset_seconds, -0.2, places=1
        )

    def test_static_or_reset_segment_cannot_be_masked_and_can_be_excluded(self):
        first, second = self._curved_batches()
        static = make_moving_batch(
            batch_id="static",
            trajectory_a=lambda t: np.array([2.0, -1.0, 0.5]),
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 4.0, 30),
            true_offset_seconds=0.7,
            noise_sigma=SIGMA,
            seed=13,
        )
        reset = make_reset_batch()
        dataset = CalibrationInput("mixed", (first, second, static, reset))
        full = calibrate_spatiotemporal(
            dataset,
            CONVENTION,
            VERSION,
            SpatiotemporalConfig(mode="per_batch_offset"),
        )
        self.assertFalse(full.success)
        self.assertIsNone(full.transform)
        self.assertIsNotNone(full.candidate_transform)
        self.assertIn("clock_reset_overlap_multiple_pairing_sources", full.reasons)
        self.assertIn("static:offset_unobservable_target_static", full.reasons)
        self.assertIsNone(full.segment_time_offsets["curved-a"].offset_seconds)
        self.assertIsNotNone(full.segment_time_offsets["curved-a"].candidate_offset_seconds)
        self.assertIsNone(full.segment_time_offsets["static"].offset_seconds)
        self.assertIsNotNone(full.segment_time_offsets["static"].candidate_offset_seconds)
        self.assertIsNone(full.segment_time_offsets["reset-batch"].offset_seconds)
        self.assertIsNotNone(full.segment_time_offsets["reset-batch"].candidate_offset_seconds)

        repaired = calibrate_spatiotemporal(
            dataset,
            CONVENTION,
            VERSION,
            SpatiotemporalConfig(
                mode="per_batch_offset",
                excluded_batches=frozenset({"static", "reset-batch"}),
            ),
        )
        self.assertTrue(repaired.success)
        self.assertIsNotNone(repaired.transform)
        np.testing.assert_allclose(repaired.transform.rotation, ROTATION, atol=0.01)
        np.testing.assert_allclose(repaired.transform.translation, TRANSLATION, atol=0.01)
        self.assertAlmostEqual(
            repaired.segment_time_offsets["curved-a"].offset_seconds, 0.3, places=1
        )
        self.assertAlmostEqual(
            repaired.segment_time_offsets["curved-b"].offset_seconds, -0.2, places=1
        )
if __name__ == "__main__":
    unittest.main()
