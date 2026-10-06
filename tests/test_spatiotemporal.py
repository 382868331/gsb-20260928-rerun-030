from __future__ import annotations

import unittest

import numpy as np

from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
    SpatiotemporalScenario,
)
from sensor_calibration.spatiotemporal import (
    SpatiotemporalConfig,
    calibrate_spatiotemporal,
    compare_spatiotemporal_scenarios,
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


class SegmentJointCalibrationTest(unittest.TestCase):
    def test_shared_transform_independent_offsets_static_and_conflict_exclusion(self):
        curve_one = make_moving_batch(
            batch_id="curve-one",
            trajectory_a=curved_trajectory,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 3.0, 32),
            true_offset_seconds=-0.20,
            noise_sigma=SIGMA,
            seed=11,
        )
        curve_two = make_moving_batch(
            batch_id="curve-two",
            trajectory_a=curved_trajectory,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(8.0, 11.0, 32),
            true_offset_seconds=0.50,
            noise_sigma=SIGMA,
            seed=12,
        )
        static = make_moving_batch(
            batch_id="static-segment",
            trajectory_a=lambda t: np.array([1.4, -0.8, 0.6]),
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(0.0, 2.0, 24),
            true_offset_seconds=0.80,
            noise_sigma=SIGMA,
            seed=13,
        )
        conflict_source = make_moving_batch(
            batch_id="conflict-segment",
            trajectory_a=curved_trajectory,
            rotation=ROTATION,
            translation=TRANSLATION,
            sample_times=np.linspace(4.0, 7.0, 32),
            true_offset_seconds=0.10,
            noise_sigma=SIGMA,
            seed=14,
        )
        bias = np.array([0.05, -0.04, 0.03])
        conflict = ObservationBatch(
            conflict_source.batch_id,
            conflict_source.sensor_a_id,
            conflict_source.sensor_b_id,
            tuple(
                Observation(
                    observation_id=observation.observation_id,
                    target_id=observation.target_id,
                    sensor_a_position=observation.sensor_a_position,
                    sensor_b_position=observation.sensor_b_position + bias,
                    position_sigma=observation.position_sigma,
                    t_a=observation.t_a,
                    t_b=observation.t_b,
                    sensor_b_position_clean=observation.sensor_b_position_clean + bias,
                )
                for observation in conflict_source.observations
            ),
        )
        reset = make_reset_batch()
        dataset = CalibrationInput(
            "multi-segment",
            (curve_one, curve_two, static, conflict, reset),
        )

        old_mode = calibrate_spatiotemporal(
            dataset,
            CONVENTION,
            VERSION,
            SpatiotemporalConfig(mode="joint_offset"),
        )
        self.assertFalse(old_mode.success)
        self.assertIn("clock_reset_overlap_multiple_pairing_sources", old_mode.reasons)

        fast_segment_config = SpatiotemporalConfig(
            mode="segment_offsets",
            search_grid=41,
            refinement_rounds=1,
        )
        full = calibrate_spatiotemporal(dataset, CONVENTION, VERSION, fast_segment_config)
        segments = {item.segment_id: item for item in full.segment_offsets}
        self.assertIn("static-segment:segment-000:offset_unobservable_target_static", full.reasons)
        self.assertIn(
            "conflict-segment:segment-000:segment_residuals_inconsistent_with_reported_noise",
            full.reasons,
        )
        self.assertEqual(segments["static-segment:segment-000"].status, "unobservable_target_static")
        self.assertIsNone(segments["static-segment:segment-000"].offset_seconds)
        self.assertIsNotNone(segments["static-segment:segment-000"].candidate_offset_seconds)
        self.assertEqual(segments["conflict-segment:segment-000"].status, "rejected_inconsistent_pairings")

        config = SpatiotemporalConfig(
            mode="segment_offsets",
            search_grid=41,
            refinement_rounds=1,
        )
        base, comparisons = compare_spatiotemporal_scenarios(
            dataset,
            CONVENTION,
            VERSION,
            {
                "drop-static-and-conflict": SpatiotemporalScenario(
                    excluded_segments=frozenset({
                        "static-segment:segment-000",
                        "conflict-segment:segment-000",
                    })
                ),
                "select-two-curved-intervals": SpatiotemporalScenario(
                    included_segments=frozenset({
                        "curve-one:segment-000",
                        "reset-batch:segment-001",
                    })
                ),
            },
            config,
        )
        self.assertFalse(base.success)
        by_label = {item.label: item for item in comparisons}
        resolved = by_label["drop-static-and-conflict"].result
        selected = by_label["select-two-curved-intervals"].result

        self.assertTrue(resolved.success, resolved.reasons)
        assert resolved.transform is not None
        np.testing.assert_allclose(resolved.transform.rotation, ROTATION, atol=2e-3)
        np.testing.assert_allclose(resolved.transform.translation, TRANSLATION, atol=2e-3)
        resolved_segments = {item.segment_id: item for item in resolved.segment_offsets}
        self.assertAlmostEqual(resolved_segments["curve-one:segment-000"].offset_seconds, -0.20, places=2)
        self.assertAlmostEqual(resolved_segments["curve-two:segment-000"].offset_seconds, 0.50, places=2)
        self.assertAlmostEqual(resolved_segments["reset-batch:segment-000"].offset_seconds, 0.30, places=2)
        self.assertAlmostEqual(resolved_segments["reset-batch:segment-001"].offset_seconds, 2.30, places=2)
        self.assertTrue(
            any("segment_status:" in item for item in by_label["drop-static-and-conflict"].judgment_basis)
        )

        self.assertTrue(selected.success, selected.reasons)
        selected_segments = {item.segment_id: item for item in selected.segment_offsets}
        self.assertEqual(set(selected_segments), {
            "curve-one:segment-000",
            "reset-batch:segment-001",
        })
        self.assertAlmostEqual(
            selected_segments["reset-batch:segment-001"].offset_seconds, 2.30, places=2
        )


if __name__ == "__main__":
    unittest.main()
