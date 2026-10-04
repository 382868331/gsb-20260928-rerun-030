from __future__ import annotations
import json
import unittest

import numpy as np

from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    ObservationBatch,
)
from sensor_calibration.spatiotemporal import (
    SpatiotemporalConfig,
    _b_segments,
    _build_pairs,
    _interpolate_segment,
    calibrate_spatiotemporal,
)
from sensor_calibration.synthetic import make_moving_batch, rotation_about_axis


VERSION = CalibrationVersion("spacetime", "2.1.0")
CONVENTION = CoordinateConvention("spacetime-v1", "a", "b", "meter")
ROTATION = rotation_about_axis(np.array([0.2, 0.7, 0.7]), 28.0)
TRANSLATION = np.array([0.42, -0.31, 0.18])
TIMES = np.linspace(0.0, 4.0, 41)


def curved_trajectory(time_seconds: float) -> np.ndarray:
    return np.array(
        [
            time_seconds,
            0.65 * np.sin(0.9 * time_seconds),
            0.4 * time_seconds + 0.28 * np.cos(0.55 * time_seconds),
        ]
    )


def straight_trajectory(time_seconds: float) -> np.ndarray:
    return np.array([0.8 * time_seconds, -0.35 * time_seconds, 0.2 * time_seconds])


def static_trajectory(_time_seconds: float) -> np.ndarray:
    return np.array([1.2, -0.7, 0.4])


def moving_batch(*, batch_id: str, trajectory=curved_trajectory, true_offset: float = 0.18, **kwargs) -> ObservationBatch:
    return make_moving_batch(
        batch_id=batch_id,
        trajectory_a=trajectory,
        rotation=ROTATION,
        translation=TRANSLATION,
        sample_times=TIMES,
        true_offset_seconds=true_offset,
        noise_sigma=1e-6,
        seed=17,
        **kwargs,
    )


class SpatiotemporalTests(unittest.TestCase):
    def test_curved_trajectory_recovers_transform_and_0_18s_offset(self):
        batch = moving_batch(batch_id="curved")
        config = SpatiotemporalConfig(search_half_range_seconds=0.5, search_grid=301)

        result = calibrate_spatiotemporal(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION, config)

        self.assertTrue(result.success)
        self.assertEqual(result.time_offset.status, "identifiable")
        self.assertAlmostEqual(result.time_offset.offset_seconds, 0.18, delta=0.002)
        self.assertLess(np.linalg.norm(result.transform.rotation - ROTATION, ord="fro"), 2e-3)
        self.assertLess(np.linalg.norm(result.transform.translation - TRANSLATION), 2e-3)

    def test_static_and_constant_velocity_targets_are_not_reported_as_success(self):
        config = SpatiotemporalConfig(search_half_range_seconds=0.5)

        static = calibrate_spatiotemporal(
            CalibrationInput("dataset", (moving_batch(batch_id="static", trajectory=static_trajectory),)),
            CONVENTION,
            VERSION,
            config,
        )
        straight = calibrate_spatiotemporal(
            CalibrationInput("dataset", (moving_batch(batch_id="straight", trajectory=straight_trajectory),)),
            CONVENTION,
            VERSION,
            config,
        )

        self.assertFalse(static.success)
        self.assertEqual(static.time_offset.status, "unobservable_target_static")
        self.assertIsNone(static.time_offset.offset_seconds)
        self.assertFalse(straight.success)
        self.assertEqual(straight.time_offset.status, "confounded_with_motion")
        self.assertIsNone(straight.time_offset.sigma_seconds)
        self.assertEqual(straight.spatial_distribution, "line")

    def test_build_pairs_uses_windows_and_segment_specific_offsets(self):
        batch = moving_batch(batch_id="curved")
        config = SpatiotemporalConfig(
            excluded_a_windows={"curved": ((2.0, 2.1), (3.9, 4.0))}
        )
        segments, gaps = _b_segments(batch, config.gap_factor)
        segments_data = [_interpolate_segment(segment, batch.seconds_per_unit) for segment in segments]

        pairs, omitted, b_ranges, pair_sources = _build_pairs(
            batch, config, segments_data, ((2.0, 2.1), (3.9, 4.0))
        )

        self.assertEqual(tuple(gaps), ())
        self.assertEqual(set(omitted), {"obs-020", "obs-021", "obs-039", "obs-040"})
        self.assertTrue(all(reason == "excluded_time_window" for reason in omitted.values()))
        self.assertEqual(len(pairs), 37)
        self.assertEqual(pair_sources["obs-000"], (0,))
        self.assertEqual(b_ranges, [(-0.18, 3.82)])
        first = next(pair for pair in pairs if pair.observation_id == "obs-000")
        self.assertEqual(first.segment_index, 0)
        self.assertAlmostEqual(first.offset_lo_seconds, 0.0 - 3.82)
        self.assertAlmostEqual(first.offset_hi_seconds, 0.0 - (-0.18))

    def test_missing_b_period_is_never_bridged_and_selection_is_traceable(self):
        batch = moving_batch(
            batch_id="gapped",
            trajectory=curved_trajectory,
            b_gap_units=((1.55, 2.55),),
        )
        config = SpatiotemporalConfig(search_half_range_seconds=0.5)

        result = calibrate_spatiotemporal(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION, config)
        selection = result.selection[0]

        self.assertTrue(result.success)
        self.assertEqual(len(batch.observations), 41)
        self.assertEqual(len(selection.gaps_b_clock), 1)
        self.assertAlmostEqual(selection.gaps_b_clock[0][0], 1.52)
        self.assertAlmostEqual(selection.gaps_b_clock[0][1], 2.62)
        self.assertEqual(selection.paired_count, 29)
        self.assertEqual(
            set(selection.omitted_reasons.values()),
            {"outside_contiguous_segment_at_chosen_offset"},
        )
        self.assertEqual(len(selection.omitted_reasons), 12)
        for interval in selection.used_intervals_b_clock:
            self.assertFalse(interval.start <= 1.52 and interval.end >= 2.62)
        encoded = json.loads(json.dumps(result.as_dict()))
        self.assertEqual(len(encoded["selection"][0]["gaps_b_clock"]), 1)
        self.assertAlmostEqual(encoded["selection"][0]["gaps_b_clock"][0][0], 1.52)
        self.assertAlmostEqual(encoded["selection"][0]["gaps_b_clock"][0][1], 2.62)
        self.assertEqual(encoded["selection"][0]["time_unit"], "second")
        self.assertLessEqual(
            min(interval["start"] for interval in encoded["selection"][0]["used_intervals_a_clock"]),
            0.0,
        )

    def test_clock_reset_with_two_pair_sources_is_rejected(self):
        base = moving_batch(batch_id="reset", trajectory=curved_trajectory)
        observations = []
        for observation in base.observations:
            if observation.t_a >= 2.0:
                reset_time_b = observation.t_b - 0.5
            else:
                reset_time_b = observation.t_b
            observations.append(
                type(observation)(
                    observation_id=observation.observation_id,
                    target_id=observation.target_id,
                    sensor_a_position=observation.sensor_a_position,
                    sensor_b_position=observation.sensor_b_position,
                    timestamp=observation.timestamp,
                    quality=observation.quality,
                    position_sigma=observation.position_sigma,
                    t_a=observation.t_a,
                    t_b=reset_time_b,
                    time_sigma=observation.time_sigma,
                    sensor_b_position_clean=observation.sensor_b_position_clean,
                )
            )
        batch = ObservationBatch("reset", base.sensor_a_id, base.sensor_b_id, tuple(observations))
        config = SpatiotemporalConfig(search_half_range_seconds=0.5)

        result = calibrate_spatiotemporal(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION, config)

        self.assertFalse(result.success)
        self.assertIsNone(result.transform)
        self.assertIn("multiple_pair_sources", result.reasons[0])
        reasons = set(result.selection[0].omitted_reasons.values())
        self.assertIn("matches_multiple_clock_segments_at_chosen_offset", reasons)


if __name__ == "__main__":
    unittest.main()
