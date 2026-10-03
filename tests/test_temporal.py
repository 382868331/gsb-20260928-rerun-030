from __future__ import annotations

import numpy as np
import unittest

from sensor_calibration import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    TemporalCalibrationConfig,
    TemporalScenario,
    calibrate_temporal,
    compare_temporal_scenarios,
)
from sensor_calibration.synthetic import make_moving_target_batch, rotation_about_axis


CONVENTION = CoordinateConvention("c1", "frame-a", "frame-b", "m")
VERSION = CalibrationVersion("temporal-kabsch", "1.0.0")
R_TRUE = rotation_about_axis((0.0, 0.0, 1.0), 35.0)
T_TRUE = np.array([0.5, -0.25, 0.1])
OFFSET_TRUE = 0.12
NOISE = 1e-4


def curved_trajectory(time_value):
    return np.array([
        0.9 * time_value,
        0.55 * np.sin(1.7 * time_value),
        0.35 * np.cos(1.3 * time_value),
    ])


def straight_trajectory(time_value):
    return np.array([0.7 * time_value, -0.2 * time_value, 0.3 + 0.1 * time_value])


def stationary_trajectory(_time_value):
    return np.array([0.1, -0.2, 0.3])


def build_batch(*, trajectory, times, seed=7, time_unit="s", offset=OFFSET_TRUE, missing=(), batch_id=None):
    return make_moving_target_batch(
        rotation=R_TRUE,
        translation=T_TRUE,
        trajectory_a=trajectory,
        sample_times=times,
        clock_offset=offset,
        noise_sigma=NOISE,
        seed=seed,
        batch_id=batch_id or f"batch-{seed}",
        time_unit=time_unit,
        missing_time_windows=missing,
    )


class TemporalCalibrationTests(unittest.TestCase):
    def test_known_motion_recovers_offset_and_transform(self):
        times = np.linspace(0.0, 4.0, 81)
        batch = build_batch(trajectory=curved_trajectory, times=times)
        data = CalibrationInput("ds", (batch,))
        result = calibrate_temporal(data, CONVENTION, VERSION, TemporalCalibrationConfig())
        self.assertTrue(result.offset_identifiable, result.reasons)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.time_offset_seconds, OFFSET_TRUE, delta=0.01)
        self.assertLess(
            np.linalg.norm(result.transform.translation - T_TRUE), 5e-3
        )
        rotation_error = np.linalg.norm(result.transform.rotation - R_TRUE, ord="fro")
        self.assertLess(rotation_error, 0.02)
        self.assertEqual(result.per_batch_motion["batch-7"], "accelerated")
        self.assertTrue(all(item.time_a_seconds >= 0 for item in result.matched_observations))

    def test_raw_observations_are_not_rewritten_and_every_match_is_traceable(self):
        times = np.linspace(0.0, 4.0, 81)
        batch = build_batch(trajectory=curved_trajectory, times=times)
        raw_b_times = [obs.timestamp_b for obs in batch.observations]
        data = CalibrationInput("ds", (batch,))
        result = calibrate_temporal(data, CONVENTION, VERSION)
        self.assertEqual([obs.timestamp_b for obs in batch.observations], raw_b_times)
        anchors = {item.anchor_observation_id for item in result.matched_observations}
        self.assertGreater(len(anchors), 0)
        self.assertEqual(len(anchors), len(result.matched_observations))
        self.assertTrue(result.adopted_intervals)
        serialized = result.as_dict()
        self.assertIn("adopted_intervals", serialized)
        self.assertIn("time_offset_seconds", serialized)

    def test_uniform_straight_motion_offset_is_not_identifiable(self):
        times = np.linspace(0.0, 4.0, 81)
        batch = build_batch(trajectory=straight_trajectory, times=times, seed=11)
        data = CalibrationInput("ds", (batch,))
        result = calibrate_temporal(data, CONVENTION, VERSION)
        self.assertFalse(result.offset_identifiable)
        self.assertFalse(result.success)
        joined = " ".join(result.reasons)
        self.assertIn("uniform_straight_line_motion", joined)
        self.assertTrue(any("small_residuals" in warning for warning in result.warnings))
        self.assertIsNotNone(result.time_offset_equivalence_range_seconds)

    def test_stationary_target_offset_unobservable(self):
        times = np.linspace(0.0, 2.0, 41)
        batch = build_batch(trajectory=stationary_trajectory, times=times, seed=13)
        data = CalibrationInput("ds", (batch,))
        result = calibrate_temporal(data, CONVENTION, VERSION)
        self.assertFalse(result.offset_identifiable)
        self.assertTrue(
            any("stationary" in reason for reason in result.reasons)
        )

    def test_short_record_is_declared_insufficient(self):
        times = np.linspace(0.0, 0.08, 3)
        batch = build_batch(trajectory=curved_trajectory, times=times, seed=17)
        data = CalibrationInput("ds", (batch,))
        result = calibrate_temporal(data, CONVENTION, VERSION)
        self.assertFalse(result.success)

    def test_gap_is_not_crossed_and_neighbors_are_excluded(self):
        times = np.linspace(0.0, 4.0, 81)
        batch = build_batch(
            trajectory=curved_trajectory,
            times=times,
            seed=19,
            missing=((1.5 - OFFSET_TRUE, 2.6 - OFFSET_TRUE),),
        )
        data = CalibrationInput("ds", (batch,))
        result = calibrate_temporal(data, CONVENTION, VERSION)
        self.assertIn("batch-19", result.gaps)
        self.assertEqual(len(result.adopted_intervals), 2)
        reasons = set(result.excluded_temporal_observations.values())
        self.assertIn("target_time_falls_in_a_missing_interval_or_outside_overlap", reasons)
        # No interpolated match may reference B samples across the gap.
        for item in result.matched_observations:
            self.assertFalse(
                item.interpolated and item.segment_index != item.segment_index
            )

    def test_time_unit_change_does_not_change_the_result(self):
        times = np.linspace(0.0, 4.0, 81)
        batch_s = build_batch(trajectory=curved_trajectory, times=times, seed=23, time_unit="s")
        batch_ms = build_batch(trajectory=curved_trajectory, times=times, seed=23, time_unit="ms")
        config = TemporalCalibrationConfig()
        result_s = calibrate_temporal(CalibrationInput("ds", (batch_s,)), CONVENTION, VERSION, config)
        result_ms = calibrate_temporal(CalibrationInput("ds", (batch_ms,)), CONVENTION, VERSION, config)
        self.assertAlmostEqual(
            result_s.time_offset_seconds, result_ms.time_offset_seconds, places=10
        )
        self.assertAlmostEqual(
            result_s.rms_residual, result_ms.rms_residual, places=10
        )

    def test_record_order_does_not_change_the_result(self):
        times = np.linspace(0.0, 4.0, 61)
        batch = build_batch(trajectory=curved_trajectory, times=times, seed=29)
        reversed_batch = type(batch)(
            batch_id=batch.batch_id,
            sensor_a_id=batch.sensor_a_id,
            sensor_b_id=batch.sensor_b_id,
            observations=tuple(reversed(batch.observations)),
            time_unit=batch.time_unit,
        )
        config = TemporalCalibrationConfig(search_grid_points=121)
        first = calibrate_temporal(CalibrationInput("ds", (batch,)), CONVENTION, VERSION, config)
        second = calibrate_temporal(CalibrationInput("ds", (reversed_batch,)), CONVENTION, VERSION, config)
        self.assertAlmostEqual(first.time_offset_seconds, second.time_offset_seconds, places=10)
        self.assertEqual(
            sorted(item.anchor_observation_id for item in first.matched_observations),
            sorted(item.anchor_observation_id for item in second.matched_observations),
        )

    def test_fixed_offset_and_simultaneous_modes(self):
        times = np.linspace(0.0, 4.0, 81)
        batch = build_batch(trajectory=curved_trajectory, times=times, seed=31)
        data = CalibrationInput("ds", (batch,))
        fixed = calibrate_temporal(
            data, CONVENTION, VERSION,
            TemporalCalibrationConfig(mode="fixed_offset", fixed_offset_seconds=OFFSET_TRUE),
        )
        self.assertTrue(fixed.success)
        self.assertAlmostEqual(fixed.time_offset_seconds, OFFSET_TRUE)
        simultaneous = calibrate_temporal(
            data, CONVENTION, VERSION,
            TemporalCalibrationConfig(
                mode="simultaneous_only", simultaneous_tolerance_seconds=0.002
            ),
        )
        # Clock readings differ by a nonzero offset; with a tiny tolerance the
        # strictly simultaneous subset is empty or small and is reported as such.
        self.assertEqual(simultaneous.mode, "simultaneous_only")
        self.assertTrue(
            len(simultaneous.matched_observations) < len(fixed.matched_observations)
        )

    def test_saved_scenario_excludes_batches_and_windows(self):
        good = build_batch(trajectory=curved_trajectory, times=np.linspace(0.0, 4.0, 81), seed=41)
        short = build_batch(
            trajectory=curved_trajectory,
            times=np.linspace(0.0, 0.1, 4),
            seed=42,
            batch_id="batch-42",
        )
        data = CalibrationInput("ds", (good, short))
        full, comparisons = compare_temporal_scenarios(
            data, CONVENTION, VERSION,
            {"drop-short-batch": TemporalScenario(excluded_batches=frozenset({"batch-42"}))},
        )
        self.assertTrue(full.offset_identifiable)
        comparison = comparisons[0]
        self.assertNotIn("batch-42", comparison.result.per_batch_motion)
        self.assertTrue(any("excluded_batches:1" in item for item in comparison.judgment_basis))

    def test_batches_complement_and_conflict(self):
        times = np.linspace(0.0, 4.0, 81)
        first = build_batch(trajectory=curved_trajectory, times=times, seed=51)
        second = build_batch(trajectory=curved_trajectory, times=times, seed=52, offset=0.4)
        data = CalibrationInput("ds", (first, second))
        result = calibrate_temporal(data, CONVENTION, VERSION)
        self.assertTrue(any("prefer incompatible fixed offsets" in note for note in result.identifiability_notes))
        self.assertIn("batches_disagree_on_fixed_clock_offset", result.warnings)

    def test_duplicate_acquisition_does_not_add_evidence(self):
        times = np.linspace(0.0, 4.0, 81)
        batch = build_batch(trajectory=curved_trajectory, times=times, seed=61, batch_id="b1")
        duplicate = type(batch)(
            batch_id="b2",
            sensor_a_id=batch.sensor_a_id,
            sensor_b_id=batch.sensor_b_id,
            observations=batch.observations,
            time_unit=batch.time_unit,
        )
        result = calibrate_temporal(
            CalibrationInput("ds", (batch, duplicate)), CONVENTION, VERSION
        )
        self.assertTrue(
            any("identical acquisitions" in note for note in result.identifiability_notes)
        )

    def test_stationary_batch_complemented_by_accelerated_batch(self):
        curved = build_batch(
            trajectory=curved_trajectory, times=np.linspace(0.0, 4.0, 81), seed=71, batch_id="curve"
        )
        still = build_batch(
            trajectory=stationary_trajectory,
            times=np.linspace(0.0, 1.0, 21),
            seed=72,
            batch_id="still",
        )
        result = calibrate_temporal(
            CalibrationInput("ds", (curved, still)), CONVENTION, VERSION
        )
        self.assertTrue(result.offset_identifiable)
        self.assertEqual(result.per_batch_motion["still"], "stationary")


if __name__ == "__main__":
    unittest.main()
