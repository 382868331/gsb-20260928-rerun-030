from __future__ import annotations

import unittest

import numpy as np

from sensor_calibration.calibration import (
    CalibrationConfig,
    calibrate,
    compare_observation_subsets,
)
from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
)
from sensor_calibration.synthetic import make_noisy_batch, rotation_about_axis


VERSION = CalibrationVersion("paired-point-kabsch", "1.1.0")
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
PLANAR_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.9, 0.8, 0.0],
    [0.4, -0.6, 0.0],
    [-0.5, 0.3, 0.0],
])


def _clean_batch(points: np.ndarray, batch_id: str = "batch") -> ObservationBatch:
    observations = tuple(
        Observation(
            observation_id=f"obs-{index:03d}",
            target_id=f"target-{index:03d}",
            sensor_a_position=point,
            sensor_b_position=ROTATION @ point + TRANSLATION,
            position_sigma=1e-6,
        )
        for index, point in enumerate(points)
    )
    return ObservationBatch(batch_id, "sensor-a", "sensor-b", observations)


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
        self.assertEqual(result.distribution, "volume")
        assert result.transform is not None
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=0.005)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=0.005)
        self.assertLess(result.rms_residual, 0.005)
        self.assertEqual(len(result.used_observations), 8)

    def test_well_spread_planar_points_recover_rigid_transform(self):
        result = calibrate(
            CalibrationInput("dataset", (_clean_batch(PLANAR_POINTS),)),
            CONVENTION,
            VERSION,
        )

        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.distribution, "plane")
        assert result.transform is not None
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=1e-8)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=1e-8)
        self.assertLess(result.rms_residual, 1e-8)

    def test_planar_points_without_noise_remain_valid_not_unstable(self):
        result = calibrate(
            CalibrationInput("dataset", (_clean_batch(PLANAR_POINTS),)),
            CONVENTION,
            VERSION,
        )

        self.assertEqual(result.status, "valid")
        self.assertNotIn(
            "planar_configuration_rotation_about_plane_normal_has_low_leverage",
            result.warnings,
        )

    def test_nearly_collinear_planar_sampling_is_unstable_not_undetermined(self):
        points = np.column_stack(
            [
                np.linspace(0.0, 1.0, 6),
                np.array([0.0, 0.00002, -0.000015, 0.00003, -0.000025, 0.00001]),
                np.zeros(6),
            ]
        )
        result = calibrate(
            CalibrationInput("dataset", (_clean_batch(points),)),
            CONVENTION,
            VERSION,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, "unstable")
        self.assertEqual(result.distribution, "plane")
        self.assertIn("planar_configuration_is_nearly_collinear", result.warnings)
        assert result.transform is not None
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=1e-6)

    def test_strictly_collinear_points_are_undetermined(self):
        points = np.column_stack([np.linspace(0.0, 1.0, 6), np.zeros(6), np.zeros(6)])
        result = calibrate(
            CalibrationInput("dataset", (_clean_batch(points),)),
            CONVENTION,
            VERSION,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "undetermined")
        self.assertIsNone(result.transform)
        self.assertEqual(result.distribution, "line")
        self.assertIn(
            "target_positions_are_collinear_rotation_about_line_is_unobservable",
            result.reasons,
        )

    def test_points_concentrated_at_one_location_are_undetermined(self):
        points = np.tile(np.array([0.4, -0.2, 0.7]), (5, 1))
        result = calibrate(
            CalibrationInput("dataset", (_clean_batch(points),)),
            CONVENTION,
            VERSION,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.status, "undetermined")
        self.assertIsNone(result.transform)
        self.assertEqual(result.distribution, "point")
        self.assertIn("target_positions_concentrate_at_one_point", result.reasons)

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
        self.assertEqual(polluted.status, "unstable")
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
        batch = _clean_batch(POINTS, batch_id="batch")
        calibration_input = CalibrationInput("dataset", (batch,))
        config = CalibrationConfig(
            additional_excluded_observations=frozenset({("batch", "obs-002")})
        )

        result = calibrate(calibration_input, CONVENTION, VERSION, config)

        self.assertTrue(result.success)
        self.assertEqual(result.excluded_observations, (("batch", "obs-002"),))
        self.assertEqual(batch.observations[2].quality, "good")

    def test_subset_comparison_flags_influential_observations_and_preserves_batch(self):
        observations = []
        for index, point in enumerate(POINTS):
            sensor_b = ROTATION @ point + TRANSLATION
            if index == 5:
                sensor_b = sensor_b + np.array([0.45, -0.35, 0.2])
            observations.append(
                Observation(
                    observation_id=f"obs-{index:03d}",
                    target_id=f"target-{index:03d}",
                    sensor_a_position=point,
                    sensor_b_position=sensor_b,
                    quality="good",
                    position_sigma=0.001,
                )
            )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", tuple(observations))
        calibration_input = CalibrationInput("dataset", (batch,))

        full_result, comparisons = compare_observation_subsets(
            calibration_input,
            CONVENTION,
            VERSION,
            {"drop-suspect": {("batch", "obs-005")}},
        )

        self.assertEqual(full_result.status, "unstable")
        self.assertIn(("batch", "obs-005"), full_result.influential_observations)
        self.assertIn(("batch", "obs-005"), full_result.influence)
        self.assertEqual(len(comparisons), 1)
        comparison = comparisons[0]
        self.assertEqual(comparison.label, "drop-suspect")
        self.assertEqual(
            comparison.additional_excluded_observations,
            (("batch", "obs-005"),),
        )
        self.assertEqual(comparison.result.status, "valid")
        self.assertGreater(comparison.rotation_change, 0.0)
        self.assertGreater(comparison.translation_change, 0.0)
        np.testing.assert_allclose(comparison.result.transform.rotation, ROTATION, atol=1e-6)
        np.testing.assert_allclose(comparison.result.transform.translation, TRANSLATION, atol=1e-6)
        self.assertEqual(len(batch.observations), 8)
        self.assertEqual(full_result.used_observations[-3], ("batch", "obs-005"))
        self.assertEqual(
            full_result.excluded_observations,
            (),
        )

    def test_small_motion_relative_to_noise_is_undetermined(self):
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
        self.assertIn("target_motion_is_small_relative_to_measurement_uncertainty", result.reasons)

    def test_too_few_observations_is_reported_as_undetermined(self):
        batch = _clean_batch(POINTS[:2])

        result = calibrate(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)

        self.assertFalse(result.success)
        self.assertIsNone(result.transform)
        self.assertEqual(result.reasons, ("at_least_3_non_excluded_observations_required",))


    def test_tilted_well_spread_plane_is_valid_not_near_degenerate(self):
        normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        axis_u = np.array([1.0, -1.0, 0.0])
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        offsets = [(0, 0), (1, 0), (0, 1), (0.9, 0.8), (0.4, -0.6), (-0.5, 0.3)]
        points = np.array([a * axis_u + b * axis_v for a, b in offsets])
        rotation = rotation_about_axis(np.array([1.0, 2.0, 3.0]), 40.0)
        observations = tuple(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=point,
                sensor_b_position=rotation @ point + TRANSLATION,
                position_sigma=1e-6,
            )
            for index, point in enumerate(points)
        )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", observations)

        result = calibrate(
            CalibrationInput("dataset", (batch,)), CONVENTION, VERSION
        )

        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.distribution, "plane")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.warnings, ())
        np.testing.assert_allclose(result.transform.rotation, rotation, atol=1e-8)

    def test_sampling_suggestions_name_missing_spatial_distribution(self):
        planar = calibrate(
            CalibrationInput("dataset", (_clean_batch(PLANAR_POINTS),)),
            CONVENTION,
            VERSION,
        )
        self.assertEqual(
            planar.sampling_suggestions,
            ("add_off_plane_observations_to_cover_the_missing_third_dimension",),
        )

        collinear_points = np.column_stack(
            [np.linspace(0.0, 1.0, 6), np.zeros(6), np.zeros(6)]
        )
        collinear = calibrate(
            CalibrationInput("dataset", (_clean_batch(collinear_points),)),
            CONVENTION,
            VERSION,
        )
        self.assertIn(
            "add_observations_off_the_dominant_line_to_span_a_plane",
            collinear.sampling_suggestions,
        )

        concentrated = calibrate(
            CalibrationInput(
                "dataset",
                (_clean_batch(np.tile(np.array([0.1, 0.2, 0.3]), (5, 1))),),
            ),
            CONVENTION,
            VERSION,
        )
        self.assertIn(
            "add_observations_spread_across_multiple_directions_around_the_target",
            concentrated.sampling_suggestions,
        )

    def test_status_is_consistent_under_global_similarity_transform(self):
        base_result = calibrate(
            CalibrationInput("dataset", (_clean_batch(PLANAR_POINTS),)),
            CONVENTION,
            VERSION,
        )

        global_rotation = rotation_about_axis(np.array([0.3, -0.7, 0.2]), 65.0)
        global_translation = np.array([12.0, -4.5, 7.25])
        global_scale = 37.5
        transformed_points = (
            global_scale * (global_rotation @ PLANAR_POINTS.T).T + global_translation
        )
        observations = tuple(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=point_a,
                sensor_b_position=ROTATION @ point_a + TRANSLATION,
                position_sigma=1e-6 * global_scale,
            )
            for index, point_a in enumerate(transformed_points)
        )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", observations)
        moved_result = calibrate(
            CalibrationInput("dataset", (batch,)), CONVENTION, VERSION
        )

        self.assertEqual(moved_result.status, base_result.status)
        self.assertEqual(moved_result.distribution, base_result.distribution)
        self.assertEqual(moved_result.warnings, base_result.warnings)
        self.assertEqual(
            moved_result.sampling_suggestions, base_result.sampling_suggestions
        )

    def test_subset_comparison_records_judgment_basis(self):
        observations = []
        for index, point in enumerate(POINTS):
            sensor_b = ROTATION @ point + TRANSLATION
            if index == 5:
                sensor_b = sensor_b + np.array([0.45, -0.35, 0.2])
            observations.append(
                Observation(
                    observation_id=f"obs-{index:03d}",
                    target_id=f"target-{index:03d}",
                    sensor_a_position=point,
                    sensor_b_position=sensor_b,
                    quality="good",
                    position_sigma=0.001,
                )
            )
        batch = ObservationBatch("batch", "sensor-a", "sensor-b", tuple(observations))
        calibration_input = CalibrationInput("dataset", (batch,))

        full_result, comparisons = compare_observation_subsets(
            calibration_input,
            CONVENTION,
            VERSION,
            {"drop-suspect": {("batch", "obs-005")}},
        )

        comparison = comparisons[0]
        self.assertEqual(comparison.full_status, "unstable")
        basis_text = " ".join(comparison.basis)
        self.assertIn("batch/obs-005_full_fit_residual_", basis_text)
        self.assertIn("batch/obs-005_full_fit_influence_score_", basis_text)
        self.assertIn("batch/obs-005_flagged_influential_in_full_fit", basis_text)
        self.assertIn("status_changed_from_unstable_to_valid", basis_text)
        self.assertEqual(len(batch.observations), 8)


if __name__ == "__main__":
    unittest.main()
