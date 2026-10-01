from __future__ import annotations

import unittest

import numpy as np

from sensor_calibration.calibration import CalibrationConfig, calibrate
from sensor_calibration.joint import (
    JointCalibrationConfig,
    calibrate_joint,
    compare_joint_scenarios,
)
from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    JointScenario,
    Observation,
    ObservationBatch,
)
from sensor_calibration.synthetic import rotation_about_axis


VERSION = CalibrationVersion("paired-point-kabsch", "1.1.0")
CONVENTION = CoordinateConvention("convention-v1", "a", "b", "meter")
ROTATION = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 18.0)
TRANSLATION = np.array([0.35, -0.12, 0.08])
VOLUME_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.8, 0.7, 0.8],
])
LINE_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
])


def make_batch(
    batch_id: str,
    points: np.ndarray,
    *,
    sigma: float = 1e-4,
    seed: int = 0,
    bias: np.ndarray | None = None,
    scale: float = 1.0,
) -> ObservationBatch:
    rng = np.random.default_rng(seed)
    observations = []
    for index, point in enumerate(points):
        position_a = np.asarray(point, dtype=float) * scale
        clean_b = ROTATION @ position_a + TRANSLATION * scale
        noise = rng.normal(0.0, sigma, size=3)
        if bias is not None:
            noise = noise + bias
        observations.append(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=position_a,
                sensor_b_position=clean_b + noise,
                position_sigma=sigma,
            )
        )
    return ObservationBatch(batch_id, "sensor-a", "sensor-b", tuple(observations))


class JointCalibrationTests(unittest.TestCase):
    def test_joint_recovers_transform_and_reports_per_batch(self):
        accurate = make_batch("accurate", VOLUME_POINTS, sigma=1e-4, seed=1)
        noisy = make_batch("noisy", VOLUME_POINTS, sigma=5e-3, seed=2)
        result = calibrate_joint(
            CalibrationInput("dataset", (noisy, accurate)), CONVENTION, VERSION
        )

        self.assertTrue(result.joint.success, result.joint.reasons)
        self.assertEqual(result.joint.status, "valid")
        assert result.joint.transform is not None
        np.testing.assert_allclose(result.joint.transform.rotation, ROTATION, atol=0.01)
        np.testing.assert_allclose(result.joint.transform.translation, TRANSLATION, atol=0.01)
        self.assertEqual(set(result.per_batch), {"accurate", "noisy"})
        relations = {item.batch_id: item.relation for item in result.compatibility}
        self.assertEqual(relations["accurate"], "supports_joint_solution")
        self.assertEqual(relations["noisy"], "supports_joint_solution")
        # Every used observation keeps its (batch, observation) identity.
        self.assertIn(("accurate", "obs-000"), dict.fromkeys(result.joint.used_observations))
        self.assertEqual(
            set(result.observation_weights), set(result.joint.used_observations)
        )

    def test_point_count_does_not_dominate_the_joint_fit(self):
        many_points = np.random.default_rng(5).normal(0.0, 1.0, size=(60, 3))
        # The large batch carries an unknown systematic bias; treating its
        # point count as the whole basis would import that bias in full.
        biased_big = make_batch(
            "biased-big", many_points, sigma=1e-3, seed=3, bias=np.array([0.05, 0.0, 0.0])
        )
        accurate = make_batch("accurate-small", VOLUME_POINTS, sigma=1e-3, seed=4)
        dataset = CalibrationInput("dataset", (biased_big, accurate))

        joint = calibrate_joint(dataset, CONVENTION, VERSION).joint
        pooled = calibrate(dataset, CONVENTION, VERSION)
        assert joint.transform is not None and pooled.transform is not None
        joint_error = np.linalg.norm(joint.transform.translation - TRANSLATION)
        pooled_error = np.linalg.norm(pooled.transform.translation - TRANSLATION)
        self.assertLess(joint_error, pooled_error)
        self.assertLess(joint_error, 0.03)
        weights = calibrate_joint(dataset, CONVENTION, VERSION).observation_weights
        big_total = sum(w for (b, _), w in weights.items() if b == "biased-big")
        small_total = sum(w for (b, _), w in weights.items() if b == "accurate-small")
        self.assertAlmostEqual(big_total, small_total, places=6)

    def test_biased_batch_is_flagged_as_conflicting(self):
        good_a = make_batch("good-a", VOLUME_POINTS, sigma=1e-4, seed=5)
        good_b = make_batch("good-b", VOLUME_POINTS, sigma=1e-4, seed=17)
        biased = make_batch(
            "biased", VOLUME_POINTS, sigma=1e-4, seed=6, bias=np.array([0.05, 0.0, 0.0])
        )
        result = calibrate_joint(
            CalibrationInput("dataset", (good_a, good_b, biased)), CONVENTION, VERSION
        )
        relations = {item.batch_id: item.relation for item in result.compatibility}
        self.assertEqual(relations["biased"], "conflicts_with_joint_solution")
        self.assertEqual(relations["good-a"], "supports_joint_solution")
        self.assertEqual(relations["good-b"], "supports_joint_solution")

    def test_collinear_batch_is_covered_by_joint_geometry(self):
        collinear = make_batch("collinear", LINE_POINTS, sigma=1e-4, seed=7)
        volume = make_batch("volume", VOLUME_POINTS, sigma=1e-4, seed=8)
        result = calibrate_joint(
            CalibrationInput("dataset", (collinear, volume)), CONVENTION, VERSION
        )

        self.assertEqual(result.per_batch["collinear"].status, "undetermined")
        self.assertEqual(result.per_batch["collinear"].distribution, "line")
        self.assertTrue(result.joint.success, result.joint.reasons)
        self.assertEqual(result.joint.status, "valid")
        relations = {item.batch_id: item.relation for item in result.compatibility}
        self.assertEqual(relations["collinear"], "indeterminate_alone_covered_by_joint")
        self.assertTrue(
            any("collinear" in note for note in result.identifiability_notes),
            result.identifiability_notes,
        )

    def test_duplicate_acquisition_does_not_add_evidence(self):
        batch = make_batch("acq", VOLUME_POINTS, sigma=1e-3, seed=9)
        duplicate = ObservationBatch(
            "acq-copy", batch.sensor_a_id, batch.sensor_b_id, batch.observations
        )
        single = calibrate_joint(CalibrationInput("dataset", (batch,)), CONVENTION, VERSION)
        doubled = calibrate_joint(
            CalibrationInput("dataset", (batch, duplicate)), CONVENTION, VERSION
        )

        self.assertEqual(doubled.effective_batch_count, 1.0)
        self.assertEqual(doubled.duplicate_groups, (("acq", "acq-copy"),))
        self.assertTrue(any("identical" in note for note in doubled.identifiability_notes))
        assert single.joint.transform is not None and doubled.joint.transform is not None
        np.testing.assert_allclose(
            doubled.joint.transform.rotation, single.joint.transform.rotation, atol=1e-12
        )
        np.testing.assert_allclose(
            doubled.joint.transform.translation, single.joint.transform.translation, atol=1e-12
        )

    def test_batch_order_does_not_change_joint_solution(self):
        batch_a = make_batch("a", VOLUME_POINTS, sigma=1e-3, seed=10)
        batch_b = make_batch("b", VOLUME_POINTS, sigma=2e-3, seed=11)
        forward = calibrate_joint(
            CalibrationInput("dataset", (batch_a, batch_b)), CONVENTION, VERSION
        )
        reverse = calibrate_joint(
            CalibrationInput("dataset", (batch_b, batch_a)), CONVENTION, VERSION
        )
        assert forward.joint.transform is not None and reverse.joint.transform is not None
        np.testing.assert_allclose(
            forward.joint.transform.rotation, reverse.joint.transform.rotation, atol=1e-12
        )
        np.testing.assert_allclose(
            forward.joint.transform.translation, reverse.joint.transform.translation, atol=1e-12
        )

    def test_unit_scale_does_not_change_joint_status_or_rotation(self):
        meters = make_batch("meters", VOLUME_POINTS, sigma=1e-3, seed=12)
        millimeters = make_batch("millimeters", VOLUME_POINTS, sigma=1.0, seed=12, scale=1e3)
        joint_m = calibrate_joint(CalibrationInput("dataset", (meters,)), CONVENTION, VERSION)
        joint_mm = calibrate_joint(
            CalibrationInput("dataset", (millimeters,)), CONVENTION, VERSION
        )
        self.assertEqual(joint_m.joint.status, joint_mm.joint.status)
        self.assertEqual(joint_m.joint.distribution, joint_mm.joint.distribution)
        assert joint_m.joint.transform is not None and joint_mm.joint.transform is not None
        np.testing.assert_allclose(
            joint_m.joint.transform.rotation, joint_mm.joint.transform.rotation, atol=1e-9
        )
        np.testing.assert_allclose(
            joint_mm.joint.transform.translation,
            joint_m.joint.transform.translation * 1e3,
            rtol=1e-6,
        )

    def test_scenario_exclusion_is_traceable_and_non_destructive(self):
        good = make_batch("good", VOLUME_POINTS, sigma=1e-4, seed=13)
        biased = make_batch(
            "biased", VOLUME_POINTS, sigma=1e-4, seed=14, bias=np.array([0.05, 0.0, 0.0])
        )
        dataset = CalibrationInput("dataset", (good, biased))
        scenarios = {
            "drop-biased-batch": JointScenario(excluded_batches=frozenset({"biased"})),
            "drop-two-correspondences": JointScenario(
                excluded_observations=frozenset({("biased", "obs-000"), ("biased", "obs-001")})
            ),
        }
        full, comparisons = compare_joint_scenarios(dataset, CONVENTION, VERSION, scenarios)

        self.assertEqual(len(comparisons), 2)
        by_label = {item.label: item for item in comparisons}
        dropped = by_label["drop-biased-batch"]
        self.assertEqual(dropped.result.excluded_batches, ("biased",))
        self.assertNotIn(
            ("biased", "obs-000"), dict.fromkeys(dropped.result.joint.used_observations)
        )
        self.assertTrue(dropped.judgment_basis)
        partial = by_label["drop-two-correspondences"]
        self.assertIn(
            ("biased", "obs-000"), dict.fromkeys(partial.result.joint.excluded_observations)
        )
        # Source acquisitions are untouched and every point stays traceable.
        self.assertEqual(len(dataset.batches[1].observations), 8)
        self.assertIn(("biased", "obs-002"), dict.fromkeys(partial.result.joint.used_observations))
        serialized = dropped.result.as_dict()
        self.assertIn("observation_weights", serialized)
        self.assertIn("joint", serialized)
        # Full joint fit still uses both batches.
        self.assertEqual(full.effective_batch_count, 2.0)

    def test_excluding_the_only_informative_batch_loses_identifiability(self):
        collinear = make_batch("collinear", LINE_POINTS, sigma=1e-4, seed=15)
        volume = make_batch("volume", VOLUME_POINTS, sigma=1e-4, seed=16)
        dataset = CalibrationInput("dataset", (collinear, volume))
        full, comparisons = compare_joint_scenarios(
            dataset,
            CONVENTION,
            VERSION,
            {"only-collinear-left": JointScenario(excluded_batches=frozenset({"volume"}))},
        )
        self.assertTrue(full.joint.success)
        scenario = comparisons[0]
        self.assertFalse(scenario.result.joint.success)
        self.assertEqual(scenario.result.joint.status, "undetermined")
        self.assertIn("scenario_loses_joint_identifiability", scenario.judgment_basis)


if __name__ == "__main__":
    unittest.main()
