from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensor_calibration.calibration import CalibrationConfig, calibrate
from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
)
from sensor_calibration.persistence import build_plan, save_comparison_bundle
from sensor_calibration.synthetic import rotation_about_axis


VERSION = CalibrationVersion("paired-point-weighted-kabsch", "2.0.0")
CONVENTION = CoordinateConvention("convention-v1", "a", "b", "meter")
ROTATION = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 22.0)
TRANSLATION = np.array([0.31, -0.07, 0.12])


def _batch(
    batch_id: str,
    points_a,
    *,
    sigma: float,
    scale: float = 1.0,
    acquisition_id: str | None = None,
    b_offset=None,
    seed: int = 0,
) -> ObservationBatch:
    rng = np.random.default_rng(seed)
    points_a = np.asarray(points_a, dtype=float)
    observations = []
    for index, point in enumerate(points_a):
        point_a_m = point * scale
        clean_b = ROTATION @ point_a_m + TRANSLATION
        noise = rng.normal(0.0, sigma, size=3)
        if b_offset is not None:
            clean_b = clean_b + np.asarray(b_offset, dtype=float)
        observations.append(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=point * scale,
                sensor_b_position=clean_b + noise,
                position_sigma=sigma,
            )
        )
    return ObservationBatch(
        batch_id=batch_id,
        sensor_a_id="sensor-a",
        sensor_b_id="sensor-b",
        observations=tuple(observations),
        length_scale_to_convention=1.0,
        length_unit="meter",
        acquisition_id=acquisition_id,
    )


VOLUME_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.1],
    [0.9, 0.1, 0.9],
])

LINE_POINTS = np.column_stack([np.linspace(0.0, 1.0, 6), np.zeros(6), np.zeros(6)])


class MultiBatchCalibrationTests(unittest.TestCase):
    def test_collinear_batch_is_rescued_by_other_batch_in_joint_fit(self):
        line_batch = _batch("line-batch", LINE_POINTS, sigma=1e-5, seed=1)
        plane_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        plane_batch = _batch("plane-batch", plane_points, sigma=1e-5, seed=2)

        result = calibrate(
            CalibrationInput("dataset", (line_batch, plane_batch)),
            CONVENTION,
            VERSION,
        )

        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.distribution, "volume")
        self.assertEqual(result.batch_evaluations["line-batch"].status, "undetermined")
        self.assertIn(
            "batch_geometry_is_individually_degenerate_but_other_batches_can_supply_missing_directions",
            result.batch_evaluations["line-batch"].reasons,
        )
        self.assertEqual(result.batch_evaluations["plane-batch"].distribution, "plane")
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=1e-4)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=1e-4)

    def test_point_count_does_not_dominate_precision_weighted_joint_fit(self):
        many_noisy_points = np.array([
            [0.2 * index, 0.0, 0.0] for index in range(8)
        ])
        # A large but biased batch: constant offset cannot be explained by R, t.
        big_batch = _batch(
            "big-noisy",
            many_noisy_points,
            sigma=0.02,
            b_offset=[0.15, -0.1, 0.0],
            seed=3,
        )
        small_precise = _batch("small-precise", VOLUME_POINTS[:4], sigma=1e-5, seed=4)

        result = calibrate(
            CalibrationInput("dataset", (big_batch, small_precise)),
            CONVENTION,
            VERSION,
        )

        self.assertTrue(result.success)
        big_eval = result.batch_evaluations["big-noisy"]
        self.assertIsNotNone(big_eval.bias_sigma_ratio)
        self.assertGreater(big_eval.bias_sigma_ratio, 4.0)
        self.assertTrue(
            any("big-noisy" in warning and "bias" in warning for warning in result.warnings)
        )
        # The joint transform stays much closer to the precise batch's truth
        # than to the biased batch's apparent translation.
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=0.02)
        self.assertLess(np.linalg.norm(result.transform.translation - TRANSLATION), 0.05)

    def test_batch_order_does_not_change_joint_fit(self):
        first = _batch("batch-1", VOLUME_POINTS, sigma=0.002, seed=11)
        second = _batch("batch-2", VOLUME_POINTS + 0.5, sigma=0.004, seed=12)

        forward = calibrate(CalibrationInput("d", (first, second)), CONVENTION, VERSION)
        reversed_fit = calibrate(
            CalibrationInput("d", (second, first)), CONVENTION, VERSION
        )

        np.testing.assert_allclose(
            forward.transform.rotation, reversed_fit.transform.rotation, atol=1e-12
        )
        np.testing.assert_allclose(
            forward.transform.translation, reversed_fit.transform.translation, atol=1e-12
        )

    def test_different_units_are_converted_before_joint_fit(self):
        metre_batch = _batch("metres", VOLUME_POINTS, sigma=1e-5, seed=21)
        millimetre_observations = []
        rng = np.random.default_rng(22)
        for index, point in enumerate(VOLUME_POINTS):
            point_mm = point * 1000.0
            clean_b_mm = (ROTATION @ point + TRANSLATION) * 1000.0
            millimetre_observations.append(
                Observation(
                    observation_id=f"obs-{index:03d}",
                    target_id=f"target-{index:03d}",
                    sensor_a_position=point_mm,
                    sensor_b_position=clean_b_mm + rng.normal(0.0, 0.01, size=3),
                    position_sigma=0.01,
                )
            )
        mm_batch = ObservationBatch(
            "millimetres",
            "sensor-a",
            "sensor-b",
            tuple(millimetre_observations),
            length_scale_to_convention=0.001,
            length_unit="millimetre",
        )

        result = calibrate(
            CalibrationInput("dataset", (metre_batch, mm_batch)),
            CONVENTION,
            VERSION,
        )

        self.assertTrue(result.success, result.reasons)
        np.testing.assert_allclose(result.transform.rotation, ROTATION, atol=1e-4)
        np.testing.assert_allclose(result.transform.translation, TRANSLATION, atol=1e-4)
        self.assertEqual(result.distribution, "volume")

    def test_excluding_a_whole_batch_is_traceable_and_changes_identifiability(self):
        line_batch = _batch("line-only", LINE_POINTS, sigma=1e-5, seed=31)
        volume_batch = _batch("volume", VOLUME_POINTS, sigma=1e-5, seed=32)
        config = CalibrationConfig(excluded_batches=frozenset({"volume"}))

        result = calibrate(
            CalibrationInput("dataset", (line_batch, volume_batch)),
            CONVENTION,
            VERSION,
            config,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.distribution, "line")
        self.assertEqual(result.batch_evaluations["volume"].status, "excluded")
        decisions = result.observation_decisions
        volume_decisions = [d for d in decisions.values() if d.batch_id == "volume"]
        self.assertTrue(all(not d.adopted and d.decision == "excluded_batch" for d in volume_decisions))
        line_decisions = [d for d in decisions.values() if d.batch_id == "line-only"]
        self.assertTrue(all(d.decision == "used" for d in line_decisions))
        self.assertIn(
            "target_positions_are_collinear_rotation_about_line_is_unobservable",
            result.reasons,
        )

    def test_repeated_acquisition_is_counted_as_one_piece_of_evidence(self):
        first = _batch(
            "run-1",
            VOLUME_POINTS,
            sigma=1e-6,
            acquisition_id="acq-X",
            seed=41,
        )
        # Exact re-addition of the same acquisition (same acquisition id,
        # same target ids), even under a new batch id.
        second = ObservationBatch(
            "run-2",
            "sensor-a",
            "sensor-b",
            first.observations,
            length_scale_to_convention=1.0,
            acquisition_id="acq-X",
        )
        duplicated = calibrate(
            CalibrationInput("dataset", (first, second)), CONVENTION, VERSION
        )
        single = calibrate(CalibrationInput("dataset", (first,)), CONVENTION, VERSION)

        np.testing.assert_allclose(
            duplicated.transform.rotation, single.transform.rotation, atol=1e-12
        )
        np.testing.assert_allclose(
            duplicated.transform.translation, single.transform.translation, atol=1e-12
        )
        evidence = duplicated.acquisition_evidence
        repeated = [item for item in evidence.values() if item.repeated]
        self.assertEqual(len(repeated), len(VOLUME_POINTS))
        for item in repeated:
            self.assertEqual(item.multiplicity, 2)
            self.assertAlmostEqual(item.independent_weight, 0.5)
        for decision in duplicated.observation_decisions.values():
            if decision.adopted:
                self.assertAlmostEqual(decision.evidence_weight, 0.5)
        self.assertIn("repeated_acquisition_copies_are_counted_once", duplicated.warnings)

    def test_saved_plan_bundle_traces_every_point_to_its_acquisition(self):
        first = _batch("run-1", VOLUME_POINTS, sigma=1e-4, seed=51)
        second = _batch("run-2", VOLUME_POINTS + 0.3, sigma=1e-4, seed=52)
        config = CalibrationConfig(
            additional_excluded_observations=frozenset({("run-2", "obs-001")}),
        )
        plan = build_plan(
            "plan-A",
            CalibrationInput("dataset", (first, second)),
            CONVENTION,
            VERSION,
            config,
            subsets={"drop-run-1-point": {("run-1", "obs-000")}},
        )
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.json"
            bundle = save_comparison_bundle(
                bundle_path,
                plan,
                CalibrationInput("dataset", (first, second)),
                CONVENTION,
                VERSION,
                config,
            )
            reloaded = json.loads(bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(reloaded["plan"]["plan_id"], "plan-A")
        self.assertEqual(
            reloaded["plan"]["excluded_observations"], [["run-2", "obs-001"]]
        )
        decisions = bundle["joint_result"]["observation_decisions"]
        self.assertEqual(decisions["run-2/obs-001"]["decision"], "excluded_manually")
        self.assertEqual(decisions["run-1/obs-000"]["acquisition_id"], "run-1:target-000")
        self.assertEqual(len(reloaded["input"]["batches"]), 2)
        # Excluded points remain in the saved raw acquisition.
        run_two = next(b for b in reloaded["input"]["batches"] if b["batch_id"] == "run-2")
        self.assertEqual(len(run_two["observations"]), len(VOLUME_POINTS))
        labels = {item["label"] for item in bundle["subset_comparisons"]}
        self.assertEqual(labels, {"drop-run-1-point"})


if __name__ == "__main__":
    unittest.main()
