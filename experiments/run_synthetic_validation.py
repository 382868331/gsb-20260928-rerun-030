from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sensor_calibration.calibration import CalibrationConfig, calibrate, calibrate_subsets
from sensor_calibration.models import (
    CalibrationInput,
    CalibrationVersion,
    CoordinateConvention,
    Observation,
    ObservationBatch,
)
from sensor_calibration.synthetic import make_noisy_batch, rotation_about_axis


VERSION = CalibrationVersion(
    method_name="paired-point-kabsch",
    method_version="1.1.0",
    parameters={
        "distance_metric": "euclidean",
        "rotation_parameterization": "rotation_matrix",
        "geometry_check": "effective-rank-with-noise-floor",
    },
)
CONVENTION = CoordinateConvention(
    convention_id="metric-right-handed-v1",
    frame_a="virtual_sensor_a",
    frame_b="virtual_sensor_b",
    length_unit="meter",
    description="Same target observations expressed independently in each sensor frame",
)


def error_summary(result, expected_rotation, expected_translation):
    summary = {
        "success": result.success,
        "status": result.status,
        "geometric_rank": result.geometric_rank,
        "effective_rank": result.effective_rank,
        "rms_residual": result.rms_residual,
        "reasons": list(result.reasons),
    }
    if result.transform is not None:
        summary["rotation_frobenius_error"] = float(
            np.linalg.norm(result.transform.rotation - expected_rotation, ord="fro")
        )
        summary["translation_error"] = float(
            np.linalg.norm(result.transform.translation - expected_translation)
        )
        summary["influential_observations"] = [list(item) for item in result.influential_observations]
        summary["warning_observations"] = [list(item) for item in result.warning_observations]
    return summary


def with_swapped_correspondence(batch: ObservationBatch, index: int, offset) -> ObservationBatch:
    original = batch.observations[index]
    swapped = Observation(
        observation_id=original.observation_id,
        target_id=original.target_id,
        sensor_a_position=original.sensor_a_position,
        sensor_b_position=original.sensor_b_position + np.asarray(offset, dtype=float),
        quality="good",
        position_sigma=original.position_sigma,
        timestamp=original.timestamp,
    )
    observations = batch.observations[:index] + (swapped,) + batch.observations[index + 1 :]
    return ObservationBatch(batch.batch_id, batch.sensor_a_id, batch.sensor_b_id, observations)


def main() -> int:
    expected_rotation = rotation_about_axis(np.array([0.0, 0.0, 1.0]), 18.0)
    expected_translation = np.array([0.35, -0.12, 0.08])

    well_spread = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.02, 0.01],
        [0.05, 1.0, -0.03],
        [0.02, -0.04, 1.0],
        [1.05, 0.95, 0.08],
        [0.9, -0.05, 0.95],
        [-0.08, 0.92, 0.9],
        [0.75, 0.72, 0.8],
    ])
    planar_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.02, 0.0],
        [0.05, 1.0, 0.0],
        [1.05, 0.95, 0.0],
        [0.9, -0.05, 0.0],
        [-0.08, 0.92, 0.0],
    ])
    line_direction = np.array([1.0, 0.2, 0.0])
    line_direction /= np.linalg.norm(line_direction)
    collinear_positions = np.outer(np.linspace(-0.5, 0.5, 8), line_direction)

    three_d_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread,
        noise_sigma=0.003,
        seed=201,
    )
    planar_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=planar_points,
        noise_sigma=0.003,
        seed=202,
    )
    collinear_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=collinear_positions,
        noise_sigma=0.003,
        seed=203,
    )

    three_d_result = calibrate(
        CalibrationInput("three-d-dataset", (three_d_batch,)), CONVENTION, VERSION
    )
    planar_result = calibrate(
        CalibrationInput("planar-dataset", (planar_batch,)), CONVENTION, VERSION
    )
    collinear_result = calibrate(
        CalibrationInput("collinear-dataset", (collinear_batch,)), CONVENTION, VERSION
    )

    assert three_d_result.status == "valid"
    assert planar_result.success and planar_result.status == "valid", planar_result.reasons
    assert planar_result.geometric_rank == 2
    assert np.linalg.norm(planar_result.transform.rotation - expected_rotation, ord="fro") < 0.01
    assert np.linalg.norm(planar_result.transform.translation - expected_translation) < 0.01
    assert not collinear_result.success
    assert collinear_result.status == "undetermined"

    # A small number of wrong correspondences: the answer is still produced but
    # flagged unstable, with the offending observations identified.
    questionable_batch = with_swapped_correspondence(three_d_batch, 3, (0.5, -0.4, 0.45))
    questionable_input = CalibrationInput("questionable-dataset", (questionable_batch,))
    subset_results = calibrate_subsets(
        questionable_input,
        CONVENTION,
        VERSION,
        {
            "baseline": frozenset(),
            "without_obs_003": frozenset({(questionable_batch.batch_id, "obs-003")}),
        },
    )
    baseline = subset_results["baseline"]
    repaired = subset_results["without_obs_003"]
    assert baseline.status == "unstable"
    assert (questionable_batch.batch_id, "obs-003") in baseline.influential_observations
    assert repaired.status == "valid"
    assert len(questionable_batch.observations) == len(three_d_batch.observations)
    assert questionable_batch.observations[3].quality == "good"

    report = {
        "three_dimensional": error_summary(
            three_d_result, expected_rotation, expected_translation
        ),
        "planar": error_summary(planar_result, expected_rotation, expected_translation),
        "collinear": error_summary(collinear_result, expected_rotation, expected_translation),
        "wrong_correspondence_baseline": error_summary(
            baseline, expected_rotation, expected_translation
        ),
        "wrong_correspondence_after_manual_exclusion": error_summary(
            repaired, expected_rotation, expected_translation
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
