from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sensor_calibration.calibration import CalibrationConfig, calibrate
from sensor_calibration.models import CalibrationInput, CalibrationVersion, CoordinateConvention
from sensor_calibration.synthetic import make_noisy_batch, rotation_about_axis


VERSION = CalibrationVersion(
    method_name="paired-point-kabsch",
    method_version="1.0.0",
    parameters={"distance_metric": "euclidean", "rotation_parameterization": "rotation_matrix"},
)
CONVENTION = CoordinateConvention(
    convention_id="metric-right-handed-v1",
    frame_a="virtual_sensor_a",
    frame_b="virtual_sensor_b",
    length_unit="meter",
    description="Same target observations expressed independently in each sensor frame",
)


def compare_transform(result, expected_rotation, expected_translation):
    assert result.success, result.reasons
    assert result.transform is not None
    rotation_error = np.linalg.norm(result.transform.rotation - expected_rotation, ord="fro")
    translation_error = np.linalg.norm(result.transform.translation - expected_translation)
    return rotation_error, translation_error


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

    normal_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread,
        noise_sigma=0.003,
        seed=101,
    )
    normal_input = CalibrationInput("normal-synthetic-dataset", (normal_batch,))
    normal_result = calibrate(normal_input, CONVENTION, VERSION)
    normal_rotation_error, normal_translation_error = compare_transform(
        normal_result, expected_rotation, expected_translation
    )
    assert normal_rotation_error < 0.01
    assert normal_translation_error < 0.01
    assert (normal_result.rms_residual or 0.0) < 0.01

    outlier_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=well_spread,
        noise_sigma=0.003,
        seed=102,
        bad_indices=(3, 6),
        outlier_sigma=0.5,
    )
    outlier_input = CalibrationInput("outlier-synthetic-dataset", (outlier_batch,))
    polluted_result = calibrate(
        outlier_input,
        CONVENTION,
        VERSION,
        CalibrationConfig(exclude_marked_bad=False),
    )
    polluted_rotation_error, polluted_translation_error = compare_transform(
        polluted_result, expected_rotation, expected_translation
    )
    assert polluted_rotation_error > 0.02 or polluted_translation_error > 0.02

    recalculated_result = calibrate(outlier_input, CONVENTION, VERSION)
    fixed_rotation_error, fixed_translation_error = compare_transform(
        recalculated_result, expected_rotation, expected_translation
    )
    assert fixed_rotation_error < 0.01
    assert fixed_translation_error < 0.01
    assert recalculated_result.excluded_observations == (
        (outlier_batch.batch_id, "obs-003"),
        (outlier_batch.batch_id, "obs-006"),
    )
    assert outlier_batch.observations[3].quality == "bad"
    assert len(outlier_batch.observations) == len(normal_batch.observations)

    base_direction = np.array([1.0, 0.2, 0.0])
    base_direction /= np.linalg.norm(base_direction)
    insufficient_positions = np.outer(np.linspace(0.0, 0.002, 8), base_direction)
    insufficient_batch = make_noisy_batch(
        rotation=expected_rotation,
        translation=expected_translation,
        target_positions_a=insufficient_positions,
        noise_sigma=0.003,
        seed=103,
    )
    insufficient_input = CalibrationInput("insufficient-motion-dataset", (insufficient_batch,))
    insufficient_result = calibrate(insufficient_input, CONVENTION, VERSION)
    assert not insufficient_result.success
    assert insufficient_result.transform is None
    assert "target_motion_is_small_relative_to_measurement_uncertainty" in insufficient_result.reasons

    report = {
        "normal": {
            **normal_result.as_dict(),
            "rotation_frobenius_error": normal_rotation_error,
            "translation_error": normal_translation_error,
        },
        "outliers_before_exclusion": {
            **polluted_result.as_dict(),
            "rotation_frobenius_error": polluted_rotation_error,
            "translation_error": polluted_translation_error,
        },
        "outliers_after_exclusion": {
            **recalculated_result.as_dict(),
            "rotation_frobenius_error": fixed_rotation_error,
            "translation_error": fixed_translation_error,
        },
        "insufficient_motion": insufficient_result.as_dict(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
