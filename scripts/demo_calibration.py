from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensor_calibration import (
    ObservationBatch,
    PoseTransform,
    calibrate,
)
from sensor_calibration.synthetic import make_observations, rotation_axis_angle
from sensor_calibration.models import CoordinateConvention


GROUND_TRUTH = PoseTransform(
    rotation=rotation_axis_angle(np.array([0.0, 0.0, 1.0]), 0.18),
    translation=np.array([0.35, -0.20, 0.60]),
)

CONVENTION = CoordinateConvention(
    convention_id="coord-v1",
    frame_a="virtual_sensor_a",
    frame_b="virtual_sensor_b",
    length_unit="meter",
    description="right-handed experimental frame",
)


def tetrahedron_targets() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.1],
            [0.8, 0.2, 0.9],
            [0.15, 0.9, 0.7],
            [0.65, 0.75, 0.35],
        ]
    )


def stationary_targets() -> np.ndarray:
    return np.repeat(np.array([[0.4, -0.2, 0.5]]), 8, axis=0)


def print_result(name: str, result) -> None:
    print(f"\n{name}")
    print(f"  determined: {result.determined}")
    print(f"  rank: {result.rank}; singular values: {tuple(round(x, 6) for x in result.singular_values)}")
    print(f"  excluded: {result.version.excluded_observation_ids}")
    if result.transform is None:
        print(f"  reason: {'; '.join(result.reasons)}")
        return
    axis, angle = result.transform.axis_angle()
    print(f"  translation: {np.round(result.transform.translation, 4).tolist()}")
    print(f"  rotation axis: {np.round(axis, 4).tolist()}; angle rad: {angle:.6f}")
    print(f"  RMS error: {result.rms_error:.6f}; max error: {result.max_error:.6f}")


def main() -> None:
    normal_obs = make_observations(tetrahedron_targets(), GROUND_TRUTH, noise_sigma=0.003, seed=11)
    normal_batch = ObservationBatch(
        batch_id="batch-normal",
        sensor_a_id="virtual-a",
        sensor_b_id="virtual-b",
        convention_id=CONVENTION.convention_id,
        observations=tuple(normal_obs),
        collected_at="2026-09-29T00:00:00Z",
    )
    normal = calibrate("cal-normal", normal_batch, CONVENTION, noise_scale=0.003)
    print_result("Normal sample", normal)

    outlier_obs = make_observations(
        tetrahedron_targets(),
        GROUND_TRUTH,
        noise_sigma=0.003,
        seed=12,
        outlier_indices=(2, 6),
        outlier_sigma=2.0,
    )
    outlier_batch = ObservationBatch(
        batch_id="batch-outlier",
        sensor_a_id="virtual-a",
        sensor_b_id="virtual-b",
        convention_id=CONVENTION.convention_id,
        observations=tuple(outlier_obs),
        collected_at="2026-09-29T00:10:00Z",
    )
    before = calibrate("cal-outlier-before", outlier_batch, CONVENTION, noise_scale=0.003)
    after = calibrate(
        "cal-outlier-after",
        outlier_batch,
        CONVENTION,
        parent_version_id="cal-outlier-before",
        noise_scale=0.003,
        extra_excluded_ids=("obs-002", "obs-006"),
        notes="reviewer explicitly excluded outlier observations for this recalculation",
    )
    print_result("Outlier sample, all observations included", before)
    print_result("Outlier sample, reviewer exclusions for recalculation", after)
    print(f"  source batch hash unchanged: {before.version.batch_hash == after.version.batch_hash}")

    stationary_obs = make_observations(stationary_targets(), GROUND_TRUTH, noise_sigma=0.003, seed=13)
    stationary_batch = ObservationBatch(
        batch_id="batch-stationary",
        sensor_a_id="virtual-a",
        sensor_b_id="virtual-b",
        convention_id=CONVENTION.convention_id,
        observations=tuple(stationary_obs),
        collected_at="2026-09-29T00:20:00Z",
    )
    degenerate = calibrate("cal-stationary", stationary_batch, CONVENTION, noise_scale=0.003)
    print_result("Insufficient target motion", degenerate)


if __name__ == "__main__":
    main()
