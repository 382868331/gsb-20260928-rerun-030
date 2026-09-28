from __future__ import annotations

import numpy as np

from .models import Observation, ObservationBatch


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=float)
    unit_axis = unit_axis / np.linalg.norm(unit_axis)
    x, y, z = unit_axis
    angle = np.radians(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def make_noisy_batch(
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    target_positions_a: np.ndarray,
    noise_sigma: float,
    seed: int,
    bad_indices: tuple[int, ...] = (),
    outlier_sigma: float = 1.0,
) -> ObservationBatch:
    rng = np.random.default_rng(seed)
    observations = []
    for index, position_a in enumerate(target_positions_a):
        clean_b = rotation @ np.asarray(position_a, dtype=float) + np.asarray(translation, dtype=float)
        noise = rng.normal(0.0, noise_sigma, size=3)
        if index in bad_indices:
            noise = noise + rng.normal(0.0, outlier_sigma, size=3)
        observations.append(
            Observation(
                observation_id=f"obs-{index:03d}",
                target_id=f"target-{index:03d}",
                sensor_a_position=np.asarray(position_a, dtype=float),
                sensor_b_position=clean_b + noise,
                quality="bad" if index in bad_indices else "good",
                position_sigma=noise_sigma,
            )
        )
    return ObservationBatch(
        batch_id=f"synthetic-seed-{seed}",
        sensor_a_id="virtual-sensor-a",
        sensor_b_id="virtual-sensor-b",
        observations=tuple(observations),
    )
