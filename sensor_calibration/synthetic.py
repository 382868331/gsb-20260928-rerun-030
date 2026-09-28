from __future__ import annotations

import numpy as np

from .models import Observation
from .transform import PoseTransform


def rotation_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    x, y, z = unit
    c = np.cos(angle)
    s = np.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + s * cross + (1.0 - c) * np.outer(unit, unit)


def make_observations(
    target_points_a: np.ndarray,
    transform: PoseTransform,
    noise_sigma: float,
    *,
    seed: int = 0,
    bad_indices: tuple[int, ...] = (),
    outlier_indices: tuple[int, ...] = (),
    outlier_sigma: float = 0.0,
) -> list[Observation]:
    rng = np.random.default_rng(seed)
    points_a = np.asarray(target_points_a, dtype=float)
    points_b = transform.apply(points_a)
    points_a = points_a + rng.normal(scale=noise_sigma, size=points_a.shape)
    points_b = points_b + rng.normal(scale=noise_sigma, size=points_b.shape)

    observations: list[Observation] = []
    for i, (point_a, point_b) in enumerate(zip(points_a, points_b, strict=True)):
        is_bad = i in bad_indices
        note = ""
        if is_bad or i in outlier_indices:
            point_b = point_b + rng.normal(scale=outlier_sigma, size=3)
        if is_bad:
            note = "manually flagged bad observation"
        observations.append(
            Observation(
                observation_id=f"obs-{i:03d}",
                target_id=f"target-{i:03d}",
                point_a=tuple(point_a),
                point_b=tuple(point_b),
                timestamp=f"2026-09-29T00:{i:02d}:00Z",
                sigma=noise_sigma,
                is_bad=is_bad,
                note=note,
            )
        )
    return observations
