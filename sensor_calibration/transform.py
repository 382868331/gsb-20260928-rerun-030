from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PoseTransform:
    """Rigid transform satisfying ``point_b = rotation @ point_a + translation``."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("expected a 3x3 rotation and a length-3 translation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation is not orthogonal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("rotation has determinant -1")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def apply(self, points_a: np.ndarray) -> np.ndarray:
        points = np.asarray(points_a, dtype=float)
        return points @ self.rotation.T + self.translation

    def inverse(self) -> "PoseTransform":
        return PoseTransform(self.rotation.T, -self.rotation.T @ self.translation)

    def axis_angle(self) -> tuple[np.ndarray, float]:
        trace = float(np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = np.arccos(trace)
        if np.isclose(angle, 0.0):
            return np.array([1.0, 0.0, 0.0]), 0.0
        if np.isclose(angle, np.pi):
            diag = np.diag(self.rotation)
            axis = np.sqrt(np.maximum((diag + 1.0) / 2.0, 0.0))
            i = int(np.argmax(axis))
            axis[i] *= np.sign(self.rotation[(i + 2) % 3, (i + 1) % 3]) or 1.0
            return axis / np.linalg.norm(axis), float(angle)
        axis = np.array(
            [
                self.rotation[2, 1] - self.rotation[1, 2],
                self.rotation[0, 2] - self.rotation[2, 0],
                self.rotation[1, 0] - self.rotation[0, 1],
            ]
        )
        return axis / np.linalg.norm(axis), float(angle)
