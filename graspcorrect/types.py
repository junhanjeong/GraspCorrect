from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

Point2D = Tuple[float, float]


@dataclass
class Action:
    """RLBench end-effector action: xyz, quaternion xyzw, gripper open state."""

    position: np.ndarray
    rotation: np.ndarray
    gripper: float

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "Action":
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 8:
            raise ValueError("Action vector must contain xyz + quat + gripper.")
        return cls(position=arr[:3].copy(), rotation=normalize_quat(arr[3:7]), gripper=float(arr[7]))

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.position, dtype=np.float32),
                normalize_quat(self.rotation),
                np.asarray([self.gripper], dtype=np.float32),
            ]
        )


@dataclass
class Observation:
    rgb: np.ndarray
    raw: Optional[Any] = None
    camera: str = "overhead"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraspCandidate:
    label: int
    point: Point2D
    contour_index: int


@dataclass
class GraspPair:
    left: GraspCandidate
    right: GraspCandidate
    description: str
    object_prompt: str = ""

    @property
    def distance_px(self) -> float:
        a = np.asarray(self.left.point, dtype=np.float32)
        b = np.asarray(self.right.point, dtype=np.float32)
        return float(np.linalg.norm(a - b))


@dataclass
class GoalMetadata:
    grasp_description: str = ""
    increased_distance_px: float = 0.0
    background_points_pre: Optional[Tuple[Point2D, Point2D]] = None
    background_points_current: Optional[Tuple[Point2D, Point2D]] = None


@dataclass
class CorrectionResult:
    corrected_action: Action
    goal_rgb: np.ndarray
    grasp_pair: GraspPair
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return arr / norm
