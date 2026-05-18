from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


Point2D = Tuple[float, float]


@dataclass
class Action:
    """End-effector action used by the paper: position, quaternion, gripper."""

    position: np.ndarray
    rotation: np.ndarray
    gripper: float = 1.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(self.rotation))
        if norm > 1e-8:
            self.rotation = self.rotation / norm
        self.gripper = float(self.gripper)

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "Action":
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size not in (7, 8):
            raise ValueError(f"Action vector must have 7 or 8 values, got {arr.size}.")
        gripper = float(arr[7]) if arr.size == 8 else 1.0
        return cls(position=arr[:3], rotation=arr[3:7], gripper=gripper)

    def as_vector(self, include_gripper: bool = True) -> np.ndarray:
        values = [self.position, self.rotation]
        if include_gripper:
            values.append(np.asarray([self.gripper], dtype=np.float32))
        return np.concatenate(values).astype(np.float32)


@dataclass
class Observation:
    """RGB-D observation from a benchmark camera."""

    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    camera: str = "unknown"
    intrinsics: Optional[np.ndarray] = None
    extrinsics: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
            raise ValueError(f"Observation rgb must be HxWx3/4, got {rgb.shape}.")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        self.rgb = rgb
        if self.depth is not None:
            self.depth = np.asarray(self.depth)


@dataclass
class GraspCandidate:
    """A numbered contour point shown to the VLM."""

    label: int
    point: Point2D
    contour_index: int
    score: float = 0.0


@dataclass
class GraspPair:
    """Left and right parallel-jaw contact points in image coordinates."""

    left: GraspCandidate
    right: GraspCandidate
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def midpoint(self) -> Point2D:
        return (
            0.5 * (self.left.point[0] + self.right.point[0]),
            0.5 * (self.left.point[1] + self.right.point[1]),
        )

    @property
    def width_px(self) -> float:
        return float(np.linalg.norm(np.asarray(self.left.point) - np.asarray(self.right.point)))


@dataclass
class GraspCorrectOutput:
    corrected_action: Action
    grasp_pair: GraspPair
    goal_rgb: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
