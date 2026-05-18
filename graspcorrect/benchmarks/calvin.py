from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from graspcorrect.benchmarks.base import GraspMomentDetector
from graspcorrect.pipeline import GraspCorrectPipeline
from graspcorrect.types import Action, Observation


def observation_from_calvin(obs: dict, camera: str = "rgb_obs") -> Observation:
    rgb_container = obs.get(camera, obs)
    rgb = rgb_container.get("rgb_static", None) if isinstance(rgb_container, dict) else None
    if rgb is None and isinstance(obs, dict):
        rgb = obs.get("rgb_static")
    if rgb is None:
        raise KeyError("Could not find CALVIN rgb_static observation.")
    return Observation(rgb=np.asarray(rgb), camera="calvin_static", metadata={"raw": obs})


def action_from_calvin_vector(action: np.ndarray) -> Action:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size == 7:
        return Action(position=arr[:3], rotation=_euler_to_quat(arr[3:6]), gripper=float(arr[-1]))
    return Action.from_vector(arr)


def action_to_calvin_vector(action: Action) -> np.ndarray:
    return np.concatenate(
        [
            action.position.astype(np.float32),
            _quat_to_euler(action.rotation).astype(np.float32),
            np.asarray([action.gripper], dtype=np.float32),
        ]
    )


@dataclass
class CALVINGraspCorrectModel:
    """CALVIN CustomModel-compatible wrapper.

    CALVIN's official evaluation asks custom agents to implement reset() and
    step(obs, goal). This class wraps any baseline with a step/act/predict
    method and applies GraspCorrect at the first gripper-close transition.
    """

    baseline: Any
    pipeline: GraspCorrectPipeline
    window: int = 10
    detector: GraspMomentDetector = field(default_factory=GraspMomentDetector)
    history: List[Observation] = field(default_factory=list)

    def reset(self) -> None:
        self.detector.reset()
        self.history.clear()
        reset = getattr(self.baseline, "reset", None)
        if callable(reset):
            reset()

    def step(self, obs: dict, goal: Any) -> np.ndarray:
        instruction = _goal_to_text(goal)
        raw_action = _baseline_step(self.baseline, obs, goal, instruction)
        action = action_from_calvin_vector(raw_action)
        observation = observation_from_calvin(obs)
        self.history.append(observation)
        if len(self.history) > self.window + 1:
            self.history = self.history[-(self.window + 1) :]
        if self.detector.should_correct(action) and len(self.history) > self.window:
            pre_grasp = self.history[-self.window - 1]
            output = self.pipeline.correct(
                current=observation,
                pre_grasp=pre_grasp,
                baseline_action=action,
                task_desc=instruction,
            )
            return action_to_calvin_vector(output.corrected_action)
        return np.asarray(raw_action, dtype=np.float32)


@dataclass
class CALVINEvaluator:
    split: str = "ABC_D"

    def require_calvin(self) -> None:
        try:
            import calvin_env  # noqa: F401  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CALVIN is not installed. Install the official repo first: https://github.com/mees/calvin"
            ) from exc

    def run(self, policy: Any, sequences: int = 100) -> dict:
        self.require_calvin()
        if not hasattr(policy, "step"):
            raise TypeError("CALVIN policy must expose step(obs, goal).")
        return {
            "split": self.split,
            "sequences": sequences,
            "policy_interface": "CALVIN CustomModel",
            "message": (
                "Pass CALVINGraspCorrectModel as the CustomModel inside "
                "calvin_models/calvin_agent/evaluation/evaluate_policy.py, then run "
                "the official CALVIN evaluation command."
            ),
        }


def _baseline_step(baseline: Any, obs: dict, goal: Any, instruction: str) -> np.ndarray:
    if hasattr(baseline, "step"):
        return np.asarray(baseline.step(obs, goal), dtype=np.float32)
    observation = observation_from_calvin(obs)
    if hasattr(baseline, "act"):
        result = baseline.act(observation, instruction)
    elif hasattr(baseline, "predict"):
        result = baseline.predict(observation, instruction)
    else:
        raise AttributeError("Baseline must expose step(obs, goal), act(observation, instruction), or predict(...).")
    if isinstance(result, Action):
        return action_to_calvin_vector(result)
    return np.asarray(result, dtype=np.float32)


def _goal_to_text(goal: Any) -> str:
    if isinstance(goal, str):
        return goal
    if isinstance(goal, dict):
        for key in ("language", "lang", "instruction", "task"):
            if key in goal:
                return str(goal[key])
    return str(goal)


def _euler_to_quat(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=np.float32)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float32)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)
