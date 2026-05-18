from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from graspcorrect.types import Action, Observation


def observation_from_rlbench(obs: Any, camera: str = "front") -> Observation:
    rgb = getattr(obs, f"{camera}_rgb")
    depth = getattr(obs, f"{camera}_depth", None)
    misc = getattr(obs, "misc", {}) or {}
    return Observation(
        rgb=np.asarray(rgb),
        depth=None if depth is None else np.asarray(depth),
        camera=camera,
        intrinsics=misc.get(f"{camera}_camera_intrinsics"),
        extrinsics=misc.get(f"{camera}_camera_extrinsics"),
        metadata={"raw": obs},
    )


def action_from_rlbench_vector(action: np.ndarray) -> Action:
    return Action.from_vector(action)


def action_to_rlbench_vector(action: Action) -> np.ndarray:
    return action.as_vector(include_gripper=True)


@dataclass
class RLBenchEvaluator:
    """Thin evaluator shell.

    This intentionally keeps benchmark-specific imports local because RLBench
    requires CoppeliaSim/PyRep system setup.
    """

    task_name: str
    camera: str = "front"
    top_camera: str = "overhead"
    headless: bool = True
    max_steps: int = 200

    def require_rlbench(self) -> None:
        try:
            import rlbench  # noqa: F401  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "RLBench is not installed or CoppeliaSim/PyRep is not configured. "
                "Install the official RLBench repo first: https://github.com/stepjam/RLBench"
            ) from exc

    def run(self, policy: Any, episodes: int = 25) -> dict:
        self.require_rlbench()
        from rlbench.action_modes.action_mode import MoveArmThenGripper  # type: ignore
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning  # type: ignore
        from rlbench.action_modes.gripper_action_modes import Discrete  # type: ignore
        from rlbench.environment import Environment  # type: ignore
        from rlbench.observation_config import ObservationConfig  # type: ignore

        task_cls = _load_task_class(self.task_name)
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )
        env = Environment(action_mode=action_mode, dataset_root="", obs_config=obs_config, headless=self.headless)
        env.launch()
        successes = 0
        try:
            task = env.get_task(task_cls)
            for _ in range(episodes):
                if hasattr(policy, "reset"):
                    policy.reset()
                descriptions, obs = task.reset()
                instruction = descriptions[0] if descriptions else self.task_name.replace("_", " ")
                success = False
                for _step in range(self.max_steps):
                    observation = observation_from_rlbench(obs, self.camera)
                    action = policy.act(observation, instruction)
                    obs, reward, terminate = task.step(action_to_rlbench_vector(action))
                    if terminate:
                        success = bool(reward)
                        break
                successes += int(success)
        finally:
            env.shutdown()
        return {"task": self.task_name, "episodes": episodes, "success_rate": successes / max(episodes, 1)}


def _load_task_class(task_name: str) -> Any:
    import importlib

    module_name = task_name
    class_name = "".join(part.capitalize() for part in task_name.split("_"))
    module = importlib.import_module(f"rlbench.tasks.{module_name}")
    return getattr(module, class_name)
