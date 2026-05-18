from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from graspcorrect.types import Action, Observation


@dataclass
class PythonClassPolicyAdapter:
    """Adapter for official baselines that expose a Python policy class."""

    import_path: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        module_name, class_name = self.import_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self.policy = cls(**self.kwargs)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Observation, instruction: str) -> Action:
        if hasattr(self.policy, "act"):
            result = self.policy.act(observation, instruction)
        elif hasattr(self.policy, "predict"):
            result = self.policy.predict(observation, instruction)
        else:
            raise AttributeError(f"{self.policy!r} has neither act nor predict.")
        return _coerce_action(result)


@dataclass
class SubprocessPolicyAdapter:
    """JSON stdin/stdout bridge for baselines with script-only inference."""

    command: list[str]
    cwd: Optional[Path] = None

    def reset(self) -> None:
        return None

    def act(self, observation: Observation, instruction: str) -> Action:
        payload = {
            "instruction": instruction,
            "camera": observation.camera,
            "rgb": observation.rgb.tolist(),
            "depth": None if observation.depth is None else observation.depth.tolist(),
        }
        proc = subprocess.run(
            self.command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=self.cwd,
            check=True,
        )
        return _coerce_action(json.loads(proc.stdout))


def _coerce_action(value: Any) -> Action:
    if isinstance(value, Action):
        return value
    if isinstance(value, dict):
        if "action" in value:
            return Action.from_vector(value["action"])
        return Action(
            position=np.asarray(value["position"], dtype=np.float32),
            rotation=np.asarray(value["rotation"], dtype=np.float32),
            gripper=float(value.get("gripper", 1.0)),
        )
    return Action.from_vector(value)
