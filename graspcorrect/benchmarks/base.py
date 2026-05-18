from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from graspcorrect.pipeline import GraspCorrectPipeline
from graspcorrect.types import Action, Observation


class PolicyAdapter(Protocol):
    def reset(self) -> None:
        ...

    def act(self, observation: Observation, instruction: str) -> Action:
        ...


@dataclass
class GraspMomentDetector:
    """Detects the first open-to-close transition in a baseline action stream."""

    close_threshold: float = 0.5
    previous_gripper: Optional[float] = None
    already_corrected: bool = False

    def reset(self) -> None:
        self.previous_gripper = None
        self.already_corrected = False

    def should_correct(self, action: Action) -> bool:
        if self.already_corrected:
            self.previous_gripper = action.gripper
            return False
        prev = self.previous_gripper
        self.previous_gripper = action.gripper
        if prev is None:
            return False
        should = prev > self.close_threshold and action.gripper <= self.close_threshold
        if should:
            self.already_corrected = True
        return should


@dataclass
class GraspCorrectPolicyWrapper:
    baseline: PolicyAdapter
    pipeline: GraspCorrectPipeline
    window: int = 10
    detector: GraspMomentDetector = field(default_factory=GraspMomentDetector)
    history: List[Observation] = field(default_factory=list)

    def reset(self) -> None:
        self.baseline.reset()
        self.detector.reset()
        self.history.clear()

    def act(self, observation: Observation, instruction: str) -> Action:
        action = self.baseline.act(observation, instruction)
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
            return output.corrected_action
        return action
