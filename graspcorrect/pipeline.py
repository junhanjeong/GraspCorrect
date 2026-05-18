from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from graspcorrect.detection.grasp_detector import GraspDetector
from graspcorrect.goal_generation.compositor import GoalComposer
from graspcorrect.types import Action, GraspCorrectOutput, Observation


@dataclass
class GraspCorrectPipeline:
    """Plug-and-play GraspCorrect module.

    The pipeline corrects a baseline policy action at the grasping moment:
    VLM/mask guided contact detection -> visual goal image -> GCBC action.
    """

    detector: GraspDetector = field(default_factory=GraspDetector)
    composer: GoalComposer = field(default_factory=GoalComposer)
    policy: Optional[object] = None

    def correct(
        self,
        current: Observation,
        pre_grasp: Observation,
        baseline_action: Action,
        task_desc: str,
        target_prompt: Optional[str] = None,
        object_mask: Optional[np.ndarray] = None,
        gripper_mask: Optional[np.ndarray] = None,
    ) -> GraspCorrectOutput:
        grasp_pair = self.detector.detect(
            pre_grasp=pre_grasp,
            task_desc=task_desc,
            target_prompt=target_prompt,
            mask=object_mask,
        )
        if object_mask is None:
            object_mask = self.detector.segmenter.segment(pre_grasp.rgb, target_prompt or task_desc)
        goal = self.composer.compose(
            current_rgb=current.rgb,
            pre_grasp_rgb=pre_grasp.rgb,
            object_mask=object_mask,
            grasp_pair=grasp_pair,
            gripper_mask=gripper_mask,
        )
        corrected = baseline_action
        if self.policy is not None:
            corrected = self.policy.predict(
                current_rgb=current.rgb,
                goal_rgb=goal,
                current_action=baseline_action,
            )
        return GraspCorrectOutput(
            corrected_action=corrected,
            grasp_pair=grasp_pair,
            goal_rgb=goal,
            metadata={"policy_used": self.policy is not None},
        )
