from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from graspcorrect.config import DetectorConfig, GoalConfig, SegmenterConfig, VLMConfig
from graspcorrect.detection import GraspDetector
from graspcorrect.goal_generation import GoalComposer
from graspcorrect.perception import LangSAMSegmenter, SubprocessLangSAMSegmenter
from graspcorrect.perception.langsam import Segmenter
from graspcorrect.policies import GCBCDiffusionPolicy
from graspcorrect.types import Action, CorrectionResult, GoalMetadata, Observation
from graspcorrect.utils.image import extract_observation_rgb
from graspcorrect.vlm import OpenAIGPT54MiniVLM
from graspcorrect.vlm.openai_client import VLMClient


@dataclass
class GraspCorrectPipeline:
    detector: GraspDetector
    composer: GoalComposer
    segmenter: Segmenter
    policy: Optional[GCBCDiffusionPolicy] = None
    goal_config: GoalConfig = field(default_factory=GoalConfig)

    @classmethod
    def build(
        cls,
        policy: Optional[GCBCDiffusionPolicy] = None,
        use_langsam_subprocess: bool = True,
        langsam_python: Optional[str] = None,
        correction_camera: str = "overhead",
        increased_distance_px: float = 0.0,
    ) -> "GraspCorrectPipeline":
        goal_cfg = GoalConfig(
            correction_camera=correction_camera,
            default_increased_distance_px=increased_distance_px,
        )
        vlm = OpenAIGPT54MiniVLM(VLMConfig())
        seg_cfg = SegmenterConfig(langsam_python=langsam_python)
        segmenter = SubprocessLangSAMSegmenter(seg_cfg) if use_langsam_subprocess else LangSAMSegmenter(seg_cfg)
        detector = GraspDetector(vlm=vlm, segmenter=segmenter, config=DetectorConfig())
        return cls(detector=detector, composer=GoalComposer(goal_cfg), segmenter=segmenter, policy=policy, goal_config=goal_cfg)

    def correct(
        self,
        current_obs: object,
        pre_grasp_obs: object,
        baseline_action: np.ndarray,
        task_desc: str,
        object_prompt: Optional[str] = None,
    ) -> CorrectionResult:
        camera = self.goal_config.correction_camera
        current_rgb = extract_observation_rgb(current_obs, camera)
        pre_rgb = extract_observation_rgb(pre_grasp_obs, camera)
        prompt = object_prompt or task_desc

        object_mask_pre = self.segmenter.segment(pre_rgb, prompt)
        object_mask_current = self.segmenter.segment(current_rgb, prompt)
        gripper_mask_current = self.segmenter.segment(current_rgb, self.goal_config.gripper_prompt)

        pair = self.detector.detect(
            pre_grasp=Observation(rgb=pre_rgb, raw=pre_grasp_obs, camera=camera),
            task_desc=task_desc,
            object_prompt=prompt,
            object_mask=object_mask_pre,
        )
        goal_meta = GoalMetadata(
            grasp_description=pair.description,
            increased_distance_px=self.goal_config.default_increased_distance_px,
        )
        goal = self.composer.compose(
            current_rgb=current_rgb,
            pre_grasp_rgb=pre_rgb,
            object_mask_pre=object_mask_pre,
            object_mask_current=object_mask_current,
            gripper_mask_current=gripper_mask_current,
            grasp_pair=pair,
            metadata=goal_meta,
        )
        action = Action.from_vector(baseline_action)
        corrected = action
        used_policy = False
        if self.policy is not None:
            corrected = self.policy.predict(current_rgb=current_rgb, goal_rgb=goal.image, current_action=action)
            used_policy = True
        return CorrectionResult(
            corrected_action=corrected,
            goal_rgb=goal.image,
            grasp_pair=pair,
            metadata={
                "policy_used": used_policy,
                "final_point_distance": goal.final_point_distance,
                "object_scale": goal.object_scale,
                "rotation_degrees": goal.rotation_degrees,
            },
        )
