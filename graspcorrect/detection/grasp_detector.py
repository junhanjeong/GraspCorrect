from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from graspcorrect.config import DetectorConfig
from graspcorrect.detection.contour import OrderedContour, select_by_labels
from graspcorrect.perception.langsam import Segmenter
from graspcorrect.types import GraspCandidate, GraspPair, Observation
from graspcorrect.utils.image import draw_candidates, draw_cross, mask_to_bool
from graspcorrect.vlm.openai_client import VLMClient


@dataclass
class GraspDetector:
    """Paper-style VLM-guided contour refinement.

    The object mask is converted into a 1D contour. Candidates are sampled on
    this contour, GPT-5.4 mini chooses top candidates, and the next candidates are
    drawn from Gaussian distributions over the contour index.
    """

    vlm: VLMClient
    segmenter: Segmenter
    config: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)

    def detect(
        self,
        pre_grasp: Observation,
        task_desc: str,
        object_prompt: Optional[str] = None,
        object_mask: Optional[np.ndarray] = None,
    ) -> GraspPair:
        prompt = object_prompt or task_desc
        mask = object_mask if object_mask is not None else self.segmenter.segment(pre_grasp.rgb, prompt)
        mask_bool = mask_to_bool(mask, min_area=self.config.min_mask_area)
        contour = OrderedContour.from_mask(mask_bool)
        description = self.vlm.describe_grasp(pre_grasp.rgb, task_desc)

        left = self._select_one(
            image=pre_grasp.rgb,
            contour=contour,
            description=_left_description(description),
            role="left",
        )
        right_image = draw_cross(pre_grasp.rgb, left.point)
        right = self._select_one(
            image=right_image,
            contour=contour,
            description=_right_description(description, left),
            role="right",
            anchor=left,
        )
        return GraspPair(left=left, right=right, description=description, object_prompt=prompt)

    def _select_one(
        self,
        image: np.ndarray,
        contour: OrderedContour,
        description: str,
        role: str,
        anchor: Optional[GraspCandidate] = None,
    ) -> GraspCandidate:
        candidates = contour.sample_uniform(self.config.candidates_per_iteration)
        selected: List[GraspCandidate] = []
        for iteration in range(self.config.iterations):
            top_n = 1 if iteration == self.config.iterations - 1 else self.config.top_n
            annotated = draw_candidates(image, candidates)
            labels = self.vlm.choose_points(annotated, description, top_n)
            selected = select_by_labels(candidates, labels)
            if not selected:
                selected = self._geometric_fallback(candidates, role, anchor, top_n)
            if iteration == self.config.iterations - 1:
                return selected[0]
            candidates = contour.sample_gaussian(
                selected=selected,
                count=self.config.candidates_per_iteration,
                sigma_fraction=self.config.gaussian_sigma_fraction,
                rng=self.rng,
            )
        return selected[0]

    def _geometric_fallback(
        self,
        candidates: Sequence[GraspCandidate],
        role: str,
        anchor: Optional[GraspCandidate],
        top_n: int,
    ) -> List[GraspCandidate]:
        pts = np.asarray([c.point for c in candidates], dtype=np.float32)
        center = pts.mean(axis=0)
        if role == "left":
            score = -(pts[:, 0] - center[0])
        elif anchor is not None:
            anchor_pt = np.asarray(anchor.point, dtype=np.float32)
            dist = np.linalg.norm(pts - anchor_pt[None], axis=1)
            opposite = pts[:, 0] - center[0]
            score = dist + 0.25 * opposite
        else:
            score = pts[:, 0] - center[0]
        order = np.argsort(-score)
        return [candidates[int(i)] for i in order[:top_n]]


def _left_description(description: str) -> str:
    return description + "\nFocus only on the left gripper contact point."


def _right_description(description: str, left: GraspCandidate) -> str:
    x, y = left.point
    return (
        description
        + "\nBe aware that the red circle indicates the left gripper's contact position "
        + f"near ({x:.1f}, {y:.1f}) in image space. Focus on the matching right gripper contact point."
    )
