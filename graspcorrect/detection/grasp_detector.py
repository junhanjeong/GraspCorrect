from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from graspcorrect.detection.contour import (
    antipodal_score,
    candidates_by_label,
    contour_centroid,
    gaussian_resample_candidates,
    sample_initial_candidates,
)
from graspcorrect.detection.segmenters import HeuristicSegmenter, Segmenter
from graspcorrect.types import GraspCandidate, GraspPair, Observation
from graspcorrect.utils.image import draw_cross, draw_numbered_circles, mask_to_bool
from graspcorrect.vlm.base import HeuristicVLMClient, VLMClient


@dataclass
class GraspDetectorConfig:
    window: int = 10
    iterations: int = 4
    num_candidates: int = 24
    top_n: int = 3
    gaussian_sigma_fraction: float = 0.08
    min_mask_area: int = 64
    seed: int = 7


@dataclass
class GraspDetector:
    """VLM-guided, object-aware grasp point detector from Section 3.1."""

    vlm: VLMClient = field(default_factory=HeuristicVLMClient)
    segmenter: Segmenter = field(default_factory=HeuristicSegmenter)
    config: GraspDetectorConfig = field(default_factory=GraspDetectorConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)

    def detect(
        self,
        pre_grasp: Observation,
        task_desc: str,
        target_prompt: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
    ) -> GraspPair:
        image = pre_grasp.rgb
        if mask is None:
            mask = self.segmenter.segment(image, target_prompt or task_desc)
        mask_bool = mask_to_bool(mask, min_area=self.config.min_mask_area)
        description = self.vlm.describe_grasp(image, task_desc)

        left = self._iterative_select(
            image=image,
            mask=mask_bool,
            description=_left_description(description),
            role="left",
        )
        right_image = draw_cross(image, left.point, color=(230, 45, 45))
        right = self._iterative_select(
            image=right_image,
            mask=mask_bool,
            description=_right_description(description, left),
            role="right",
            anchor=left,
        )

        return GraspPair(
            left=left,
            right=right,
            description=description,
            metadata={"mask_area": int(mask_bool.sum()), "task_desc": task_desc},
        )

    def _iterative_select(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        description: str,
        role: str,
        anchor: Optional[GraspCandidate] = None,
    ) -> GraspCandidate:
        candidates = sample_initial_candidates(mask, self.config.num_candidates, self.rng)
        selected: Sequence[GraspCandidate] = []
        for iteration in range(self.config.iterations):
            annotated = draw_numbered_circles(image, candidates)
            top_n = 1 if iteration == self.config.iterations - 1 else self.config.top_n
            labels = self._choose_labels(annotated, candidates, description, top_n, role, anchor)
            selected = candidates_by_label(candidates, labels)
            if not selected:
                selected = self._heuristic_select(candidates, top_n, role, anchor, mask)
            if iteration == self.config.iterations - 1:
                return selected[0]
            candidates = gaussian_resample_candidates(
                mask,
                selected,
                self.config.num_candidates,
                sigma_fraction=self.config.gaussian_sigma_fraction,
                rng=self.rng,
            )
        return list(selected)[0]

    def _choose_labels(
        self,
        annotated: np.ndarray,
        candidates: Sequence[GraspCandidate],
        description: str,
        top_n: int,
        role: str,
        anchor: Optional[GraspCandidate],
    ) -> list[int]:
        if isinstance(self.vlm, HeuristicVLMClient):
            return [candidate.label for candidate in self._heuristic_select(candidates, top_n, role, anchor, None)]
        try:
            labels = self.vlm.choose_points(annotated, description, top_n)
        except Exception:
            labels = []
        valid = {candidate.label for candidate in candidates}
        return [label for label in labels if label in valid][:top_n]

    def _heuristic_select(
        self,
        candidates: Sequence[GraspCandidate],
        top_n: int,
        role: str,
        anchor: Optional[GraspCandidate],
        mask: Optional[np.ndarray],
    ) -> list[GraspCandidate]:
        if not candidates:
            raise ValueError("No candidates to select from.")
        points = np.asarray([candidate.point for candidate in candidates], dtype=np.float32)
        center = points.mean(axis=0)
        if mask is not None:
            center = np.asarray(contour_centroid(mask), dtype=np.float32)

        if role == "left":
            scores = -(points[:, 0] - center[0]) - 0.15 * np.abs(points[:, 1] - center[1])
        elif anchor is not None:
            scores = np.asarray(
                [antipodal_score(anchor.point, candidate.point, tuple(center)) for candidate in candidates],
                dtype=np.float32,
            )
            scores += 0.05 * (points[:, 0] - center[0])
        else:
            scores = (points[:, 0] - center[0]) - 0.15 * np.abs(points[:, 1] - center[1])
        order = np.argsort(-scores)
        return [candidates[int(i)] for i in order[:top_n]]


def _left_description(description: str) -> str:
    return f"{description}\nFocus only on the left gripper contact point."


def _right_description(description: str, left: GraspCandidate) -> str:
    x, y = left.point
    return (
        f"{description}\nBe aware that the red circle indicates the left gripper's contact position "
        f"near ({x:.1f}, {y:.1f}) in image space. Focus on the matching right gripper contact point."
    )
