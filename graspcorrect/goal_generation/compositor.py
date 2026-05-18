from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import ImageDraw

from graspcorrect.types import GraspPair
from graspcorrect.utils.image import (
    bbox_from_mask,
    blur_mask_region,
    ensure_uint8_rgb,
    foreground_rgba,
    mask_to_bool,
    median_fill,
    paste_rgba,
    to_pil,
)


@dataclass
class GoalComposerConfig:
    gripper_width: int = 12
    gripper_length: int = 48
    contact_radius: int = 5
    inpaint_blur_radius: int = 7
    background_fill: str = "median"


@dataclass
class GoalComposer:
    """Image-composition goal generator from Section 3.2.

    The paper uses LaMa to restore the occluded background. This implementation
    uses a deterministic inpainting fallback by default and allows callers to
    pass an external inpainted background if they use LaMa.
    """

    config: GoalComposerConfig = field(default_factory=GoalComposerConfig)

    def compose(
        self,
        current_rgb: np.ndarray,
        pre_grasp_rgb: np.ndarray,
        object_mask: np.ndarray,
        grasp_pair: GraspPair,
        gripper_mask: Optional[np.ndarray] = None,
        inpainted_background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        current = ensure_uint8_rgb(current_rgb)
        pre = ensure_uint8_rgb(pre_grasp_rgb)
        mask = mask_to_bool(object_mask)

        if inpainted_background is not None:
            background = ensure_uint8_rgb(inpainted_background)
        elif gripper_mask is not None:
            background = blur_mask_region(median_fill(current, gripper_mask), gripper_mask, self.config.inpaint_blur_radius)
        else:
            background = current.copy()

        object_rgba = foreground_rgba(pre, mask)
        x0, y0, x1, y1 = bbox_from_mask(mask)
        object_crop = object_rgba.crop((x0, y0, x1, y1))
        composed = paste_rgba(background, object_crop, (x0, y0))
        composed = self._draw_parallel_jaw_gripper(composed, grasp_pair)
        return composed

    def _draw_parallel_jaw_gripper(self, image: np.ndarray, pair: GraspPair) -> np.ndarray:
        pil = to_pil(image)
        draw = ImageDraw.Draw(pil, "RGBA")
        left = np.asarray(pair.left.point, dtype=np.float32)
        right = np.asarray(pair.right.point, dtype=np.float32)
        axis = right - left
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            axis = np.asarray([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        axis = axis / norm
        tangent = np.asarray([-axis[1], axis[0]], dtype=np.float32)
        for point, side in ((left, -1.0), (right, 1.0)):
            center = point + side * axis * (0.5 * self.config.gripper_width)
            polygon = _oriented_rectangle(
                center=center,
                long_axis=tangent,
                width=self.config.gripper_width,
                length=self.config.gripper_length,
            )
            draw.polygon([tuple(p) for p in polygon], fill=(45, 45, 45, 230), outline=(245, 245, 245, 255))
            r = self.config.contact_radius
            draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=(35, 150, 255, 230))
        return np.asarray(pil.convert("RGB"))


def _oriented_rectangle(
    center: np.ndarray,
    long_axis: np.ndarray,
    width: float,
    length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    long_axis = np.asarray(long_axis, dtype=np.float32)
    long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-6)
    short_axis = np.asarray([long_axis[1], -long_axis[0]], dtype=np.float32)
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    return (
        center - half_l * long_axis - half_w * short_axis,
        center + half_l * long_axis - half_w * short_axis,
        center + half_l * long_axis + half_w * short_axis,
        center - half_l * long_axis + half_w * short_axis,
    )
