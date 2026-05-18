from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from graspcorrect.config import GoalConfig
from graspcorrect.types import GoalMetadata, GraspCandidate, GraspPair, Point2D
from graspcorrect.utils.image import (
    connected_components,
    ensure_rgb,
    inpaint_background,
    mask_to_bool,
    rgba_from_mask,
    rotate_points,
    scale_layer_about_point,
    translate_layer,
)


@dataclass
class GoalComposition:
    image: np.ndarray
    aligned_pair: GraspPair
    final_point_distance: float
    object_scale: float
    rotation_degrees: float


@dataclass
class GoalComposer:
    """Visual goal generation by conventional image composition.

    The paper only specifies the high-level transformations. This implementation
    follows the explicit parts first: inpaint current background, transform the
    pre-grasp object layer using contact points, then align gripper masks to the
    final contact-point distance.
    """

    config: GoalConfig = field(default_factory=GoalConfig)

    def compose(
        self,
        current_rgb: np.ndarray,
        pre_grasp_rgb: np.ndarray,
        object_mask_pre: np.ndarray,
        object_mask_current: np.ndarray,
        gripper_mask_current: np.ndarray,
        grasp_pair: GraspPair,
        metadata: Optional[GoalMetadata] = None,
    ) -> GoalComposition:
        current = ensure_rgb(current_rgb)
        pre = ensure_rgb(pre_grasp_rgb)
        object_mask_pre = mask_to_bool(object_mask_pre)
        object_mask_current = mask_to_bool(object_mask_current)
        gripper_mask_current = mask_to_bool(gripper_mask_current)
        metadata = metadata or GoalMetadata(grasp_description=grasp_pair.description)

        remove_mask = object_mask_current | gripper_mask_current
        background = inpaint_background(current, remove_mask, radius=self.config.inpaint_radius)

        object_layer = rgba_from_mask(pre, object_mask_pre)
        rotation = estimate_rotation_degrees(grasp_pair, metadata)
        center = image_center(pre.shape)
        rotated_layer = object_layer.rotate(rotation, resample=Image.BICUBIC, center=center, expand=False)
        left_point, right_point = rotate_points(
            [grasp_pair.left.point, grasp_pair.right.point],
            rotation,
            center,
        )
        rotated_pair = replace_pair_points(grasp_pair, left_point, right_point)

        aligned_object, aligned_pair, distance, scale = align_object_to_gripper(
            layer=rotated_layer,
            pair=rotated_pair,
            image_shape=current.shape,
            gripper_mask=gripper_mask_current,
            increased_distance_px=metadata.increased_distance_px or self.config.default_increased_distance_px,
        )
        aligned_gripper = align_gripper_to_distance(
            current=current,
            gripper_mask=gripper_mask_current,
            desired_inner_distance=distance,
        )

        composed = Image.fromarray(background).convert("RGBA")
        if aligned_gripper is not None:
            composed.alpha_composite(aligned_gripper)
        composed.alpha_composite(aligned_object)
        return GoalComposition(
            image=np.asarray(composed.convert("RGB")),
            aligned_pair=aligned_pair,
            final_point_distance=distance,
            object_scale=scale,
            rotation_degrees=rotation,
        )


def estimate_rotation_degrees(pair: GraspPair, metadata: GoalMetadata) -> float:
    if metadata.background_points_pre is not None and metadata.background_points_current is not None:
        pre_vec = np.asarray(metadata.background_points_pre[1]) - np.asarray(metadata.background_points_pre[0])
        cur_vec = np.asarray(metadata.background_points_current[1]) - np.asarray(metadata.background_points_current[0])
        if np.linalg.norm(pre_vec) > 1e-6 and np.linalg.norm(cur_vec) > 1e-6:
            a0 = np.degrees(np.arctan2(pre_vec[1], pre_vec[0]))
            a1 = np.degrees(np.arctan2(cur_vec[1], cur_vec[0]))
            return float(a1 - a0)

    # Paper: object alignment uses contact-point information. With parallel jaws
    # rendered left/right in the image, the contact line should become horizontal.
    p0 = np.asarray(pair.left.point, dtype=np.float32)
    p1 = np.asarray(pair.right.point, dtype=np.float32)
    angle = float(np.degrees(np.arctan2((p1 - p0)[1], (p1 - p0)[0])))
    return -angle


def align_object_to_gripper(
    layer: Image.Image,
    pair: GraspPair,
    image_shape: Tuple[int, ...],
    gripper_mask: np.ndarray,
    increased_distance_px: float,
) -> Tuple[Image.Image, GraspPair, float, float]:
    left = np.asarray(pair.left.point, dtype=np.float32)
    right = np.asarray(pair.right.point, dtype=np.float32)
    midpoint = 0.5 * (left + right)
    base_distance = float(np.linalg.norm(right - left))
    final_distance = max(1.0, base_distance + float(increased_distance_px))
    scale = final_distance / max(base_distance, 1e-6)

    scaled_layer = scale_layer_about_point(layer, tuple(midpoint), scale)
    scaled_points = midpoint + (np.stack([left, right]) - midpoint) * scale

    h, w = image_shape[:2]
    target_x = w * 0.5
    target_y = h * 0.5
    center_shift = np.asarray([target_x, target_y], dtype=np.float32) - scaled_points.mean(axis=0)
    centered_layer = translate_layer(scaled_layer, float(center_shift[0]), float(center_shift[1]))
    centered_points = scaled_points + center_shift

    target_gripper_y = gripper_contact_height(gripper_mask, image_shape)
    lower_shift = target_gripper_y - float(centered_points[:, 1].mean())
    lowered_layer = translate_layer(centered_layer, 0.0, lower_shift)
    lowered_points = centered_points + np.asarray([0.0, lower_shift], dtype=np.float32)
    return lowered_layer, replace_pair_points(pair, lowered_points[0], lowered_points[1]), final_distance, scale


def align_gripper_to_distance(
    current: np.ndarray,
    gripper_mask: np.ndarray,
    desired_inner_distance: float,
) -> Optional[Image.Image]:
    mask = mask_to_bool(gripper_mask)
    comps = [c for c in connected_components(mask) if len(c) > 20]
    if len(comps) < 2:
        return rgba_from_mask(current, mask)
    comps = sorted(comps, key=lambda c: np.asarray(c)[:, 1].mean())
    left_comp, right_comp = comps[0], comps[-1]
    left_arr = np.asarray(left_comp, dtype=np.int32)
    right_arr = np.asarray(right_comp, dtype=np.int32)
    left_inner_x = float(left_arr[:, 1].max())
    right_inner_x = float(right_arr[:, 1].min())
    center_x = current.shape[1] * 0.5
    target_left = center_x - desired_inner_distance * 0.5
    target_right = center_x + desired_inner_distance * 0.5

    rgba = np.zeros((*current.shape[:2], 4), dtype=np.uint8)
    src = np.dstack([ensure_rgb(current), np.full(current.shape[:2], 255, dtype=np.uint8)])
    _place_component(rgba, src, left_arr, int(round(target_left - left_inner_x)))
    _place_component(rgba, src, right_arr, int(round(target_right - right_inner_x)))
    extend_components_to_bottom(rgba)
    if not rgba[..., 3].any():
        return rgba_from_mask(current, mask)
    return Image.fromarray(rgba, mode="RGBA")


def gripper_contact_height(gripper_mask: np.ndarray, image_shape: Tuple[int, ...]) -> float:
    ys = np.nonzero(mask_to_bool(gripper_mask))[0]
    if len(ys) == 0:
        return float(image_shape[0]) * 0.72
    return float(ys.min())


def extend_components_to_bottom(rgba: np.ndarray) -> None:
    alpha = rgba[..., 3] > 0
    h = rgba.shape[0]
    for comp in connected_components(alpha):
        arr = np.asarray(comp, dtype=np.int32)
        if arr.size == 0:
            continue
        bottom_y = int(arr[:, 0].max())
        for x in np.unique(arr[arr[:, 0] == bottom_y][:, 1]):
            rgba[bottom_y:h, int(x)] = rgba[bottom_y, int(x)]


def image_center(shape: Tuple[int, ...]) -> Point2D:
    return (float(shape[1]) * 0.5, float(shape[0]) * 0.5)


def replace_pair_points(pair: GraspPair, left: Point2D, right: Point2D) -> GraspPair:
    left_c = GraspCandidate(pair.left.label, (float(left[0]), float(left[1])), pair.left.contour_index)
    right_c = GraspCandidate(pair.right.label, (float(right[0]), float(right[1])), pair.right.contour_index)
    return GraspPair(left=left_c, right=right_c, description=pair.description, object_prompt=pair.object_prompt)


def _place_component(out: np.ndarray, src: np.ndarray, comp_yx: np.ndarray, dx: int) -> None:
    h, w = out.shape[:2]
    ys = comp_yx[:, 0]
    xs = comp_yx[:, 1]
    nx = xs + dx
    valid = (nx >= 0) & (nx < w) & (ys >= 0) & (ys < h)
    out[ys[valid], nx[valid]] = src[ys[valid], xs[valid]]
