import numpy as np

from graspcorrect.goal_generation import GoalComposer
from graspcorrect.types import GoalMetadata, GraspCandidate, GraspPair


def test_goal_composer_aligns_object_and_gripper():
    current = np.full((96, 96, 3), 180, dtype=np.uint8)
    pre = current.copy()
    pre[28:52, 34:62] = [20, 180, 220]
    object_mask_pre = np.zeros((96, 96), dtype=bool)
    object_mask_pre[28:52, 34:62] = True
    object_mask_current = np.zeros((96, 96), dtype=bool)
    object_mask_current[50:74, 34:62] = True
    gripper_mask = np.zeros((96, 96), dtype=bool)
    gripper_mask[70:95, 12:28] = True
    gripper_mask[70:95, 68:84] = True
    current[gripper_mask] = [10, 10, 10]

    pair = GraspPair(
        left=GraspCandidate(1, (34, 40), 0),
        right=GraspCandidate(2, (62, 40), 1),
        description="grasp sides",
    )
    result = GoalComposer().compose(
        current,
        pre,
        object_mask_pre,
        object_mask_current,
        gripper_mask,
        pair,
        GoalMetadata(increased_distance_px=10.0),
    )
    assert result.image.shape == current.shape
    assert result.final_point_distance > pair.distance_px
    assert abs(result.aligned_pair.left.point[1] - result.aligned_pair.right.point[1]) < 1.0
