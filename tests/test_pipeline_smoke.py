import numpy as np

from graspcorrect.pipeline import GraspCorrectPipeline
from graspcorrect.types import Action, Observation


def test_pipeline_smoke_generates_goal_image():
    image = np.full((96, 96, 3), 230, dtype=np.uint8)
    image[32:64, 28:68] = np.asarray([40, 130, 220], dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[32:64, 28:68] = 255
    action = Action([0, 0, 0], [0, 0, 0, 1], 0)
    output = GraspCorrectPipeline().correct(
        current=Observation(image),
        pre_grasp=Observation(image),
        baseline_action=action,
        task_desc="pick up the blue block",
        object_mask=mask,
    )
    assert output.goal_rgb.shape == image.shape
    assert output.grasp_pair.width_px > 10
