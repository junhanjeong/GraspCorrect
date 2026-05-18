import numpy as np

from graspcorrect.config import DetectorConfig
from graspcorrect.detection.contour import OrderedContour
from graspcorrect.detection.grasp_detector import GraspDetector
from graspcorrect.types import Observation


class FakeVLM:
    def describe_grasp(self, image, task_desc):
        return "Left: Position the left gripper on the left side. Right: Position the right gripper opposite it."

    def choose_points(self, annotated_image, description, top_n):
        return list(range(1, top_n + 1))


class FakeSegmenter:
    def __init__(self, mask):
        self.mask = mask

    def segment(self, image, text_prompt):
        return self.mask


def test_ordered_contour_resamples_on_boundary():
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 20:44] = True
    contour = OrderedContour.from_mask(mask)
    first = contour.sample_uniform(12)
    refined = contour.sample_gaussian(first[:3], 12, 0.05, np.random.default_rng(0))
    assert len(first) == 12
    assert len(refined) == 12
    for cand in refined:
        x, y = map(int, cand.point)
        assert mask[y, x]


def test_detector_uses_vlm_and_mask_contour():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 20:44] = True
    detector = GraspDetector(
        vlm=FakeVLM(),
        segmenter=FakeSegmenter(mask),
        config=DetectorConfig(iterations=2, candidates_per_iteration=8, top_n=3),
    )
    pair = detector.detect(Observation(image), "pick up the block", object_mask=mask)
    assert pair.left.point != pair.right.point
    assert pair.distance_px > 0
