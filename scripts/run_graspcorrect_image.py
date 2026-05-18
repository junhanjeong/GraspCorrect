#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.detection.grasp_detector import GraspDetector, GraspDetectorConfig
from graspcorrect.detection.segmenters import HeuristicSegmenter
from graspcorrect.goal_generation.compositor import GoalComposer, GoalComposerConfig
from graspcorrect.pipeline import GraspCorrectPipeline
from graspcorrect.types import Action, Observation
from graspcorrect.utils.image import load_rgb, save_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraspCorrect detection and goal composition on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None, help="Optional binary mask image.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    image = load_rgb(args.image)
    mask = None
    if args.mask:
        mask = load_rgb(args.mask)[..., 0] > 0

    detector = GraspDetector(
        segmenter=HeuristicSegmenter(),
        config=GraspDetectorConfig(seed=args.seed),
    )
    composer = GoalComposer(GoalComposerConfig())
    pipeline = GraspCorrectPipeline(detector=detector, composer=composer, policy=None)
    dummy_action = Action(position=np.zeros(3), rotation=np.asarray([0, 0, 0, 1], dtype=np.float32), gripper=0.0)
    obs = Observation(rgb=image, camera="image")
    output = pipeline.correct(
        current=obs,
        pre_grasp=obs,
        baseline_action=dummy_action,
        task_desc=args.task,
        object_mask=mask,
    )
    save_rgb(output.goal_rgb, args.output)
    print(f"wrote {Path(args.output)}")
    print(f"left={output.grasp_pair.left.point} right={output.grasp_pair.right.point}")


if __name__ == "__main__":
    main()
