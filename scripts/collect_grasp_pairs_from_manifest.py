#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.data.dataset import identify_grasp_index, perturb_action, save_npz_sample
from graspcorrect.detection.grasp_detector import GraspDetector
from graspcorrect.detection.segmenters import HeuristicSegmenter
from graspcorrect.goal_generation.compositor import GoalComposer
from graspcorrect.types import Observation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GraspCorrect training pairs from saved episode npz files."
    )
    parser.add_argument("--episodes", required=True, help="JSONL with {'path': episode.npz, 'task': text}.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--samples-per-episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = GraspDetector(segmenter=HeuristicSegmenter())
    composer = GoalComposer()

    entries = []
    with open(args.episodes, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            episode_path = Path(item["path"])
            task = item.get("task", "")
            with np.load(episode_path) as ep:
                rgbs = ep["rgbs"]
                actions = ep["actions"]
                masks = ep["masks"] if "masks" in ep else None
            grasp_idx = identify_grasp_index(actions)
            if grasp_idx is None or grasp_idx - args.window < 0:
                print(f"[skip] {episode_path}: no usable grasp transition")
                continue
            pre_idx = grasp_idx - args.window
            pre = Observation(rgb=rgbs[pre_idx])
            current = rgbs[grasp_idx]
            mask = masks[pre_idx] if masks is not None else detector.segmenter.segment(pre.rgb, task)
            pair = detector.detect(pre, task_desc=task, mask=mask)
            goal = composer.compose(current, pre.rgb, mask, pair)
            for sample_i in range(args.samples_per_episode):
                current_action = perturb_action(actions[grasp_idx], rng=rng)
                target_action = actions[grasp_idx]
                out_path = out_dir / f"sample_{len(entries):06d}.npz"
                save_npz_sample(out_path, current, goal, current_action, target_action)
                entries.append({"path": str(out_path), "task": task})

    manifest = Path(args.manifest_out)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"wrote {len(entries)} samples -> {manifest}")


if __name__ == "__main__":
    main()
