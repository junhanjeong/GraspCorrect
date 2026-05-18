#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.baselines.base import PythonClassPolicyAdapter
from graspcorrect.benchmarks.base import GraspCorrectPolicyWrapper
from graspcorrect.benchmarks.rlbench import RLBenchEvaluator
from graspcorrect.policies.gcbc import GCBCDiffusionPolicy
from graspcorrect.pipeline import GraspCorrectPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="RLBench GraspCorrect evaluation entry point.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline", default="3d_diffuser_actor")
    parser.add_argument("--baseline-root", default="external/3d_diffuser_actor")
    parser.add_argument("--gcbc-checkpoint", required=True)
    parser.add_argument("--task", default="insert_peg")
    parser.add_argument("--camera", default="overhead")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--policy-import", default=None, help="Baseline class as module:ClassName.")
    parser.add_argument("--policy-kwargs-json", default="{}", help="JSON kwargs passed to the baseline class.")
    args = parser.parse_args()

    if args.policy_import is None:
        raise SystemExit(
            "Pass --policy-import module:ClassName for the local baseline inference class. "
            "Official baselines expose different APIs, so this CLI uses an explicit adapter."
        )
    kwargs = json.loads(args.policy_kwargs_json)
    baseline = PythonClassPolicyAdapter(args.policy_import, kwargs=kwargs)
    gcbc = GCBCDiffusionPolicy.from_checkpoint(args.gcbc_checkpoint)
    pipeline = GraspCorrectPipeline(policy=gcbc)
    policy = GraspCorrectPolicyWrapper(baseline=baseline, pipeline=pipeline)
    evaluator = RLBenchEvaluator(task_name=args.task, camera=args.camera)
    print(evaluator.run(policy, episodes=args.episodes))


if __name__ == "__main__":
    main()
