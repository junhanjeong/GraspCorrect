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
from graspcorrect.benchmarks.calvin import CALVINEvaluator, CALVINGraspCorrectModel
from graspcorrect.policies.gcbc import GCBCDiffusionPolicy
from graspcorrect.pipeline import GraspCorrectPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CALVIN GraspCorrect evaluation entry point.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline", default="3d_diffuser_actor")
    parser.add_argument("--baseline-root", default="external/3d_diffuser_actor")
    parser.add_argument("--gcbc-checkpoint", required=True)
    parser.add_argument("--policy-import", default=None, help="Baseline class as module:ClassName.")
    parser.add_argument("--policy-kwargs-json", default="{}", help="JSON kwargs passed to the baseline class.")
    args = parser.parse_args()

    if args.policy_import is None:
        raise SystemExit(
            "Pass --policy-import module:ClassName for the local baseline inference class, then use the "
            "printed CALVIN CustomModel wrapper in the official evaluate_policy.py flow."
        )
    baseline = PythonClassPolicyAdapter(args.policy_import, kwargs=json.loads(args.policy_kwargs_json))
    gcbc = GCBCDiffusionPolicy.from_checkpoint(args.gcbc_checkpoint)
    model = CALVINGraspCorrectModel(baseline=baseline, pipeline=GraspCorrectPipeline(policy=gcbc))
    evaluator = CALVINEvaluator()
    print(evaluator.run(model))


if __name__ == "__main__":
    main()
