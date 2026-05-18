#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.rlbench.collect_gcbc import (
    WaypointPerturbConfig,
    collect_rlbench_gcbc_dataset,
    collect_rlbench_gcbc_dataset_from_stored_demos,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RLBench GCBC training pairs with waypoint perturbation.")
    parser.add_argument("--tasks", nargs="+", default=["insert_onto_square_peg"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["stored", "live"], default="stored")
    parser.add_argument("--data-dir", default="external/3d_diffuser_actor/data/peract/raw/test")
    parser.add_argument("--samples-per-task", type=int, default=200, help="Total paired examples to collect per task.")
    parser.add_argument("--variations", nargs="*", type=int, default=[0])
    parser.add_argument("--correction-camera", default="overhead")
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--position-sigma", type=float, default=0.015)
    parser.add_argument("--rotation-sigma", type=float, default=0.08)
    parser.add_argument("--perturb-all-waypoints", action="store_true")
    parser.add_argument("--demo-max-attempts", type=int, default=10)
    parser.add_argument("--replay-max-tries", type=int, default=2)
    args = parser.parse_args()

    perturb = WaypointPerturbConfig(
        position_sigma=args.position_sigma,
        rotation_sigma=args.rotation_sigma,
        perturb_all_waypoints=args.perturb_all_waypoints,
    )
    if args.source == "stored":
        manifest = collect_rlbench_gcbc_dataset_from_stored_demos(
            tasks=args.tasks,
            output_dir=Path(args.output_dir),
            samples_per_task=args.samples_per_task,
            variations=args.variations,
            data_dir=Path(args.data_dir),
            correction_camera=args.correction_camera,
            headless=bool(args.headless),
            seed=args.seed,
            perturb_config=perturb,
            replay_max_tries=args.replay_max_tries,
        )
    else:
        manifest = collect_rlbench_gcbc_dataset(
            tasks=args.tasks,
            output_dir=Path(args.output_dir),
            samples_per_task=args.samples_per_task,
            variations=args.variations,
            correction_camera=args.correction_camera,
            headless=bool(args.headless),
            seed=args.seed,
            perturb_config=perturb,
            demo_max_attempts=args.demo_max_attempts,
        )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
