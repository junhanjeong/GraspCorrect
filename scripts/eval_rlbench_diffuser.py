#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.rlbench.diffuser_eval import run_diffuser_rlbench_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3D Diffuser Actor on RLBench with optional GraspCorrect.")
    parser.add_argument("--checkpoint", default="external/3d_diffuser_actor/train_logs/diffuser_actor_peract.pth")
    parser.add_argument("--tasks", nargs="+", default=["insert_onto_square_peg"])
    parser.add_argument("--data-dir", default="external/3d_diffuser_actor/data/peract/raw/test")
    parser.add_argument("--instructions", default="external/3d_diffuser_actor/instructions/peract/instructions.pkl")
    parser.add_argument("--output-file", default="runs/rlbench_eval.json")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--variations", nargs="*", type=int, default=list(range(61)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--enable-graspcorrect", action="store_true")
    parser.add_argument("--gcbc-checkpoint", default=None)
    parser.add_argument("--langsam-python", default=None)
    parser.add_argument("--correction-camera", default="overhead")
    parser.add_argument("--object-prompt", default=None)
    parser.add_argument("--increased-distance-px", type=float, default=0.0)
    parser.add_argument("--inprocess-langsam", action="store_true")
    args = parser.parse_args()

    result = run_diffuser_rlbench_eval(
        checkpoint=Path(args.checkpoint),
        tasks=args.tasks,
        output_file=Path(args.output_file),
        data_dir=Path(args.data_dir),
        instructions=Path(args.instructions),
        num_episodes=args.num_episodes,
        variations=args.variations,
        device=args.device,
        seed=args.seed,
        headless=bool(args.headless),
        enable_graspcorrect=args.enable_graspcorrect,
        gcbc_checkpoint=None if args.gcbc_checkpoint is None else Path(args.gcbc_checkpoint),
        langsam_python=args.langsam_python,
        correction_camera=args.correction_camera,
        object_prompt=args.object_prompt,
        use_langsam_subprocess=not args.inprocess_langsam,
        increased_distance_px=args.increased_distance_px,
    )
    print(result)


if __name__ == "__main__":
    main()
