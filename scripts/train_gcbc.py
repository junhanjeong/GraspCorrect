#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.data import GraspCorrectionDataset
from graspcorrect.policies import GCBCConfig, GCBCDiffusionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraspCorrect GCBC.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = GCBCConfig(image_size=args.image_size, diffusion_steps=args.diffusion_steps)
    dataset = GraspCorrectionDataset(args.manifest, image_size=cfg.image_size, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    policy = GCBCDiffusionPolicy(cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def lr_lambda(step: int) -> float:
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        policy.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses = policy.training_loss(batch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            scheduler.step()
            total += float(losses["loss"].detach().cpu())
            global_step += 1
        mean_loss = total / max(1, len(loader))
        print(f"epoch={epoch + 1}/{args.epochs} step={global_step} loss={mean_loss:.6f}")
        policy.save_checkpoint(out / "latest.pt")
    policy.save_checkpoint(out / "policy.pt")


if __name__ == "__main__":
    main()
