#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graspcorrect.data.dataset import GraspCorrectionDataset
from graspcorrect.policies.gcbc import GCBCConfig, GCBCDiffusionPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GraspCorrect GCBC DDPM policy.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise ImportError("Training requires torch. Install with `pip install -e .[train]`.") from exc

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    gcfg = cfg.get("gcbc", {})
    policy_cfg = GCBCConfig(
        image_size=int(gcfg.get("image_size", 224)),
        action_dim=int(gcfg.get("action_dim", 7)),
        hidden_dim=int(gcfg.get("hidden_dim", 256)),
        diffusion_steps=int(gcfg.get("diffusion_steps", 100)),
        beta_start=float(gcfg.get("beta_start", 0.0001)),
        beta_end=float(gcfg.get("beta_end", 0.02)),
        position_loss_weight=float(gcfg.get("position_loss_weight", 0.2)),
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = GraspCorrectionDataset(args.manifest, image_size=policy_cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(gcfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(gcfg.get("num_workers", 4)),
        drop_last=True,
    )
    policy = GCBCDiffusionPolicy(policy_cfg).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=float(gcfg.get("learning_rate", 5e-4)))
    epochs = int(gcfg.get("epochs", 50))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(epochs):
        policy.train()
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses = policy.training_loss(batch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            opt.step()
            running += float(losses["loss"].detach().cpu())
            step += 1
        mean_loss = running / max(len(loader), 1)
        print(f"epoch={epoch + 1}/{epochs} loss={mean_loss:.6f}")
        policy.save_checkpoint(out / "policy.pt")


if __name__ == "__main__":
    main()
