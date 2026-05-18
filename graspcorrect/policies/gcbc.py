from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from graspcorrect.types import Action

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass
class GCBCConfig:
    image_size: int = 224
    action_dim: int = 7
    current_action_dim: int = 8
    hidden_dim: int = 256
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    position_loss_weight: float = 0.2


if torch is not None:

    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


    def group_norm(channels: int) -> nn.GroupNorm:
        groups = min(32, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)


    class ImageEncoder(nn.Module):
        def __init__(self, in_channels: int = 6) -> None:
            super().__init__()
            try:
                from torchvision.models import resnet34  # type: ignore

                self.net = resnet34(weights=None, norm_layer=group_norm)
                self.net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.net.fc = nn.Identity()
                self.out_dim = 512
            except Exception:
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
                    group_norm(32),
                    Swish(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    group_norm(64),
                    Swish(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    group_norm(128),
                    Swish(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.out_dim = 128

        def forward(self, x):
            return self.net(x)


    class TimeEmbedding(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
            emb = torch.cat([torch.sin(t.float()[:, None] * freqs[None]), torch.cos(t.float()[:, None] * freqs[None])], dim=-1)
            if self.dim % 2:
                emb = F.pad(emb, (0, 1))
            return emb


    class Denoiser(nn.Module):
        def __init__(self, config: GCBCConfig) -> None:
            super().__init__()
            self.encoder = ImageEncoder(6)
            self.time = nn.Sequential(TimeEmbedding(64), nn.Linear(64, config.hidden_dim), Swish())
            in_dim = self.encoder.out_dim + config.hidden_dim + config.current_action_dim + config.action_dim
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, config.hidden_dim),
                Swish(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                Swish(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                Swish(),
                nn.Linear(config.hidden_dim, config.action_dim),
            )

        def forward(self, current_image, goal_image, current_action, noisy_action, timesteps):
            visual = self.encoder(torch.cat([current_image, goal_image], dim=1))
            t = self.time(timesteps)
            return self.mlp(torch.cat([visual, t, current_action, noisy_action], dim=-1))


    class DDPMScheduler(nn.Module):
        def __init__(self, config: GCBCConfig) -> None:
            super().__init__()
            betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
            self.register_buffer("betas", betas)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
            self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
            self.register_buffer("posterior_variance", (betas * (1.0 - prev) / (1.0 - alphas_cumprod)).clamp(min=1e-20))

        def q_sample(self, x0, t, noise):
            return extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(
                self.sqrt_one_minus_alphas_cumprod, t, x0.shape
            ) * noise


    class GCBCDiffusionPolicy(nn.Module):
        def __init__(self, config: Optional[GCBCConfig] = None) -> None:
            super().__init__()
            self.config = config or GCBCConfig()
            self.denoiser = Denoiser(self.config)
            self.scheduler = DDPMScheduler(self.config)

        def training_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            current = batch["current_image"]
            goal = batch["goal_image"]
            cur_action = batch["current_action"]
            target = batch["target_action"][:, : self.config.action_dim]
            bsz = target.shape[0]
            t = torch.randint(0, self.config.diffusion_steps, (bsz,), device=target.device)
            noise = torch.randn_like(target)
            noisy = self.scheduler.q_sample(target, t, noise)
            pred = self.denoiser(current, goal, cur_action, noisy, t)
            pos_loss = F.mse_loss(pred[:, :3], noise[:, :3])
            rot_loss = F.mse_loss(pred[:, 3:7], noise[:, 3:7])
            loss = self.config.position_loss_weight * pos_loss + rot_loss
            return {"loss": loss, "position_loss": pos_loss.detach(), "rotation_loss": rot_loss.detach()}

        @torch.no_grad()
        def sample(self, current_image, goal_image, current_action):
            bsz = current_action.shape[0]
            x = torch.randn(bsz, self.config.action_dim, device=current_action.device)
            for step in reversed(range(self.config.diffusion_steps)):
                t = torch.full((bsz,), step, dtype=torch.long, device=current_action.device)
                pred_noise = self.denoiser(current_image, goal_image, current_action, x, t)
                beta = extract(self.scheduler.betas, t, x.shape)
                sqrt_one_minus = extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip = extract(self.scheduler.sqrt_recip_alphas, t, x.shape)
                mean = sqrt_recip * (x - beta * pred_noise / sqrt_one_minus)
                if step > 0:
                    var = extract(self.scheduler.posterior_variance, t, x.shape)
                    x = mean + torch.sqrt(var) * torch.randn_like(x)
                else:
                    x = mean
            quat = x[:, 3:7]
            x[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return x

        @torch.no_grad()
        def predict(self, current_rgb: np.ndarray, goal_rgb: np.ndarray, current_action: Action) -> Action:
            self.eval()
            device = next(self.parameters()).device
            current = preprocess_image(current_rgb, self.config.image_size).to(device)
            goal = preprocess_image(goal_rgb, self.config.image_size).to(device)
            action = torch.from_numpy(current_action.as_vector()).float().unsqueeze(0).to(device)
            pred = self.sample(current, goal, action)[0].detach().cpu().numpy()
            return Action(position=pred[:3], rotation=pred[3:7], gripper=current_action.gripper)

        def save_checkpoint(self, path: Union[str, Path]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

        @classmethod
        def from_checkpoint(cls, path: Union[str, Path], map_location: str = "cpu") -> "GCBCDiffusionPolicy":
            ckpt = torch.load(path, map_location=map_location)
            model = cls(GCBCConfig(**ckpt["config"]))
            model.load_state_dict(ckpt["state_dict"])
            return model


    def preprocess_image(image: np.ndarray, image_size: int):
        from PIL import Image

        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


    def extract(values, timesteps, shape):
        out = values.gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

else:

    class GCBCDiffusionPolicy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for GCBC.")
