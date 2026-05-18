from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from graspcorrect.types import Action

try:  # pragma: no cover - exercised only when torch is installed
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
    beta_start: float = 0.0001
    beta_end: float = 0.02
    position_loss_weight: float = 0.2


if torch is not None:

    class Swish(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)


    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
            half = self.dim // 2
            freqs = torch.exp(
                -np.log(10000.0) * torch.arange(0, half, device=timesteps.device).float() / max(half - 1, 1)
            )
            args = timesteps.float()[:, None] * freqs[None]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.dim % 2:
                emb = F.pad(emb, (0, 1))
            return emb


    def _group_norm(channels: int) -> nn.GroupNorm:
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)


    class ImageEncoder(nn.Module):
        def __init__(self, in_channels: int = 6) -> None:
            super().__init__()
            try:
                from torchvision.models import resnet34  # type: ignore

                self.encoder = resnet34(weights=None, norm_layer=_group_norm)
                self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.encoder.fc = nn.Identity()
                self.out_dim = 512
            except Exception:
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
                    _group_norm(32),
                    Swish(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    _group_norm(64),
                    Swish(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    _group_norm(128),
                    Swish(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                self.out_dim = 128

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)


    class ActionDenoiser(nn.Module):
        def __init__(self, config: GCBCConfig) -> None:
            super().__init__()
            self.config = config
            self.encoder = ImageEncoder(in_channels=6)
            time_dim = 64
            self.time = nn.Sequential(
                SinusoidalTimeEmbedding(time_dim),
                nn.Linear(time_dim, config.hidden_dim),
                Swish(),
            )
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

        def forward(
            self,
            current_image: torch.Tensor,
            goal_image: torch.Tensor,
            current_action: torch.Tensor,
            noisy_action: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            image = torch.cat([current_image, goal_image], dim=1)
            features = self.encoder(image)
            time_features = self.time(timesteps)
            x = torch.cat([features, time_features, current_action, noisy_action], dim=-1)
            return self.mlp(x)


    class DDPMScheduler(nn.Module):
        def __init__(self, config: GCBCConfig) -> None:
            super().__init__()
            betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
            self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
            self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

        def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + _extract(
                self.sqrt_one_minus_alphas_cumprod, t, x0.shape
            ) * noise


    class GCBCDiffusionPolicy(nn.Module):
        def __init__(self, config: Optional[GCBCConfig] = None) -> None:
            super().__init__()
            self.config = config or GCBCConfig()
            self.denoiser = ActionDenoiser(self.config)
            self.scheduler = DDPMScheduler(self.config)

        def training_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            current_image = batch["current_image"]
            goal_image = batch["goal_image"]
            current_action = batch["current_action"]
            target = batch["target_action"][:, : self.config.action_dim]
            bsz = target.shape[0]
            timesteps = torch.randint(0, self.config.diffusion_steps, (bsz,), device=target.device)
            noise = torch.randn_like(target)
            noisy = self.scheduler.q_sample(target, timesteps, noise)
            pred = self.denoiser(current_image, goal_image, current_action, noisy, timesteps)
            pos_loss = F.mse_loss(pred[:, :3], noise[:, :3])
            rot_loss = F.mse_loss(pred[:, 3:7], noise[:, 3:7])
            loss = self.config.position_loss_weight * pos_loss + rot_loss
            return {"loss": loss, "position_loss": pos_loss.detach(), "rotation_loss": rot_loss.detach()}

        @torch.no_grad()
        def sample(
            self,
            current_image: torch.Tensor,
            goal_image: torch.Tensor,
            current_action: torch.Tensor,
        ) -> torch.Tensor:
            bsz = current_action.shape[0]
            x = torch.randn(bsz, self.config.action_dim, device=current_action.device)
            for step in reversed(range(self.config.diffusion_steps)):
                t = torch.full((bsz,), step, device=current_action.device, dtype=torch.long)
                pred_noise = self.denoiser(current_image, goal_image, current_action, x, t)
                beta_t = _extract(self.scheduler.betas, t, x.shape)
                sqrt_one_minus = _extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
                sqrt_recip_alpha = _extract(self.scheduler.sqrt_recip_alphas, t, x.shape)
                mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus)
                if step > 0:
                    var = _extract(self.scheduler.posterior_variance, t, x.shape)
                    x = mean + torch.sqrt(var) * torch.randn_like(x)
                else:
                    x = mean
            quat = x[:, 3:7]
            x[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            return x

        @torch.no_grad()
        def predict(self, current_rgb: np.ndarray, goal_rgb: np.ndarray, current_action: Action) -> Action:
            device = next(self.parameters()).device
            self.eval()
            current = preprocess_image(current_rgb, self.config.image_size).to(device)
            goal = preprocess_image(goal_rgb, self.config.image_size).to(device)
            action = torch.from_numpy(current_action.as_vector(include_gripper=True)).float().unsqueeze(0).to(device)
            pred = self.sample(current, goal, action)[0].detach().cpu().numpy()
            return Action(position=pred[:3], rotation=pred[3:7], gripper=current_action.gripper)

        def save_checkpoint(self, path: str | Path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

        @classmethod
        def from_checkpoint(cls, path: str | Path, map_location: str = "cpu") -> "GCBCDiffusionPolicy":
            ckpt = torch.load(path, map_location=map_location)
            config = GCBCConfig(**ckpt["config"])
            model = cls(config)
            model.load_state_dict(ckpt["state_dict"])
            return model


    def preprocess_image(image: np.ndarray, image_size: int) -> torch.Tensor:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        pil = pil.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor


    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


else:

    class GCBCDiffusionPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("GCBCDiffusionPolicy requires torch. Install with `pip install -e .[train]`.")
