import numpy as np
import pytest


def test_gcbc_forward_if_torch_available():
    torch = pytest.importorskip("torch")
    from graspcorrect.policies import GCBCConfig, GCBCDiffusionPolicy

    cfg = GCBCConfig(image_size=32, diffusion_steps=4, hidden_dim=32)
    policy = GCBCDiffusionPolicy(cfg)
    batch = {
        "current_image": torch.rand(2, 3, 32, 32),
        "goal_image": torch.rand(2, 3, 32, 32),
        "current_action": torch.rand(2, 8),
        "target_action": torch.rand(2, 8),
    }
    losses = policy.training_loss(batch)
    assert np.isfinite(float(losses["loss"].detach()))
