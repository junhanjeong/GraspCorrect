from graspcorrect.policies.gcbc import GCBCConfig


def test_gcbc_config_defaults():
    cfg = GCBCConfig()
    assert cfg.position_loss_weight == 0.2
    assert cfg.diffusion_steps > 0
