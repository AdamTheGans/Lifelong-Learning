import torch

from scripts.eval_brain import _upgrade_legacy_brain_state_dict


def test_upgrade_legacy_brain_state_dict_pads_action_heads():
    legacy = {
        "actor_mean.weight": torch.randn(7, 128),
        "actor_mean.bias": torch.randn(7),
        "actor_logstd": torch.randn(7),
    }

    upgraded = _upgrade_legacy_brain_state_dict(legacy, target_act_dim=15)

    assert upgraded["actor_mean.weight"].shape == (15, 128)
    assert upgraded["actor_mean.bias"].shape == (15,)
    assert upgraded["actor_log_std"].shape == (15,)
    assert "actor_logstd" not in upgraded
    assert torch.allclose(upgraded["actor_mean.weight"][:7], legacy["actor_mean.weight"])
    assert torch.allclose(upgraded["actor_mean.bias"][:7], legacy["actor_mean.bias"])
    assert torch.allclose(upgraded["actor_log_std"][:7], legacy["actor_logstd"])
