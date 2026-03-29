from unittest import mock

import torch

from scripts.eval_brain import _load_brain_checkpoint, _upgrade_legacy_brain_state_dict


def test_upgrade_legacy_brain_state_dict_pads_action_heads():
    legacy = {
        "actor_mean.weight": torch.randn(7, 128),
        "actor_mean.bias": torch.randn(7),
        "actor_logstd": torch.randn(7),
    }

    upgraded, notices = _upgrade_legacy_brain_state_dict(
        legacy,
        target_obs_dim=19,
        target_act_dim=15,
    )

    assert upgraded["actor_mean.weight"].shape == (15, 128)
    assert upgraded["actor_mean.bias"].shape == (15,)
    assert upgraded["actor_log_std"].shape == (15,)
    assert "actor_logstd" not in upgraded
    assert notices
    assert torch.allclose(upgraded["actor_mean.weight"][:7], legacy["actor_mean.weight"])
    assert torch.allclose(upgraded["actor_mean.bias"][:7], legacy["actor_mean.bias"])
    assert torch.allclose(upgraded["actor_log_std"][:7], legacy["actor_logstd"])


def test_load_brain_checkpoint_uses_full_pickle_mode():
    device = torch.device("cpu")
    with mock.patch("scripts.eval_brain.torch.load", return_value={"ok": True}) as load_mock:
        ckpt = _load_brain_checkpoint("brain.pt", device)

    assert ckpt == {"ok": True}
    load_mock.assert_called_once_with("brain.pt", map_location=device, weights_only=False)