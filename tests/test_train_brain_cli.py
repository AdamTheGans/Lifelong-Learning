import sys
from functools import partial

import torch

import gymnasium as gym
import scripts.train_brain as train_brain_script


def test_train_brain_parses_schedule_controls(monkeypatch):
    captured = {}

    def fake_train_brain(args):
        captured["args"] = args

    monkeypatch.setattr(train_brain_script, "train_brain", fake_train_brain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_brain.py",
            "--start_regime",
            "1",
            "--randomize_start_regime",
            "--save_every_episodes",
            "3",
            "--plot_every_episodes",
            "4",
            "--generate_high_scale_plots",
            "--brain_vectorization",
            "sync",
        ],
    )

    train_brain_script.main()

    args = captured["args"]
    assert args.start_regime == 1
    assert args.randomize_start_regime is True
    assert args.save_every_episodes == 3
    assert args.plot_every_episodes == 4
    assert args.generate_high_scale_plots is True
    assert args.brain_vectorization == "sync"


def test_build_meta_vector_env_async():
    env = train_brain_script.build_meta_vector_env([partial(gym.make, "CartPole-v1")], "async")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == 1
        assert env.autoreset_mode == gym.vector.AutoresetMode.DISABLED
    finally:
        env.close()


def test_build_meta_vector_env_sync_uses_disabled_autoreset():
    env = train_brain_script.build_meta_vector_env([partial(gym.make, "CartPole-v1")], "sync")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] == 1
        assert env.autoreset_mode == gym.vector.AutoresetMode.DISABLED
    finally:
        env.close()


def test_async_brain_vectorization_caps_worker_cpu_threads():
    assert train_brain_script.get_meta_env_runtime_cpu_threads("async") == 1
    assert train_brain_script.get_meta_env_runtime_cpu_threads("sync") is None


def test_should_refresh_brain_trends_on_schedule_only():
    assert train_brain_script.should_refresh_brain_trends(True, False) is True
    assert train_brain_script.should_refresh_brain_trends(False, False) is False


def test_should_refresh_brain_trends_for_per_episode_high_scale_plots():
    assert train_brain_script.should_refresh_brain_trends(False, True) is True

def test_upgrade_legacy_brain_state_dict_pads_old_action_heads():
    legacy = {
        "actor_mean.weight": torch.randn(7, 128),
        "actor_mean.bias": torch.randn(7),
        "actor_logstd": torch.randn(7),
    }
    upgraded = train_brain_script._upgrade_legacy_brain_state_dict(legacy, target_act_dim=15)
    assert upgraded["actor_mean.weight"].shape == (15, 128)
    assert upgraded["actor_mean.bias"].shape == (15,)
    assert upgraded["actor_log_std"].shape == (15,)
    assert "actor_logstd" not in upgraded
    assert torch.allclose(upgraded["actor_mean.weight"][:7], legacy["actor_mean.weight"])
    assert torch.allclose(upgraded["actor_mean.bias"][:7], legacy["actor_mean.bias"])
    assert torch.allclose(upgraded["actor_log_std"][:7], legacy["actor_logstd"])


def test_extract_episode_average_success_rate_averages_final_info():
    infos = {
        "final_info": [
            {"inner_stats": {"overall_success_rate": 0.25}},
            {"inner_stats": {"success_rate": 0.75}},
            None,
        ]
    }
    avg_success_rate = train_brain_script._extract_episode_average_success_rate(infos)
    assert avg_success_rate == 0.5

