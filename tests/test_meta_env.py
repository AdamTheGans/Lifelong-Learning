"""Tests for the MetaEnv (Brain's gym environment wrapping the inner loop)."""
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
import torch

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.brain.meta_env import MetaEnv
from lifelong_learning.agents.brain.signals import NUM_SIGNALS
from lifelong_learning.agents.ppo.train import configure_runtime_threads


class TestMetaEnv(unittest.TestCase):
    """Test that MetaEnv behaves as a valid Gymnasium environment."""

    @classmethod
    def setUpClass(cls):
        """Create a MetaEnv with very short inner runs for speed."""
        inner_cfg = PPOConfig(
            total_timesteps=10_000,   # Very short inner run
            num_envs=4,
            num_steps=32,
            seed=42,
            device="cpu",
            mode="dyna",
        )
        cls.env = MetaEnv(
            env_id="MiniGrid-MultiGoal-8x8-v0",
            inner_cfg=inner_cfg,
            decision_interval=2,      # Only 2 inner updates per Brain step
            steps_per_regime=3000,
            intrinsic_coef=0.015,
            imagined_horizon=3,
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_observation_space(self):
        """Observation space should be Box(NUM_SIGNALS,)."""
        self.assertEqual(self.env.observation_space.shape, (NUM_SIGNALS,))

    def test_action_space(self):
        """Action space should be Box(15,) in [-1, 1]."""
        self.assertEqual(self.env.action_space.shape, (15,))
        np.testing.assert_array_equal(self.env.action_space.low, -1.0 * np.ones(15))
        np.testing.assert_array_equal(self.env.action_space.high, 1.0 * np.ones(15))

    def test_reset_returns_correct_shape(self):
        """reset() should return (obs, info) with correct obs shape."""
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (NUM_SIGNALS,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIn("inner_stats", info)

    def test_step_returns_correct_shape(self):
        """step() should return (obs, reward, term, trunc, info)."""
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(next_obs.shape, (NUM_SIGNALS,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("inner_stats", info)

    def test_step_applies_hp_adjustments(self):
        """Stepping with action [1,1,1,1] should increase HPs."""
        self.env.reset()
        state = self.env._state

        lr_before = state.optimizer.param_groups[0]["lr"]
        ent_before = state.cfg.ent_coef
        ic_before = state.intrinsic_coef
        hz_before = state.imagined_horizon

        # Action [1,1,...,1] (15-dim) → max scale everything up
        action = np.ones(15, dtype=np.float32)
        self.env.step(action)

        lr_after = state.optimizer.param_groups[0]["lr"]
        ent_after = state.cfg.ent_coef
        ic_after = state.intrinsic_coef
        hz_after = state.imagined_horizon

        # All should increase (or hit bounds)
        self.assertGreaterEqual(lr_after, lr_before)
        self.assertGreaterEqual(ent_after, ent_before)
        self.assertGreaterEqual(ic_after, ic_before)
        self.assertGreaterEqual(hz_after, hz_before)

    def test_reset_restores_inner_config_template(self):
        """A fresh reset should start from the template config, not the last Brain action."""
        inner_cfg = PPOConfig(
            total_timesteps=10_000,
            num_envs=4,
            num_steps=32,
            seed=123,
            device="cpu",
            mode="dyna",
            ent_coef=0.02,
        )
        env = MetaEnv(
            env_id="MiniGrid-MultiGoal-8x8-v0",
            inner_cfg=inner_cfg,
            decision_interval=1,
            steps_per_regime=10_000,
            intrinsic_coef=0.015,
            imagined_horizon=3,
        )
        try:
            env.reset(seed=0)
            env.step(np.ones(15, dtype=np.float32))

            self.assertAlmostEqual(inner_cfg.ent_coef, 0.02)

            env.reset(seed=1)
            self.assertAlmostEqual(env._state.cfg.ent_coef, 0.02)
        finally:
            env.close()

    def test_reset_honors_fixed_start_regime(self):
        """Configured start_regime should be preserved unless randomization is enabled."""
        inner_cfg = PPOConfig(
            total_timesteps=10_000,
            num_envs=4,
            num_steps=32,
            seed=7,
            device="cpu",
            mode="dyna",
        )
        env = MetaEnv(
            env_id="MiniGrid-MultiGoal-8x8-v0",
            inner_cfg=inner_cfg,
            decision_interval=1,
            steps_per_regime=10_000,
            start_regime=1,
            randomize_start_regime=False,
            num_regimes=2,
            intrinsic_coef=0.015,
            imagined_horizon=3,
        )
        try:
            env.reset(seed=123)
            self.assertEqual(env._state.current_mode_regime, 1)
        finally:
            env.close()

    def test_interval_summary_uses_completed_episode_counts(self):
        """Meta-step summaries should aggregate the full decision interval, not just the last update."""
        env = MetaEnv(inner_cfg=PPOConfig(total_timesteps=10_000, num_envs=4, num_steps=32, seed=0, device="cpu", mode="dyna"))
        env._state = object()
        interval_updates = [
            {
                "done": False,
                "mean_episodic_return": 1.0,
                "episodic_return_sum": 2.0,
                "completed_episodes": 2,
                "successful_episodes": 1,
                "failed_episodes": 1,
                "timeout_episodes": 0,
                "success_rate": 0.4,
                "failure_rate": 0.3,
                "recent_success_rate": 0.5,
                "recent_failure_rate": 0.25,
                "recent_timeout_rate": 0.25,
                "mean_surprise": 0.2,
                "wm_loss_state": 0.3,
                "wm_loss_reward": 0.4,
                "policy_entropy": 0.5,
                "policy_loss": 0.6,
                "value_loss": 0.7,
            },
            {
                "done": False,
                "mean_episodic_return": 9.0,
                "episodic_return_sum": 18.0,
                "completed_episodes": 2,
                "successful_episodes": 2,
                "failed_episodes": 0,
                "timeout_episodes": 0,
                "success_rate": 0.9,
                "failure_rate": 0.05,
                "recent_success_rate": 0.8,
                "recent_failure_rate": 0.1,
                "recent_timeout_rate": 0.1,
                "mean_surprise": 0.6,
                "wm_loss_state": 0.7,
                "wm_loss_reward": 0.8,
                "policy_entropy": 0.9,
                "policy_loss": 1.0,
                "value_loss": 1.1,
            },
        ]

        with mock.patch("lifelong_learning.agents.brain.meta_env.run_inner_update", side_effect=interval_updates):
            stats = env._run_n_updates(2)

        self.assertEqual(stats["interval_completed_episodes"], 4)
        self.assertAlmostEqual(stats["mean_episodic_return"], 5.0)
        self.assertAlmostEqual(stats["success_rate"], 0.75)
        self.assertAlmostEqual(stats["failure_rate"], 0.25)
        self.assertAlmostEqual(stats["overall_success_rate"], 0.9)
        self.assertAlmostEqual(stats["mean_surprise"], 0.4)

    def test_configure_runtime_threads_sets_torch_limits(self):
        """Async workers should be able to cap PyTorch CPU thread pools."""
        with mock.patch("lifelong_learning.agents.ppo.train.torch.set_num_threads") as set_num_threads, \
             mock.patch("lifelong_learning.agents.ppo.train.torch.set_num_interop_threads") as set_num_interop_threads:
            configure_runtime_threads(1)

        set_num_threads.assert_called_once_with(1)
        set_num_interop_threads.assert_called_once_with(1)

    def test_reset_respects_requested_inner_num_envs(self):
        """The inner trainer should honor the configured env count instead of forcing 16."""
        inner_cfg = PPOConfig(
            total_timesteps=10_000,
            num_envs=2,
            num_steps=32,
            seed=99,
            device="cpu",
            mode="dyna",
        )
        env = MetaEnv(
            env_id="MiniGrid-MultiGoal-8x8-v0",
            inner_cfg=inner_cfg,
            decision_interval=1,
            steps_per_regime=10_000,
            intrinsic_coef=0.015,
            imagined_horizon=3,
        )
        try:
            env.reset(seed=0)
            self.assertEqual(env._state.num_envs, 2)
        finally:
            env.close()

    def test_apply_action_logs_neuromodulation_metrics(self):
        """Applying a context code should log the decoded mask and its effect."""
        env = MetaEnv(inner_cfg=PPOConfig(total_timesteps=10_000, num_envs=4, num_steps=32, seed=0, device="cpu", mode="dyna"))
        logger = mock.Mock()
        model = mock.Mock()
        model.describe_neuromodulation.return_value = {
            "mask_mean": 0.5,
            "mask_std": 0.1,
            "mask_min": 0.2,
            "mask_max": 0.9,
            "policy_kl_vs_unmasked": 0.03,
            "entropy_delta_vs_unmasked": -0.02,
            "value_delta_abs_vs_unmasked": 0.04,
            "channel_means": torch.linspace(0.0, 1.0, 64),
        }
        state = SimpleNamespace(
            optimizer=SimpleNamespace(param_groups=[{"lr": 3e-4}]),
            cfg=SimpleNamespace(ent_coef=0.01, anchoring_weight=0.0),
            intrinsic_coef=0.015,
            imagined_horizon=3,
            replay_ratio=0.0,
            replay_prioritization=0.0,
            device=torch.device("cpu"),
            model=model,
            logger=logger,
            obs_t=torch.zeros(2, 21, 8, 8),
            global_step=128,
        )
        env._state = state

        action = np.zeros(15, dtype=np.float32)
        action[7:15] = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        env._apply_action(action)

        model.set_context_code.assert_called_once()
        logged_tags = [call.args[0] for call in logger.scalar.call_args_list]
        self.assertIn("brain_context/context_0", logged_tags)
        self.assertIn("brain_neuromod/policy_kl_vs_unmasked", logged_tags)
        self.assertIn("brain_neuromod/channel_mean_63", logged_tags)

    def test_episode_terminates(self):
        """Episode should terminate once inner training is done."""
        self.env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 100:
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            steps += 1
        # Should terminate within a reasonable number of steps
        self.assertTrue(terminated, f"Episode did not terminate within {steps} steps")




if __name__ == "__main__":
    unittest.main()
