"""Tests for the MetaEnv (Brain's gym environment wrapping the inner loop)."""
import unittest
import numpy as np

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.brain.meta_env import MetaEnv
from lifelong_learning.agents.brain.signals import NUM_SIGNALS


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
            env_id="MiniGrid-DualGoal-8x8-v0",
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
        """Action space should be Box(4,) in [-1, 1]."""
        self.assertEqual(self.env.action_space.shape, (4,))
        np.testing.assert_array_equal(self.env.action_space.low, -1.0 * np.ones(4))
        np.testing.assert_array_equal(self.env.action_space.high, 1.0 * np.ones(4))

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

        # Action [1,1,1,1] → max scale everything up
        action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
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

    def test_action_to_multiplier(self):
        """Verify the action-to-multiplier mapping."""
        self.assertAlmostEqual(MetaEnv._action_to_multiplier(-1.0), 0.5, places=5)
        self.assertAlmostEqual(MetaEnv._action_to_multiplier(0.0), 1.0, places=5)
        self.assertAlmostEqual(MetaEnv._action_to_multiplier(1.0), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
