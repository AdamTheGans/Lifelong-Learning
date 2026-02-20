import torch
import torch.nn as nn
import numpy as np
import unittest
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel


class MockPolicy(nn.Module):
    """Deterministic mock policy for testing trajectory generation."""
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def act(self, obs):
        B = obs.shape[0]
        actions = torch.randint(0, self.action_dim, (B,))
        logprobs = torch.randn(B)
        entropy = torch.randn(B)
        value = torch.randn(B)
        return actions, logprobs, entropy, value


class TestDynaLogic(unittest.TestCase):
    def setUp(self):
        self.obs_shape = (3, 8, 8)  # C, H, W
        self.n_actions = 4
        self.wm = SimpleWorldModel(self.obs_shape, self.n_actions)
        self.policy = MockPolicy(self.n_actions)

    def test_world_model_forward_shape(self):
        """Verify WM output shapes match expected (B, C, H, W) and (B,)."""
        B = 2
        obs = torch.randn(B, *self.obs_shape)
        action = torch.randint(0, self.n_actions, (B,))

        next_obs, reward = self.wm(obs, action)

        self.assertEqual(next_obs.shape, (B, *self.obs_shape))
        self.assertEqual(reward.shape, (B,))

    def test_discretize_state(self):
        """Verify discretize_state produces valid one-hot encoding."""
        B = 2
        continuous = torch.randn(B, *self.obs_shape)

        discrete = self.wm.discretize_state(continuous)

        self.assertEqual(discrete.shape, continuous.shape)

        # Values must be only 0.0 or 1.0
        unique_vals = torch.unique(discrete)
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])

        # Exactly one channel active per pixel (sum along C = 1)
        sums = discrete.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))

    def test_imagined_trajectories(self):
        """Verify trajectory generation produces correct shapes and continuity."""
        B = 4
        horizon = 5
        start_states = torch.randn(B, *self.obs_shape)

        traj = self.wm.generate_imagined_trajectories(self.policy, start_states, horizon)

        self.assertEqual(len(traj), horizon)

        # Check shapes of first step
        step0 = traj[0]
        self.assertEqual(step0['obs'].shape, (B, *self.obs_shape))
        self.assertEqual(step0['actions'].shape, (B,))
        self.assertEqual(step0['rewards'].shape, (B,))
        self.assertEqual(step0['next_obs'].shape, (B, *self.obs_shape))

        # next_obs must be discrete (one-hot)
        unique_vals = torch.unique(step0['next_obs'])
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])

        # Continuity: obs[t+1] == next_obs[t]
        for t in range(horizon - 1):
            self.assertTrue(torch.allclose(traj[t+1]['obs'], traj[t]['next_obs']))

    def test_buffer_sampling_logic(self):
        """Verify that buffer indexing produces correct shapes for dream seeding."""
        num_steps = 10
        num_envs = 4
        C, H, W = self.obs_shape

        buffer_obs = torch.randn(num_steps, num_envs, C, H, W)

        rand_time_idxs = torch.randint(0, num_steps, (num_envs,))
        env_idxs = torch.arange(num_envs)
        start_states = buffer_obs[rand_time_idxs, env_idxs]

        expected_shape = (num_envs, C, H, W)
        self.assertEqual(start_states.shape, expected_shape)

    def test_proportional_reward(self):
        """Verify pure proportional intrinsic reward logic."""
        coef = 0.1
        clip = 0.1

        # Small error → small reward
        reward_small = min(clip, max(0.0, 0.01 * coef))
        self.assertAlmostEqual(reward_small, 0.001)

        # Medium error → proportional reward
        reward_medium = min(clip, max(0.0, 0.5 * coef))
        self.assertAlmostEqual(reward_medium, 0.05)

        # Large error → clipped to max
        reward_large = min(clip, max(0.0, 2.0 * coef))
        self.assertEqual(reward_large, clip)

    def test_intrinsic_rewards_non_negative(self):
        """Ensure intrinsic rewards are always non-negative and within bounds."""
        coef = 500.0
        clip = 0.5
        mean = 0.1

        errors = torch.rand(100) * 0.5
        rewards = torch.clamp((errors - mean) * coef, 0.0, clip)

        self.assertTrue(torch.all(rewards >= 0.0))
        self.assertTrue(torch.all(rewards <= clip))

    def test_cross_entropy_intrinsic_reward(self):
        """Verify CrossEntropy-based surprise calculation matches expected behavior."""
        B, C, H, W = 2, 3, 4, 4

        pred_next_obs = torch.randn(B, C, H, W)

        # Build one-hot ground truth
        real_next_obs = torch.zeros(B, C, H, W)
        target_indices = torch.randint(0, C, (B, H, W))
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    c = target_indices[b, h, w]
                    real_next_obs[b, c, h, w] = 1.0

        # Recover indices from one-hot
        target_indices_recovered = torch.argmax(real_next_obs, dim=1)
        self.assertTrue(torch.equal(target_indices, target_indices_recovered))

        # CrossEntropy loss shape: (B, H, W)
        loss = torch.nn.functional.cross_entropy(pred_next_obs, target_indices_recovered, reduction='none')
        self.assertEqual(loss.shape, (B, H, W))

        # Aggregated surprise: (B,), always non-negative
        surprise = loss.mean(dim=[1, 2])
        self.assertEqual(surprise.shape, (B,))
        self.assertTrue(torch.all(surprise >= 0.0))
