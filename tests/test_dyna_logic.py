import torch
import torch.nn as nn
import unittest
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel

class MockPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
    
    def act(self, obs):
        B = obs.shape[0]
        # Return random actions, logprobs, entropy, values
        actions = torch.randint(0, self.action_dim, (B,))
        logprobs = torch.randn(B)
        entropy = torch.randn(B)
        value = torch.randn(B)
        return actions, logprobs, entropy, value

class TestDynaLogic(unittest.TestCase):
    def setUp(self):
        self.obs_shape = (3, 8, 8) # C, H, W
        self.n_actions = 4
        self.wm = SimpleWorldModel(self.obs_shape, self.n_actions)
        self.policy = MockPolicy(self.n_actions)

    def test_world_model_forward_shape(self):
        B = 2
        obs = torch.randn(B, *self.obs_shape)
        action = torch.randint(0, self.n_actions, (B,))
        
        next_obs, reward = self.wm(obs, action)
        
        self.assertEqual(next_obs.shape, (B, *self.obs_shape))
        self.assertEqual(reward.shape, (B,)) # Should be scalar per batch item
        
    def test_discretize_state(self):
        B = 2
        # Create random continuous output [B, C, H, W]
        continuous = torch.randn(B, *self.obs_shape)
        
        discrete = self.wm.discretize_state(continuous)
        
        # Check shape matches
        self.assertEqual(discrete.shape, continuous.shape)
        
        # Check values are only 0 or 1
        unique_vals = torch.unique(discrete)
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])
            
        # Check it is one-hot along channel dim (sum of channels = 1 for each pixel)
        # sum(dim=1) -> [B, H, W] should be all ones
        sums = discrete.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))
        
    def test_imagined_trajectories(self):
        B = 4
        horizon = 5
        start_states = torch.randn(B, *self.obs_shape) # Random start (doesn't have to be one-hot for this test to run, but conceptually should)
        
        traj = self.wm.generate_imagined_trajectories(self.policy, start_states, horizon)
        
        # Check length
        self.assertEqual(len(traj), horizon)
        
        # Check contents of first step
        step0 = traj[0]
        self.assertEqual(step0['obs'].shape, (B, *self.obs_shape))
        self.assertEqual(step0['actions'].shape, (B,))
        self.assertEqual(step0['rewards'].shape, (B,))
        self.assertEqual(step0['next_obs'].shape, (B, *self.obs_shape))
        
        # Check next_obs is discrete
        unique_vals = torch.unique(step0['next_obs'])
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])
            
        # Check continuity: obs[t+1] == next_obs[t]
        for t in range(horizon - 1):
            obs_next = traj[t+1]['obs']
            next_obs_curr = traj[t]['next_obs']
            self.assertTrue(torch.allclose(obs_next, next_obs_curr))

    def test_buffer_sampling_logic(self):
        # Simulation of the bug fix in train.py
        num_steps = 10
        num_envs = 4
        C, H, W = self.obs_shape
        
        # Mock buffer.obs: (num_steps, num_envs, C, H, W)
        buffer_obs = torch.randn(num_steps, num_envs, C, H, W)
        
        # The logic we implemented:
        rand_time_idxs = torch.randint(0, num_steps, (num_envs,))
        env_idxs = torch.arange(num_envs)
        
        # Sampling
        start_states = buffer_obs[rand_time_idxs, env_idxs]
        
        # Verify shape
        expected_shape = (num_envs, C, H, W)

import numpy as np

class TestDynaLogic(unittest.TestCase):
    def setUp(self):
        self.obs_shape = (3, 8, 8) # C, H, W
        self.n_actions = 4
        self.wm = SimpleWorldModel(self.obs_shape, self.n_actions)
        self.policy = MockPolicy(self.n_actions)
        
    def test_discretize_state(self):
        B = 2
        # Create random continuous output [B, C, H, W]
        continuous = torch.randn(B, *self.obs_shape)
        
        discrete = self.wm.discretize_state(continuous)
        
        # Check shape matches
        self.assertEqual(discrete.shape, continuous.shape)
        
        # Check values are only 0 or 1
        unique_vals = torch.unique(discrete)
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])
            
        # Check it is one-hot along channel dim (sum of channels = 1 for each pixel)
        # sum(dim=1) -> [B, H, W] should be all ones
        sums = discrete.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))
        
    def test_imagined_trajectories(self):
        B = 4
        horizon = 5
        start_states = torch.randn(B, *self.obs_shape) # Random start (doesn't have to be one-hot for this test to run, but conceptually should)
        
        traj = self.wm.generate_imagined_trajectories(self.policy, start_states, horizon)
        
        # Check length
        self.assertEqual(len(traj), horizon)
        
        # Check contents of first step
        step0 = traj[0]
        self.assertEqual(step0['obs'].shape, (B, *self.obs_shape))
        self.assertEqual(step0['actions'].shape, (B,))
        self.assertEqual(step0['rewards'].shape, (B,))
        self.assertEqual(step0['next_obs'].shape, (B, *self.obs_shape))
        
        # Check next_obs is discrete
        unique_vals = torch.unique(step0['next_obs'])
        for v in unique_vals:
            self.assertIn(v.item(), [0.0, 1.0])
            
        # Check continuity: obs[t+1] == next_obs[t]
        for t in range(horizon - 1):
            obs_next = traj[t+1]['obs']
            next_obs_curr = traj[t]['next_obs']
            self.assertTrue(torch.allclose(obs_next, next_obs_curr))

    def test_buffer_sampling_logic(self):
        # Simulation of the bug fix in train.py
        num_steps = 10
        num_envs = 4
        C, H, W = self.obs_shape
        
        # Mock buffer.obs: (num_steps, num_envs, C, H, W)
        buffer_obs = torch.randn(num_steps, num_envs, C, H, W)
        
        # The logic we implemented:
        rand_time_idxs = torch.randint(0, num_steps, (num_envs,))
        env_idxs = torch.arange(num_envs)
        
        # Sampling
        start_states = buffer_obs[rand_time_idxs, env_idxs]
        
        # Verify shape
        expected_shape = (num_envs, C, H, W)
        self.assertEqual(start_states.shape, expected_shape)

    def test_proportional_reward(self):
        # [NEW] Verify Pure Proportional Reward Logic
        
        coef = 0.1
        clip = 0.1
        
        # 1. Small Error (0.01)
        # reward = 0.01 * 0.1 = 0.001
        small_error = 0.01
        reward_small = min(clip, max(0.0, small_error * coef))
        self.assertAlmostEqual(reward_small, 0.001)
        
        # 2. Large Error (0.5) (e.g., Regime Switch)
        # reward = 0.05 (No gating, immediate signal)
        large_error = 0.5
        reward_large = min(clip, max(0.0, large_error * coef))
        self.assertAlmostEqual(reward_large, 0.05)
        
        # 3. Huge Error (2.0)
        # reward = 0.2 -> clipped to 0.1
        huge_error = 2.0
        reward_huge = min(clip, max(0.0, huge_error * coef))
        self.assertEqual(reward_huge, clip)

    def test_intrinsic_rewards_non_negative(self):
        # Ensure logic doesn't produce negative rewards
        coef = 500.0
        clip = 0.5
        mean = 0.1
        
        # Random errors 
        errors = torch.rand(100) * 0.5 # [0, 0.5]
        
        rewards = torch.clamp((errors - mean) * coef, 0.0, clip)
        
        self.assertTrue(torch.all(rewards >= 0.0))
        self.assertTrue(torch.all(rewards <= clip))

    def test_cross_entropy_intrinsic_reward(self):
        # [NEW] Verify CrossEntropy Logic used in train.py
        B, C, H, W = 2, 3, 4, 4
        
        # 1. Mock Preds (Logits)
        pred_next_obs = torch.randn(B, C, H, W)
        
        # 2. Mock Real (One-Hot)
        real_next_obs = torch.zeros(B, C, H, W)
        # Set some "correct" classes
        target_indices = torch.randint(0, C, (B, H, W))
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    c = target_indices[b, h, w]
                    real_next_obs[b, c, h, w] = 1.0
                    
        # 3. Logic from train.py
        # Convert One-Hot target to indices
        target_indices_recovered = torch.argmax(real_next_obs, dim=1)
        self.assertTrue(torch.equal(target_indices, target_indices_recovered))
        
        # Compute CrossEntropy Loss (no reduction)
        # Expected shape: (B, H, W)
        loss = torch.nn.functional.cross_entropy(pred_next_obs, target_indices_recovered, reduction='none')
        self.assertEqual(loss.shape, (B, H, W))
        
        # Aggregate spatial dimensions -> (B,)
        surprise = loss.mean(dim=[1, 2])
        self.assertEqual(surprise.shape, (B,))
        
        # Verify values are positive (CrossEntropy is always non-negative)
        self.assertTrue(torch.all(surprise >= 0.0))

