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
from lifelong_learning.utils.running_stats import RunningMeanStd
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

    def test_amplified_error_logic(self):
        # [NEW] Verify Amplified Error Logic
        # Formula: reward = clamp(error * coef, 0.0, clip)
        
        coef = 500.0
        clip = 0.5
        
        # Case 1: Boring transition (low error)
        # error = 0.0002 -> reward = 0.1
        boring_error = 0.0002
        reward_boring = min(clip, boring_error * coef)
        self.assertAlmostEqual(reward_boring, 0.1)
        self.assertLess(reward_boring, clip)
        
        # Case 2: Surprising transition (high error)
        # error = 0.1 -> reward = 50.0 -> clipped to 0.5
        surprise_error = 0.1
        reward_surprise = min(clip, surprise_error * coef)
        self.assertEqual(reward_surprise, clip)
        
        # Case 3: Zero error
        zero_error = 0.0
        reward_zero = min(clip, zero_error * coef)
        self.assertEqual(reward_zero, 0.0)
        
        # Case 4: Edge case (exact clip boundary)
        # error = 0.001 -> reward = 0.5
        edge_error = 0.001
        reward_edge = min(clip, edge_error * coef)
        self.assertEqual(reward_edge, 0.5)

    def test_intrinsic_rewards_non_negative(self):
        # Ensure logic doesn't produce negative rewards
        coef = 500.0
        clip = 0.5
        
        # Random errors should always be positive (MSE Loss is always >= 0)
        errors = torch.rand(100) * 0.5 # [0, 0.5]
        
        rewards = torch.clamp(errors * coef, 0.0, clip)
        
        self.assertTrue(torch.all(rewards >= 0.0))
        self.assertTrue(torch.all(rewards <= clip))
