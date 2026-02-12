from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """
    Stores (T, N) rollout for PPO, then computes GAE advantages and returns.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_shape, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((num_steps, num_envs), device=device, dtype=torch.long)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.next_obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device) # [NEW] For World Model

        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)

        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values, next_obs):
        t = self.step
        self.obs[t].copy_(obs)
        self.next_obs[t].copy_(next_obs) # [NEW]
        self.actions[t].copy_(actions)
        self.logprobs[t].copy_(logprobs)
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.values[t].copy_(values)
        self.step += 1

    def compute_returns_and_advantages(self, last_value, gamma: float, gae_lambda: float):
        # GAE-Lambda
        last_adv = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                # If this was the last step of the buffer, we check if IT was terminal.
                next_nonterminal = 1.0 - self.dones[t]
                next_values = last_value
            else:
                # We must use dones[t] to mask the transition from t -> t+1.
                # The original code erroneously used dones[t+1].
                next_nonterminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
            self.advantages[t] = last_adv

        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size: int, shuffle: bool = True):
        # Flatten (T, N) -> (T*N)
        T, N = self.num_steps, self.num_envs
        batch_size = T * N

        b_obs = self.obs.reshape((batch_size,) + self.obs_shape)
        b_next_obs = self.next_obs.reshape((batch_size,) + self.obs_shape) # [NEW]
        b_actions = self.actions.reshape(batch_size)
        b_logprobs = self.logprobs.reshape(batch_size)
        b_advantages = self.advantages.reshape(batch_size)
        b_returns = self.returns.reshape(batch_size)
        b_values = self.values.reshape(batch_size)

        b_rewards = self.rewards.reshape(batch_size) # [NEW]
        b_dones = self.dones.reshape(batch_size) # [NEW] (might as well)

        # Normalize advantage
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        idxs = np.arange(batch_size)
        if shuffle:
            np.random.shuffle(idxs)

        for start in range(0, batch_size, minibatch_size):
            mb = idxs[start:start + minibatch_size]
            yield b_obs[mb], b_actions[mb], b_logprobs[mb], b_advantages[mb], b_returns[mb], b_values[mb], b_next_obs[mb], b_rewards[mb]

    def reset(self):
        self.step = 0
