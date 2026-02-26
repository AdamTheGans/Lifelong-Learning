"""
Episodic Memory: a persistent ring buffer that stores transitions across
regime switches for experience replay.

The Brain meta-agent controls when and how much to replay from this buffer.
"""
from __future__ import annotations

import torch
import numpy as np


class EpisodicMemory:
    """
    Fixed-capacity ring buffer for storing (obs, action, reward, next_obs, done)
    transitions across the lifetime of the inner agent.

    Oldest transitions are overwritten when capacity is reached.
    """

    def __init__(self, capacity: int, obs_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device

        self.obs = torch.zeros((capacity,) + obs_shape, device=device)
        self.actions = torch.zeros(capacity, device=device, dtype=torch.long)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_obs = torch.zeros((capacity,) + obs_shape, device=device)
        self.dones = torch.zeros(capacity, device=device)

        self._size = 0          # Number of valid entries
        self._write_idx = 0     # Next write position

    @property
    def size(self) -> int:
        return self._size

    @property
    def fullness(self) -> float:
        """Fraction of capacity filled, in [0, 1]."""
        return self._size / self.capacity

    def store_from_rollout(self, rollout_buffer) -> None:
        """
        Copy all transitions from a RolloutBuffer into episodic memory.

        Args:
            rollout_buffer: a RolloutBuffer with shape (T, N, ...) tensors
        """
        T = rollout_buffer.num_steps
        N = rollout_buffer.num_envs

        # Flatten (T, N) → (T*N)
        n_transitions = T * N
        flat_obs = rollout_buffer.obs[:T].reshape((n_transitions,) + self.obs_shape)
        flat_actions = rollout_buffer.actions[:T].reshape(n_transitions)
        flat_rewards = rollout_buffer.rewards[:T].reshape(n_transitions)
        flat_next_obs = rollout_buffer.next_obs[:T].reshape((n_transitions,) + self.obs_shape)
        flat_dones = rollout_buffer.dones[:T].reshape(n_transitions)

        # Write in chunks (handle wrap-around)
        for i in range(n_transitions):
            idx = self._write_idx
            self.obs[idx] = flat_obs[i]
            self.actions[idx] = flat_actions[i]
            self.rewards[idx] = flat_rewards[i]
            self.next_obs[idx] = flat_next_obs[i]
            self.dones[idx] = flat_dones[i]

            self._write_idx = (self._write_idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int) -> dict | None:
        """
        Sample n random transitions from the buffer.

        Returns None if fewer than n transitions are stored.

        Returns:
            dict with keys: obs, actions, rewards, next_obs, dones
            All tensors on self.device.
        """
        if self._size < n:
            return None

        idxs = np.random.randint(0, self._size, size=n)

        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }
