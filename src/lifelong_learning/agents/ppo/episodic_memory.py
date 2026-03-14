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
        self.regimes = torch.zeros(capacity, device=device, dtype=torch.long)

        self._size = 0          # Number of valid entries
        self._write_idx = 0     # Next write position

    @property
    def size(self) -> int:
        return self._size

    @property
    def fullness(self) -> float:
        """Fraction of capacity filled, in [0, 1]."""
        return self._size / self.capacity

    def store_from_rollout(self, rollout_buffer, regime_ids: np.ndarray | torch.Tensor | int | None = None) -> None:
        """
        Copy all transitions from a RolloutBuffer into episodic memory.

        Args:
            rollout_buffer: a RolloutBuffer with shape (T, N, ...) tensors
            regime_ids: optionally track the regime IDs for the transitions.
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

        if regime_ids is not None:
            if isinstance(regime_ids, int):
                flat_regimes = torch.full((n_transitions,), regime_ids, device=self.device, dtype=torch.long)
            else:
                if isinstance(regime_ids, np.ndarray):
                    regime_ids = torch.tensor(regime_ids, device=self.device, dtype=torch.long)
                # Expand (N,) to (T, N) then flatten
                if regime_ids.dim() == 1:
                    regime_ids = regime_ids.unsqueeze(0).expand(T, -1)
                flat_regimes = regime_ids.reshape(n_transitions)
        else:
            flat_regimes = torch.zeros(n_transitions, device=self.device, dtype=torch.long)

        # Write in chunks (handle wrap-around)
        for i in range(n_transitions):
            idx = self._write_idx
            self.obs[idx] = flat_obs[i]
            self.actions[idx] = flat_actions[i]
            self.rewards[idx] = flat_rewards[i]
            self.next_obs[idx] = flat_next_obs[i]
            self.dones[idx] = flat_dones[i]
            self.regimes[idx] = flat_regimes[i]

            self._write_idx = (self._write_idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, prioritization: float = 0.0, current_regime: int | None = None) -> dict | None:
        """
        Sample n transitions from the buffer.

        If prioritization > 0 and current_regime is provided, samples
        a fraction `prioritization` of experiences exclusively from regimes
        different from `current_regime`.
        
        Returns None if fewer than n transitions are stored.

        Returns:
            dict with keys: obs, actions, rewards, next_obs, dones
            All tensors on self.device.
        """
        if self._size < n:
            return None

        # Prioritized sampling based on regimes
        n_priority = 0
        priority_idxs = None
        uniform_n = n

        if prioritization > 0.0 and current_regime is not None:
            n_priority = int(n * prioritization)
            uniform_n = n - n_priority

            if n_priority > 0:
                valid_regimes = self.regimes[:self._size]
                old_regime_mask = valid_regimes != current_regime
                old_regime_idxs = torch.where(old_regime_mask)[0]

                if len(old_regime_idxs) > 0:
                    # Sample with replacement if we don't have enough old experiences
                    replace = len(old_regime_idxs) < n_priority
                    sampled_old_idxs = np.random.choice(
                        old_regime_idxs.cpu().numpy(), size=n_priority, replace=replace
                    )
                    priority_idxs = sampled_old_idxs
                else:
                    # Fallback to uniform if no valid old regimes exist yet
                    uniform_n = n
                    n_priority = 0

        uniform_idxs = np.random.randint(0, self._size, size=uniform_n)
        
        if priority_idxs is not None:
            idxs = np.concatenate([uniform_idxs, priority_idxs])
            np.random.shuffle(idxs)
        else:
            idxs = uniform_idxs

        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }
