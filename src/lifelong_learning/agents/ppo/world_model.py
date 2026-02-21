from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleWorldModel(nn.Module):
    """
    Lightweight feed-forward World Model for symbolic (One-Hot) observations.

    Predicts next state and reward from (state, action).

    Architecture:
        - State encoder: Flatten → Linear → ReLU
        - Action embedding: Embedding(n_actions, 32)
        - Core: Concat([state, action_emb]) → 2-layer MLP
        - Heads:
            - Next State: Linear → reshape to (C, H, W) logits
            - Reward: Linear → scalar
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_shape = obs_shape
        self.c, self.h, self.w = obs_shape
        self.n_actions = n_actions
        self.flat_obs_dim = self.c * self.h * self.w

        # CNN encoder — mirrors CNNActorCritic architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically compute CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, self.c, self.h, self.w)
            cnn_out_dim = self.cnn(dummy).shape[1]

        # Action embedding (late fusion)
        self.action_emb = nn.Embedding(n_actions, 32)

        # Fuse CNN features + action embedding → MLP trunk
        self.trunk = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.next_state_head = nn.Linear(hidden_dim, self.flat_obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Args:
            obs:    (B, C, H, W) one-hot observation
            action: (B,) integer actions

        Returns:
            next_obs_pred: (B, C, H, W) logits (for CrossEntropy loss)
            reward_pred:   (B,) scalar reward prediction
        """
        B = obs.shape[0]
        cnn_features = self.cnn(obs)                          # (B, cnn_out_dim)
        act_emb = self.action_emb(action)                     # (B, 32)
        x = torch.cat([cnn_features, act_emb], dim=1)         # (B, cnn_out_dim + 32)
        features = self.trunk(x)                               # (B, hidden_dim)

        next_obs_flat = self.next_state_head(features)
        reward_pred = self.reward_head(features).squeeze(-1)
        next_obs_pred = next_obs_flat.reshape(B, self.c, self.h, self.w)

        return next_obs_pred, reward_pred

    def discretize_state(self, continuous_obs: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous WM output back into a valid one-hot state.

        Takes argmax along the channel dimension and re-encodes as one-hot,
        preventing "blurry dreams" from accumulating prediction noise.

        Args:
            continuous_obs: (B, C, H, W) float logits
        Returns:
            discrete_obs: (B, C, H, W) float tensor containing only 0.0 and 1.0
        """
        max_indices = torch.argmax(continuous_obs, dim=1)          # (B, H, W)
        one_hot = F.one_hot(max_indices, num_classes=self.c)       # (B, H, W, C)
        discrete_obs = one_hot.permute(0, 3, 1, 2).float()        # (B, C, H, W)
        return discrete_obs

    def generate_imagined_trajectories(
        self,
        policy_net: nn.Module,
        start_states: torch.Tensor,
        horizon: int
    ) -> list[dict]:
        """
        Roll out imagined trajectories using the WM as a simulator (Dyna-style).

        The policy selects actions, the WM predicts next states/rewards, and
        states are discretized each step to maintain valid one-hot encoding.
        All operations are under no_grad — PPO treats imagined data the same
        as real data (fixed collection, then update).

        Args:
            policy_net:   PPO Actor-Critic network
            start_states: (B, C, H, W) seed states sampled from the replay buffer
            horizon:      number of imagination steps

        Returns:
            List of transition dicts, one per timestep:
                {obs, actions, logprobs, rewards, dones, values, next_obs}
        """
        trajectories = []
        curr_obs = start_states

        for _ in range(horizon):
            with torch.no_grad():
                action, logprob, _, value = policy_net.act(curr_obs)
                next_obs_pred, reward_pred = self.forward(curr_obs, action)

            next_obs_discrete = self.discretize_state(next_obs_pred)

            # Termination heuristic: reward > 0.5 implies goal reached
            dones = (reward_pred > 0.5).float()

            trajectories.append({
                "obs": curr_obs,
                "actions": action,
                "logprobs": logprob,
                "rewards": reward_pred,
                "dones": dones,
                "values": value,
                "next_obs": next_obs_discrete
            })

            curr_obs = next_obs_discrete

        return trajectories
