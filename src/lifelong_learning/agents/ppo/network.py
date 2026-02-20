from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CNNActorCritic(nn.Module):
    """
    Actor-Critic network for MiniGrid with One-Hot encoded observations.

    Architecture:
        - Shared 3-layer CNN encoder (input channels → 32 → 64 → 64)
        - Decoupled Actor head (policy logits) and Critic head (state value)

    Input:  (B, C, H, W) one-hot tensor from OneHotPartialObsWrapper
    Output: (logits, value)
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        self.c, self.h, self.w = obs_shape

        # Shared CNN feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.c, self.h, self.w)
            flat_size = self.encoder(dummy).shape[1]

        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Orthogonal init with role-specific gains for output layers."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Actor output: gain=0.01 → near-uniform initial policy
        for layer in self.actor_head:
            if isinstance(layer, nn.Linear):
                gain = 0.01 if layer == self.actor_head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Critic output: gain=1.0 → standard for value function
        for layer in self.critic_head:
            if isinstance(layer, nn.Linear):
                gain = 1.0 if layer == self.critic_head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)

    def act(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions and return (log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value