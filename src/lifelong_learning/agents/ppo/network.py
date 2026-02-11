from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNActorCritic(nn.Module):
    """
    Stronger architecture for MiniGrid.
    Expects One-Hot encoded input from environment wrapper.
    Decouples Actor and Critic heads for better convergence.
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        # obs_shape is now (20, H, W) -> pre-encoded one-hot
        self.c, self.h, self.w = obs_shape
        
        # Input is already One-Hot encoded by wrapper
        input_channels = self.c

        # Shared Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flat size
        with torch.no_grad():
            # Dummy input to calculate size
            dummy = torch.zeros(1, input_channels, self.h, self.w)
            flat_size = self.encoder(dummy).shape[1]

        # Decoupled Heads
        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        # Input is already float one-hot, just pass to encoder
        features = self.encoder(obs)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value