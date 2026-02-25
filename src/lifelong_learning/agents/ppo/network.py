from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CNNActorCritic(nn.Module):
    """
    Actor-Critic network for MiniGrid with One-Hot encoded observations.

    Architecture:
        - Shared 3-layer CNN encoder (input channels → 32 → 64 → 64)
        - Multiple Actor heads (one per regime, switchable)
        - Multiple Critic heads (one per regime, switchable)

    Input:  (B, C, H, W) one-hot tensor from OneHotPartialObsWrapper
    Output: (logits, value)
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        self.c, self.h, self.w = obs_shape
        self.n_actions = n_actions

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
            self.flat_size = self.encoder(dummy).shape[1]

        # Multi-head actor (one per regime)
        self.actor_heads = nn.ModuleList([self._make_actor_head()])

        # Multi-head critic (one per regime)
        self.critic_heads = nn.ModuleList([self._make_critic_head()])

        self.active_policy_head: int = 0

        # Weight initialization
        self.apply(self._init_weights)

    def _make_actor_head(self) -> nn.Sequential:
        """Create a new actor head with proper initialization."""
        head = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )
        for layer in head:
            if isinstance(layer, nn.Linear):
                gain = 0.01 if layer == head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        return head

    def _make_critic_head(self) -> nn.Sequential:
        """Create a new critic head with proper initialization."""
        head = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        for layer in head:
            if isinstance(layer, nn.Linear):
                gain = 1.0 if layer == head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        return head

    def spawn_policy_head(self) -> int:
        """Add new actor + critic heads and return the index."""
        device = next(self.parameters()).device
        self.actor_heads.append(self._make_actor_head().to(device))
        self.critic_heads.append(self._make_critic_head().to(device))
        return len(self.actor_heads) - 1

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list[dict]:
        """Return param groups with separate LRs for encoder vs heads."""
        return [
            {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            {"params": list(self.actor_heads.parameters()) +
                       list(self.critic_heads.parameters()), "lr": head_lr},
        ]

    def _init_weights(self, m):
        """Orthogonal init for conv/linear layers."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        logits = self.actor_heads[self.active_policy_head](features)
        value = self.critic_heads[self.active_policy_head](features).squeeze(-1)
        return logits, value

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