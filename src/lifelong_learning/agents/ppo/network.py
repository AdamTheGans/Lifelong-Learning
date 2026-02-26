from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CNNActorCritic(nn.Module):
    """
    Multi-head Actor-Critic with shared CNN trunk for lifelong learning.

    Architecture:
        - Shared CNN encoder (input channels → 32 → 64 → 64)
        - Per-regime actor heads (policy logits)   — nn.ModuleList
        - Per-regime critic heads (state value)    — nn.ModuleList

    The shared trunk learns visual features that transfer across regimes,
    while separate heads prevent catastrophic forgetting of regime-specific
    policy/value decisions.

    Input:  (B, C, H, W) one-hot tensor from OneHotPartialObsWrapper
    Output: (logits, value) — routed through self.active_head
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, num_initial_heads: int = 1):
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

        # Per-regime heads
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        for _ in range(num_initial_heads):
            self.actor_heads.append(self._make_actor_head())
            self.critic_heads.append(self._make_critic_head())

        self.active_head = 0

        # Weight initialization
        self.apply(self._init_weights)

    def _make_actor_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

    def _make_critic_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def spawn_head(self) -> int:
        """Add a new actor + critic head pair. Returns new head index."""
        device = next(self.parameters()).device
        actor = self._make_actor_head().to(device)
        critic = self._make_critic_head().to(device)
        self._init_head_weights(actor, role="actor")
        self._init_head_weights(critic, role="critic")
        self.actor_heads.append(actor)
        self.critic_heads.append(critic)
        return len(self.actor_heads) - 1

    def _init_weights(self, m):
        """Orthogonal init with role-specific gains for output layers."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Apply role-specific gains to all existing heads
        for actor in self.actor_heads:
            self._init_head_weights(actor, role="actor")
        for critic in self.critic_heads:
            self._init_head_weights(critic, role="critic")

    def _init_head_weights(self, head: nn.Sequential, role: str):
        """Apply role-specific orthogonal init to a head."""
        for layer in head:
            if isinstance(layer, nn.Linear):
                if role == "actor":
                    gain = 0.01 if layer == head[-1] else np.sqrt(2)
                else:  # critic
                    gain = 1.0 if layer == head[-1] else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        logits = self.actor_heads[self.active_head](features)
        value = self.critic_heads[self.active_head](features).squeeze(-1)
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