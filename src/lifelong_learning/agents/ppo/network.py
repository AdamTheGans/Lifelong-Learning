from __future__ import annotations

import torch
import torch.nn as nn


class CNNActorCritic(nn.Module):
    """
    Simple CNN -> MLP -> (policy logits, value).
    Works with MiniGrid image obs: (C, H, W) = (3, 7, 7).
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.cnn(dummy).shape[-1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        # Orthogonal init often helps PPO stability
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        # Normalize input. MiniGrid object IDs are small integers (0-10).
        # Dividing by 10.0 scales them to roughly [0, 1] range.
        # This prevents large input values from saturating ReLU or gradients.
        z = self.cnn(obs / 10.0) 
        z = self.mlp(z)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value
