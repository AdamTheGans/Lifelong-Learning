from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleWorldModel(nn.Module):
    """
    A lightweight World Model valid for PPO sidekick duties.
    Predicts Next State and Reward from (State, Action).
    
    Architecture:
    - State Encoder: Flatten -> Linear -> ReLU
    - Action Embedding: Embedding
    - Core: Concat(State, Action) -> MLP -> ReLU
    - Heads:
        - Next State: Linear -> Reshape to (C, H, W)
        - Reward: Linear -> Scalar
    """
    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_shape = obs_shape
        self.c, self.h, self.w = obs_shape
        self.n_actions = n_actions
        self.flat_obs_dim = self.c * self.h * self.w

        # Action Embedding
        self.action_emb = nn.Embedding(original_count := n_actions, embedding_dim := 32)

        # Joint Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.flat_obs_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Heads
        self.next_state_head = nn.Linear(hidden_dim, self.flat_obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        obs: (B, C, H, W)
        action: (B,) int
        
        Returns:
            next_obs_pred: (B, C, H, W) logits (or raw values for MSE)
            reward_pred: (B,)
        """
        B = obs.shape[0]
        
        # Flatten Obs
        flat_obs = obs.reshape(B, -1)
        
        # Embed Action
        act_emb = self.action_emb(action)
        
        # Concat
        x = torch.cat([flat_obs, act_emb], dim=1)
        
        # Encode
        features = self.encoder(x)
        
        # Predict
        next_obs_flat = self.next_state_head(features)
        reward_pred = self.reward_head(features).squeeze(-1)
        
        # Reshape next_obs
        next_obs_pred = next_obs_flat.reshape(B, self.c, self.h, self.w)
        
        return next_obs_pred, reward_pred
