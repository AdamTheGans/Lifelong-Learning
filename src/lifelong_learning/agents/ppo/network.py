from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNActorCritic(nn.Module):
    """
    Stronger architecture for MiniGrid.
    Uses One-Hot encoding for Object IDs and Colors instead of scaling.
    Decouples Actor and Critic heads for better convergence.
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int):
        super().__init__()
        # obs_shape is (3, H, W) -> (Type, Color, State)
        self.c, self.h, self.w = obs_shape
        
        # MiniGrid constants (usually max object ID is ~11, max color is ~6)
        # We'll set safe upper bounds for One-Hot encoding.
        self.max_obj_id = 11
        self.max_col_id = 6
        
        # Calculate input channels for CNN after One-Hot concatenation
        # Channel 0 (Type) -> expands to max_obj_id channels
        # Channel 1 (Color) -> expands to max_col_id channels
        # We ignore Channel 2 (State) for this simple task, or we could add it.
        # Total channels = 11 + 6 = 17
        input_channels = self.max_obj_id + self.max_col_id

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

    def _preprocess(self, obs: torch.Tensor):
        """
        Converts MiniGrid (B, 3, H, W) symbolic obs into One-Hot tensors.
        """
        # Ensure integer type for one_hot
        obs = obs.long()
        
        # Split channels: (B, 3, H, W)
        objects = obs[:, 0, :, :] # (B, H, W)
        colors  = obs[:, 1, :, :] # (B, H, W)
        
        # One-Hot Encode
        # F.one_hot adds a last dimension: (B, H, W, Num_Classes)
        # We need to permute to (B, Num_Classes, H, W)
        self.num_obj_classes = 12  # e.g. max_id + 1
        self.num_col_classes = 7
        obj_hot = F.one_hot(objects, num_classes=self.num_obj_classes).permute(0, 3, 1, 2).float()
        col_hot = F.one_hot(colors, num_classes=self.num_col_classes).permute(0, 3, 1, 2).float()
        
        # Concatenate along channel dimension
        return torch.cat([obj_hot, col_hot], dim=1)

    def forward(self, obs: torch.Tensor):
        x = self._preprocess(obs)
        features = self.encoder(x)
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