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

    def discretize_state(self, continuous_obs: torch.Tensor) -> torch.Tensor:
        """
        Converts continuous World Model outputs back into a valid One-Hot encoded state.
        
        Args:
            continuous_obs: (B, C, H, W) float tensor
            
        Returns:
            discrete_obs: (B, C, H, W) float tensor (containing only 0.0 and 1.0)
        """
        # 1. Identify the most likely channel (category) for each pixel
        # shape: (B, H, W)
        max_indices = torch.argmax(continuous_obs, dim=1)
        
        # 2. Convert back to One-Hot
        # shape: (B, H, W, C)
        one_hot = F.one_hot(max_indices, num_classes=self.c)
        
        # 3. Permute back to (B, C, H, W) and float
        # shape: (B, C, H, W)
        discrete_obs = one_hot.permute(0, 3, 1, 2).float()
        
        return discrete_obs

    def generate_imagined_trajectories(
        self, 
        policy_net: nn.Module, 
        start_states: torch.Tensor, 
        horizon: int
    ) -> list[dict]:
        """
        Generates imagined trajectories using the World Model and Policy.
        
        Args:
            policy_net: The PPO Actor-Critic network
            start_states: (B, C, H, W) tensor from the real buffer
            horizon: Int, number of steps to dream
            
        Returns:
            list of transition dicts: {
                'obs': (B, ...), 
                'actions': (B,), 
                'logprobs': (B,), 
                'rewards': (B,), 
                'dones': (B,), 
                'values': (B,), 
                'next_obs': (B, ...)
            }
        """
        trajectories = []
        curr_obs = start_states
        
        # We process 'horizon' steps
        for _ in range(horizon):
            # 1. Action Selection (Policy)
            # Note: distinct from real env step, we use the policy on the *imagined* observation
            # We must ensure gradients flow through the policy for PPO, but NOT through the world model
            # For the *input* to the policy, we treat it as ground truth state.
            with torch.no_grad():
                # We typically don't backprop through the *generation* of the data for PPO 
                # (PPO assumes fixed data collection).
                # So we can run this completely in no_grad for data generation efficiency, 
                # OR we might want gradients for solving differentiable planning?
                # User request: "gradients are strictly blocked from updating the World Model's weights".
                # Standard Dyna-PPO: Collect imagined data (no grad), then train PPO on it (grad).
                action, logprob, _, value = policy_net.act(curr_obs)
            
            # 2. World Model Prediction
            # We need the prediction to advance the state
            # For "Dyna", we treat the World Model as an Environment.
            # We do NOT propagate gradients from the policy loss into the World Model.
            # So this forward pass is also effectively no_grad or detached.
            with torch.no_grad():
                next_obs_pred, reward_pred = self.forward(curr_obs, action)
            
            # 3. Handle Hallucinations (Discrete State Enforcement)
            # This is crucial for the "Symbolic" environment consistency
            next_obs_discrete = self.discretize_state(next_obs_pred)
            
            # 4. Store transition
            # Using 0 for done/truncated for simplicity in dreams (infinite horizon assumption or fixed horizon)
            dones = torch.zeros(curr_obs.shape[0], device=curr_obs.device)
            
            trajectories.append({
                "obs": curr_obs,
                "actions": action,
                "logprobs": logprob,
                "rewards": reward_pred,
                "dones": dones,
                "values": value,
                "next_obs": next_obs_discrete
            })
            
            # 5. Advance
            curr_obs = next_obs_discrete
            
        return trajectories
