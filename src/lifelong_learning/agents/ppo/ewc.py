"""
Elastic Weight Consolidation (EWC) for PPO policy.

Adds a quadratic penalty that discourages changing weights that were
important for previous regimes:

    L_ewc = (lambda / 2) * sum_i  F_i * (theta_i - theta_i*)^2

where F_i is the diagonal Fisher information and theta_i* are the
reference parameters from the previous regime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """Tracks Fisher information and reference parameters for EWC."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.fisher: dict[str, torch.Tensor] = {}
        self.ref_params: dict[str, torch.Tensor] = {}
        self._initialized = False

    @property
    def is_active(self) -> bool:
        return self._initialized

    @torch.no_grad()
    def update(self, model: nn.Module, obs: torch.Tensor,
               actions: torch.Tensor) -> None:
        """
        Snapshot current params as reference and estimate diagonal Fisher.

        Call this when a regime switch is detected.

        Args:
            model:   PPO actor-critic network
            obs:     (N, C, H, W) observations from the current regime
            actions: (N,) actions taken in those observations
        """
        # 1. Snapshot current parameters as theta*
        self.ref_params = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # 2. Estimate diagonal Fisher via policy log-prob gradients
        self.fisher = {
            name: torch.zeros_like(p)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        model.eval()
        n_samples = obs.shape[0]

        # Process in mini-batches to save memory
        batch_size = min(256, n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_obs = obs[start:end]
            batch_act = actions[start:end]

            model.zero_grad()
            log_probs, _, _ = model.evaluate_actions(batch_obs, batch_act)
            loss = log_probs.mean()
            loss.backward()

            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    # Fisher = E[grad(log pi)^2]
                    self.fisher[name] += (p.grad.data ** 2) * (end - start)

        # Normalize by total samples
        for name in self.fisher:
            self.fisher[name] /= n_samples

        model.train()
        self._initialized = True

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: sum_i F_i * (theta_i - theta_i*)^2

        Returns scalar tensor (on model device).
        """
        if not self._initialized:
            return torch.tensor(0.0, device=self.device)

        total = torch.tensor(0.0, device=self.device)
        for name, p in model.named_parameters():
            if name in self.fisher:
                total += (self.fisher[name] * (p - self.ref_params[name]) ** 2).sum()

        return total
