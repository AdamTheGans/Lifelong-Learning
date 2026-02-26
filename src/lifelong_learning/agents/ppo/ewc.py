"""
Elastic Weight Consolidation (EWC) for shared-trunk policy networks.

Applies a quadratic penalty ONLY to the shared encoder parameters,
preventing the CNN feature extractor from drifting when switching
regimes.  Per-regime heads are excluded (they're already isolated).

    L_ewc = (lambda / 2) * sum_i  F_i * (theta_i - theta_i*)^2

where F_i is the diagonal Fisher information and theta_i* are the
reference parameters from the previous regime.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """Tracks Fisher information and reference parameters for EWC.

    Args:
        model:        The multi-head actor-critic network.
        device:       Torch device.
        param_prefix: Only parameters whose names start with this prefix
                      are protected.  Default ``"encoder."`` targets the
                      shared CNN trunk while ignoring per-regime heads.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 param_prefix: str = "encoder."):
        self.device = device
        self.param_prefix = param_prefix
        self.fisher: dict[str, torch.Tensor] = {}
        self.ref_params: dict[str, torch.Tensor] = {}
        self._initialized = False

    def _is_tracked(self, name: str) -> bool:
        """Return True if this parameter should be protected by EWC."""
        return name.startswith(self.param_prefix)

    @property
    def is_active(self) -> bool:
        return self._initialized

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
        # 1. Snapshot current encoder parameters as theta*
        with torch.no_grad():
            self.ref_params = {
                name: p.data.clone()
                for name, p in model.named_parameters()
                if p.requires_grad and self._is_tracked(name)
            }

        # 2. Estimate diagonal Fisher via policy log-prob gradients
        self.fisher = {
            name: torch.zeros_like(p)
            for name, p in model.named_parameters()
            if p.requires_grad and self._is_tracked(name)
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

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and self._is_tracked(name):
                        # Fisher = E[grad(log pi)^2]
                        self.fisher[name] += (p.grad.data ** 2) * (end - start)

        # Normalize by total samples
        with torch.no_grad():
            for name in self.fisher:
                self.fisher[name] /= n_samples

        model.train()
        self._initialized = True

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: sum_i F_i * (theta_i - theta_i*)^2

        Only sums over tracked (encoder) parameters.
        Returns scalar tensor (on model device).
        """
        if not self._initialized:
            return torch.tensor(0.0, device=self.device)

        total = torch.tensor(0.0, device=self.device)
        for name, p in model.named_parameters():
            if name in self.fisher:
                total += (self.fisher[name] * (p - self.ref_params[name]) ** 2).sum()

        return total
