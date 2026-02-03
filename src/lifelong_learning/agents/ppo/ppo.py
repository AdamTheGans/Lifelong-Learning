from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    total_timesteps: int = 300_000
    num_envs: int = 8
    num_steps: int = 128  # rollout length
    update_epochs: int = 4
    minibatch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    lr: float = 2.5e-4
    max_grad_norm: float = 0.5

    seed: int = 0
    device: str = "cuda"


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    minibatches,
    cfg: PPOConfig,
):
    """
    Performs PPO updates for one rollout buffer.
    Returns logging scalars.
    """
    total_pg, total_v, total_ent, total_loss = 0.0, 0.0, 0.0, 0.0
    n = 0

    for obs, actions, old_logprobs, advantages, returns, old_values in minibatches:
        new_logprobs, entropy, new_values = model.evaluate_actions(obs, actions)
        logratio = new_logprobs - old_logprobs
        ratio = torch.exp(logratio)

        # Policy loss (clipped)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * (returns - new_values).pow(2).mean()

        # Entropy
        ent_loss = entropy.mean()

        loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_pg += float(pg_loss.detach().cpu())
        total_v += float(v_loss.detach().cpu())
        total_ent += float(ent_loss.detach().cpu())
        total_loss += float(loss.detach().cpu())
        n += 1

    return {
        "loss/policy": total_pg / max(n, 1),
        "loss/value": total_v / max(n, 1),
        "loss/entropy": total_ent / max(n, 1),
        "loss/total": total_loss / max(n, 1),
    }
