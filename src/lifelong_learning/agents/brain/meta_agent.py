"""
Brain meta-agent: an MLP-based PPO agent that learns to adjust
the inner Dyna-PPO agent's hyperparameters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from lifelong_learning.agents.brain.signals import NUM_SIGNALS


@dataclass
class BrainConfig:
    """Hyperparameters for the Brain meta-agent's own PPO training."""
    brain_episodes: int = 100        # Number of inner training runs (Brain episodes)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 32
    device: str = "cuda"


class MLPActorCritic(nn.Module):
    """
    Simple MLP Actor-Critic for the Brain.

    Input: NUM_SIGNALS-dim observation (training signals)
    Output: (action_mean, action_log_std, value)

    Uses a continuous Gaussian policy since the action space is Box(4,).
    """

    def __init__(self, obs_dim: int = NUM_SIGNALS, act_dim: int = 5, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        action_mean = torch.tanh(self.actor_mean(features))  # Squash to [-1, 1]
        value = self.critic(features).squeeze(-1)
        return action_mean, value

    def act(self, obs: torch.Tensor):
        """Sample an action from the Gaussian policy."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)  # Squash to [-1, 1]

        # Log prob with tanh correction
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions under the current policy."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_mean, std)

        # Inverse tanh to get raw actions
        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))

        log_prob = dist.log_prob(raw_actions).sum(-1)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(-1)

        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class BrainRolloutBuffer:
    """
    Simple rollout buffer for the Brain's PPO training.

    Stores transitions as flat lists since the Brain's episodes have
    variable length (depends on inner training run length).
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE advantages for the collected trajectory."""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        T = len(rewards)
        num_envs = rewards.shape[1] if len(rewards.shape) > 1 else 1

        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)

        last_adv = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values
                next_nonterminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_nonterminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
            advantages[t] = last_adv

        returns = advantages + values
        self._advantages = advantages
        self._returns = returns

    def get_batches(self, device: torch.device):
        """Return all data as flattened tensors for PPO update."""
        obs = np.array(self.obs, dtype=np.float32).reshape(-1, NUM_SIGNALS)
        actions = np.array(self.actions, dtype=np.float32).reshape(-1, 5)
        log_probs = np.array(self.log_probs, dtype=np.float32).reshape(-1)
        advantages = self._advantages.reshape(-1)
        returns = self._returns.reshape(-1)
        values = np.array(self.values, dtype=np.float32).reshape(-1)

        return {
            "obs": torch.tensor(obs, device=device),
            "actions": torch.tensor(actions, device=device),
            "log_probs": torch.tensor(log_probs, device=device),
            "advantages": torch.tensor(advantages, device=device),
            "returns": torch.tensor(returns, device=device),
            "values": torch.tensor(values, device=device),
        }

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


def brain_ppo_update(
    model: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    cfg: BrainConfig,
) -> dict:
    """
    Run PPO update epochs on a Brain rollout batch.

    Returns averaged loss statistics.
    """
    device = next(model.parameters()).device
    batch_size = batch["obs"].shape[0]

    total_pg, total_v, total_ent, total_loss = 0.0, 0.0, 0.0, 0.0
    n = 0

    for epoch in range(cfg.update_epochs):
        idxs = np.random.permutation(batch_size)

        for start in range(0, batch_size, cfg.minibatch_size):
            mb = idxs[start:start + cfg.minibatch_size]

            obs = batch["obs"][mb]
            actions = batch["actions"][mb]
            old_logprobs = batch["log_probs"][mb]
            advantages = batch["advantages"][mb]
            returns = batch["returns"][mb]
            old_values = batch["values"][mb]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            new_logprobs, entropy, new_values = model.evaluate_actions(obs, actions)

            logratio = new_logprobs - old_logprobs
            ratio = torch.exp(logratio)

            # Clipped policy loss
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * (returns - new_values).pow(2).mean()

            # Entropy bonus
            ent_loss = entropy.mean()

            loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_pg += pg_loss.item()
            total_v += v_loss.item()
            total_ent += ent_loss.item()
            total_loss += loss.item()
            n += 1

    return {
        "brain/policy_loss": total_pg / max(n, 1),
        "brain/value_loss": total_v / max(n, 1),
        "brain/entropy": total_ent / max(n, 1),
        "brain/total_loss": total_loss / max(n, 1),
    }
