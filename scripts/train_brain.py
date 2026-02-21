"""
Training script for the Brain meta-agent.

Usage:
    python scripts/train_brain.py --inner_total_timesteps 250000 --brain_episodes 50
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import torch

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.brain.meta_env import MetaEnv
from lifelong_learning.agents.brain.meta_agent import (
    MLPActorCritic,
    BrainConfig,
    BrainRolloutBuffer,
    brain_ppo_update,
)
from lifelong_learning.utils.logger import TBLogger


def train_brain(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Inner agent config (passed to MetaEnv)
    # -----------------------------------------------------------------
    inner_cfg = PPOConfig(
        total_timesteps=args.inner_total_timesteps,
        num_envs=args.inner_num_envs,
        num_steps=args.inner_num_steps,
        seed=args.seed,
        device=args.device,
        mode=args.inner_mode,
    )

    # -----------------------------------------------------------------
    # Meta-environment
    # -----------------------------------------------------------------
    meta_env = MetaEnv(
        env_id=args.env_id,
        inner_cfg=inner_cfg,
        decision_interval=args.decision_interval,
        steps_per_regime=args.inner_steps_per_regime,
        start_regime=0,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        anneal_lr=False,  # Brain controls LR
        intrinsic_coef=args.inner_intrinsic_coef,
        imagined_horizon=args.inner_imagined_horizon,
        wm_lr=args.inner_wm_lr,
    )

    # -----------------------------------------------------------------
    # Brain agent
    # -----------------------------------------------------------------
    brain_cfg = BrainConfig(
        brain_episodes=args.brain_episodes,
        lr=args.brain_lr,
        device=args.device,
    )

    brain_model = MLPActorCritic().to(device)
    brain_optimizer = torch.optim.Adam(brain_model.parameters(), lr=brain_cfg.lr, eps=1e-5)

    logger = TBLogger(run_name=args.run_name or "brain_training")

    # -----------------------------------------------------------------
    # Training loop: each episode = one full inner training run
    # -----------------------------------------------------------------
    print(f"Training Brain on {device} for {brain_cfg.brain_episodes} episodes")
    print(f"  Inner: {args.inner_total_timesteps} timesteps, "
          f"regime switch every {args.inner_steps_per_regime} steps")
    print(f"  Decision interval: {args.decision_interval} inner updates")

    all_episode_rewards = []

    for episode in range(1, brain_cfg.brain_episodes + 1):
        ep_start = time.time()
        rollout = BrainRolloutBuffer()
        total_reward = 0.0
        steps = 0

        obs, info = meta_env.reset()

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, entropy, value = brain_model.act(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = meta_env.step(action_np)

            rollout.add(
                obs=obs,
                action=action_np,
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=float(terminated or truncated),
            )

            total_reward += reward
            steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        # Compute advantages
        rollout.compute_returns_and_advantages(
            last_value=0.0,  # Episode ended
            gamma=brain_cfg.gamma,
            gae_lambda=brain_cfg.gae_lambda,
        )

        # PPO update on this episode's data
        batch = rollout.get_batches(device)
        update_stats = brain_ppo_update(brain_model, brain_optimizer, batch, brain_cfg)

        # Log
        ep_time = time.time() - ep_start
        all_episode_rewards.append(total_reward)
        avg_reward_10 = np.mean(all_episode_rewards[-10:])

        logger.scalar("brain/episode_reward", total_reward, episode)
        logger.scalar("brain/episode_steps", steps, episode)
        logger.scalar("brain/episode_time_s", ep_time, episode)
        logger.scalar("brain/avg_reward_10ep", avg_reward_10, episode)

        for k, v in update_stats.items():
            logger.scalar(k, v, episode)

        # Log HP trajectories from last inner stats
        inner_stats = info.get("inner_stats", {})
        logger.scalar("brain/inner_final_success_rate", inner_stats.get("success_rate", 0.0), episode)
        logger.scalar("brain/inner_final_lr", inner_stats.get("current_lr", 0.0), episode)
        logger.scalar("brain/inner_final_ent_coef", inner_stats.get("current_ent_coef", 0.0), episode)
        logger.scalar("brain/inner_final_intrinsic_coef", inner_stats.get("current_intrinsic_coef", 0.0), episode)
        logger.scalar("brain/inner_final_imagined_horizon", inner_stats.get("current_imagined_horizon", 0.0), episode)

        print(f"Episode {episode}/{brain_cfg.brain_episodes} | "
              f"reward={total_reward:.4f} | avg10={avg_reward_10:.4f} | "
              f"steps={steps} | time={ep_time:.1f}s")

    meta_env.close()
    logger.close()
    print(f"\nBrain training complete. Total episodes: {brain_cfg.brain_episodes}")


def main():
    p = argparse.ArgumentParser(description="Train the Brain meta-agent")

    # Inner agent settings
    p.add_argument("--env_id", type=str, default="MiniGrid-DualGoal-8x8-v0")
    p.add_argument("--inner_total_timesteps", type=int, default=250_000)
    p.add_argument("--inner_num_envs", type=int, default=8)
    p.add_argument("--inner_num_steps", type=int, default=128)
    p.add_argument("--inner_steps_per_regime", type=int, default=15000)
    p.add_argument("--inner_mode", type=str, default="dyna", choices=["dyna", "passive"])
    p.add_argument("--inner_intrinsic_coef", type=float, default=0.015)
    p.add_argument("--inner_imagined_horizon", type=int, default=10)
    p.add_argument("--inner_wm_lr", type=float, default=1e-4)

    # Brain meta-agent settings
    p.add_argument("--brain_episodes", type=int, default=50)
    p.add_argument("--brain_lr", type=float, default=3e-4)
    p.add_argument("--decision_interval", type=int, default=10,
                   help="Number of inner PPO updates per Brain decision")

    # Reward shaping
    p.add_argument("--reward_alpha", type=float, default=0.1)
    p.add_argument("--reward_beta", type=float, default=0.5)

    # General
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run_name", type=str, default=None)

    args = p.parse_args()
    train_brain(args)


if __name__ == "__main__":
    main()
