from __future__ import annotations

import argparse
from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import train_ppo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # Deterministic Schedule
    p.add_argument("--steps_per_regime", type=int, default=None, help="Steps per env before switch")
    p.add_argument("--episodes_per_regime", type=int, default=None, help="Episodes per env before switch")
    p.add_argument("--start_regime", type=int, default=0)

    # Legacy / Random
    p.add_argument("--switch_on_reset", action="store_true")
    p.add_argument("--switch_mid_episode", action="store_true")
    p.add_argument("--switch_lo", type=int, default=20)
    p.add_argument("--switch_hi", type=int, default=80)

    p.add_argument("--run_name", type=str, default=None)
    args = p.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
        device=args.device,
    )

    # NOTE: Ensure agents/ppo/train.py:train_ppo accepts these new kwargs
    # and passes them to make_env!
    train_ppo(
        env_id=args.env_id,
        cfg=cfg,
        steps_per_regime=args.steps_per_regime,
        episodes_per_regime=args.episodes_per_regime,
        start_regime=args.start_regime,
        switch_on_reset=args.switch_on_reset,
        switch_mid_episode=args.switch_mid_episode,
        mid_episode_switch_step_range=(args.switch_lo, args.switch_hi),
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
