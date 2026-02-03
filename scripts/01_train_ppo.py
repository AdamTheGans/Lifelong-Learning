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

    train_ppo(
        env_id=args.env_id,
        cfg=cfg,
        switch_on_reset=args.switch_on_reset,
        switch_mid_episode=args.switch_mid_episode,
        mid_episode_switch_step_range=(args.switch_lo, args.switch_hi),
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
