from __future__ import annotations

import argparse
from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import train_ppo


def main():
    p = argparse.ArgumentParser(description="Train PPO / Dyna-PPO on MiniGrid-DualGoal")
    p.add_argument("--env_id", type=str, default="MiniGrid-DualGoal-8x8-v0")
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # Regime switching schedule
    p.add_argument("--steps_per_regime", type=int, default=None, help="Steps per env before regime switch")
    p.add_argument("--episodes_per_regime", type=int, default=None, help="Episodes per env before regime switch")
    p.add_argument("--start_regime", type=int, default=0)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint.pt to resume from")

    # Learning rate annealing
    p.add_argument("--anneal_lr", action="store_true", default=False, help="Enable linear LR decay to 0")
    p.add_argument("--no-anneal_lr", action="store_false", dest="anneal_lr", help="Constant LR (recommended for lifelong)")

    # Dyna-PPO / Passive mode
    p.add_argument("--mode", type=str, default="dyna", choices=["dyna", "passive"], help="'dyna' (WM + curiosity) or 'passive' (baseline PPO)")
    p.add_argument("--intrinsic_coef", type=float, default=0.015, help="Coefficient for intrinsic curiosity reward")
    p.add_argument("--intrinsic_reward_clip", type=float, default=0.1, help="Max intrinsic reward per step")
    p.add_argument("--imagined_horizon", type=int, default=10, help="Length of imagined trajectories")
    p.add_argument("--wm_lr", type=float, default=1e-4, help="World Model learning rate")

    args = p.parse_args()

    # In passive mode, disable curiosity and dreaming
    if args.mode == "passive":
        print("Running in PASSIVE mode (No curiosity, No dreaming).")
        intrinsic_coef = 0.0
        imagined_horizon = 0
    else:
        print(f"Running in DYNA mode (intrinsic_coef={args.intrinsic_coef}, horizon={args.imagined_horizon}).")
        intrinsic_coef = args.intrinsic_coef
        imagined_horizon = args.imagined_horizon

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
        device=args.device,
        mode=args.mode,
    )

    train_ppo(
        env_id=args.env_id,
        cfg=cfg,
        steps_per_regime=args.steps_per_regime,
        episodes_per_regime=args.episodes_per_regime,
        start_regime=args.start_regime,
        run_name=args.run_name,
        anneal_lr=args.anneal_lr,
        resume_path=args.resume_path,
        intrinsic_coef=intrinsic_coef,
        imagined_horizon=imagined_horizon,
        intrinsic_reward_clip=args.intrinsic_reward_clip,
        wm_lr=args.wm_lr,
    )


if __name__ == "__main__":
    main()
