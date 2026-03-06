"""
Evaluate pretrain inner agent hyperparameters on a fresh run.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import torch

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import train_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to inner checkpoint .pt file")
    parser.add_argument("--env_id", type=str, default="MiniGrid-MultiGoal-8x8-v0")
    parser.add_argument("--total_timesteps", type=int, default=450000)
    parser.add_argument("--num_regimes", type=int, default=3)
    parser.add_argument("--steps_per_regime", type=int, default=22500)
    args = parser.parse_args()

    # Workaround for NumPy 2.0 vs 1.x version mismatch when unpickling
    try:
        import numpy.core
        sys.modules['numpy._core'] = sys.modules['numpy.core']
    except ImportError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint_path}")
    
    # Extract hyperparameters
    try:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    cfg_dict = ckpt.get("cfg", {})
    lr = cfg_dict.get("lr", 3e-4)
    ent_coef = cfg_dict.get("ent_coef", 0.01)

    intrinsic_coef = 0.015
    imagined_horizon = 10

    # Attempt to read other hyperparams (intrinsic_coef, imagined_horizon) from the JSON log
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
    run_dir = os.path.dirname(ckpt_dir)
    basename = os.path.basename(args.checkpoint_path)
    env_name = basename.split("_update")[0]  # e.g., ep1_env0

    json_path = None
    if os.path.exists(run_dir):
        for d in os.listdir(run_dir):
            if d.startswith(env_name + "_") and os.path.isdir(os.path.join(run_dir, d)):
                j = os.path.join(run_dir, d, f"{env_name}_data.json")
                if os.path.exists(j):
                    json_path = j
                    break

    if json_path:
        print(f"Reading additional hyperparams from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
        
        if "brain/intrinsic_coef" in data and len(data["brain/intrinsic_coef"]) > 0:
            intrinsic_coef = data["brain/intrinsic_coef"][-1][1]
        elif "brain_hyperparams/mean_inner_intrinsic_coef" in data and len(data["brain_hyperparams/mean_inner_intrinsic_coef"]) > 0:
            intrinsic_coef = data["brain_hyperparams/mean_inner_intrinsic_coef"][-1][1]
            
        if "brain/imagined_horizon" in data and len(data["brain/imagined_horizon"]) > 0:
            imagined_horizon = int(data["brain/imagined_horizon"][-1][1])
        elif "brain_hyperparams/mean_inner_imagined_horizon" in data and len(data["brain_hyperparams/mean_inner_imagined_horizon"]) > 0:
            imagined_horizon = int(data["brain_hyperparams/mean_inner_imagined_horizon"][-1][1])

    print("\n--- Extracted Hyperparameters ---")
    print(f"  lr: {lr}")
    print(f"  ent_coef: {ent_coef}")
    print(f"  intrinsic_coef: {intrinsic_coef}")
    print(f"  imagined_horizon: {imagined_horizon}")
    print("---------------------------------\n")

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=8,
        num_steps=128,
        seed=0,
        device=str(device),
        mode="dyna",
        lr=lr,
        ent_coef=ent_coef,
    )

    run_name = f"eval_hyperparams_{env_name}"
    print(f"Starting evaluation run: {run_name} on {args.env_id}")
    print(f"Total steps: {args.total_timesteps}, Regimes: {args.num_regimes}, Steps/regime: {args.steps_per_regime}")
    
    from lifelong_learning.agents.ppo.train import init_inner_training, run_inner_update, close_inner_training
    
    state = init_inner_training(
        env_id=args.env_id,
        cfg=cfg,
        steps_per_regime=args.steps_per_regime,
        episodes_per_regime=None,
        start_regime=0,
        num_regimes=args.num_regimes,
        run_name=run_name,
        log_dir="evals",
        save_dir="evals",
        anneal_lr=False,  
        resume_path=None,
        intrinsic_coef=intrinsic_coef,
        imagined_horizon=imagined_horizon,
        intrinsic_reward_clip=0.1,
        wm_lr=1e-4,
    )
    
    while True:
        stats = run_inner_update(state)
        if stats["done"]:
            break
            
    close_inner_training(state)


if __name__ == "__main__":
    main()
