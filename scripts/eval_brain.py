"""
Evaluate a trained Brain model on a fresh inner Dyna-PPO training run.

The Brain controls hyperparameters in inference mode (no gradient updates)
while the inner agent trains from scratch.

Usage:
    python scripts/eval_brain.py \
        --brain_checkpoint runs/brain_episodic_run_5_.../brain_model.pt \
        --total_timesteps 1150000 \
        --steps_per_regime 18500 \
        --run_name eval_brain_run5
"""
from __future__ import annotations

import os
import argparse
import time
import json
import numpy as np
import torch

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    init_inner_training,
    run_inner_update,
    close_inner_training,
)
from lifelong_learning.agents.brain.signals import SignalExtractor, NUM_SIGNALS
from lifelong_learning.agents.brain.meta_agent import MLPActorCritic
from lifelong_learning.utils.logger import DataLogger


def eval_brain(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Load trained Brain model
    # -----------------------------------------------------------------
    print(f"Loading Brain model from: {args.brain_checkpoint}")
    ckpt = torch.load(args.brain_checkpoint, map_location=device)
    
    brain_model = MLPActorCritic().to(device)
    
    # Handle backward compatibility: older checkpoints may have 5 or 7 action dims, we need 15
    state_dict = ckpt["model_state_dict"]
    
    old_act_dim = state_dict["actor_mean.weight"].shape[0] if "actor_mean.weight" in state_dict else 15
    if old_act_dim < 15:
        print(f"DETECTED LEGACY {old_act_dim}-DIM ACTION SPACE MODEL. PADDING TO 15-DIM...")
        old_weight = state_dict["actor_mean.weight"]
        old_bias = state_dict["actor_mean.bias"]
        old_logstd = state_dict["actor_logstd"]
        
        new_weight = torch.zeros(15, old_weight.shape[1], device=old_weight.device)
        new_weight[:old_act_dim, :] = old_weight
        state_dict["actor_mean.weight"] = new_weight
        
        new_bias = torch.zeros(15, device=old_bias.device)
        new_bias[:old_act_dim] = old_bias
        state_dict["actor_mean.bias"] = new_bias
        
        logstd_dim = old_logstd.shape[-1] if old_logstd.dim() > 1 else old_logstd.shape[0]
        new_logstd = torch.zeros(1, 15, device=old_logstd.device) - 0.5
        new_logstd[0, :logstd_dim] = old_logstd.view(-1)[:logstd_dim]
        state_dict["actor_logstd"] = new_logstd

    brain_model.load_state_dict(state_dict)
    brain_model.eval()  # Inference mode — no dropout, batchnorm etc.

    train_args = ckpt.get("args", {})
    episodes_trained = ckpt.get("episodes_trained", "?")
    print(f"  Brain was trained for {episodes_trained} episodes")
    print(f"  Original training config: {json.dumps(train_args, indent=2, default=str)}")

    # -----------------------------------------------------------------
    # Inner agent config
    # -----------------------------------------------------------------
    inner_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        mode=args.mode,
        device=device,
        lr=3e-4,
        ent_coef=0.01,
    )

    # -----------------------------------------------------------------
    # Logger
    # -----------------------------------------------------------------
    logger = DataLogger(run_name=args.run_name or "eval_brain", log_dir="evals")

    # Save eval config
    config_path = os.path.join(logger.full_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Brain Evaluation Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"brain_checkpoint: {args.brain_checkpoint}\n")
        f.write(f"brain_episodes_trained: {episodes_trained}\n")
        f.write("-" * 40 + "\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
    print(f"Saved eval config to: {config_path}")

    # -----------------------------------------------------------------
    # Initialize inner agent
    # -----------------------------------------------------------------
    state = init_inner_training(
        env_id=args.env_id,
        cfg=inner_cfg,
        steps_per_regime=args.steps_per_regime,
        start_regime=0,
        num_regimes=args.num_regimes,
        run_name=args.run_name or "eval_brain",
        save_every_updates=args.save_every_updates,
        anneal_lr=False,  # Brain controls LR
        intrinsic_coef=args.intrinsic_coef,
        imagined_horizon=args.imagined_horizon,
        wm_lr=args.wm_lr,
        log_dir=logger.full_dir,
        episodic_memory_capacity=args.episodic_memory_capacity,
    )

    sig = SignalExtractor(
        max_inner_lr=args.max_inner_lr,
        min_inner_lr=args.min_inner_lr,
        min_ent_coef=args.min_ent_coef,
        max_ent_coef=args.max_ent_coef,
        min_intrinsic_coef=args.min_intrinsic_coef,
        max_intrinsic_coef=args.max_intrinsic_coef,
    )

    # HP bounds (same as MetaEnv)
    lr_bounds = (args.min_inner_lr, args.max_inner_lr)
    ent_coef_bounds = (args.min_ent_coef, args.max_ent_coef)
    intrinsic_coef_bounds = (args.min_intrinsic_coef, args.max_intrinsic_coef)
    imagined_horizon_bounds = (1, 30)
    replay_ratio_bounds = (0.0, 0.5)
    replay_prioritization_bounds = (0.0, 1.0)
    anchoring_weight_bounds = (0.0, 0.5)

    print(f"\nRunning Brain-controlled evaluation:")
    print(f"  Environment: {args.env_id}")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Regime switch every: {args.steps_per_regime} steps")
    print(f"  Decision interval: {args.decision_interval} inner updates")
    print(f"  Mode: {args.mode}")

    # -----------------------------------------------------------------
    # Main loop: run inner updates with Brain inference
    # -----------------------------------------------------------------
    start_time = time.time()
    update_count = 0
    prev_success_rate = 0.0

    while True:
        stats = run_inner_update(state)

        if stats.get("done"):
            break

        update_count += 1

        # Every decision_interval updates, let the Brain adjust hyperparams
        if update_count % args.decision_interval == 0:
            # Extract observation signals
            obs = sig.extract(stats)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Brain inference (no gradients)
            with torch.no_grad():
                action_mean, value = brain_model.forward(obs_t)
                action = action_mean.squeeze(0).cpu().numpy()

            # Helper to map [-1, 1] to [min_val, max_val]
            def map_to_range(a_val: float, bounds: tuple[float, float]) -> float:
                return float(bounds[0] + (a_val + 1.0) / 2.0 * (bounds[1] - bounds[0]))

            # Action[0]: lr scale
            new_lr = map_to_range(float(action[0]), lr_bounds)
            state.optimizer.param_groups[0]["lr"] = new_lr

            # Action[1]: entropy coefficient
            new_ent = map_to_range(float(action[1]), ent_coef_bounds)
            state.cfg.ent_coef = new_ent

            # Action[2]: intrinsic curiosity coefficient
            new_ic = map_to_range(float(action[2]), intrinsic_coef_bounds)
            state.intrinsic_coef = new_ic

            # Action[3]: imagined horizon
            horizon_float = map_to_range(float(action[3]), imagined_horizon_bounds)
            new_ih = int(np.clip(round(horizon_float), *imagined_horizon_bounds))
            state.imagined_horizon = new_ih

            # Action[4]: replay ratio
            new_rr = map_to_range(float(action[4]), replay_ratio_bounds)
            state.replay_ratio = new_rr

            # Backward compatibility for models trained before levers 5 and 6 were added
            if len(action) > 5:
                # Action[5]: replay prioritization
                new_rp = map_to_range(float(action[5]), replay_prioritization_bounds)
                state.replay_prioritization = new_rp

                # Action[6]: anchoring weight
                new_aw = map_to_range(float(action[6]), anchoring_weight_bounds)
                state.cfg.anchoring_weight = new_aw
            else:
                new_rp = 0.0
                new_aw = 0.0
                state.replay_prioritization = new_rp
                state.cfg.anchoring_weight = new_aw

            # Action[7:15]: neuromodulation context code
            if len(action) > 7:
                context_code = torch.tensor(action[7:15], dtype=torch.float32, device=device)
                state.model.set_context_code(context_code)

            # Log Brain decisions
            success_rate = stats.get("success_rate", 0.0)
            state.logger.scalar("brain/lr", new_lr, state.global_step)
            state.logger.scalar("brain/ent_coef", new_ent, state.global_step)
            state.logger.scalar("brain/intrinsic_coef", new_ic, state.global_step)
            state.logger.scalar("brain/imagined_horizon", new_ih, state.global_step)
            state.logger.scalar("brain/replay_ratio", new_rr, state.global_step)
            state.logger.scalar("brain/replay_prioritization", new_rp, state.global_step)
            state.logger.scalar("brain/anchoring_weight", new_aw, state.global_step)
            state.logger.scalar("brain/value_estimate", value.item(), state.global_step)

    elapsed = time.time() - start_time
    print(f"Evaluation complete in {elapsed:.1f}s ({update_count} updates)")

    # Plot results
    plot_dir = state.logger.full_dir
    state.logger.plot(save_dir=plot_dir, title=f"Brain Eval: {args.run_name}")
    print(f"Charts saved to: {plot_dir}")
    
    # Generate extremely high scale specific plotting script automatically
    try:
        import subprocess
        import sys
        
        plot_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "plot_high_scale.py")
        print(f"Generating high resolution charts via {plot_script}...")
        
        subprocess.run([sys.executable, plot_script, "--folder", plot_dir], check=True)
        print("High resolution charts generated successfully.")
    except Exception as e:
        print(f"Failed to generate high-res plots automatically: {e}")

    close_inner_training(state)
    logger.close()


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Brain on a fresh inner agent")

    # Brain checkpoint
    p.add_argument("--brain_checkpoint", type=str, required=True,
                   help="Path to brain_model.pt from a training run")

    # Environment
    p.add_argument("--env_id", type=str, default="MiniGrid-MultiGoal-5x5-v0")
    p.add_argument("--total_timesteps", type=int, default=1150000)
    p.add_argument("--steps_per_regime", type=int, default=296000)
    p.add_argument("--num_regimes", type=int, default=2)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--mode", type=str, default="dyna", choices=["dyna", "passive"])

    # Inner agent defaults
    p.add_argument("--intrinsic_coef", type=float, default=0.015)
    p.add_argument("--imagined_horizon", type=int, default=10)
    p.add_argument("--wm_lr", type=float, default=1e-4)
    p.add_argument("--episodic_memory_capacity", type=int, default=50000)
    p.add_argument("--max_inner_lr", type=float, default=0.003, help="Maximum absolute bound for the inner agent's learning rate")
    p.add_argument("--min_inner_lr", type=float, default=1e-4, help="Minimum absolute bound for the inner agent's learning rate")
    p.add_argument("--min_ent_coef", type=float, default=0.001, help="Minimum bound for inner entropy coefficient")
    p.add_argument("--max_ent_coef", type=float, default=0.1, help="Maximum bound for inner entropy coefficient")
    p.add_argument("--min_intrinsic_coef", type=float, default=0.001, help="Minimum bound for inner intrinsic curiosity coefficient")
    p.add_argument("--max_intrinsic_coef", type=float, default=0.5, help="Maximum bound for inner intrinsic curiosity coefficient")

    # Brain settings
    p.add_argument("--decision_interval", type=int, default=10)

    # General
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_every_updates", type=int, default=9999)

    args = p.parse_args()
    eval_brain(args)


if __name__ == "__main__":
    main()
