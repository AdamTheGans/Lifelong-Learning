"""
Training script for the Brain meta-agent.

Usage:
    python scripts/train_brain.py --inner_total_timesteps 150000 --brain_episodes 50
"""
from __future__ import annotations

import os
import argparse
import random
import time
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.brain.meta_env import MetaEnv
from lifelong_learning.agents.brain.meta_agent import (
    MLPActorCritic,
    BrainConfig,
    BrainRolloutBuffer,
    brain_ppo_update,
)
from lifelong_learning.utils.logger import DataLogger


def should_run_periodic(episode: int, every: int, *, final_episode: int | None = None) -> bool:
    """Return True when an artifact should be refreshed on this episode."""
    if every <= 0:
        return final_episode is not None and episode == final_episode
    return episode % every == 0 or (final_episode is not None and episode == final_episode)


def launch_high_scale_plots(episode_dir: str):
    """Spawn high-scale plot generation for each inner-env log folder."""
    import glob
    import subprocess
    import sys

    if not os.path.isdir(episode_dir):
        return

    for env_folder in sorted(glob.glob(os.path.join(episode_dir, "ep*_env*"))):
        if not os.path.isdir(env_folder):
            continue
        try:
            print(f"  [plot] Generating charts for {os.path.basename(env_folder)}...")
            subprocess.Popen(
                [sys.executable, "scripts/plot_high_scale.py", "--folder", env_folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"  [plot] Failed to launch high-scale plot for {env_folder}: {e}")


def build_meta_vector_env(env_fns, vectorization: str):
    """Create the Brain's vectorized outer environment."""
    autoreset_mode = gym.vector.AutoresetMode.DISABLED
    if vectorization == "async":
        return gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            autoreset_mode=autoreset_mode,
        )
    if vectorization == "sync":
        return gym.vector.SyncVectorEnv(
            env_fns,
            autoreset_mode=autoreset_mode,
        )
    raise ValueError(f"Unsupported brain vectorization mode: {vectorization}")


def get_meta_env_runtime_cpu_threads(vectorization: str) -> int | None:
    """Limit per-worker CPU thread pools when outer envs run in subprocesses."""
    if vectorization == "async":
        return 1
    return None


def should_refresh_brain_trends(plot_episode_artifacts: bool, generate_high_scale_plots: bool) -> bool:
    """Refresh aggregate Brain plots on schedule or alongside per-episode high-scale plots."""
    return plot_episode_artifacts or generate_high_scale_plots


def capture_brain_rng_state() -> dict:
    """Snapshot Python, NumPy, and Torch RNG state for a resumable Brain checkpoint."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_brain_rng_state(state: dict | None) -> bool:
    """Restore RNG state captured alongside a Brain checkpoint."""
    if not state:
        return False

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    return True


def capture_meta_env_resume_state(meta_env) -> list[dict] | None:
    """Collect per-worker MetaEnv state that should survive a resume."""
    if not hasattr(meta_env, "call"):
        return None
    return meta_env.call("get_resume_state")


def restore_meta_env_resume_state(meta_env, resume_state) -> bool:
    """Push a saved MetaEnv resume payload back into each vectorized worker."""
    if resume_state is None or not hasattr(meta_env, "call"):
        return False
    meta_env.call("load_resume_state", resume_state)
    return True


def build_brain_checkpoint_payload(brain_model, brain_optimizer, meta_env, *, episodes_trained: int, args, episode: int | None = None, avg_reward_10: float | None = None) -> dict:
    """Build a Brain checkpoint payload including resume-sensitive outer state."""
    payload = {
        "model_state_dict": brain_model.state_dict(),
        "optimizer_state_dict": brain_optimizer.state_dict(),
        "episodes_trained": episodes_trained,
        "args": vars(args),
        "rng_state": capture_brain_rng_state(),
        "meta_env_resume_state": capture_meta_env_resume_state(meta_env),
    }
    if episode is not None:
        payload["episode"] = episode
    if avg_reward_10 is not None:
        payload["avg_reward_10"] = avg_reward_10
    return payload

def _extract_episode_average_success_rate(infos: dict) -> float | None:
    """Average final inner-agent success rate across Brain vector envs for one episode."""
    success_rates = []
    final_infos = infos.get("final_info", [])
    if isinstance(final_infos, dict):
        final_infos = [final_infos]
    for final_info in final_infos:
        if not isinstance(final_info, dict):
            continue
        inner_stats = final_info.get("inner_stats")
        if not isinstance(inner_stats, dict):
            continue
        if "overall_success_rate" in inner_stats:
            success_rates.append(float(inner_stats["overall_success_rate"]))
        elif "success_rate" in inner_stats:
            success_rates.append(float(inner_stats["success_rate"]))
    if success_rates:
        return float(np.mean(success_rates))
    inner_stats = infos.get("inner_stats")
    if isinstance(inner_stats, dict):
        if "overall_success_rate" in inner_stats:
            return float(np.mean(np.asarray(inner_stats["overall_success_rate"], dtype=np.float32)))
        if "success_rate" in inner_stats:
            return float(np.mean(np.asarray(inner_stats["success_rate"], dtype=np.float32)))
    if isinstance(inner_stats, (list, tuple)):
        fallback_rates = []
        for stats in inner_stats:
            if not isinstance(stats, dict):
                continue
            if "overall_success_rate" in stats:
                fallback_rates.append(float(stats["overall_success_rate"]))
            elif "success_rate" in stats:
                fallback_rates.append(float(stats["success_rate"]))
        if fallback_rates:
            return float(np.mean(fallback_rates))
    return None

def _upgrade_legacy_brain_state_dict(
    state_dict: dict,
    target_obs_dim: int,
    target_act_dim: int,
) -> tuple[dict, list[str]]:
    """Pad or trim older Brain checkpoints to the current observation/action shapes."""
    upgraded = dict(state_dict)
    notices: list[str] = []

    shared_key = "shared.0.weight"
    if shared_key in upgraded:
        old_obs_dim = upgraded[shared_key].shape[1]
        if old_obs_dim != target_obs_dim:
            old_weight = upgraded[shared_key]
            new_weight = old_weight.new_zeros((old_weight.shape[0], target_obs_dim))
            limit = min(old_obs_dim, target_obs_dim)
            new_weight[:, :limit] = old_weight[:, :limit]
            upgraded[shared_key] = new_weight
            verb = "PADDING" if old_obs_dim < target_obs_dim else "TRUNCATING"
            notices.append(
                f"DETECTED LEGACY {old_obs_dim}-DIM OBS SPACE MODEL. "
                f"{verb} INPUT LAYER TO {target_obs_dim}-DIM..."
            )

    if "actor_mean.weight" not in upgraded:
        return upgraded, notices

    old_act_dim = upgraded["actor_mean.weight"].shape[0]
    logstd_key = "actor_log_std" if "actor_log_std" in upgraded else "actor_logstd"
    needs_action_upgrade = old_act_dim != target_act_dim
    needs_logstd_rename = logstd_key != "actor_log_std"

    if not needs_action_upgrade and not needs_logstd_rename:
        return upgraded, notices

    old_weight = upgraded["actor_mean.weight"]
    old_bias = upgraded["actor_mean.bias"]
    old_logstd = upgraded.get(logstd_key)

    new_weight = old_weight.new_zeros((target_act_dim, old_weight.shape[1]))
    row_limit = min(old_act_dim, target_act_dim)
    new_weight[:row_limit, :] = old_weight[:row_limit, :]
    upgraded["actor_mean.weight"] = new_weight

    new_bias = old_bias.new_zeros(target_act_dim)
    new_bias[:row_limit] = old_bias[:row_limit]
    upgraded["actor_mean.bias"] = new_bias

    if old_logstd is None:
        new_logstd = old_weight.new_full((target_act_dim,), -0.5)
    else:
        old_logstd_flat = old_logstd.reshape(-1)
        new_logstd = old_logstd_flat.new_full((target_act_dim,), -0.5)
        logstd_limit = min(target_act_dim, old_logstd_flat.shape[0])
        new_logstd[:logstd_limit] = old_logstd_flat[:logstd_limit]

    upgraded["actor_log_std"] = new_logstd
    upgraded.pop("actor_logstd", None)

    if needs_action_upgrade:
        verb = "PADDING" if old_act_dim < target_act_dim else "TRUNCATING"
        notices.append(
            f"DETECTED LEGACY {old_act_dim}-DIM ACTION SPACE MODEL. "
            f"{verb} TO {target_act_dim}-DIM..."
        )
    elif needs_logstd_rename:
        notices.append("DETECTED LEGACY BRAIN LOGSTD KEY. RENAMING TO actor_log_std...")

    return upgraded, notices

def train_brain(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Checkpoint Loading & Config Restoration
    # -----------------------------------------------------------------
    start_episode = 1
    checkpoint = None
    resume_dir = None
    if args.resume_path:
        if not getattr(args, "new_run_dir", False):
            abs_path = os.path.abspath(args.resume_path)
            parent_dir = os.path.dirname(abs_path)
            parent_dir_name = os.path.basename(parent_dir)
            if parent_dir_name.startswith("episode_") or parent_dir_name.startswith("pretrain_"):
                resume_dir = os.path.dirname(parent_dir)
            else:
                resume_dir = parent_dir
            print(f"Resuming in existing folder: {resume_dir}")

        print(f"Loading checkpoint configs from: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if "episodes_trained" in checkpoint:
                start_episode = checkpoint.get("episode", checkpoint["episodes_trained"]) + 1
                
            if "args" in checkpoint:
                saved_args = checkpoint["args"]
                
                # Find arguments explicitly passed in the command line
                import sys
                explicit_args = set()
                for arg_str in sys.argv[1:]:
                    if arg_str.startswith("--"):
                        key = arg_str[2:].split("=")[0]
                        # Handle if the user passes dashes instead of underscores
                        explicit_args.add(key.replace("-", "_"))
                
                # Restore saved arguments ONLY IF they were not explicitly passed
                for k, v in saved_args.items():
                    if k not in explicit_args:
                        setattr(args, k, v)
                        
                print(f"Successfully restored environment and inner agent config from checkpoint.")
                print(f"Kept explicit CLI arguments: {list(explicit_args.intersection(saved_args.keys()))}")

    # -----------------------------------------------------------------
    # Inner agent config (passed to MetaEnv)
    # -----------------------------------------------------------------
    inner_cfg = PPOConfig(
        seed=args.seed,
        total_timesteps=args.inner_total_timesteps,
        num_envs=args.inner_num_envs,
        num_steps=args.inner_num_steps,
        mode=args.inner_mode,
        device=device,
        # Default starting values (Brain will override these)
        lr=3e-4,
        ent_coef=0.01,
    )

    # -----------------------------------------------------------------
    # Meta-environment Vectorization
    # -----------------------------------------------------------------
    logger = DataLogger(run_name=args.run_name or "brain_training", full_dir=resume_dir)

    if args.resume_path and not getattr(args, "new_run_dir", False) and resume_dir:
        import glob
        import json
        trends_dir = os.path.join(resume_dir, "brain_trends")
        json_path = os.path.join(trends_dir, f"{logger.run_name}_data.json")
        if not os.path.exists(json_path):
            json_files = glob.glob(os.path.join(trends_dir, "*_data.json"))
            if json_files:
                json_path = json_files[0]
                logger.run_name = os.path.basename(json_path).replace("_data.json", "")
        
        if os.path.exists(json_path):
            print(f"Loading previous logger data: {json_path}")
            try:
                with open(json_path, "r") as f:
                    old_data = json.load(f)
                    from collections import defaultdict
                    new_data = defaultdict(list)
                    for k, v in old_data.items():
                        new_data[k] = [tuple(item) for item in v]
                    logger.data = new_data
            except Exception as e:
                print(f"Warning: failed to load previous logger data: {e}")
    
    # Save all configuration arguments to a txt file in the run directory
    config_path = os.path.join(logger.full_dir, "config.txt")
    with open(config_path, "w") as f:
        f.write("Brain Training Configuration:\n")
        f.write("-" * 40 + "\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
    print(f"Saved configuration to: {config_path}")

    env_fns = [
        partial(
            MetaEnv,
            env_id=args.env_id,
            inner_cfg=inner_cfg,
            decision_interval=args.decision_interval,
            steps_per_regime=args.inner_steps_per_regime,
            start_regime=args.start_regime,
            randomize_start_regime=args.randomize_start_regime,
            num_regimes=args.num_regimes,
            reward_alpha=args.reward_alpha,
            reward_beta=args.reward_beta,
            reward_mode=args.reward_mode,
            anneal_lr=False,
            intrinsic_coef=args.inner_intrinsic_coef,
            imagined_horizon=args.inner_imagined_horizon,
            wm_lr=args.inner_wm_lr,
            inner_log_dir=logger.full_dir,
            env_index=env_idx,
            episodic_memory_capacity=args.episodic_memory_capacity,
            max_inner_lr=args.max_inner_lr,
            min_inner_lr=args.min_inner_lr,
            min_ent_coef=args.min_ent_coef,
            max_ent_coef=args.max_ent_coef,
            min_intrinsic_coef=args.min_intrinsic_coef,
            max_intrinsic_coef=args.max_intrinsic_coef,
            start_episode=start_episode,
            disable_neuromodulation=args.disable_neuromodulation,
            runtime_cpu_threads=get_meta_env_runtime_cpu_threads(args.brain_vectorization),
        )
        for env_idx in range(args.brain_num_envs)
    ]

    meta_env = build_meta_vector_env(env_fns, args.brain_vectorization)

    # -----------------------------------------------------------------
    # Brain agent
    # -----------------------------------------------------------------
    brain_cfg = BrainConfig(
        brain_episodes=args.brain_episodes,
        lr=args.brain_lr,
        ent_coef=args.brain_ent_coef,
        device=args.device,
    )

    brain_model = MLPActorCritic().to(device)
    brain_optimizer = torch.optim.Adam(brain_model.parameters(), lr=brain_cfg.lr, eps=1e-5)

    if checkpoint:
        print("Restoring Brain model and optimizer weights...")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        state_dict, upgrade_messages = _upgrade_legacy_brain_state_dict(
            state_dict,
            brain_model.shared[0].in_features,
            brain_model.actor_mean.out_features,
        )
        is_legacy_checkpoint = bool(upgrade_messages)
        for message in upgrade_messages:
            print(message)
        brain_model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            if is_legacy_checkpoint:
                print("Skipping optimizer state restore for legacy Brain checkpoint.")
            else:
                try:
                    brain_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except (ValueError, RuntimeError) as exc:
                    print(f"Warning: failed to restore optimizer state: {exc}")

        if isinstance(checkpoint, dict):
            if restore_meta_env_resume_state(meta_env, checkpoint.get("meta_env_resume_state")):
                print("Restored MetaEnv resume state.")
            else:
                print("Warning: checkpoint missing MetaEnv resume state; signal normalizers will warm up from scratch.")

            if restore_brain_rng_state(checkpoint.get("rng_state")):
                print("Restored Brain RNG state.")
            else:
                print("Warning: checkpoint missing Brain RNG state; action sampling will resume from a fresh RNG stream.")

        print(f"Resuming at episode {start_episode}")
    print(f"Training Brain on {device} for {brain_cfg.brain_episodes} RL episodes")
    print(f"  Inner: {args.inner_total_timesteps} timesteps, "
          f"regime switch every {args.inner_steps_per_regime} steps")
    print(f"  Parallel Envs: {args.brain_num_envs}")
    print(f"  Brain vectorization: {args.brain_vectorization}")
    worker_cpu_threads = get_meta_env_runtime_cpu_threads(args.brain_vectorization)
    if worker_cpu_threads is not None:
        print(f"  Async worker CPU threads: {worker_cpu_threads}")
    print(f"  Decision interval: {args.decision_interval} inner updates")

    # -----------------------------------------------------------------
    # Imitation Learning: Pretraining
    # -----------------------------------------------------------------
    # Only run pretraining if starting from scratch
    if args.pretrain_episodes > 0 and start_episode == 1:
        print(f"\n--- Starting Imitation Learning Pretraining for {args.pretrain_episodes} episodes ---")
        meta_env.set_attr("_episode_prefix", "pretrain")
        meta_env.set_attr("_episode_counter", 0)
        for pre_ep in range(1, args.pretrain_episodes + 1):
            ep_start = time.time()
            obs, info = meta_env.reset()
            steps = 0
            
            # Since all vectorized inner envs step synchronously and have deterministic max steps,
            # they will all return True for done at the exact same time.
            while True:
                # Observation shape: [num_envs, 19]
                # Index 1 is success_rate (from signals.py)
                success_rates = obs[:, 1]
                
                target_actions = np.zeros((args.brain_num_envs, 15), dtype=np.float32)
                for i in range(args.brain_num_envs):
                    if args.pretrain_mode == "recovery":
                        # Recovery pretrain: multi-tier, surprise-reactive heuristic
                        # success_rate is statically normalized: raw sr * 2.0 - 1.0
                        # so raw_sr = (normed + 1) / 2
                        raw_sr = (success_rates[i] + 1.0) / 2.0
                        # steps_since_surprise_spike at index 14 (running normalized)
                        # Negative = recent spike (below average)
                        spike_signal = obs[i, 14]

                        if spike_signal < -0.5:
                            # Just saw a surprise spike = likely regime switch
                            # Slam into explore mode regardless of success rate
                            target_actions[i] = [0.7, 0.7, 0.7, 0.7, -0.3, 0.8, -0.8, 0,0,0,0,0,0,0,0]
                        elif raw_sr < 0.3:
                            # Deep recovery: strong explore
                            target_actions[i] = [0.5, 0.6, 0.6, 0.5, -0.2, 0.6, -0.6, 0,0,0,0,0,0,0,0]
                        elif raw_sr < 0.6:
                            # Mid recovery: moderate explore
                            target_actions[i] = [0.3, 0.3, 0.4, 0.3, 0.0, 0.3, -0.2, 0,0,0,0,0,0,0,0]
                        elif raw_sr < 0.8:
                            # Almost recovered: start tapering
                            target_actions[i] = [0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.2, 0,0,0,0,0,0,0,0]
                        else:
                            # Recovered: moderate exploit (not extreme)
                            target_actions[i] = [-0.2, -0.3, -0.1, -0.1, 0.3, -0.5, 0.5, 0,0,0,0,0,0,0,0]
                    else:
                        # Basic pretrain: original binary heuristic
                        if success_rates[i] < 0.5:
                            # Explore: map towards higher values (actions > 0)
                            target_actions[i] = [0.8, 0.8, 0.8, 0.8, -0.9, 0.8, -0.8, 0,0,0,0,0,0,0,0]
                        else:
                            # Exploit: map towards lower values (actions < 0)
                            target_actions[i] = [-0.8, -0.8, -0.8, -0.8, 0.8, -0.8, 0.8, 0,0,0,0,0,0,0,0]

                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                target_a_t = torch.tensor(target_actions, dtype=torch.float32, device=device)

                # MSE Loss for pretraining
                action_mean, value = brain_model.forward(obs_t)
                loss = F.mse_loss(action_mean, target_a_t)

                brain_optimizer.zero_grad()
                loss.backward()
                brain_optimizer.step()

                next_obs, rewards, terminations, truncations, infos = meta_env.step(target_actions)
                obs = next_obs
                steps += 1

                if terminations.any() or truncations.any():
                    break
                    
            ep_time = time.time() - ep_start
            print(f"Pretrain Episode {pre_ep}/{args.pretrain_episodes} | steps={steps} | time={ep_time:.1f}s | last_loss={loss.item():.4f}")

            if args.generate_high_scale_plots:
                launch_high_scale_plots(os.path.join(logger.full_dir, f"pretrain_{pre_ep}"))

    # -----------------------------------------------------------------
    # Training loop: RL
    # -----------------------------------------------------------------
    target_episodes = brain_cfg.brain_episodes
    if start_episode > target_episodes:
        print(f"Note: Checkpoint is already at episode {start_episode - 1}.")
        print(f"Adding {args.brain_episodes} additional episodes to target.")
        target_episodes = (start_episode - 1) + args.brain_episodes
        brain_cfg.brain_episodes = target_episodes

    print(f"\n--- Starting Meta-RL PPO Training (Target: {target_episodes}) ---")
    all_episode_rewards = []
    if "brain/episode_reward" in logger.data:
        all_episode_rewards = [val for step, val in logger.data["brain/episode_reward"]]
    
    # Restore the prefix and starting episode counter count since pretraining modified it
    # AND ensure that when resuming from a checkpoint, the environment uses the correct episode number.
    if args.pretrain_episodes > 0 and start_episode == 1:
        meta_env.set_attr("_episode_prefix", "episode")
        meta_env.set_attr("_episode_counter", start_episode - 1)
    elif start_episode > 1:
        meta_env.set_attr("_episode_prefix", "episode")
        meta_env.set_attr("_episode_counter", start_episode - 1)

    for episode in range(start_episode, target_episodes + 1):
        ep_start = time.time()
        rollout = BrainRolloutBuffer()
        total_rewards = np.zeros(args.brain_num_envs, dtype=np.float32)
        steps = 0
        
        episode_lrs = []
        episode_ent_coefs = []
        episode_intrinsic_coefs = []
        episode_horizons = []
        
        episode_action_lrs = []
        episode_action_ents = []
        episode_action_intrs = []
        episode_action_horizons = []
        episode_action_replays = []
        episode_replay_ratios = []

        obs, info = meta_env.reset()

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                action, log_prob, entropy, value = brain_model.act(obs_t)

            action_np = action.cpu().numpy()  # Shape: [num_envs, 15]
            next_obs, rewards, terminations, truncations, infos = meta_env.step(action_np)
            
            # Track inner hyperparameter stats from envs
            if "inner_stats" in infos:
                inner_stats = infos["inner_stats"]
                if isinstance(inner_stats, dict) and "current_lr" in inner_stats:
                    # SyncVectorEnv batches dicts so that value is an array
                    episode_lrs.append(float(np.mean(inner_stats.get("current_lr", 0.0))))
                    episode_ent_coefs.append(float(np.mean(inner_stats.get("current_ent_coef", 0.0))))
                    episode_intrinsic_coefs.append(float(np.mean(inner_stats.get("current_intrinsic_coef", 0.0))))
                    episode_horizons.append(float(np.mean(inner_stats.get("current_imagined_horizon", 0.0))))
                    if "current_replay_ratio" in inner_stats:
                        episode_replay_ratios.append(float(np.mean(inner_stats.get("current_replay_ratio", 0.0))))
                elif isinstance(inner_stats, (list, tuple)):
                    lrs = [s.get("current_lr", 0.0) for s in inner_stats if isinstance(s, dict) and "current_lr" in s]
                    if lrs: episode_lrs.append(float(np.mean(lrs)))
                    ents = [s.get("current_ent_coef", 0.0) for s in inner_stats if isinstance(s, dict) and "current_ent_coef" in s]
                    if ents: episode_ent_coefs.append(float(np.mean(ents)))
                    intrs = [s.get("current_intrinsic_coef", 0.0) for s in inner_stats if isinstance(s, dict) and "current_intrinsic_coef" in s]
                    if intrs: episode_intrinsic_coefs.append(float(np.mean(intrs)))
                    horz = [s.get("current_imagined_horizon", 0.0) for s in inner_stats if isinstance(s, dict) and "current_imagined_horizon" in s]
                    if horz: episode_horizons.append(float(np.mean(horz)))
                    rr = [s.get("current_replay_ratio", 0.0) for s in inner_stats if isinstance(s, dict) and "current_replay_ratio" in s]
                    if rr: episode_replay_ratios.append(float(np.mean(rr)))
                
            # Track average Brain action taken
            episode_action_lrs.append(float(np.mean(action_np[:, 0])))
            episode_action_ents.append(float(np.mean(action_np[:, 1])))
            episode_action_intrs.append(float(np.mean(action_np[:, 2])))
            episode_action_horizons.append(float(np.mean(action_np[:, 3])))
            episode_action_replays.append(float(np.mean(action_np[:, 4])))

            # Store batched experience
            dones = (terminations | truncations).astype(np.float32)
            rollout.add(
                obs=obs,
                action=action_np,
                log_prob=log_prob.cpu().numpy(),
                reward=rewards,
                value=value.cpu().numpy(),
                done=dones,
            )

            total_rewards += rewards
            steps += 1
            obs = next_obs

            # All inner runs share the exact same lifespan configured by inner_total_timesteps
            if dones.any():
                break

        # Compute advantages
        # Since episode just ended, next value is exactly 0
        rollout.compute_returns_and_advantages(
            last_values=np.zeros(args.brain_num_envs, dtype=np.float32),
            gamma=brain_cfg.gamma,
            gae_lambda=brain_cfg.gae_lambda,
        )

        # PPO update on this episode's data
        batch = rollout.get_batches(device)
        update_stats = brain_ppo_update(brain_model, brain_optimizer, batch, brain_cfg)

        # Log Mean across vector environments
        ep_time = time.time() - ep_start
        mean_reward_across_envs = float(np.mean(total_rewards))
        all_episode_rewards.append(mean_reward_across_envs)
        avg_reward_10 = float(np.mean(all_episode_rewards[-10:]))

        logger.scalar("brain/episode_reward", mean_reward_across_envs, episode)
        logger.scalar("brain/episode_steps", steps, episode)
        logger.scalar("brain/episode_time_s", ep_time, episode)
        logger.scalar("brain/avg_reward_10ep", avg_reward_10, episode)

        for k, v in update_stats.items():
            logger.scalar(k, v, episode)

        # Log mean episodic action choices
        if episode_action_lrs:
            logger.scalar("brain_action/mean_lr_adjustment", float(np.mean(episode_action_lrs)), episode)
            logger.scalar("brain_action/mean_ent_adjustment", float(np.mean(episode_action_ents)), episode)
            logger.scalar("brain_action/mean_intrinsic_adjustment", float(np.mean(episode_action_intrs)), episode)
            logger.scalar("brain_action/mean_horizon_adjustment", float(np.mean(episode_action_horizons)), episode)
            logger.scalar("brain_action/mean_replay_adjustment", float(np.mean(episode_action_replays)), episode)

        # Log mean inner hyperparameters actually realized during the episode
        if episode_lrs:
            logger.scalar("brain_hyperparams/mean_inner_lr", float(np.mean(episode_lrs)), episode)
        if episode_ent_coefs:
            logger.scalar("brain_hyperparams/mean_inner_ent_coef", float(np.mean(episode_ent_coefs)), episode)
        if episode_intrinsic_coefs:
            logger.scalar("brain_hyperparams/mean_inner_intrinsic_coef", float(np.mean(episode_intrinsic_coefs)), episode)
        if episode_horizons:
            logger.scalar("brain_hyperparams/mean_inner_imagined_horizon", float(np.mean(episode_horizons)), episode)
        if episode_replay_ratios:
            logger.scalar("brain_hyperparams/mean_inner_replay_ratio", float(np.mean(episode_replay_ratios)), episode)

        # Preserve the legacy single-env metric for continuity, and add the per-episode mean across envs.
        final_info = infos.get("final_info", [{}])[0]
        if final_info and "inner_stats" in final_info:
            inner_stats = final_info["inner_stats"]
            logger.scalar("brain/inner_final_success_rate", inner_stats.get("overall_success_rate", inner_stats.get("success_rate", 0.0)), episode)
        episode_avg_success_rate = _extract_episode_average_success_rate(infos)
        if episode_avg_success_rate is not None:
            logger.scalar("brain/episode_avg_success_rate", episode_avg_success_rate, episode)
        episode_summary = (
            f"Episode {episode}/{brain_cfg.brain_episodes} | "
            f"reward={mean_reward_across_envs:.4f} | avg10={avg_reward_10:.4f}"
        )
        if episode_avg_success_rate is not None:
            episode_summary += f" | avg_success_rate={episode_avg_success_rate:.4f}"
        episode_summary += f" | steps={steps} | time={ep_time:.1f}s"
        print(episode_summary)

        save_episode_artifacts = should_run_periodic(
            episode,
            args.save_every_episodes,
            final_episode=target_episodes,
        )
        plot_episode_artifacts = should_run_periodic(
            episode,
            args.plot_every_episodes,
            final_episode=target_episodes,
        )

        if save_episode_artifacts:
            ckpt_dir = os.path.join(logger.full_dir, f"episode_{episode}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"brain_ep{episode}.pt")
            torch.save(
                build_brain_checkpoint_payload(
                    brain_model,
                    brain_optimizer,
                    meta_env,
                    episode=episode,
                    avg_reward_10=avg_reward_10,
                    episodes_trained=episode,
                    args=args,
                ),
                ckpt_path,
            )
            print(f"  [brain ckpt] {ckpt_path}")

        refresh_brain_trends = should_refresh_brain_trends(
            plot_episode_artifacts,
            args.generate_high_scale_plots,
        )
        if refresh_brain_trends:
            plot_dir = os.path.join(logger.full_dir, "brain_trends")
            logger.plot(save_dir=plot_dir, title="Brain Overall Trends")

        if args.generate_high_scale_plots:
            launch_high_scale_plots(os.path.join(logger.full_dir, f"episode_{episode}"))
              
    # Generate final overall Brain trend charts
    plot_dir = os.path.join(logger.full_dir, "brain_trends")
    logger.plot(save_dir=plot_dir, title="Brain Overall Trends")

    # Save Brain model checkpoint for evaluation
    brain_save_path = os.path.join(logger.full_dir, "brain_model.pt")
    torch.save(
        build_brain_checkpoint_payload(
            brain_model,
            brain_optimizer,
            meta_env,
            episodes_trained=brain_cfg.brain_episodes,
            args=args,
        ),
        brain_save_path,
    )
    print(f"Saved Brain model to: {brain_save_path}")

    meta_env.close()
    logger.close()
    print(f"\nBrain training complete. Total RL episodes: {brain_cfg.brain_episodes}")


def main():
    p = argparse.ArgumentParser(description="Train the Brain meta-agent")

    # Inner agent settings
    p.add_argument("--env_id", type=str, default="MiniGrid-MultiGoal-5x5-v0")
    p.add_argument("--inner_total_timesteps", type=int, default=1150000)
    p.add_argument("--inner_num_envs", type=int, default=8)
    p.add_argument("--inner_num_steps", type=int, default=128)
    p.add_argument("--num_regimes", type=int, default=2)
    p.add_argument("--start_regime", type=int, default=0,
                   help="Fixed starting regime for each inner run unless randomization is enabled")
    p.add_argument("--randomize_start_regime", action="store_true",
                   help="Sample a fresh starting regime on each MetaEnv reset")
    p.add_argument("--inner_steps_per_regime", type=int, default=296000)
    p.add_argument("--inner_mode", type=str, default="dyna", choices=["dyna", "passive"])
    p.add_argument("--inner_intrinsic_coef", type=float, default=0.015)
    p.add_argument("--inner_imagined_horizon", type=int, default=10)
    p.add_argument("--inner_wm_lr", type=float, default=1e-4)
    p.add_argument("--max_inner_lr", type=float, default=0.003, help="Maximum absolute bound for the inner agent's learning rate")
    p.add_argument("--min_inner_lr", type=float, default=1e-4, help="Minimum absolute bound for the inner agent's learning rate")
    p.add_argument("--min_ent_coef", type=float, default=0.001, help="Minimum bound for inner entropy coefficient")
    p.add_argument("--max_ent_coef", type=float, default=0.1, help="Maximum bound for inner entropy coefficient")
    p.add_argument("--min_intrinsic_coef", type=float, default=0.001, help="Minimum bound for inner intrinsic curiosity coefficient")
    p.add_argument("--max_intrinsic_coef", type=float, default=0.5, help="Maximum bound for inner intrinsic curiosity coefficient")

    # Brain meta-agent settings
    p.add_argument("--brain_num_envs", type=int, default=16, help="Number of parallel MetaEnvs run simultaneously")
    p.add_argument("--brain_vectorization", type=str, default="async", choices=["async", "sync"],
                   help="Outer Brain env batching mode; async uses subprocesses for real parallel inner runs")
    p.add_argument("--pretrain_episodes", type=int, default=5, help="Number of Imitation Learning pretrain episodes")
    p.add_argument("--pretrain_mode", type=str, default="recovery", choices=["basic", "recovery"],
                   help="Pretrain heuristic: 'basic' (binary explore/exploit) or 'recovery' (multi-tier, surprise-reactive)")
    p.add_argument("--brain_episodes", type=int, default=50)
    p.add_argument("--brain_lr", type=float, default=1e-4)
    p.add_argument("--brain_ent_coef", type=float, default=0.0,
                   help="Entropy coefficient for the Brain to encourage exploration")
    p.add_argument("--decision_interval", type=int, default=10,
                   help="Number of inner PPO updates per Brain decision")

    # Reward shaping
    p.add_argument("--reward_alpha", type=float, default=0.1)
    p.add_argument("--reward_beta", type=float, default=0.5)
    p.add_argument("--reward_mode", type=str, default="auc", choices=["auc", "recovery", "curriculum"],
                   help="Brain reward mode: 'auc' (original), 'recovery' (hybrid delta + urgency), or 'curriculum' (exponential multiplier for returning regimes)")
    p.add_argument("--disable_neuromodulation", action="store_true",
                   help="Disable neuromodulation gating (Brain still outputs 15 dims but context code is ignored)")

    # Episodic Memory
    p.add_argument("--episodic_memory_capacity", type=int, default=50000,
                   help="Capacity of the episodic memory ring buffer")

    # General
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume_path", type=str, default=None, help="Path to brain checkpoint.pt to resume from")
    p.add_argument("--new_run_dir", action="store_true", help="If resuming, create a new run folder instead of continuing in the same folder")
    p.add_argument("--save_every_episodes", type=int, default=1,
                   help="Write per-episode Brain checkpoints every N episodes; 0 keeps only the final episode checkpoint")
    p.add_argument("--plot_every_episodes", type=int, default=5,
                   help="Refresh aggregate Brain plots every N episodes; 0 plots only at the end")
    p.add_argument("--generate_high_scale_plots", action="store_true",
                   help="Spawn plot_high_scale.py for inner env logs after every pretrain and RL episode")

    args = p.parse_args()
    train_brain(args)


if __name__ == "__main__":
    main()
