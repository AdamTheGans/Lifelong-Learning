from __future__ import annotations

import os
import warnings

# [FIX] Silence TensorFlow OneDNN warning (must be before torch/tensorflow imports)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# [FIX] Silence pkg_resources deprecation warning from pygame
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import time
import numpy as np
import torch
import gymnasium as gym
from collections import deque

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel
from lifelong_learning.agents.ppo.ewc import EWC

from lifelong_learning.agents.ppo.buffers import RolloutBuffer
from lifelong_learning.utils.seeding import seed_everything
from lifelong_learning.utils.logger import TBLogger
from lifelong_learning.envs.make_env import make_env


def train_ppo(
    env_id: str,
    cfg: PPOConfig,
    *,
    steps_per_regime: int | None = None,
    episodes_per_regime: int | None = None,
    start_regime: int = 0,
    run_name: str | None = None,
    save_dir: str = "checkpoints",
    save_every_updates: int = 50,
    anneal_lr: bool = True,
    resume_path: str | None = None,
    intrinsic_coef: float = 0.1,
    intrinsic_reward_clip: float = 0.1,
    imagined_horizon: int = 5,
    wm_lr: float = 1e-4,
    surprise_threshold: float = 0.1,
    max_heads: int = 4,
    ewc_lambda: float = 1000.0,
):
    """
    Main Dyna-PPO training loop.

    Each update cycle has three phases:
        A) Collect real experience (with intrinsic curiosity reward)
        B) Train World Model on real transitions (supervised)
        C) Generate imagined trajectories and update policy on dreams
    """

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_envs = max(cfg.num_envs, 16)

    # -------------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------------

    def make_thunk(i: int):
        def thunk():
            return make_env(
                env_id=env_id,
                seed=cfg.seed + i,
                steps_per_regime=steps_per_regime,
                episodes_per_regime=episodes_per_regime,
                start_regime=start_regime,
                record_stats=False,
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(num_envs)])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    # -------------------------------------------------------------------------
    # Model & Optimizer Setup
    # -------------------------------------------------------------------------

    # Shared-trunk multi-head network (shared CNN, per-regime actor/critic heads)
    model = CNNActorCritic(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    ewc = EWC(model, device, param_prefix="encoder.")

    world_model = SimpleWorldModel(obs_shape, n_actions).to(device)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_lr)
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)

    # -------------------------------------------------------------------------
    # Resume from Checkpoint
    # -------------------------------------------------------------------------

    start_global_step = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Optimizer state loaded.")
        else:
            print("WARNING: Optimizer state not found in checkpoint.")

        if "world_model_state_dict" in ckpt:
            world_model.load_state_dict(ckpt["world_model_state_dict"])
            print("World Model state loaded.")
        else:
            print("WARNING: World Model state not found in checkpoint.")

        if "wm_optimizer_state_dict" in ckpt:
            wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
            print("World Model Optimizer state loaded.")

        if "global_step" in ckpt:
            start_global_step = ckpt["global_step"]
            print(f"Resuming from global_step={start_global_step}")

    # -------------------------------------------------------------------------
    # Logging & Tracking
    # -------------------------------------------------------------------------

    if run_name is None:
        run_name = f"ppo_{env_id}_s{cfg.seed}"

    logger = TBLogger(run_name=run_name)
    os.makedirs(save_dir, exist_ok=True)

    obs, info = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = cfg.total_timesteps // (num_envs * cfg.num_steps)
    global_step = start_global_step
    start_update = global_step // (num_envs * cfg.num_steps) + 1
    start_time = time.time()

    # Manual episode stats (per-env accumulators)
    running_returns = np.zeros(num_envs)
    running_lengths = np.zeros(num_envs, dtype=int)

    # Rolling window for outcome tracking (good goal / bad goal / timeout)
    outcome_window = deque(maxlen=100)

    print(f"Training on {device} with {num_envs} envs for {num_updates} updates (starting from update {start_update}).")

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def collect_real_experience(global_step):
        """Phase A: Interact with real environments and collect transitions."""
        nonlocal obs_t, running_returns, running_lengths

        episodic_intrinsic_rewards = []
        episodic_intrinsic_rewards_max = []

        for t in range(cfg.num_steps):
            global_step += num_envs

            with torch.no_grad():
                action, logprob, entropy, value = model.act(obs_t)
                pred_next_obs, pred_reward = world_model(obs_t, action)

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Handle autoreset: use final_observation for surprise calc on done envs
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                # Use _final_observation mask if available, or assume all done envs have it
                final_obs_mask = infos.get("_final_observation", done)
                for i, is_final in enumerate(final_obs_mask):
                    if is_final and i < len(infos["final_observation"]):
                         real_next_obs[i] = infos["final_observation"][i]
            
            real_next_obs_t = torch.tensor(real_next_obs, dtype=torch.float32, device=device)

            # Compute intrinsic reward (surprise signal)
            with torch.no_grad():
                # State surprise: CrossEntropy between predicted and actual one-hot
                target_indices = torch.argmax(real_next_obs_t, dim=1)  # (B, H, W)
                state_loss = torch.nn.functional.cross_entropy(
                    pred_next_obs, target_indices, reduction='none'
                )  # (B, H, W)
                state_surprise = state_loss.mean(dim=[1, 2])  # (B,)

                logger.scalar("debug/raw_cross_entropy_loss", state_surprise.mean().item(), global_step)

                # Reward surprise: MSE between predicted and actual reward
                real_reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
                reward_surprise = torch.nn.functional.mse_loss(
                    pred_reward, real_reward_t, reduction='none'
                )  # (B,)

                # Combined surprise → intrinsic reward (zero in passive mode)
                if cfg.mode == "passive":
                    total_surprise = torch.zeros_like(state_surprise)
                else:
                    total_surprise = state_surprise + reward_surprise

                # Pure proportional: scale by coefficient and clip
                raw_intrinsic = total_surprise * intrinsic_coef
                intrinsic_reward = torch.clamp(raw_intrinsic, 0.0, intrinsic_reward_clip)

                # Debug logging
                logger.scalar("debug/wm_raw_error_mean", total_surprise.mean().item(), global_step)
                logger.scalar("debug/intrinsic_reward_raw", raw_intrinsic.mean().item(), global_step)

                episodic_intrinsic_rewards.append(intrinsic_reward.mean().item())
                episodic_intrinsic_rewards_max.append(intrinsic_reward.max().item())

            # Update per-env episode trackers
            running_returns += reward
            running_lengths += 1

            # Store transition (extrinsic + intrinsic reward)
            total_reward = torch.tensor(reward, dtype=torch.float32, device=device) + intrinsic_reward

            buffer.add(
                obs=obs_t,
                actions=action,
                logprobs=logprob,
                rewards=total_reward,
                dones=torch.tensor(done, dtype=torch.float32, device=device),
                values=value,
                next_obs=real_next_obs_t,
            )

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # Log finished episodes
            if np.any(done):
                done_indices = np.where(done)[0]
                for i in done_indices:
                    logger.scalar("charts/episodic_return", running_returns[i], global_step)
                    logger.scalar("charts/episodic_length", running_lengths[i], global_step)

                    # Extract Goal Outcomes
                    outcome = 0  # timeout/other
                    
                    # Safe access: default to 0.0 if key missing
                    reached_good = infos["reached_good_goal"][i] if "reached_good_goal" in infos else 0.0
                    reached_bad = infos["reached_bad_goal"][i] if "reached_bad_goal" in infos else 0.0
                    # timed_out check can be inferred or explicit if key exists
                    
                    if reached_good > 0:
                        outcome = 1  # success
                    elif reached_bad > 0:
                        outcome = -1  # failure
                    
                    outcome_window.append(outcome)

                    if len(outcome_window) > 0:
                        success_rate = sum(1 for x in outcome_window if x == 1) / len(outcome_window)
                        failure_rate = sum(1 for x in outcome_window if x == -1) / len(outcome_window)
                        timeout_rate = sum(1 for x in outcome_window if x == 0) / len(outcome_window)

                        logger.scalar("charts/success_rate", success_rate, global_step)
                        logger.scalar("charts/failure_rate", failure_rate, global_step)
                        logger.scalar("charts/timeout_rate", timeout_rate, global_step)

                    # Regime tracking
                    if "regime_id" in infos:
                        regime = infos["regime_id"][i]
                        logger.scalar(f"charts/r_regime_{regime}", running_returns[i], global_step)

                    running_returns[i] = 0
                    running_lengths[i] = 0

        return {
            "episodic_intrinsic_rewards": episodic_intrinsic_rewards,
            "episodic_intrinsic_rewards_max": episodic_intrinsic_rewards_max,
            "global_step": global_step,
            "buffer_rewards_mean": buffer.rewards.mean().item(),
            "buffer_rewards_max": buffer.rewards.max().item(),
            "buffer_rewards_std": buffer.rewards.std().item(),
            "buffer_rewards_abs_mean": buffer.rewards.abs().mean().item(),
        }

    def update_world_model():
        """Phase B: Supervised training on real transitions."""
        wm_stats = []
        for epoch in range(cfg.update_epochs):
            minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
            for obs, actions, _, _, _, _, next_obs, rewards in minibatches:
                pred_next_obs, pred_reward = world_model(obs, actions)

                # State loss: CrossEntropy on one-hot → class indices
                target_indices = torch.argmax(next_obs, dim=1)
                loss_state = torch.nn.functional.cross_entropy(pred_next_obs, target_indices)

                # Reward loss: MSE
                loss_reward = torch.nn.functional.mse_loss(pred_reward, rewards)

                wm_loss = loss_state + loss_reward

                wm_optimizer.zero_grad()
                wm_loss.backward()
                wm_optimizer.step()

                wm_stats.append({
                    "world_model/loss_total": wm_loss.item(),
                    "world_model/loss_state": loss_state.item(),
                    "world_model/loss_reward": loss_reward.item(),
                })
        return wm_stats

    def generate_dream_experience():
        """Phase C: Generate imagined trajectories using WM as simulator."""
        # Sample random start states from the real buffer
        rand_time_idxs = torch.randint(0, cfg.num_steps, (num_envs,), device=device)
        env_idxs = torch.arange(num_envs, device=device)
        start_states = buffer.obs[rand_time_idxs, env_idxs]

        # Roll out imagined trajectories
        imagined_trajectories = world_model.generate_imagined_trajectories(
            policy_net=model,
            start_states=start_states,
            horizon=imagined_horizon
        )

        # Build a dream buffer from the imagined data
        dream_buffer = RolloutBuffer(imagined_horizon, num_envs, obs_shape, device)
        for t, traj in enumerate(imagined_trajectories):
            dream_buffer.add(
                obs=traj["obs"],
                actions=traj["actions"],
                logprobs=traj["logprobs"],
                rewards=traj["rewards"],
                dones=traj["dones"],
                values=traj["values"],
                next_obs=traj["next_obs"]
            )

        # Bootstrap value for GAE computation
        if imagined_trajectories:
            last_dream_obs = imagined_trajectories[-1]["next_obs"]
            with torch.no_grad():
                _, last_dream_value = model.forward(last_dream_obs)

            dream_buffer.compute_returns_and_advantages(last_dream_value, cfg.gamma, cfg.gae_lambda)

        return dream_buffer, imagined_trajectories

    def update_policy(target_buffer, epochs):
        """Run PPO updates on a buffer (real or imagined) for N epochs."""
        update_stats = []
        for epoch in range(epochs):
            minibatches = target_buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
            for obs, actions, logprobs, advantages, returns, values, _, _ in minibatches:
                ppo_batch = [obs, actions, logprobs, advantages, returns, values]
                stats = ppo_update(model, optimizer, [ppo_batch], cfg, ewc=ewc, ewc_lambda=ewc_lambda)
                update_stats.append(stats)
        return update_stats

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    # Spawn guards for multi-head routing
    spawn_warmup = 10     # don't spawn before this many updates
    spawn_cooldown = 20   # min updates between spawns
    last_spawn_update = -999
    running_reward_loss = 0.0  # tracks baseline reward loss for relative threshold

    # Track regime for network switching
    prev_regime_id = -1

    for update in range(start_update, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lrnow
        else:
            lrnow = cfg.lr

        buffer.reset()

        # Phase A: Collect real experience
        collect_stats = collect_real_experience(global_step)
        global_step = collect_stats["global_step"]

        with torch.no_grad():
            _, last_value = model.forward(obs_t)
        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)

        # Head routing: select best head or spawn a new one
        with torch.no_grad():
            n_sample = min(64, cfg.num_steps)
            sample_obs = buffer.obs[:n_sample].reshape(-1, *obs_shape)
            sample_act = buffer.actions[:n_sample].reshape(-1)
            sample_next = buffer.next_obs[:n_sample].reshape(-1, *obs_shape)
            sample_rew = buffer.rewards[:n_sample].reshape(-1)
            best_head, best_loss = world_model.select_best_head(
                sample_obs, sample_act, sample_next, sample_rew
            )
            spawn_allowed = (
                update >= spawn_warmup
                and len(world_model.state_heads) < max_heads
                and (update - last_spawn_update) >= spawn_cooldown
            )
            # Relative + absolute threshold: loss must exceed both the absolute
            # threshold AND 3× the running average (adapts to loss scale)
            surprise_spike = best_loss > surprise_threshold
            if running_reward_loss > 0:
                surprise_spike = surprise_spike and (best_loss > 3.0 * running_reward_loss)

            # Update running average (EMA of best-head reward loss)
            if running_reward_loss == 0.0:
                running_reward_loss = best_loss
            else:
                running_reward_loss = 0.05 * best_loss + 0.95 * running_reward_loss

            if surprise_spike and spawn_allowed:
                best_head = world_model.spawn_head()
                last_spawn_update = update
                # Refresh optimizer to include new head parameters
                wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_lr)
                print(f"[multihead] Spawned head {best_head} (loss={best_loss:.3f} > threshold={surprise_threshold}, update={update})")
            world_model.active_head = best_head

        # Switch active head on regime change
        if steps_per_regime and steps_per_regime > 0:
            current_regime_id = (global_step // steps_per_regime) % 2
        else:
            current_regime_id = 0
        if current_regime_id != prev_regime_id and prev_regime_id >= 0:
            # EWC: snapshot Fisher on encoder before switching
            flat_obs = buffer.obs.reshape(-1, *obs_shape)
            flat_actions = buffer.actions.reshape(-1)
            ewc.update(model, flat_obs, flat_actions)
            print(f"[ewc] Fisher updated on regime switch {prev_regime_id} → {current_regime_id} (lambda={ewc_lambda})")
            # Spawn a new head pair if this regime hasn't been seen before
            while current_regime_id >= len(model.actor_heads):
                new_head = model.spawn_head()
                # Refresh optimizer to include new head parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=lrnow, eps=1e-5)
                print(f"[policy] Spawned head {new_head}")
            model.active_head = current_regime_id
            print(f"[policy] Regime switch → regime {current_regime_id}, using head {current_regime_id}")
        prev_regime_id = current_regime_id

        logger.scalar("world_model/active_head", world_model.active_head, global_step)
        logger.scalar("world_model/num_heads", len(world_model.state_heads), global_step)
        logger.scalar("world_model/best_head_loss", best_loss, global_step)
        logger.scalar("policy/active_head", model.active_head, global_step)
        logger.scalar("policy/num_heads", len(model.actor_heads), global_step)
        logger.scalar("ewc/active", float(ewc.is_active), global_step)

        # Phase B: Update policy on real data
        update_stats = update_policy(buffer, cfg.update_epochs)

        # Phase C: Train World Model (active head + trunk)
        wm_stats = update_world_model()

        # Phase C.1: Adapt inactive heads to trunk drift (1 epoch, keeps them routable)
        if len(world_model.state_heads) > 1:
            saved_head = world_model.active_head
            for head_idx in range(len(world_model.state_heads)):
                if head_idx == saved_head:
                    continue
                world_model.active_head = head_idx
                minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
                for obs, actions, _, _, _, _, next_obs, rewards in minibatches:
                    pred_next_obs, pred_reward = world_model(obs, actions)
                    target_indices = torch.argmax(next_obs, dim=1)
                    loss_state = torch.nn.functional.cross_entropy(pred_next_obs, target_indices)
                    loss_reward = torch.nn.functional.mse_loss(pred_reward, rewards)
                    # Lower weight to prevent overwriting the head's regime specialization
                    wm_loss = 0.1 * (loss_state + loss_reward)
                    wm_optimizer.zero_grad()
                    wm_loss.backward()
                    wm_optimizer.step()
            world_model.active_head = saved_head

        # Phase D: Dream & update policy on imagined data (skipped in passive mode)
        dream_stats = []
        dream_buffer = None
        if imagined_horizon > 0:
            # Dream on the ACTIVE head first
            dream_buffer, _ = generate_dream_experience()
            dream_stats = update_policy(dream_buffer, epochs=1)

            # Cross-regime dreaming: dream on all OTHER WM heads
            saved_wm_head = world_model.active_head
            saved_policy_head = model.active_head
            for head_idx in range(len(world_model.state_heads)):
                if head_idx == saved_wm_head:
                    continue
                world_model.active_head = head_idx
                # Route policy through matching head if it exists
                model.active_head = head_idx % len(model.actor_heads)
                cross_dream_buffer, _ = generate_dream_experience()
                cross_stats = update_policy(cross_dream_buffer, epochs=1)
                dream_stats.extend(cross_stats)
            world_model.active_head = saved_wm_head
            model.active_head = saved_policy_head

        # -----------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------

        avg_stats = {k: np.mean([s[k] for s in update_stats]) for k in update_stats[0]} if update_stats else {}
        avg_wm_stats = {k: np.mean([s[k] for s in wm_stats]) for k in wm_stats[0]} if wm_stats else {}

        for k, v in avg_stats.items():
            logger.scalar(k, v, global_step)

        for k, v in avg_wm_stats.items():
            logger.scalar(k, v, global_step)

        if dream_stats and dream_buffer is not None:
            avg_dream_stats = {f"ppo/imagined_{k.split('/')[-1]}": np.mean([s[k] for s in dream_stats]) for k in dream_stats[0]}
            for k, v in avg_dream_stats.items():
                logger.scalar(k, v, global_step)

            logger.scalar("ppo/imagined_value_mean", dream_buffer.values.mean().item(), global_step)
            logger.scalar("ppo/imagined_return_mean", dream_buffer.returns.mean().item(), global_step)

        # Intrinsic reward stats
        episodic_intrinsic_rewards = collect_stats["episodic_intrinsic_rewards"]
        episodic_intrinsic_rewards_max = collect_stats["episodic_intrinsic_rewards_max"]

        mean_intrinsic = np.mean(episodic_intrinsic_rewards) if episodic_intrinsic_rewards else 0.0
        max_intrinsic = np.max(episodic_intrinsic_rewards_max) if episodic_intrinsic_rewards_max else 0.0
        mean_total_abs = collect_stats["buffer_rewards_abs_mean"]

        logger.scalar("ppo/intrinsic_reward_mean", mean_intrinsic, global_step)
        logger.scalar("ppo/intrinsic_reward_max", max_intrinsic, global_step)

        if mean_total_abs > 1e-6:
            logger.scalar("ppo/intrinsic_reward_ratio", mean_intrinsic / mean_total_abs, global_step)
        else:
            logger.scalar("ppo/intrinsic_reward_ratio", 0.0, global_step)

        logger.scalar("charts/learning_rate", lrnow, global_step)
        logger.scalar("charts/heartbeat", global_step, global_step)
        logger.scalar("charts/reward_step_mean", collect_stats["buffer_rewards_mean"], global_step)
        logger.scalar("charts/reward_step_max", collect_stats["buffer_rewards_max"], global_step)
        logger.scalar("charts/reward_step_std", collect_stats["buffer_rewards_std"], global_step)

        sps = int(global_step / max(1e-9, (time.time() - start_time)))
        logger.scalar("charts/SPS", sps, global_step)

        # Checkpointing
        if update % save_every_updates == 0 or update == num_updates:
            ckpt_path = os.path.join(save_dir, f"{run_name}_update{update}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "world_model_state_dict": world_model.state_dict(),
                    "wm_optimizer_state_dict": wm_optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")

        if update % 10 == 0:
            print(f"update {update}/{num_updates} | step={global_step} | SPS={sps}")

    envs.close()
    logger.close()
