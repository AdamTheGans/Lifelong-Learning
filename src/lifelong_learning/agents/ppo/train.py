from __future__ import annotations

import os
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from collections import deque

register(
    id="MiniGrid-DualGoal-8x8-v0",
    entry_point="lifelong_learning.envs.dual_goal:DualGoalEnv",
)

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
# [REMOVED] RunningMeanStd - switching to simple EMA



from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel 
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
    switch_on_reset: bool = False,
    switch_mid_episode: bool = False,
    mid_episode_switch_step_range: tuple[int, int] = (20, 80),
    run_name: str | None = None,
    save_dir: str = "checkpoints",
    save_every_updates: int = 50,
    anneal_lr: bool = True,
    resume_path: str | None = None,

    intrinsic_coef: float = 10.0, # [NEW] Amplified Error Coefficient (Reduced for CE Loss)
    intrinsic_reward_clip: float = 0.5, # [NEW] Cap for intrinsic reward
    intrinsic_noise_threshold: float = 0.02, # [NEW] Hard Noise Gate
    imagined_horizon: int = 5,    # [NEW] Dream length
    wm_lr: float = 1e-4,          # [NEW] Slower World Model
):

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_envs = max(cfg.num_envs, 16)

    def make_thunk(i: int):
        def thunk():
            return make_env(
                env_id=env_id,
                seed=cfg.seed + i,
                steps_per_regime=steps_per_regime,
                episodes_per_regime=episodes_per_regime,
                start_regime=start_regime,
                switch_on_reset=switch_on_reset,
                switch_mid_episode=switch_mid_episode,
                mid_episode_switch_step_range=mid_episode_switch_step_range,
                record_stats=False, # [FIX] Disable internal wrapper stats, we do it manually
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(num_envs)])
    
    # [FIX] Do NOT use RecordEpisodeStatistics here. It's causing the empty graphs.
    
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    model = CNNActorCritic(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # [NEW] Sidekick World Model
    world_model = SimpleWorldModel(obs_shape, n_actions).to(device)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_lr) # [MODIFIED] Slow down WM
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)
    
    # [NEW] Hard Noise Gate Logic
    # No more EMA. We use a static threshold to filter noise.




    # [RESUME LOGIC]
    start_global_step = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Optimizer state loaded.")
        else:
            print("WARNING: Optimizer state not found in checkpoint. Starting fresh (with potential LR mismatch if annealing).")

        # [NEW] Load World Model
        if "world_model_state_dict" in ckpt:
            world_model.load_state_dict(ckpt["world_model_state_dict"])
            print("World Model state loaded.")
        else:
            print("WARNING: World Model state not found in checkpoint (likely from a pre-Dyna run). Starting Fresh.")
            
        if "wm_optimizer_state_dict" in ckpt:
            wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
            print("World Model Optimizer state loaded.")

        if "global_step" in ckpt:
            start_global_step = ckpt["global_step"]
            print(f"Resuming from global_step={start_global_step}")


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

    # [NEW] Manual Stats Tracking
    running_returns = np.zeros(num_envs)
    running_lengths = np.zeros(num_envs, dtype=int)
    outcome_window = deque(maxlen=100)

    print(f"Training on {device} with {num_envs} envs for {num_updates} updates (starting from update {start_update}).")

    # =========================================================================
    # HELPER FUNCTIONS (WAKE / SLEEP Phases)
    # =========================================================================

    def collect_experience(obs_curr, global_step_curr):
        """Phase 1: Wake - Collect Real Experience"""
        episodic_intrinsic_rewards = []
        episodic_intrinsic_rewards_max = []
        
        for t in range(cfg.num_steps):
            global_step_curr += num_envs

            with torch.no_grad():
                action, logprob, entropy, value = model.act(obs_curr)
                # Intrinsic Motivation (Prediction)
                pred_next_obs, pred_reward = world_model(obs_curr, action) 

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # Auto-reset handling for surprise calculation
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        real_next_obs[idx] = final_obs
            
            real_next_obs_t = torch.tensor(real_next_obs, dtype=torch.float32, device=device)

            # Compute Surprise (State + Reward)
            with torch.no_grad():
                # 1. State Surprise (Cross Entropy)
                real_next_indices = torch.argmax(real_next_obs_t, dim=1)
                state_surprise_map = torch.nn.functional.cross_entropy(pred_next_obs, real_next_indices, reduction='none')
                state_surprise = state_surprise_map.view(num_envs, -1).mean(dim=1) 
                
                # 2. Reward Surprise (MSE)
                real_reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
                reward_surprise = torch.nn.functional.mse_loss(pred_reward, real_reward_t, reduction='none')
                
                total_surprise = state_surprise + reward_surprise
                
                # Hard Noise Gate
                noise_mask = (total_surprise > intrinsic_noise_threshold).float()
                signal_surprise = total_surprise * noise_mask
                raw_intrinsic = signal_surprise * intrinsic_coef
                intrinsic_reward = torch.clamp(raw_intrinsic, 0.0, intrinsic_reward_clip)
 
                # Debug Logging
                logger.scalar("debug/wm_raw_error_mean", total_surprise.mean().item(), global_step_curr)
                logger.scalar("debug/intrinsic_reward_raw", raw_intrinsic.mean().item(), global_step_curr)

                episodic_intrinsic_rewards.append(intrinsic_reward.mean().item())
                episodic_intrinsic_rewards_max.append(intrinsic_reward.max().item())

            # Update manual trackers
            nonlocal running_returns, running_lengths
            running_returns += reward
            running_lengths += 1

            # Combined Reward for PPO
            total_reward = torch.tensor(reward, dtype=torch.float32, device=device) + intrinsic_reward

            buffer.add(
                obs=obs_curr,
                actions=action,
                logprobs=logprob,
                rewards=total_reward,
                dones=torch.tensor(done, dtype=torch.float32, device=device),
                values=value,
                next_obs=real_next_obs_t, 
            )

            obs_curr = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # Log finished episodes
            if np.any(done):
                done_indices = np.where(done)[0]
                for i in done_indices:
                    logger.scalar("charts/episodic_return", running_returns[i], global_step_curr)
                    logger.scalar("charts/episodic_length", running_lengths[i], global_step_curr)
                    
                    if "final_info" in infos:
                        for final_info in infos["final_info"]:
                            if final_info and "reached_good_goal" in final_info:
                                outcome = 0
                                if final_info.get("reached_good_goal", 0) > 0: outcome = 1
                                elif final_info.get("reached_bad_goal", 0) > 0: outcome = -1
                                elif final_info.get("timed_out", 0) > 0: outcome = 0
                                outcome_window.append(outcome)
                                
                                if len(outcome_window) > 0:
                                    logger.scalar("charts/success_rate", sum(1 for x in outcome_window if x==1)/len(outcome_window), global_step_curr)
                                    logger.scalar("charts/failure_rate", sum(1 for x in outcome_window if x==-1)/len(outcome_window), global_step_curr)
                                    logger.scalar("charts/timeout_rate", sum(1 for x in outcome_window if x==0)/len(outcome_window), global_step_curr)

                    if "regime_id" in infos:
                        logger.scalar(f"charts/r_regime_{infos['regime_id'][i]}", running_returns[i], global_step_curr)

                    running_returns[i] = 0
                    running_lengths[i] = 0
        
        # Compute GAE
        with torch.no_grad():
            _, last_value = model.forward(obs_curr)
        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)
        
        return obs_curr, global_step_curr, episodic_intrinsic_rewards, episodic_intrinsic_rewards_max

    def train_world_model():
        """Phase 2: Train World Model (Supervised on Buffer)"""
        wm_stats_list = []
        for _ in range(cfg.update_epochs):
            minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
            for obs, actions, _, _, _, _, next_obs, rewards in minibatches:
                pred_next_obs, pred_reward = world_model(obs, actions)
                
                next_obs_indices = torch.argmax(next_obs, dim=1)
                loss_state = torch.nn.functional.cross_entropy(pred_next_obs, next_obs_indices)
                loss_reward = torch.nn.functional.mse_loss(pred_reward, rewards)
                
                wm_loss = loss_state + loss_reward
                
                wm_optimizer.zero_grad()
                wm_loss.backward()
                wm_optimizer.step()
                
                wm_stats_list.append({
                    "world_model/loss_total": wm_loss.item(),
                    "world_model/loss_state": loss_state.item(),
                    "world_model/loss_reward": loss_reward.item(),
                })
        return wm_stats_list

    def train_policy_on_real():
        """Phase 3a: Train Policy on Real Buffer"""
        update_stats_list = []
        for _ in range(cfg.update_epochs):
            minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
            for obs, actions, logprobs, advantages, returns, values, _, _ in minibatches:
                ppo_batch = [obs, actions, logprobs, advantages, returns, values]
                stats = ppo_update(model, optimizer, [ppo_batch], cfg)
                update_stats_list.append(stats)
        return update_stats_list

    def dream_and_train_policy(global_step_curr):
        """Phase 3b: Sleep - Dream and Train Policy"""
        # 1. Sample seeds
        rand_time_idxs = torch.randint(0, cfg.num_steps, (num_envs,), device=device)
        env_idxs = torch.arange(num_envs, device=device)
        start_states = buffer.obs[rand_time_idxs, env_idxs] 
        
        # 2. Dream
        imagined_trajectories = world_model.generate_imagined_trajectories(
            policy_net=model,
            start_states=start_states,
            horizon=imagined_horizon
        )
        
        # 3. Create Dream Buffer
        dream_buffer = RolloutBuffer(imagined_horizon, num_envs, obs_shape, device)
        for traj in imagined_trajectories:
            dream_buffer.add(**traj)
            
        # 4. Bootstrap Value
        last_dream_obs = imagined_trajectories[-1]["next_obs"]
        with torch.no_grad():
            _, last_dream_value = model.forward(last_dream_obs)
        dream_buffer.compute_returns_and_advantages(last_dream_value, cfg.gamma, cfg.gae_lambda)
        
        # 5. Train PPO on Dream Data
        dream_stats_list = []
        dream_minibatches = dream_buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
        
        for obs, actions, logprobs, advantages, returns, values, _, _ in dream_minibatches:
            ppo_batch = [obs, actions, logprobs, advantages, returns, values]
            stats = ppo_update(model, optimizer, [ppo_batch], cfg)
            dream_stats_list.append(stats)

        # Log Dream Stats
        if len(dream_stats_list) > 0:
            avg_dream_stats = {f"ppo/imagined_{k.split('/')[-1]}": np.mean([s[k] for s in dream_stats_list]) for k in dream_stats_list[0]}
            for k, v in avg_dream_stats.items():
                logger.scalar(k, v, global_step_curr)
            
            logger.scalar("ppo/imagined_value_mean", dream_buffer.values.mean().item(), global_step_curr)
            logger.scalar("ppo/imagined_return_mean", dream_buffer.returns.mean().item(), global_step_curr)

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================

    for update in range(start_update, num_updates + 1):
        # Anneal LR
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = cfg.lr

        buffer.reset()

        # 1. Collect (WAKE)
        obs_t, global_step, int_rewards, int_rewards_max = collect_experience(obs_t, global_step)

        # 2. Train World Model
        wm_stats = train_world_model()

        # 3. Train Policy (REAL)
        ppo_stats = train_policy_on_real()

        # 4. Train Policy (DREAM)
        if imagined_horizon > 0:
            dream_and_train_policy(global_step)

        # 5. Logging
        avg_ppo_stats = {k: np.mean([s[k] for s in ppo_stats]) for k in ppo_stats[0]}
        avg_wm_stats = {k: np.mean([s[k] for s in wm_stats]) for k in wm_stats[0]} 

        for k, v in avg_ppo_stats.items(): logger.scalar(k, v, global_step)
        for k, v in avg_wm_stats.items(): logger.scalar(k, v, global_step)

        logger.scalar("ppo/intrinsic_reward_mean", np.mean(int_rewards), global_step)
        logger.scalar("ppo/intrinsic_reward_max", np.max(int_rewards_max), global_step)
        
        mean_total_abs = buffer.rewards.abs().mean().item()
        logger.scalar("ppo/intrinsic_reward_ratio", (np.mean(int_rewards) / mean_total_abs) if mean_total_abs > 1e-6 else 0.0, global_step)
        
        logger.scalar("charts/learning_rate", lrnow, global_step)
        logger.scalar("charts/heartbeat", global_step, global_step)
        logger.scalar("charts/reward_step_mean", buffer.rewards.mean().item(), global_step)
        logger.scalar("charts/reward_step_max", buffer.rewards.max().item(), global_step)
        logger.scalar("charts/reward_step_std", buffer.rewards.std().item(), global_step)

        sps = int(global_step / max(1e-9, (time.time() - start_time)))
        logger.scalar("charts/SPS", sps, global_step)

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
