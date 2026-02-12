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
from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel # [NEW]
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
    intrinsic_coef: float = 0.02, # [NEW] Curiosity
    imagined_horizon: int = 5,    # [NEW] Dream length
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
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3) # Standard Adam LR
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)

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
    # We maintain these arrays to track progress for each environment individually
    running_returns = np.zeros(num_envs)
    running_lengths = np.zeros(num_envs, dtype=int)
    
    # Outcome tracking (rolling window of last 100 episodes)
    outcome_window = deque(maxlen=100)

    print(f"Training on {device} with {num_envs} envs for {num_updates} updates (starting from update {start_update}).")

    for update in range(start_update, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = cfg.lr

        buffer.reset()

        # [NEW] Track intrinsic rewards
        episodic_intrinsic_rewards = []

        for t in range(cfg.num_steps):
            global_step += num_envs

            with torch.no_grad():
                action, logprob, entropy, value = model.act(obs_t)
                
                # [NEW] Intrinsic Motivation Calculation
                # We do this inside no_grad because we don't want gradients during collection
                pred_next_obs, _ = world_model(obs_t, action) 

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            
            # [NEW] Handle Correct Next Obs (Autoreset) for surprise calc
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        real_next_obs[idx] = final_obs
            
            real_next_obs_t = torch.tensor(real_next_obs, dtype=torch.float32, device=device)

            # [NEW] Compute Surprise
            with torch.no_grad():
                # We use MSE as surprise. 
                # Note: This is an aggregation over (C, H, W). 
                # We want a scalar per environment.
                # pred_next_obs: (B, C, H, W), real_next_obs_t: (B, C, H, W)
                surprise = torch.nn.functional.mse_loss(pred_next_obs, real_next_obs_t, reduction='none')
                # Sum over C, H, W to get surprise per env
                surprise_per_env = surprise.view(num_envs, -1).mean(dim=1) 
                
                intrinsic_reward = intrinsic_coef * surprise_per_env
                episodic_intrinsic_rewards.append(intrinsic_reward.mean().item())

            # [NEW] Update manual trackers
            running_returns += reward
            running_lengths += 1

            # Combined Reward for PPO
            total_reward = torch.tensor(reward, dtype=torch.float32, device=device) + intrinsic_reward

            buffer.add(
                obs=obs_t,
                actions=action,
                logprobs=logprob,
                rewards=total_reward, # [MODIFIED] Now includes intrinsic curiosity
                dones=torch.tensor(done, dtype=torch.float32, device=device),
                values=value,
                next_obs=real_next_obs_t, 
            )

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # [NEW] Check for finished episodes and log
            if np.any(done):
                # Identify which envs finished
                done_indices = np.where(done)[0]
                for i in done_indices:
                    # Log the collected stats
                    logger.scalar("charts/episodic_return", running_returns[i], global_step)
                    logger.scalar("charts/episodic_length", running_lengths[i], global_step)
                    
                    # Optional: Attempt to log regime. 
                    if "final_info" in infos:
                        for final_info in infos["final_info"]:
                            if final_info is not None and "reached_good_goal" in final_info:
                                # This was a valid episode end
                                outcome = 0 # unknown
                                if final_info.get("reached_good_goal", 0.0) > 0:
                                    outcome = 1 # success
                                elif final_info.get("reached_bad_goal", 0.0) > 0:
                                    outcome = -1 # failure
                                elif final_info.get("timed_out", 0.0) > 0:
                                    outcome = 0 # timeout 
                                
                                outcome_window.append(outcome)
                                
                                # Log rolling stats
                                if len(outcome_window) > 0:
                                    success_rate = sum(1 for x in outcome_window if x == 1) / len(outcome_window)
                                    failure_rate = sum(1 for x in outcome_window if x == -1) / len(outcome_window)
                                    timeout_rate = sum(1 for x in outcome_window if x == 0) / len(outcome_window)
                                    
                                    logger.scalar("charts/success_rate", success_rate, global_step)
                                    logger.scalar("charts/failure_rate", failure_rate, global_step)
                                    logger.scalar("charts/timeout_rate", timeout_rate, global_step)

                    if "regime_id" in infos:
                        regime = infos["regime_id"][i]
                        logger.scalar(f"charts/r_regime_{regime}", running_returns[i], global_step)

                    # Reset trackers for this env
                    running_returns[i] = 0
                    running_lengths[i] = 0

        with torch.no_grad():
            _, last_value = model.forward(obs_t)

        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)

        update_stats = []
        wm_stats = [] 

        for epoch in range(cfg.update_epochs):
            minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
            # PPO Update needs obs, actions, logprobs, advantages, returns, values
            # World Model needs obs, actions, next_obs, rewards
            for obs, actions, logprobs, advantages, returns, values, next_obs, rewards in minibatches:
                
                # 1. PPO Update
                # Re-package for ppo_update signature
                ppo_batch = [obs, actions, logprobs, advantages, returns, values]
                stats = ppo_update(model, optimizer, [ppo_batch], cfg)
                update_stats.append(stats)

                # 2. World Model Update (Sidekick)
                # Predict
                pred_next_obs, pred_reward = world_model(obs, actions)
                
                # Loss
                loss_state = torch.nn.functional.mse_loss(pred_next_obs, next_obs)
                loss_reward = torch.nn.functional.mse_loss(pred_reward, rewards)
                
                wm_loss = loss_state + loss_reward
                
                wm_optimizer.zero_grad()
                wm_loss.backward()
                wm_optimizer.step()
                
                wm_stats.append({
                    "world_model/loss_total": wm_loss.item(),
                    "world_model/loss_state": loss_state.item(), # Surprise
                    "world_model/loss_reward": loss_reward.item(),
                })
        
        # [NEW] Imagined Phase (Active Dyna)
        # -------------------------------
        # Perform 1 epoch of PPO update on imagined data
        # Do this AFTER real updates to ensure policy is up to date before dreaming? 
        # Or before? Usually mixed. We'll do it after.
        
        # 1. Sample start states from the Buffer (randomly from what we just experienced)
        # Sample one start state per env
        # [FIX] Correctly sample (time, env) pairs to get (Num_Envs, C, H, W)
        rand_time_idxs = torch.randint(0, cfg.num_steps, (num_envs,), device=device)
        env_idxs = torch.arange(num_envs, device=device)
        start_states = buffer.obs[rand_time_idxs, env_idxs] 
        
        # 2. Dream
        # This uses the specific simple world model methods we added
        imagined_trajectories = world_model.generate_imagined_trajectories(
            policy_net=model,
            start_states=start_states,
            horizon=imagined_horizon
        )
        
        # 3. Create Dream Buffer
        dream_buffer = RolloutBuffer(imagined_horizon, num_envs, obs_shape, device)
        
        # 4. Fill Buffer
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
            
        # 5. Bootstrap Value for the last step
        # The last trajectory item has 'next_obs' which is the state *after* the horizon
        last_dream_obs = imagined_trajectories[-1]["next_obs"]
        with torch.no_grad():
            _, last_dream_value = model.forward(last_dream_obs)
            
        # 6. Compute GAE for Dream
        dream_buffer.compute_returns_and_advantages(last_dream_value, cfg.gamma, cfg.gae_lambda)
        
        # 7. Update PPO on Dream Data
        # We use a reduced number of epochs (e.g., 1) to avoid over-optimizing on dreams
        dream_stats = []
        dream_minibatches = dream_buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
        
        for obs, actions, logprobs, advantages, returns, values, _, _ in dream_minibatches:
            # We must ensure NO gradients flow to world model. 
            # generate_imagined_trajectories uses 'torch.no_grad()' for world model forward, so we are safe.
            # The 'obs' in dream_buffer are detached tensors.
            
            ppo_batch = [obs, actions, logprobs, advantages, returns, values]
            stats = ppo_update(model, optimizer, [ppo_batch], cfg)
            dream_stats.append(stats)

        # Log Dream Stats
        avg_dream_stats = {f"ppo/imagined_{k.split('/')[-1]}": np.mean([s[k] for s in dream_stats]) for k in dream_stats[0]}
        for k, v in avg_dream_stats.items():
            logger.scalar(k, v, global_step)
            
        # [NEW] Log Dream Value/Return stats
        if len(dream_stats) > 0:
             logger.scalar("ppo/imagined_value_mean", dream_buffer.values.mean().item(), global_step)
             logger.scalar("ppo/imagined_return_mean", dream_buffer.returns.mean().item(), global_step)

        # -------------------------------

        avg_stats = {k: np.mean([s[k] for s in update_stats]) for k in update_stats[0]}
        avg_wm_stats = {k: np.mean([s[k] for s in wm_stats]) for k in wm_stats[0]} 

        for k, v in avg_stats.items():
            logger.scalar(k, v, global_step)
        
        # [NEW] Log World Model Stats
        for k, v in avg_wm_stats.items():
            logger.scalar(k, v, global_step)
            
        # [NEW] Log Intrinsic Reward Components
        mean_intrinsic = np.mean(episodic_intrinsic_rewards)
        mean_total_abs = buffer.rewards.abs().mean().item()
        
        logger.scalar("ppo/intrinsic_reward_mean", mean_intrinsic, global_step)
        # Ratio of intrinsic to total signal magnitude
        if mean_total_abs > 1e-6:
            logger.scalar("ppo/intrinsic_reward_ratio", mean_intrinsic / mean_total_abs, global_step)
        else:
            logger.scalar("ppo/intrinsic_reward_ratio", 0.0, global_step)

        logger.scalar("charts/learning_rate", lrnow, global_step)
        logger.scalar("charts/heartbeat", global_step, global_step)
        logger.scalar("charts/reward_step_mean", buffer.rewards.mean().item(), global_step) # Note: this includes intrinsic

        sps = int(global_step / max(1e-9, (time.time() - start_time)))
        logger.scalar("charts/SPS", sps, global_step)

        if update % save_every_updates == 0 or update == num_updates:
            ckpt_path = os.path.join(save_dir, f"{run_name}_update{update}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
