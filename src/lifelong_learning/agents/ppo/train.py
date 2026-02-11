from __future__ import annotations

import os
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="MiniGrid-DualGoal-8x8-v0",
    entry_point="lifelong_learning.envs.dual_goal:DualGoalEnv",
)

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
from lifelong_learning.agents.ppo.network import CNNActorCritic
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

    print(f"Training on {device} with {num_envs} envs for {num_updates} updates (starting from update {start_update}).")

    for update in range(start_update, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = cfg.lr

        buffer.reset()

        for t in range(cfg.num_steps):
            global_step += num_envs

            with torch.no_grad():
                action, logprob, entropy, value = model.act(obs_t)

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # [NEW] Update manual trackers
            running_returns += reward
            running_lengths += 1

            buffer.add(
                obs=obs_t,
                actions=action,
                logprobs=logprob,
                rewards=torch.tensor(reward, dtype=torch.float32, device=device),
                dones=torch.tensor(done, dtype=torch.float32, device=device),
                values=value,
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
                    # Note: 'infos' here is from the RESET env, so this might be the *next* regime 
                    # if the switch happened exactly now. But it's close enough for visualization.
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
        for epoch in range(cfg.update_epochs):
            minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
            stats = ppo_update(model, optimizer, minibatches, cfg)
            update_stats.append(stats)
        
        avg_stats = {k: np.mean([s[k] for s in update_stats]) for k in update_stats[0]}

        for k, v in avg_stats.items():
            logger.scalar(k, v, global_step)
        logger.scalar("charts/learning_rate", lrnow, global_step)
        logger.scalar("charts/heartbeat", global_step, global_step)
        logger.scalar("charts/reward_step_mean", buffer.rewards.mean().item(), global_step)

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