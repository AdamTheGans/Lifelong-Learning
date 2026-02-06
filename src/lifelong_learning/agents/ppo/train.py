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

def log_vec_episodic_stats(logger, infos, global_step: int):
    """
    Robustly log episodic return/length.
    """
    # Check if any episode finished
    if "_episode" not in infos:
        return

    # iterate over all environments that finished an episode this step
    for i in np.where(infos["_episode"])[0]:
        
        # 1. Try to get SMOOTH stats (from RollingAvgWrapper)
        # We check 'final_info' first (standard), then fallback to infos dict
        found_smooth = False
        
        # Try Method A: final_info
        if "final_info" in infos:
            f_info = infos["final_info"][i]
            if f_info and "episode_avg" in f_info:
                avg = f_info["episode_avg"]
                logger.scalar("charts/episodic_return_smooth", avg["r"], global_step)
                logger.scalar("charts/episodic_length_smooth", avg["l"], global_step)
                found_smooth = True
                
        # Try Method B: direct key (if SyncVectorEnv didn't pack final_info correctly)
        if not found_smooth and "episode_avg" in infos:
             # If it's a dictionary of arrays
             avg_root = infos["episode_avg"]
             if "r" in avg_root:
                 # Check if it's an array (vectorized) or scalar
                 val = avg_root["r"][i] if isinstance(avg_root["r"], np.ndarray) else avg_root["r"]
                 logger.scalar("charts/episodic_return_smooth", val, global_step)
                 
                 val_l = avg_root["l"][i] if isinstance(avg_root["l"], np.ndarray) else avg_root["l"]
                 logger.scalar("charts/episodic_length_smooth", val_l, global_step)

        # 2. Always log RAW stats (from RecordEpisodeStatistics)
        if "episode" in infos:
            ep_data = infos["episode"]
            if "r" in ep_data:
                logger.scalar("charts/episodic_return_raw", ep_data["r"][i], global_step)
            if "l" in ep_data:
                logger.scalar("charts/episodic_length_raw", ep_data["l"][i], global_step)

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
                record_stats=True,
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(num_envs)])

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    model = CNNActorCritic(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)

    if run_name is None:
        run_name = f"ppo_{env_id}_s{cfg.seed}"

    logger = TBLogger(run_name=run_name)
    os.makedirs(save_dir, exist_ok=True)

    obs, info = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = cfg.total_timesteps // (num_envs * cfg.num_steps)
    global_step = 0
    start_time = time.time()

    print(f"Training on {device} with {num_envs} envs for {num_updates} updates.")

    for update in range(1, num_updates + 1):
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

            buffer.add(
                obs=obs_t,
                actions=action,
                logprobs=logprob,
                rewards=torch.tensor(reward, dtype=torch.float32, device=device),
                dones=torch.tensor(done, dtype=torch.float32, device=device),
                values=value,
            )

            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

            log_vec_episodic_stats(logger, infos, global_step)

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

        sps = int(global_step / max(1e-9, (time.time() - start_time)))
        logger.scalar("charts/SPS", sps, global_step)

        if update % save_every_updates == 0 or update == num_updates:
            ckpt_path = os.path.join(save_dir, f"{run_name}_update{update}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
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