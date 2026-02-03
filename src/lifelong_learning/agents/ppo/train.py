from __future__ import annotations

import os
import time
import numpy as np
import torch
import gymnasium as gym

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.buffers import RolloutBuffer
from lifelong_learning.utils.seeding import seed_everything
from lifelong_learning.utils.logger import TBLogger


def log_vec_episodic_stats(logger, infos, global_step: int):
    """
    Robustly log episodic return/length from Gymnasium vector env info dict.

    Depending on gymnasium version/wrappers, episode stats may appear in:
      - infos["final_info"] : list of dicts (one per env) when an episode ends
      - infos["episode"]    : sometimes list/array of dicts
      - infos["final_info"][i]["episode"] : dict with keys "r" and "l"
    """
    # Most reliable in Gymnasium vector envs:
    if "final_info" in infos and infos["final_info"] is not None:
        for finfo in infos["final_info"]:
            if finfo is None:
                continue
            if isinstance(finfo, dict) and "episode" in finfo and isinstance(finfo["episode"], dict):
                ep = finfo["episode"]
                if "r" in ep:
                    logger.scalar("charts/episodic_return", float(ep["r"]), global_step)
                if "l" in ep:
                    logger.scalar("charts/episodic_length", float(ep["l"]), global_step)

            extra = finfo.get("episode_extra", None)
            if isinstance(extra, dict):
                rid = extra.get("regime_id_end", None)
                if rid is not None:
                    logger.scalar(f"charts/episodic_return_regime_{rid}", float(ep["r"]), global_step)
        return

    # Fallback: sometimes episode info is directly here
    if "episode" in infos and infos["episode"] is not None:
        ep_container = infos["episode"]

        # Case A: list of dicts / Nones
        if isinstance(ep_container, (list, tuple)):
            for ep in ep_container:
                if isinstance(ep, dict):
                    if "r" in ep:
                        logger.scalar("charts/episodic_return", float(ep["r"]), global_step)
                    if "l" in ep:
                        logger.scalar("charts/episodic_length", float(ep["l"]), global_step)
            return

        # Case B: dict of arrays (rare)
        if isinstance(ep_container, dict):
            if "r" in ep_container:
                # could be scalar or array
                r = ep_container["r"]
                if np.isscalar(r):
                    logger.scalar("charts/episodic_return", float(r), global_step)
            if "l" in ep_container:
                l = ep_container["l"]
                if np.isscalar(l):
                    logger.scalar("charts/episodic_length", float(l), global_step)
            return


def train_ppo(
    env_id: str,
    cfg: PPOConfig,
    *,
    switch_on_reset: bool = False,
    switch_mid_episode: bool = False,
    mid_episode_switch_step_range: tuple[int, int] = (20, 80),
    run_name: str | None = None,
    save_dir: str = "checkpoints",
    save_every_updates: int = 50,
):
    """
    PPO training loop using SyncVectorEnv (stable on Windows).
    """
    from lifelong_learning.envs.make_env import make_env  # local import to avoid circulars

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    def make_thunk(i: int):
        def thunk():
            return make_env(
                env_id=env_id,
                seed=cfg.seed + i,
                switch_on_reset=switch_on_reset,
                switch_mid_episode=switch_mid_episode,
                mid_episode_switch_step_range=mid_episode_switch_step_range,
                record_stats=True,
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(cfg.num_envs)])

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    model = CNNActorCritic(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, obs_shape, device)

    if run_name is None:
        run_name = f"ppo_{env_id}_s{cfg.seed}"

    logger = TBLogger(run_name=run_name)

    os.makedirs(save_dir, exist_ok=True)

    obs, info = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = cfg.total_timesteps // (cfg.num_envs * cfg.num_steps)
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        buffer.reset()

        # Rollout
        for t in range(cfg.num_steps):
            global_step += cfg.num_envs

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

            # Episodic logging
            log_vec_episodic_stats(logger, infos, global_step)

        # Bootstrap value for GAE
        with torch.no_grad():
            _, last_value = model.forward(obs_t)

        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)

        # PPO Update
        minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
        logs = ppo_update(model, optimizer, minibatches, cfg)

        # Training diagnostics
        for k, v in logs.items():
            logger.scalar(k, v, global_step)

        sps = int(global_step / max(1e-9, (time.time() - start_time)))
        logger.scalar("charts/SPS", sps, global_step)

        # Save checkpoint
        if update % save_every_updates == 0 or update == num_updates:
            ckpt_path = os.path.join(save_dir, f"{run_name}_update{update}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "env_id": env_id,
                    "switch_on_reset": switch_on_reset,
                    "switch_mid_episode": switch_mid_episode,
                    "mid_episode_switch_step_range": mid_episode_switch_step_range,
                    "global_step": global_step,
                    "update": update,
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")

        if update % 10 == 0:
            print(f"update {update}/{num_updates} | step={global_step} | SPS={sps}")

    envs.close()
    logger.close()
