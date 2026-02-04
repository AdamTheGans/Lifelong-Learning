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
    Handles the 'Dict of Arrays' format where `_episode` is a boolean mask.
    """
    # 1. Standard Gymnasium Vector Env format (Dict of Arrays)
    # We look for '_episode', which is a boolean mask array indicating which envs finished.
    if "_episode" in infos:
        mask = infos["_episode"]
        # Iterate over all environments that finished an episode in this step
        for i in np.where(mask)[0]:
            if "episode" in infos:
                ep_data = infos["episode"]
                
                # Extract standard return/length
                # We simply grab the value at index `i`
                if "r" in ep_data:
                    logger.scalar("charts/episodic_return", float(ep_data["r"][i]), global_step)
                if "l" in ep_data:
                    logger.scalar("charts/episodic_length", float(ep_data["l"][i]), global_step)

                # Extract custom regime stats (e.g., r_regime_0, l_regime_1)
                # These were injected into the 'episode' dict by our RegimeStatsWrapper
                for k, v in ep_data.items():
                    if k.startswith("r_regime_") or k.startswith("l_regime_"):
                        # v is likely an array, so we take the i-th element
                        val = v[i] if hasattr(v, "__getitem__") and v.ndim > 0 else v
                        logger.scalar(f"charts/{k}", float(val), global_step)
        return

    # 2. Legacy/List-based format (Fallbacks)
    # ... (Keep existing fallback logic if you want, but the above block covers your case)
    if "final_info" in infos and infos["final_info"] is not None:
        for finfo in infos["final_info"]:
            if finfo and "episode" in finfo:
                logger.scalar("charts/episodic_return", float(finfo["episode"]["r"]), global_step)
                logger.scalar("charts/episodic_length", float(finfo["episode"]["l"]), global_step)


def train_ppo(
    env_id: str,
    cfg: PPOConfig,
    *,
    # Deterministic Schedule
    steps_per_regime: int | None = None,
    episodes_per_regime: int | None = None,
    start_regime: int = 0,
    # Random / Legacy
    switch_on_reset: bool = False,
    switch_mid_episode: bool = False,
    mid_episode_switch_step_range: tuple[int, int] = (20, 80),
    # Training
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
                steps_per_regime=steps_per_regime,
                episodes_per_regime=episodes_per_regime,
                start_regime=start_regime,
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
                    "schedule": {
                        "steps_per_regime": steps_per_regime,
                        "episodes_per_regime": episodes_per_regime,
                        "start_regime": start_regime,
                        "switch_on_reset": switch_on_reset,
                        "switch_mid_episode": switch_mid_episode,
                    },
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
