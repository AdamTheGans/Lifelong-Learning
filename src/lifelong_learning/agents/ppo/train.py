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
from dataclasses import dataclass, field
from collections import deque

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel
from lifelong_learning.agents.ppo.buffers import RolloutBuffer
from lifelong_learning.utils.seeding import seed_everything
from lifelong_learning.utils.logger import TBLogger
from lifelong_learning.envs.make_env import make_env


# =========================================================================
# Inner Training State (exposed for Meta-RL "Brain" controller)
# =========================================================================

@dataclass
class InnerTrainState:
    """
    Mutable state of the inner Dyna-PPO training loop.

    Exposed so the Brain meta-agent can:
      1. Read training signals (stats, counters)
      2. Write hyperparameter adjustments (lr, ent_coef, intrinsic_coef, imagined_horizon)
    """
    # --- Models & optimizers ---
    model: CNNActorCritic = field(repr=False)
    optimizer: torch.optim.Adam = field(repr=False)
    world_model: SimpleWorldModel = field(repr=False)
    wm_optimizer: torch.optim.Adam = field(repr=False)
    buffer: RolloutBuffer = field(repr=False)

    # --- Environments ---
    envs: gym.vector.VectorEnv = field(repr=False)
    obs_shape: tuple = field(default=())
    n_actions: int = 0
    num_envs: int = 16

    # --- Config (mutable hyperparams the Brain can adjust) ---
    cfg: PPOConfig = field(default_factory=PPOConfig)
    intrinsic_coef: float = 0.1
    intrinsic_reward_clip: float = 0.1
    imagined_horizon: int = 5
    anneal_lr: bool = True

    # --- Counters ---
    global_step: int = 0
    current_update: int = 0
    num_updates: int = 0
    start_time: float = field(default_factory=time.time)

    # --- Live tensors ---
    obs_t: torch.Tensor = field(default=None, repr=False)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # --- Per-env episode trackers ---
    running_returns: np.ndarray = field(default=None, repr=False)
    running_lengths: np.ndarray = field(default=None, repr=False)

    # --- Outcome tracking ---
    outcome_window: deque = field(default_factory=lambda: deque(maxlen=100))

    # --- Logger ---
    logger: TBLogger = field(default=None, repr=False)
    run_name: str = ""

    # --- Checkpointing ---
    save_dir: str = "checkpoints"
    save_every_updates: int = 50


def init_inner_training(
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
) -> InnerTrainState:
    """
    Initialize all components of the Dyna-PPO inner training loop.

    Returns an InnerTrainState that can be driven step-by-step via
    run_inner_update(), or used internally by train_ppo().
    """

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_envs = max(cfg.num_envs, 16)

    # -----------------------------------------------------------------
    # Environment Setup
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Model & Optimizer Setup
    # -----------------------------------------------------------------

    model = CNNActorCritic(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    world_model = SimpleWorldModel(obs_shape, n_actions).to(device)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=wm_lr)
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)

    # -----------------------------------------------------------------
    # Resume from Checkpoint
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Logging & Tracking
    # -----------------------------------------------------------------

    if run_name is None:
        run_name = f"ppo_{env_id}_s{cfg.seed}"

    logger = TBLogger(run_name=run_name)
    os.makedirs(save_dir, exist_ok=True)

    obs, info = envs.reset(seed=cfg.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = cfg.total_timesteps // (num_envs * cfg.num_steps)
    global_step = start_global_step
    start_update = global_step // (num_envs * cfg.num_steps) + 1

    return InnerTrainState(
        model=model,
        optimizer=optimizer,
        world_model=world_model,
        wm_optimizer=wm_optimizer,
        buffer=buffer,
        envs=envs,
        obs_shape=obs_shape,
        n_actions=n_actions,
        num_envs=num_envs,
        cfg=cfg,
        intrinsic_coef=intrinsic_coef,
        intrinsic_reward_clip=intrinsic_reward_clip,
        imagined_horizon=imagined_horizon,
        anneal_lr=anneal_lr,
        global_step=global_step,
        current_update=start_update,
        num_updates=num_updates,
        start_time=time.time(),
        obs_t=obs_t,
        device=device,
        running_returns=np.zeros(num_envs),
        running_lengths=np.zeros(num_envs, dtype=int),
        outcome_window=deque(maxlen=100),
        logger=logger,
        run_name=run_name,
        save_dir=save_dir,
        save_every_updates=save_every_updates,
    )


# =========================================================================
# Single Update Step (the core unit the Brain calls)
# =========================================================================

def run_inner_update(state: InnerTrainState) -> dict:
    """
    Run ONE Dyna-PPO update cycle (collect → train WM → dream → PPO update).

    Mutates `state` in place (global_step, current_update, obs_t, etc.)
    and returns a stats dict with all signals the Brain needs.

    Returns:
        dict with keys like:
          - "mean_episodic_return", "success_rate", "failure_rate"
          - "mean_surprise", "wm_loss_state", "wm_loss_reward"
          - "policy_entropy", "policy_loss", "value_loss"
          - "current_lr", "current_ent_coef", "current_intrinsic_coef"
          - "current_imagined_horizon", "global_step", "done"
    """

    s = state  # alias
    update = s.current_update

    if update > s.num_updates:
        return {"done": True, "global_step": s.global_step}

    # -----------------------------------------------------------------
    # Learning rate annealing
    # -----------------------------------------------------------------
    if s.anneal_lr:
        frac = 1.0 - (update - 1.0) / s.num_updates
        lrnow = frac * s.cfg.lr
        s.optimizer.param_groups[0]["lr"] = lrnow
    else:
        lrnow = s.optimizer.param_groups[0]["lr"]

    s.buffer.reset()

    # =====================================================================
    # Phase A: Collect real experience
    # =====================================================================

    episodic_intrinsic_rewards = []
    episodic_intrinsic_rewards_max = []
    episodic_returns_this_update = []

    for t in range(s.cfg.num_steps):
        s.global_step += s.num_envs

        with torch.no_grad():
            action, logprob, entropy, value = s.model.act(s.obs_t)
            pred_next_obs, pred_reward = s.world_model(s.obs_t, action)

        next_obs, reward, terminated, truncated, infos = s.envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        # Handle autoreset: use final_observation for surprise calc on done envs
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            final_obs_mask = infos.get("_final_observation", done)
            for i, is_final in enumerate(final_obs_mask):
                if is_final and i < len(infos["final_observation"]):
                     real_next_obs[i] = infos["final_observation"][i]

        real_next_obs_t = torch.tensor(real_next_obs, dtype=torch.float32, device=s.device)

        # Compute intrinsic reward (surprise signal)
        with torch.no_grad():
            target_indices = torch.argmax(real_next_obs_t, dim=1)
            state_loss = torch.nn.functional.cross_entropy(
                pred_next_obs, target_indices, reduction='none'
            )
            state_surprise = state_loss.mean(dim=[1, 2])

            s.logger.scalar("debug/raw_cross_entropy_loss", state_surprise.mean().item(), s.global_step)

            real_reward_t = torch.tensor(reward, dtype=torch.float32, device=s.device)
            reward_surprise = torch.nn.functional.mse_loss(
                pred_reward, real_reward_t, reduction='none'
            )

            if s.cfg.mode == "passive":
                total_surprise = torch.zeros_like(state_surprise)
            else:
                total_surprise = state_surprise + reward_surprise

            raw_intrinsic = total_surprise * s.intrinsic_coef
            intrinsic_reward = torch.clamp(raw_intrinsic, 0.0, s.intrinsic_reward_clip)

            s.logger.scalar("debug/wm_raw_error_mean", total_surprise.mean().item(), s.global_step)
            s.logger.scalar("debug/intrinsic_reward_raw", raw_intrinsic.mean().item(), s.global_step)

            episodic_intrinsic_rewards.append(intrinsic_reward.mean().item())
            episodic_intrinsic_rewards_max.append(intrinsic_reward.max().item())

        # Update per-env episode trackers
        s.running_returns += reward
        s.running_lengths += 1

        # Store transition (extrinsic + intrinsic reward)
        total_reward = torch.tensor(reward, dtype=torch.float32, device=s.device) + intrinsic_reward

        s.buffer.add(
            obs=s.obs_t,
            actions=action,
            logprobs=logprob,
            rewards=total_reward,
            dones=torch.tensor(done, dtype=torch.float32, device=s.device),
            values=value,
            next_obs=real_next_obs_t,
        )

        s.obs_t = torch.tensor(next_obs, dtype=torch.float32, device=s.device)

        # Log finished episodes
        if np.any(done):
            done_indices = np.where(done)[0]
            for i in done_indices:
                s.logger.scalar("charts/episodic_return", s.running_returns[i], s.global_step)
                s.logger.scalar("charts/episodic_length", s.running_lengths[i], s.global_step)
                episodic_returns_this_update.append(s.running_returns[i])

                # Extract Goal Outcomes
                outcome = 0  # timeout/other
                reached_good = infos["reached_good_goal"][i] if "reached_good_goal" in infos else 0.0
                reached_bad = infos["reached_bad_goal"][i] if "reached_bad_goal" in infos else 0.0

                if reached_good > 0:
                    outcome = 1  # success
                elif reached_bad > 0:
                    outcome = -1  # failure

                s.outcome_window.append(outcome)

                if len(s.outcome_window) > 0:
                    success_rate = sum(1 for x in s.outcome_window if x == 1) / len(s.outcome_window)
                    failure_rate = sum(1 for x in s.outcome_window if x == -1) / len(s.outcome_window)
                    timeout_rate = sum(1 for x in s.outcome_window if x == 0) / len(s.outcome_window)

                    s.logger.scalar("charts/success_rate", success_rate, s.global_step)
                    s.logger.scalar("charts/failure_rate", failure_rate, s.global_step)
                    s.logger.scalar("charts/timeout_rate", timeout_rate, s.global_step)

                if "regime_id" in infos:
                    regime = infos["regime_id"][i]
                    s.logger.scalar(f"charts/r_regime_{regime}", s.running_returns[i], s.global_step)

                s.running_returns[i] = 0
                s.running_lengths[i] = 0

    # GAE computation
    with torch.no_grad():
        _, last_value = s.model.forward(s.obs_t)
    s.buffer.compute_returns_and_advantages(last_value, s.cfg.gamma, s.cfg.gae_lambda)

    # =====================================================================
    # Phase B: Update policy on real data
    # =====================================================================

    update_stats = []
    for epoch in range(s.cfg.update_epochs):
        minibatches = s.buffer.get_minibatches(s.cfg.minibatch_size, shuffle=True)
        for obs, actions, logprobs, advantages, returns, values, _, _ in minibatches:
            ppo_batch = [obs, actions, logprobs, advantages, returns, values]
            stats = ppo_update(s.model, s.optimizer, [ppo_batch], s.cfg)
            update_stats.append(stats)

    # =====================================================================
    # Phase C: Train World Model
    # =====================================================================

    wm_stats = []
    for epoch in range(s.cfg.update_epochs):
        minibatches = s.buffer.get_minibatches(s.cfg.minibatch_size, shuffle=True)
        for obs, actions, _, _, _, _, next_obs, rewards in minibatches:
            pred_next_obs, pred_reward = s.world_model(obs, actions)
            target_indices = torch.argmax(next_obs, dim=1)
            loss_state = torch.nn.functional.cross_entropy(pred_next_obs, target_indices)
            loss_reward = torch.nn.functional.mse_loss(pred_reward, rewards)
            wm_loss = loss_state + loss_reward

            s.wm_optimizer.zero_grad()
            wm_loss.backward()
            s.wm_optimizer.step()

            wm_stats.append({
                "world_model/loss_total": wm_loss.item(),
                "world_model/loss_state": loss_state.item(),
                "world_model/loss_reward": loss_reward.item(),
            })

    # =====================================================================
    # Phase D: Dream & update policy on imagined data
    # =====================================================================

    dream_stats = []
    dream_buffer = None
    if s.imagined_horizon > 0:
        rand_time_idxs = torch.randint(0, s.cfg.num_steps, (s.num_envs,), device=s.device)
        env_idxs = torch.arange(s.num_envs, device=s.device)
        start_states = s.buffer.obs[rand_time_idxs, env_idxs]

        imagined_trajectories = s.world_model.generate_imagined_trajectories(
            policy_net=s.model,
            start_states=start_states,
            horizon=s.imagined_horizon
        )

        dream_buffer = RolloutBuffer(s.imagined_horizon, s.num_envs, s.obs_shape, s.device)
        for t_step, traj in enumerate(imagined_trajectories):
            dream_buffer.add(
                obs=traj["obs"],
                actions=traj["actions"],
                logprobs=traj["logprobs"],
                rewards=traj["rewards"],
                dones=traj["dones"],
                values=traj["values"],
                next_obs=traj["next_obs"]
            )

        if imagined_trajectories:
            last_dream_obs = imagined_trajectories[-1]["next_obs"]
            with torch.no_grad():
                _, last_dream_value = s.model.forward(last_dream_obs)
            dream_buffer.compute_returns_and_advantages(last_dream_value, s.cfg.gamma, s.cfg.gae_lambda)

        for epoch in range(1):
            minibatches = dream_buffer.get_minibatches(s.cfg.minibatch_size, shuffle=True)
            for obs, actions, logprobs, advantages, returns, values, _, _ in minibatches:
                ppo_batch = [obs, actions, logprobs, advantages, returns, values]
                ds = ppo_update(s.model, s.optimizer, [ppo_batch], s.cfg)
                dream_stats.append(ds)

    # =====================================================================
    # Logging
    # =====================================================================

    avg_stats = {k: np.mean([st[k] for st in update_stats]) for k in update_stats[0]} if update_stats else {}
    avg_wm_stats = {k: np.mean([st[k] for st in wm_stats]) for k in wm_stats[0]} if wm_stats else {}

    for k, v in avg_stats.items():
        s.logger.scalar(k, v, s.global_step)
    for k, v in avg_wm_stats.items():
        s.logger.scalar(k, v, s.global_step)

    if dream_stats and dream_buffer is not None:
        avg_dream_stats = {f"ppo/imagined_{k.split('/')[-1]}": np.mean([ds[k] for ds in dream_stats]) for k in dream_stats[0]}
        for k, v in avg_dream_stats.items():
            s.logger.scalar(k, v, s.global_step)
        s.logger.scalar("ppo/imagined_value_mean", dream_buffer.values.mean().item(), s.global_step)
        s.logger.scalar("ppo/imagined_return_mean", dream_buffer.returns.mean().item(), s.global_step)

    mean_intrinsic = np.mean(episodic_intrinsic_rewards) if episodic_intrinsic_rewards else 0.0
    max_intrinsic = np.max(episodic_intrinsic_rewards_max) if episodic_intrinsic_rewards_max else 0.0
    buffer_rewards_abs_mean = s.buffer.rewards.abs().mean().item()

    s.logger.scalar("ppo/intrinsic_reward_mean", mean_intrinsic, s.global_step)
    s.logger.scalar("ppo/intrinsic_reward_max", max_intrinsic, s.global_step)

    if buffer_rewards_abs_mean > 1e-6:
        s.logger.scalar("ppo/intrinsic_reward_ratio", mean_intrinsic / buffer_rewards_abs_mean, s.global_step)
    else:
        s.logger.scalar("ppo/intrinsic_reward_ratio", 0.0, s.global_step)

    s.logger.scalar("charts/learning_rate", lrnow, s.global_step)
    s.logger.scalar("charts/heartbeat", s.global_step, s.global_step)
    s.logger.scalar("charts/reward_step_mean", s.buffer.rewards.mean().item(), s.global_step)
    s.logger.scalar("charts/reward_step_max", s.buffer.rewards.max().item(), s.global_step)
    s.logger.scalar("charts/reward_step_std", s.buffer.rewards.std().item(), s.global_step)

    sps = int(s.global_step / max(1e-9, (time.time() - s.start_time)))
    s.logger.scalar("charts/SPS", sps, s.global_step)

    # Checkpointing
    if update % s.save_every_updates == 0 or update == s.num_updates:
        ckpt_path = os.path.join(s.save_dir, f"{s.run_name}_update{update}.pt")
        torch.save(
            {
                "model_state_dict": s.model.state_dict(),
                "optimizer_state_dict": s.optimizer.state_dict(),
                "world_model_state_dict": s.world_model.state_dict(),
                "wm_optimizer_state_dict": s.wm_optimizer.state_dict(),
                "cfg": s.cfg.__dict__,
                "global_step": s.global_step,
            },
            ckpt_path,
        )
        print(f"[save] {ckpt_path}")

    if update % 10 == 0:
        print(f"update {update}/{s.num_updates} | step={s.global_step} | SPS={sps}")

    s.current_update += 1

    # -----------------------------------------------------------------
    # Build stats dict for the Brain
    # -----------------------------------------------------------------

    # Outcome rates from the rolling window
    n_outcomes = len(s.outcome_window)
    success_rate = sum(1 for x in s.outcome_window if x == 1) / max(n_outcomes, 1)
    failure_rate = sum(1 for x in s.outcome_window if x == -1) / max(n_outcomes, 1)

    mean_return = float(np.mean(episodic_returns_this_update)) if episodic_returns_this_update else 0.0

    return {
        "done": update >= s.num_updates,
        "global_step": s.global_step,
        "update": update,

        # Performance signals
        "mean_episodic_return": mean_return,
        "success_rate": success_rate,
        "failure_rate": failure_rate,

        # Surprise / world model signals
        "mean_surprise": float(np.mean(episodic_intrinsic_rewards)) if episodic_intrinsic_rewards else 0.0,
        "wm_loss_state": avg_wm_stats.get("world_model/loss_state", 0.0),
        "wm_loss_reward": avg_wm_stats.get("world_model/loss_reward", 0.0),

        # Policy signals
        "policy_entropy": avg_stats.get("loss/entropy", 0.0),
        "policy_loss": avg_stats.get("loss/policy", 0.0),
        "value_loss": avg_stats.get("loss/value", 0.0),

        # Current hyperparameter values
        "current_lr": lrnow,
        "current_ent_coef": s.cfg.ent_coef,
        "current_intrinsic_coef": s.intrinsic_coef,
        "current_imagined_horizon": s.imagined_horizon,
    }


def close_inner_training(state: InnerTrainState):
    """Clean up environments and logger."""
    state.envs.close()
    if state.logger is not None:
        state.logger.close()


# =========================================================================
# Original train_ppo() — now delegates to init/step functions
# =========================================================================

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
):
    """
    Main Dyna-PPO training loop.

    Each update cycle has three phases:
        A) Collect real experience (with intrinsic curiosity reward)
        B) Train World Model on real transitions (supervised)
        C) Generate imagined trajectories and update policy on dreams
    """

    state = init_inner_training(
        env_id=env_id,
        cfg=cfg,
        steps_per_regime=steps_per_regime,
        episodes_per_regime=episodes_per_regime,
        start_regime=start_regime,
        run_name=run_name,
        save_dir=save_dir,
        save_every_updates=save_every_updates,
        anneal_lr=anneal_lr,
        resume_path=resume_path,
        intrinsic_coef=intrinsic_coef,
        intrinsic_reward_clip=intrinsic_reward_clip,
        imagined_horizon=imagined_horizon,
        wm_lr=wm_lr,
    )

    print(f"Training on {state.device} with {state.num_envs} envs "
          f"for {state.num_updates} updates (starting from update {state.current_update}).")

    while True:
        stats = run_inner_update(state)
        if stats["done"]:
            break

    close_inner_training(state)
