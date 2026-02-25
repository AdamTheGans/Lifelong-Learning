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
from collections import deque, defaultdict

from lifelong_learning.agents.ppo.ppo import PPOConfig, ppo_update
from lifelong_learning.agents.ppo.network import CNNActorCritic
from lifelong_learning.agents.ppo.world_model import SimpleWorldModel
from lifelong_learning.agents.ppo.mowm import MixtureOfWorldModels
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
    dreaming_ratio: float = 1.0,
    oracle_mode: bool = False,
):
    """
    Main Dyna-PPO training loop.

    Each update cycle has three phases:
        A) Collect real experience (with intrinsic curiosity reward)
        B) Train World Model on real transitions (supervised)
        C) Generate imagined trajectories and update policy on dreams
    """

    print("MoWM Dyna-PPO Trainer Version: 0.7.17")
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

    model = CNNActorCritic(obs_shape, n_actions, max_regimes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    world_model = MixtureOfWorldModels(obs_shape, n_actions, max_regimes=2).to(device)
    wm_optimizers = [torch.optim.Adam(world_model.models[0].parameters(), lr=wm_lr)]
    buffer = RolloutBuffer(cfg.num_steps, num_envs, obs_shape, device)

    # -------------------------------------------------------------------------
    # Oracle Mode Initialization
    # -------------------------------------------------------------------------
    if oracle_mode:
        dreaming_ratio = 0.0
        # Give the MoWM a second world model immediately
        new_model = SimpleWorldModel(obs_shape, n_actions, world_model.hidden_dim).to(device)
        world_model.models.append(new_model)
        
        world_model.ema_losses.append(1.0)
        world_model.routing_emas.append(1.0)
        world_model.ema_history.append(deque([1.0], maxlen=10))
        world_model.has_mastered.append(False)
        world_model.steps_under_threshold.append(0)
        world_model.spawn_steps.append(0)
        
        wm_optimizers.append(torch.optim.Adam(world_model.models[-1].parameters(), lr=wm_lr))

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
            wm_state = ckpt["world_model_state_dict"]
            num_models = 1 + max([int(k.split('.')[1]) for k in wm_state.keys() if k.startswith('models.')] + [-1])
            while len(world_model.models) < num_models:
                new_model = SimpleWorldModel(obs_shape, n_actions, world_model.hidden_dim).to(device)
                world_model.models.append(new_model)
            world_model.load_state_dict(wm_state)
            print("World Model state loaded.")
        else:
            print("WARNING: World Model state not found in checkpoint.")

        if "wm_optimizers_state_dict" in ckpt:
            wm_optimizers = []
            for i, opt_state in enumerate(ckpt["wm_optimizers_state_dict"]):
                opt = torch.optim.Adam(world_model.models[i].parameters(), lr=wm_lr)
                opt.load_state_dict(opt_state)
                wm_optimizers.append(opt)
            print("World Model Optimizers state loaded.")
        elif "wm_optimizer_state_dict" in ckpt:
            wm_optimizers[0].load_state_dict(ckpt["wm_optimizer_state_dict"])
            print("Legacy World Model Optimizer state loaded.")

        if "mowm_state" in ckpt:
            world_model.active_regime_id = ckpt["mowm_state"]["active_regime_id"]
            if "ema_losses" in ckpt["mowm_state"]:
                world_model.ema_losses = ckpt["mowm_state"]["ema_losses"]
            elif "ema_loss" in ckpt["mowm_state"]:
                # Backwards compatibility
                world_model.ema_alpha = ckpt["mowm_state"]["ema_alpha"]
            
            if "routing_emas" in ckpt["mowm_state"]:
                world_model.routing_emas = ckpt["mowm_state"]["routing_emas"]
            else:
                world_model.routing_emas = list(world_model.ema_losses)
                
            if "fast_routing_emas" in ckpt["mowm_state"]:
                world_model.fast_routing_emas = ckpt["mowm_state"]["fast_routing_emas"]
            else:
                world_model.fast_routing_emas = list(world_model.routing_emas)
            
            if "ema_history" in ckpt["mowm_state"]:
                # Convert list of lists back to list of deques
                world_model.ema_history = [deque(h, maxlen=10) for h in ckpt["mowm_state"]["ema_history"]]
            else:
                # Backwards compatibility
                world_model.ema_history = [deque([ema], maxlen=10) for ema in world_model.ema_losses]
            
            if "has_mastered" in ckpt["mowm_state"]:
                world_model.has_mastered = ckpt["mowm_state"]["has_mastered"]
                world_model.steps_under_threshold = ckpt["mowm_state"]["steps_under_threshold"]
            else:
                # Backwards compatibility: Assume models initialized before this feature have already mastered
                world_model.has_mastered = [True] * len(world_model.models)
                world_model.steps_under_threshold = [world_model.mastery_buffer_steps] * len(world_model.models)

            if "spawn_steps" in ckpt["mowm_state"]:
                world_model.spawn_steps = ckpt["mowm_state"]["spawn_steps"]
            else:
                world_model.spawn_steps = [0] * len(world_model.models)

            if "force_active_until" in ckpt["mowm_state"]:
                world_model.force_active_until = ckpt["mowm_state"]["force_active_until"]

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

    # Initialize global regime tracker (used throughout the update loop)
    current_regime = torch.zeros(num_envs, dtype=torch.long, device=device)

    # State reservoir for generative replay
    state_reservoir = defaultdict(lambda: deque(maxlen=1000))
    # Temporary storage for states in the current episode, tracked per environment
    temp_env_states = [[] for _ in range(num_envs)]

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def collect_real_experience(global_step):
        """Phase A: Interact with real environments and collect transitions."""
        nonlocal obs_t, running_returns, running_lengths, temp_env_states

        episodic_intrinsic_rewards = []
        episodic_intrinsic_rewards_max = []
        spawn_occurred = False
        last_true_regime = None  # Track ground truth regime for transition logging

        # Diagnostics: Context-Ignorance Lazy Policy Check
        with torch.no_grad():
            r0 = torch.zeros(num_envs, dtype=torch.long, device=device)
            r1 = torch.ones(num_envs, dtype=torch.long, device=device)
            logits0, _ = model(obs_t, r0)
            logits1, _ = model(obs_t, r1)
            p0 = torch.nn.functional.softmax(logits0, dim=-1)
            p1 = torch.nn.functional.softmax(logits1, dim=-1)
            # KL(P0 || P1) = sum(P0 * log(P0 / P1))
            kl_div = (p0 * (torch.log(p0 + 1e-8) - torch.log(p1 + 1e-8))).sum(dim=-1).mean().item()

        for t in range(cfg.num_steps):
            global_step += num_envs

            with torch.no_grad():
                action, logprob, entropy, value = model.act(obs_t, current_regime)
                pred_next_obs, pred_reward = world_model(obs_t, action, world_model.active_regime_id)

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Log ground truth regime transitions
            if "regime_id" in infos:
                true_regime = infos["regime_id"][0]
                if last_true_regime is not None and true_regime != last_true_regime:
                    print(f"\n[ENV] True Regime Switch: Regime {last_true_regime} → Regime {true_regime} at step {global_step}.")
                last_true_regime = true_regime

            # Handle autoreset: use final_observation for surprise calc on done envs
            real_next_obs = next_obs.copy()
            if "final_observation" in infos:
                # Use _final_observation mask if available, or assume all done envs have it
                final_obs_mask = infos.get("_final_observation", done)
                for i, is_final in enumerate(final_obs_mask):
                    if is_final and i < len(infos["final_observation"]):
                         real_next_obs[i] = infos["final_observation"][i]
            
            real_next_obs_t = torch.tensor(real_next_obs, dtype=torch.float32, device=device)

            # Infer best regime and check spawn
            real_reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
            if oracle_mode:
                true_regime = infos["regime_id"][0]
                world_model.active_regime_id = true_regime
            else:
                best_id, lowest_loss, raw_losses = world_model.infer_regime(obs_t, action, real_next_obs_t, real_reward_t, global_step)
                old_active_id = world_model.active_regime_id
                did_spawn = world_model.check_and_spawn(raw_losses, best_id, global_step)
                if did_spawn:
                    wm_optimizers.append(torch.optim.Adam(world_model.models[-1].parameters(), lr=wm_lr))
                    # ROLLBACK: Undo corruption on the old model before shelving it
                    world_model.rollback_safe_state(old_active_id, world_model.models[old_active_id], wm_optimizers[old_active_id], global_step=global_step)
                    spawn_occurred = True
                else:
                    if best_id != world_model.active_regime_id:
                        # ROLLBACK: Undo corruption on the outgoing model before switching away
                        old_id = world_model.active_regime_id
                        world_model.rollback_safe_state(old_id, world_model.models[old_id], wm_optimizers[old_id], global_step=global_step)
                        print(f"\n[MoWM] Regime Switch: Model {old_id} → Model {best_id} at step {global_step}.")
                    world_model.active_regime_id = best_id
            
            # Update the current_regime tensor for the buffer and for next step's policy
            current_regime.fill_(world_model.active_regime_id)

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
                extrinsic_rewards=torch.tensor(reward, dtype=torch.float32, device=device),
                dones=torch.tensor(done, dtype=torch.float32, device=device),
                values=value,
                next_obs=real_next_obs_t,
                regime_ids=current_regime,
            )

            # Add to temporary state storage for the current episode
            for i in range(num_envs):
                temp_env_states[i].append(obs_t[i].clone().cpu())

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
                    
                    if outcome == 1:
                        # Append the successful episode's states to the permanent reservoir
                        state_reservoir[world_model.active_regime_id].extend(temp_env_states[i])

                    # Always clear temporary states on episode end
                    temp_env_states[i].clear()

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
            "spawn_occurred": spawn_occurred,
            "regime_kl_divergence": kl_div,
        }

    def update_world_model():
        """Phase B: Supervised training on real transitions."""
        active_id = world_model.active_regime_id
        # In Oracle mode, train both models because they may both appear in the buffer frequently
        model_ids_to_train = [0, 1] if oracle_mode else [active_id]
        
        active_wm_stats = []
        for m_id in model_ids_to_train:
            wm_stats = []
            epoch_losses = []
            m_model = world_model.models[m_id]
            m_opt = wm_optimizers[m_id]

            for epoch in range(cfg.update_epochs):
                minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
                for obs, actions, _, _, _, _, next_obs, rewards, extrinsic_rewards, regime_ids in minibatches:
                    mask = (regime_ids == m_id)
                    if not mask.any():
                        continue

                    m_obs = obs[mask]
                    m_actions = actions[mask]
                    m_next_obs = next_obs[mask]
                    m_rewards = extrinsic_rewards[mask]

                    pred_next_obs, pred_reward = m_model(m_obs, m_actions)

                    # State loss: CrossEntropy on one-hot → class indices
                    target_indices = torch.argmax(m_next_obs, dim=1)
                    loss_state = torch.nn.functional.cross_entropy(pred_next_obs, target_indices, reduction='none')
                    loss_state_mean = loss_state.mean()

                    # Reward loss: MSE
                    loss_reward = torch.nn.functional.mse_loss(pred_reward, m_rewards, reduction='none')
                    loss_reward_mean = loss_reward.mean()

                    # Backpropagate on means to keep gradients stable
                    wm_loss_mean = loss_state_mean + loss_reward_mean

                    m_opt.zero_grad()
                    wm_loss_mean.backward()
                    m_opt.step()

                    # Track the MEAN loss to establish the EMA baseline correctly!
                    with torch.no_grad():
                        batch_mean_loss = wm_loss_mean.item()
                    
                    epoch_losses.append(batch_mean_loss)

                    wm_stats.append({
                        "world_model/loss_total": wm_loss_mean.item(),
                        "world_model/loss_mean": batch_mean_loss,
                        "world_model/loss_state": loss_state_mean.item(),
                        "world_model/loss_reward": loss_reward_mean.item(),
                    })

            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                # Temporarily swap active_regime_id to update the right EMA
                original_id = world_model.active_regime_id
                world_model.active_regime_id = m_id
                world_model.update_ema(avg_loss, steps_added=num_envs * cfg.num_steps, global_step=global_step)
                world_model.active_regime_id = original_id

                # Refresh safe state if this model is currently stable
                world_model.refresh_safe_state(m_id, m_model, m_opt, global_step=global_step)
                
            if m_id == active_id:
                active_wm_stats = wm_stats

        # Phase B-2: Shadow Tracking (Telemetry) for all models
        # Grab exactly one minibatch to act as a representative sample of the current transition distribution
        minibatches = buffer.get_minibatches(cfg.minibatch_size, shuffle=True)
        try:
            shadow_obs, shadow_acts, _, _, _, _, shadow_next, shadow_rews, shadow_ext_rews, _ = next(minibatches)
        except StopIteration:
            pass # Failsafe if buffer completely empty
        else:
            with torch.no_grad():
                shadow_targets = torch.argmax(shadow_next, dim=1)
                for i, model in enumerate(world_model.models):
                    p_next, p_rew = model(shadow_obs, shadow_acts)
                    l_s = torch.nn.functional.cross_entropy(p_next, shadow_targets, reduction='none').mean(dim=[1,2])
                    l_r = torch.nn.functional.mse_loss(p_rew, shadow_ext_rews, reduction='none')
                    shadow_loss = (l_s + l_r).mean().item() # We log mean() for smooth telemetry visibility
                    # Inject it into active_wm_stats so it's guaranteed to be logged
                    if active_wm_stats:
                        for stat in active_wm_stats:
                            stat[f"mowm/ShadowLoss_Model_{i}"] = shadow_loss

        return active_wm_stats if active_wm_stats else [{"world_model/loss_total": 0.0, "world_model/loss_state": 0.0, "world_model/loss_reward": 0.0}]

    def generate_dream_experience(reservoir, min_states=16):
        """Phase C: Generate imagined trajectories using WM as simulator."""
        num_models = len(world_model.models)
        if num_models > 1:
            inactive_regimes = [i for i in range(num_models) if i != world_model.active_regime_id]
            dream_regime_id = inactive_regimes[torch.randint(0, len(inactive_regimes), (1,)).item()]
        else:
            dream_regime_id = world_model.active_regime_id
            
        dream_regime_tensor = torch.full((num_envs,), dream_regime_id, dtype=torch.long, device=device)

        # Sample start states from reservoir if available, else abort dream
        if len(reservoir[dream_regime_id]) >= min_states:
            res_list = list(reservoir[dream_regime_id])
            idxs = np.random.choice(len(res_list), num_envs, replace=True)
            start_states = torch.stack([res_list[i] for i in idxs]).to(device)
        else:
            # Prevent Out-Of-Distribution Dream Seeds by aborting dream sequence
            print(f"[Warning] Aborting dream sequence: insufficient seed states for Regime {dream_regime_id} ({len(reservoir[dream_regime_id])}/{min_states}).")
            return None, []

        # Roll out imagined trajectories
        imagined_trajectories = world_model.models[dream_regime_id].generate_imagined_trajectories(
            policy_net=model,
            start_states=start_states,
            horizon=imagined_horizon,
            regime_tensor=dream_regime_tensor,
            reservoir_states=list(reservoir[dream_regime_id])
        )

        # Build a dream buffer from the imagined data
        dream_buffer = RolloutBuffer(imagined_horizon, num_envs, obs_shape, device)
        for t, traj in enumerate(imagined_trajectories):
            dream_buffer.add(
                obs=traj["obs"],
                actions=traj["actions"],
                logprobs=traj["logprobs"],
                rewards=traj["rewards"],
                extrinsic_rewards=traj["rewards"],
                dones=traj["dones"],
                values=traj["values"],
                next_obs=traj["next_obs"],
                regime_ids=dream_regime_tensor
            )

        # Bootstrap value for GAE computation
        if imagined_trajectories:
            last_dream_obs = imagined_trajectories[-1]["next_obs"]
            with torch.no_grad():
                _, last_dream_value = model.forward(last_dream_obs, dream_regime_tensor)

            dream_buffer.compute_returns_and_advantages(last_dream_value, cfg.gamma, cfg.gae_lambda)

        return dream_buffer, imagined_trajectories

    def update_policy(buffers, epochs):
        """Run PPO updates on multiple buffers (real and/or imagined) for N epochs."""
        if not isinstance(buffers, list):
            buffers = [buffers]
            
        update_stats = []
        for epoch in range(epochs):
            all_obs, all_actions, all_logprobs, all_advantages = [], [], [], []
            all_returns, all_values, all_regime_ids = [], [], []
            
            for buf in buffers:
                batch_size = buf.num_steps * buf.num_envs
                b_obs = buf.obs.reshape((batch_size,) + buf.obs_shape)
                b_actions = buf.actions.reshape(batch_size)
                b_logprobs = buf.logprobs.reshape(batch_size)
                b_advantages = buf.advantages.reshape(batch_size)
                b_returns = buf.returns.reshape(batch_size)
                b_values = buf.values.reshape(batch_size)
                b_regime_ids = buf.regime_ids.reshape(batch_size)
                
                all_obs.append(b_obs)
                all_actions.append(b_actions)
                all_logprobs.append(b_logprobs)
                all_advantages.append(b_advantages)
                all_returns.append(b_returns)
                all_values.append(b_values)
                all_regime_ids.append(b_regime_ids)
                
            c_obs = torch.cat(all_obs, dim=0)
            c_actions = torch.cat(all_actions, dim=0)
            c_logprobs = torch.cat(all_logprobs, dim=0)
            c_advantages = torch.cat(all_advantages, dim=0)
            c_returns = torch.cat(all_returns, dim=0)
            c_values = torch.cat(all_values, dim=0)
            c_regime_ids = torch.cat(all_regime_ids, dim=0)
            
            # Normalize advantages AFTER concatenation across all buffers
            # so real and dream signals retain their relative magnitudes.
            std = c_advantages.std()
            if std < 1e-6:
                std = torch.tensor(1.0, device=c_advantages.device)
            c_advantages = (c_advantages - c_advantages.mean()) / (std + 1e-8)
            
            total_size = c_obs.size(0)
            idxs = np.arange(total_size)
            np.random.shuffle(idxs)
            
            for start in range(0, total_size, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                ppo_batch = [
                    c_obs[mb], c_actions[mb], c_logprobs[mb], c_advantages[mb], 
                    c_returns[mb], c_values[mb], c_regime_ids[mb]
                ]
                stats = ppo_update(model, optimizer, [ppo_batch], cfg)
                update_stats.append(stats)
                
        return update_stats

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    for update in range(start_update, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            optimizer.param_groups[0]["lr"] = lrnow
        else:
            lrnow = cfg.lr

        buffer.reset()

        # Phase A: Collect real experience
        collect_stats = collect_real_experience(global_step)
        global_step = collect_stats["global_step"]

        with torch.no_grad():
            _, last_value = model.forward(obs_t, current_regime)
        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)

        # Phase B: Train World Model (trains on freshly collected buffer)
        wm_stats = update_world_model()

        # Phase C: Dream and build mixed buffers (skipped in passive mode)
        dream_buffers = []
        if cfg.mode == "dyna" and imagined_horizon > 0:
            current_dreaming_ratio = 1.0 if len(world_model.models) > 1 else 0.25
            num_dream_rollouts = int(max(1, cfg.num_steps // imagined_horizon) * current_dreaming_ratio)
            if num_dream_rollouts > 0:
                for _ in range(num_dream_rollouts):
                    db, _ = generate_dream_experience(state_reservoir)
                    if db is not None:
                        dream_buffers.append(db)

        # Phase D: Update policy on mixed real + imagined data
        buffers_to_train = [buffer] + dream_buffers
        update_stats = update_policy(buffers_to_train, cfg.update_epochs)

        # -----------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------

        avg_stats = {k: np.mean([s[k] for s in update_stats]) for k in update_stats[0]} if update_stats else {}
        avg_wm_stats = {k: np.mean([s[k] for s in wm_stats]) for k in wm_stats[0]} if wm_stats else {}

        for k, v in avg_stats.items():
            logger.scalar(k, v, global_step)

        for k, v in avg_wm_stats.items():
            logger.scalar(k, v, global_step)

        # MoWM metrics
        logger.scalar("mowm/active_regime_id", world_model.active_regime_id, global_step)
        logger.scalar("mowm/num_regimes", len(world_model.models), global_step)
        logger.scalar("mowm/ema_loss_active", world_model.ema_losses[world_model.active_regime_id], global_step)
        logger.scalar("mowm/routing_ema_active", world_model.routing_emas[world_model.active_regime_id], global_step)
        
        for i, ema_l in enumerate(world_model.ema_losses):
            logger.scalar(f"mowm/ema_loss_model_{i}", ema_l, global_step)
            logger.scalar(f"mowm/routing_ema_model_{i}", world_model.routing_emas[i], global_step)
            logger.scalar(f"mowm/has_mastered_model_{i}", float(world_model.has_mastered[i]), global_step)
            
        avg_total_loss = avg_wm_stats.get("world_model/loss_total", 0.0)
        avg_max_loss = avg_wm_stats.get("world_model/loss_max", avg_total_loss)
        logger.scalar("mowm/epoch_avg_loss", avg_total_loss, global_step)
        logger.scalar("mowm/epoch_avg_max_loss", avg_max_loss, global_step)
        logger.scalar("mowm/spawn_occurred", float(collect_stats["spawn_occurred"]), global_step)

        if dream_buffers:
            avg_dream_val = np.mean([db.values.mean().item() for db in dream_buffers])
            avg_dream_ret = np.mean([db.returns.mean().item() for db in dream_buffers])
            logger.scalar("ppo/imagined_value_mean", avg_dream_val, global_step)
            logger.scalar("ppo/imagined_return_mean", avg_dream_ret, global_step)

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

        # Log "Laziness" Diagnostic Metric
        logger.scalar("ppo/regime_kl_divergence", collect_stats["regime_kl_divergence"], global_step)

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
            wm_opts_state = [opt.state_dict() for opt in wm_optimizers]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "world_model_state_dict": world_model.state_dict(),
                    "wm_optimizers_state_dict": wm_opts_state,
                    "mowm_state": {
                        "active_regime_id": world_model.active_regime_id,
                        "routing_emas": world_model.routing_emas,
                        "fast_routing_emas": world_model.fast_routing_emas,
                        "ema_alpha": world_model.ema_alpha,
                        "has_mastered": world_model.has_mastered,
                        "steps_under_threshold": world_model.steps_under_threshold,
                        "spawn_steps": world_model.spawn_steps,
                        "force_active_until": world_model.force_active_until,
                    },
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
