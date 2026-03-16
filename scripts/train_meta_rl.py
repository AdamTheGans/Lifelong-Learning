from __future__ import annotations

import argparse
import os
import warnings
from collections import deque

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

import time
import numpy as np
import torch
import gymnasium as gym

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.sequence_memory_buffer import SequenceMemoryBuffer
from lifelong_learning.agents.ppo.recurrent_world_model import RecurrentWorldModel
from lifelong_learning.agents.ppo.context_aware_network import ContextAwarePPONetwork
from lifelong_learning.agents.ppo.meta_rl_trainer import MetaRLTrainer
from lifelong_learning.envs.make_env import make_env
from lifelong_learning.utils.seeding import seed_everything
from lifelong_learning.utils.logger import TBLogger


def main():
    p = argparse.ArgumentParser(description="Train Context-Aware Meta-RL (Solution 5)")
    p.add_argument("--env_id", type=str, default="MiniGrid-DualGoal-8x8-v0")
    p.add_argument("--total_timesteps", type=int, default=2250_000)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # Regime switching schedule
    p.add_argument("--steps_per_regime", type=int, default=18500, help="Steps per env before regime switch")
    p.add_argument("--total_env_regimes", type=int, default=4, choices=[2, 4], help="How many underlying env regimes to cycle through.")

    p.add_argument("--run_name", type=str, default=None)
    
    # Meta-RL Specific Configs
    p.add_argument("--wm_warmup_steps", type=int, default=75000)
    p.add_argument("--dream_horizon", type=int, default=5)
    p.add_argument("--wm_lr", type=float, default=1e-4)
    p.add_argument("--ppo_lr", type=float, default=3e-4)

    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()

    if args.run_name is None:
        run_name = f"meta_rl_{args.env_id}_s{args.seed}_{int(time.time())}"
    else:
        run_name = args.run_name

    logger = TBLogger(run_name=run_name)

    print(f"==================================================")
    print(f" Starting Context-Aware Meta-RL (Solution 5) Run")
    print(f" Run Name: {run_name}")
    print(f" Device: {device}")
    print(f" Timesteps: {args.total_timesteps}")
    print(f"==================================================")

    # 1. Environment Setup
    def make_thunk(i: int):
        def thunk():
            return make_env(
                env_id=args.env_id,
                seed=args.seed + i,
                steps_per_regime=args.steps_per_regime,
                episodes_per_regime=None, # Only step-based implemented
                start_regime=0,
                record_stats=False,
                total_regimes=args.total_env_regimes,
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(args.num_envs)])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    # 2. Component Initialization
    cfg = {
        'num_envs': args.num_envs,
        'num_steps': args.num_steps,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 4,
        'minibatch_size': 256,
        'clip_coef': 0.2,
        'ent_coef': 0.05, # Increased from 0.01 to prevent entropy collapse
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'wm_batch_size': 16, # Controls the 50/50 split chunking
        'wm_epochs': 2,
        'wm_warmup_steps': args.wm_warmup_steps,
        'dream_horizon': args.dream_horizon,
        'dream_batch_size': args.num_envs 
    }

    # Use default dimensions (21, 8, 8), 256-D context, 3 actions
    ppo_net = ContextAwarePPONetwork(obs_shape=obs_shape, context_dim=256, n_actions=n_actions).to(device)
    wm = RecurrentWorldModel(obs_shape=obs_shape, hidden_dim=256, n_actions=n_actions).to(device)
    buffer = SequenceMemoryBuffer(max_capacity=4000, seq_len=30)
    
    ppo_opt = torch.optim.Adam(ppo_net.parameters(), lr=args.ppo_lr, eps=1e-5)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=args.wm_lr)

    trainer = MetaRLTrainer(ppo_net, wm, buffer, ppo_opt, wm_opt, cfg)

    # 3. Master Training Loop
    global_step = 0
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    
    print(f"Starting training loop for {num_updates} updates...")
    
    # Pre-reset environment. SyncVectorEnv returns a numpy array, we need a Tensor.
    obs, info = envs.reset(seed=args.seed)
    
    # Modify MetaRLTrainer's train_iteration to accept state locally to handle SyncVectorEnv
    # Actually, MetaRLTrainer logic assumes env.reset() returns a tensor or expects DummyEnv behavior.
    # Let's adjust env wrapper directly. SyncVectorEnv yields ndarrays.
    class EnvTensorWrapper:
        def __init__(self, envs, device, logger, total_regimes=4):
            self.envs = envs
            self.device = device
            self.logger = logger
            self.total_regimes = total_regimes
            
            # Tracking
            self.running_returns = np.zeros(envs.num_envs)
            self.running_lengths = np.zeros(envs.num_envs, dtype=int)
            self.outcome_window = deque(maxlen=100)
            # Per-regime outcome windows for continual learning diagnostics
            self.outcome_windows = {r: deque(maxlen=100) for r in range(total_regimes)}
            self.global_step = 0
            
            # Auto-reset state tracker
            self._obs = None
            
        def reset(self):
            obs, info = self.envs.reset()
            self._obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            return self._obs
            
        def step(self, action_tensor):
            # SyncVectorEnv expects numpy array
            action_np = action_tensor.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = self.envs.step(action_np)
            
            done = np.logical_or(terminated, truncated)
            
            self.running_returns += reward
            self.running_lengths += 1
            
            for i in range(self.envs.num_envs):
                if done[i]:
                    self.logger.scalar("charts/episodic_return", self.running_returns[i], self.global_step)
                    self.logger.scalar("charts/episodic_length", self.running_lengths[i], self.global_step)
                    
                    # Outcome
                    outcome = 0
                    if infos.get("reached_good_goal", [False]*self.envs.num_envs)[i]: outcome = 1
                    elif infos.get("reached_bad_goal", [False]*self.envs.num_envs)[i]: outcome = -1
                    self.outcome_window.append(outcome)
                    
                    # Per-regime outcome window for continual learning diagnostics
                    regime_id = infos.get("regime_id", np.zeros(self.envs.num_envs, dtype=int))[i]
                    if regime_id < self.total_regimes:
                        self.outcome_windows[regime_id].append(outcome)
                    
                    if len(self.outcome_window) > 0:
                        self.logger.scalar("charts/success_rate", sum(1 for x in self.outcome_window if x == 1)/len(self.outcome_window), self.global_step)
                        self.logger.scalar("charts/failure_rate", sum(1 for x in self.outcome_window if x == -1)/len(self.outcome_window), self.global_step)
                        self.logger.scalar("charts/timeout_rate", sum(1 for x in self.outcome_window if x == 0)/len(self.outcome_window), self.global_step)

                    # Regime-specific return and success rate (proves retention across regimes)
                    if "regime_id" in infos:
                        rid = infos["regime_id"][i]
                        self.logger.scalar(f"charts/return_regime_{rid}", self.running_returns[i], self.global_step)
                        self.logger.scalar(f"charts/r_regime_{rid}", self.running_returns[i], self.global_step)
                    for r in range(self.total_regimes):
                        w = self.outcome_windows[r]
                        if len(w) > 0:
                            self.logger.scalar(f"charts/success_rate_regime_{r}", sum(1 for x in w if x == 1) / len(w), self.global_step)
                        
                    self.running_returns[i] = 0
                    self.running_lengths[i] = 0
            
            # Identify dummy steps and handle true "next" states if necessary.
            # MetaRLTrainer's buffer chunking handles dones naturally.
            
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_t = torch.tensor(done, dtype=torch.bool, device=self.device)
            
            self._obs = next_obs_t
            return next_obs_t, reward_t, done_t

    wrapped_envs = EnvTensorWrapper(envs, device, logger, total_regimes=args.total_env_regimes)

    for update in range(1, num_updates + 1):
        # 1. Execute 1 full iteration (Phase 1 -> Phase 4)
        stats = trainer.train_iteration(wrapped_envs)
        global_step = stats["global_step"]
        wrapped_envs.global_step = global_step
        
        # 2. Basic Logging
        sps = int(global_step / max(1e-9, (time.time() - start_time)))
        logger.scalar("charts/SPS", sps, global_step)
        logger.scalar("charts/heartbeat", global_step, global_step)
        
        for k, v in stats.items():
            if k != "global_step":
                logger.scalar(k, v, global_step)
        
        if update % 10 == 0:
            ctx_norm = stats.get("ppo/context_ht_norm", 0.0)
            ent = stats.get("ppo/entropy", 0.0)
            rew_max = stats.get("charts/reward_step_max", 0.0)
            v_loss = stats.get("ppo/value_loss_live", 0.0)
            p_loss = stats.get("ppo/policy_loss_live", 0.0)
            print(f"Update {update}/{num_updates} | step={global_step} | SPS={sps} | Entropy={ent:.3f} | CtxNorm={ctx_norm:.3f} | R_max={rew_max:.2f} | V_loss={v_loss:.3f} | P_loss={p_loss:.3f}")
            
        # 3. Checkpointing
        if update % 50 == 0 or update == num_updates:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"{run_name}_update{update}.pt")
            torch.save(
                {
                    "ppo_net_state_dict": ppo_net.state_dict(),
                    "world_model_state_dict": wm.state_dict(),
                    "ppo_optimizer_state_dict": ppo_opt.state_dict(),
                    "wm_optimizer_state_dict": wm_opt.state_dict(),
                    "cfg": cfg,
                    "global_step": global_step,
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")

    envs.close()
    logger.close()
    print("Training Complete!")

if __name__ == "__main__":
    main()
