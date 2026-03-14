"""
MetaEnv: Gymnasium environment wrapping the inner Dyna-PPO training loop.

The Brain meta-agent interacts with this environment. Each "step" runs
N inner PPO updates and returns training signals as observations.
"""
from __future__ import annotations

import copy
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lifelong_learning.agents.ppo.ppo import PPOConfig
from lifelong_learning.agents.ppo.train import (
    InnerTrainState,
    init_inner_training,
    run_inner_update,
    close_inner_training,
)
from lifelong_learning.agents.brain.signals import SignalExtractor, NUM_SIGNALS


class MetaEnv(gym.Env):
    """
    Gymnasium environment where:
      - Observation: 19-dim vector of normalized training signals
      - Action: 15-dim continuous vector controlling hyperparameter adjustments
        [0]  lr scale             ∈ [-1, 1] → mapped linearly to [lr_min, lr_max]
        [1]  ent_coef scale       ∈ [-1, 1] → mapped linearly to [ent_min, ent_max]
        [2]  intrinsic_coef       ∈ [-1, 1] → mapped linearly to [intr_min, intr_max]
        [3]  imagined_horizon     ∈ [-1, 1] → mapped linearly to [1, 30]
        [4]  replay_ratio         ∈ [-1, 1] → mapped linearly to [0.0, 0.5]
        [5]  replay_prioritization ∈ [-1, 1] → mapped linearly to [0.0, 1.0]
        [6]  anchoring_weight     ∈ [-1, 1] → mapped linearly to [0.0, 0.5]
        [7:15] context_code       ∈ [-1, 1]^8 → neuromodulation mask via ContextDecoder
      - Reward: recovery-based metric (Δ success_rate + α·Δ return - β·failure_rate)
      - Episode: one full inner training run
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: str = "MiniGrid-MultiGoal-8x8-v0",
        inner_cfg: PPOConfig | None = None,
        decision_interval: int = 10,
        steps_per_regime: int | None = 15000,
        episodes_per_regime: int | None = None,
        start_regime: int = 0,
        num_regimes: int = 2,
        reward_alpha: float = 0.1,
        reward_beta: float = 0.5,
        reward_mode: str = "auc",
        inner_run_name: str | None = None,
        save_checkpoints: bool = False,
        anneal_lr: bool = False,
        intrinsic_coef: float = 0.015,
        intrinsic_reward_clip: float = 0.1,
        imagined_horizon: int = 10,
        wm_lr: float = 1e-4,
        inner_log_dir: str | None = None,
        env_index: int = 0,
        episodic_memory_capacity: int = 50000,
        max_inner_lr: float = 0.003,
        min_inner_lr: float = 1e-4,
        min_ent_coef: float = 0.001,
        max_ent_coef: float = 0.1,
        min_intrinsic_coef: float = 0.001,
        max_intrinsic_coef: float = 0.5,
        start_episode: int = 1,
        disable_neuromodulation: bool = False,
    ):
        super().__init__()

        self.start_episode = start_episode
        self.env_index = env_index
        self.env_id = env_id
        self.inner_cfg = inner_cfg or PPOConfig()
        self.decision_interval = decision_interval
        self.steps_per_regime = steps_per_regime
        self.episodes_per_regime = episodes_per_regime
        self.start_regime = start_regime
        self.num_regimes = num_regimes
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_mode = reward_mode
        self.inner_run_name = inner_run_name
        self.save_checkpoints = save_checkpoints
        self.anneal_lr = anneal_lr
        self.intrinsic_coef_init = intrinsic_coef
        self.intrinsic_reward_clip = intrinsic_reward_clip
        self.imagined_horizon_init = imagined_horizon
        self.wm_lr = wm_lr
        self.inner_log_dir = inner_log_dir
        self.episodic_memory_capacity = episodic_memory_capacity
        self.max_inner_lr = max_inner_lr
        self.min_inner_lr = min_inner_lr
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.min_intrinsic_coef = min_intrinsic_coef
        self.max_intrinsic_coef = max_intrinsic_coef
        self.disable_neuromodulation = disable_neuromodulation

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(NUM_SIGNALS,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )

        # HP bounds (absolute min/max)
        self.lr_bounds = (self.min_inner_lr, self.max_inner_lr)
        self.ent_coef_bounds = (self.min_ent_coef, self.max_ent_coef)
        self.intrinsic_coef_bounds = (self.min_intrinsic_coef, self.max_intrinsic_coef)
        self.imagined_horizon_bounds = (1, 30)
        self.replay_ratio_bounds = (0.0, 0.5)
        self.replay_prioritization_bounds = (0.0, 1.0)
        self.anchoring_weight_bounds = (0.0, 0.5)

        # Will be initialized on reset()
        self._state: InnerTrainState | None = None
        self._signal_extractor: SignalExtractor | None = None
        self._prev_success_rate = 0.0
        self._prev_mean_return = 0.0
        self._prev_failure_rate = 0.0
        self._episode_counter = self.start_episode - 1
        self._episode_prefix = "episode"
        self._regime_exposures = {}
        self._current_regime = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start_regime = int(self.np_random.integers(0, self.num_regimes))

        # Clean up any previous inner training
        if self._state is not None:
            close_inner_training(self._state)

        self._episode_counter += 1
        run_name = self.inner_run_name or f"ep{self._episode_counter}_env{self.env_index}"

        init_kwargs = dict(
            env_id=self.env_id,
            cfg=self.inner_cfg,
            steps_per_regime=self.steps_per_regime,
            episodes_per_regime=self.episodes_per_regime,
            start_regime=self.start_regime,
            num_regimes=self.num_regimes,
            run_name=run_name,
            save_every_updates=9999 if not self.save_checkpoints else 50,
            anneal_lr=self.anneal_lr,
            intrinsic_coef=self.intrinsic_coef_init,
            intrinsic_reward_clip=self.intrinsic_reward_clip,
            imagined_horizon=self.imagined_horizon_init,
            wm_lr=self.wm_lr,
            episodic_memory_capacity=self.episodic_memory_capacity,
        )
        if self.inner_log_dir is not None:
            ep_log_dir = os.path.join(self.inner_log_dir, f"{self._episode_prefix}_{self._episode_counter}")
            init_kwargs["log_dir"] = ep_log_dir
            init_kwargs["save_dir"] = os.path.join(ep_log_dir, "inner_checkpoints")
            
        self._state = init_inner_training(**init_kwargs)

        self._signal_extractor = SignalExtractor(
            max_inner_lr=self.max_inner_lr,
            min_inner_lr=self.min_inner_lr,
            min_ent_coef=self.min_ent_coef,
            max_ent_coef=self.max_ent_coef,
            min_intrinsic_coef=self.min_intrinsic_coef,
            max_intrinsic_coef=self.max_intrinsic_coef,
        )
        self._prev_success_rate = 0.0
        self._prev_mean_return = 0.0
        self._prev_failure_rate = 0.0
        self._regime_exposures = {}
        self._current_regime = None

        # Run initial updates to get a meaningful first observation
        stats = self._run_n_updates(self.decision_interval)
        obs = self._signal_extractor.extract(stats)
        info = {"inner_stats": stats}

        return obs, info

    def step(self, action: np.ndarray):
        assert self._state is not None, "Must call reset() first"

        # Apply hyperparameter adjustments
        self._apply_action(action)

        # Run N inner updates
        stats = self._run_n_updates(self.decision_interval)

        # Compute observation
        obs = self._signal_extractor.extract(stats)

        # Compute reward
        success_rate = stats.get("success_rate", 0.0)
        mean_return = stats.get("mean_episodic_return", 0.0)
        failure_rate = stats.get("failure_rate", 0.0)

        if self.reward_mode == "recovery":
            # Hybrid recovery reward: incentivizes fast recovery after regime switches
            delta_sr = success_rate - self._prev_success_rate
            recovery = max(0.0, delta_sr) * 5.0
            maintenance = success_rate * 0.3
            shortfall = max(0.0, 0.8 - success_rate)
            urgency_penalty = -shortfall * 3.0
            failure_penalty = -failure_rate * 0.5
            reward = recovery + maintenance + urgency_penalty + failure_penalty
        elif self.reward_mode == "curriculum":
            # Tracks exposure to regimes. Re-visiting an old regime grants a massive multiplier.
            current_regime = stats.get("current_mode_regime", 0)
            if self._current_regime != current_regime:
                self._regime_exposures[current_regime] = self._regime_exposures.get(current_regime, 0) + 1
                self._current_regime = current_regime
            
            exposures = self._regime_exposures.get(current_regime, 1)
            # Give a 10x multiplier if this is the 2nd+ time we've seen this regime
            multiplier = 10.0 if exposures > 1 else 1.0
            
            base_reward = success_rate + self.reward_alpha * mean_return - self.reward_beta * failure_rate
            reward = base_reward * multiplier
        else:
            # Original AUC reward
            reward = success_rate + self.reward_alpha * mean_return - self.reward_beta * failure_rate

        self._prev_success_rate = success_rate
        self._prev_mean_return = mean_return
        self._prev_failure_rate = failure_rate

        terminated = stats.get("done", False)
        truncated = False
        info = {"inner_stats": stats}

        return obs, float(reward), terminated, truncated, info

    def _run_n_updates(self, n: int) -> dict:
        """Run n inner PPO updates and return the last stats dict."""
        stats = {}
        for _ in range(n):
            stats = run_inner_update(self._state)
            if stats.get("done", False):
                break
        return stats

    def _apply_action(self, action: np.ndarray):
        """Map Brain action [-1, 1]^15 to absolute HP values and apply to inner state."""
        s = self._state

        # Helper to map [-1, 1] to [min_val, max_val]
        def map_to_range(a: float, bounds: tuple[float, float]) -> float:
            return float(bounds[0] + (a + 1.0) / 2.0 * (bounds[1] - bounds[0]))

        # Action[0]: lr scale
        new_lr = map_to_range(action[0], self.lr_bounds)
        s.optimizer.param_groups[0]["lr"] = new_lr

        # Action[1]: ent_coef
        s.cfg.ent_coef = map_to_range(action[1], self.ent_coef_bounds)

        # Action[2]: intrinsic_coef
        s.intrinsic_coef = map_to_range(action[2], self.intrinsic_coef_bounds)

        # Action[3]: imagined_horizon (integer mapping)
        horizon_float = map_to_range(action[3], self.imagined_horizon_bounds)
        s.imagined_horizon = int(np.clip(round(horizon_float), *self.imagined_horizon_bounds))

        # Action[4]: replay_ratio
        s.replay_ratio = map_to_range(action[4], self.replay_ratio_bounds)

        # Action[5]: replay_prioritization
        s.replay_prioritization = map_to_range(action[5], self.replay_prioritization_bounds)

        # Action[6]: anchoring_weight
        s.cfg.anchoring_weight = map_to_range(action[6], self.anchoring_weight_bounds)

        # Action[7:15]: neuromodulation context code
        if not self.disable_neuromodulation:
            import torch
            context_code = torch.tensor(action[7:15], dtype=torch.float32, device=s.device)
            s.model.set_context_code(context_code)

    def close(self):
        if self._state is not None:
            close_inner_training(self._state)
            self._state = None
        super().close()
