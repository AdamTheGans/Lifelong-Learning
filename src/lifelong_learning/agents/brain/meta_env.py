"""
MetaEnv: Gymnasium environment wrapping the inner Dyna-PPO training loop.

The Brain meta-agent interacts with this environment. Each "step" runs
N inner PPO updates and returns training signals as observations.
"""
from __future__ import annotations

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
      - Observation: 15-dim vector of normalized training signals
      - Action: 4-dim continuous vector controlling hyperparameter adjustments
        [0] lr scale        ∈ [-1, 1] → mapped to multiply by [0.5, 2.0]
        [1] ent_coef scale  ∈ [-1, 1] → mapped to multiply by [0.5, 2.0]
        [2] intrinsic_coef  ∈ [-1, 1] → mapped to multiply by [0.5, 2.0]
        [3] imagined_horizon ∈ [-1, 1] → mapped to delta {-2, -1, 0, +1, +2}
      - Reward: recovery-based metric (Δ success_rate + α·Δ return - β·failure_rate)
      - Episode: one full inner training run
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: str = "MiniGrid-DualGoal-8x8-v0",
        inner_cfg: PPOConfig | None = None,
        decision_interval: int = 10,
        steps_per_regime: int | None = 15000,
        episodes_per_regime: int | None = None,
        start_regime: int = 0,
        reward_alpha: float = 0.1,
        reward_beta: float = 0.5,
        inner_run_name: str | None = None,
        save_checkpoints: bool = False,
        anneal_lr: bool = False,
        intrinsic_coef: float = 0.015,
        intrinsic_reward_clip: float = 0.1,
        imagined_horizon: int = 10,
        wm_lr: float = 1e-4,
    ):
        super().__init__()

        self.env_id = env_id
        self.inner_cfg = inner_cfg or PPOConfig()
        self.decision_interval = decision_interval
        self.steps_per_regime = steps_per_regime
        self.episodes_per_regime = episodes_per_regime
        self.start_regime = start_regime
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.inner_run_name = inner_run_name
        self.save_checkpoints = save_checkpoints
        self.anneal_lr = anneal_lr
        self.intrinsic_coef_init = intrinsic_coef
        self.intrinsic_reward_clip = intrinsic_reward_clip
        self.imagined_horizon_init = imagined_horizon
        self.wm_lr = wm_lr

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(NUM_SIGNALS,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # HP bounds (absolute min/max)
        self.lr_bounds = (1e-5, 1e-2)
        self.ent_coef_bounds = (0.001, 0.1)
        self.intrinsic_coef_bounds = (0.001, 0.5)
        self.imagined_horizon_bounds = (1, 30)

        # Will be initialized on reset()
        self._state: InnerTrainState | None = None
        self._signal_extractor: SignalExtractor | None = None
        self._prev_success_rate = 0.0
        self._prev_mean_return = 0.0
        self._episode_counter = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Clean up any previous inner training
        if self._state is not None:
            close_inner_training(self._state)

        self._episode_counter += 1
        run_name = self.inner_run_name or f"brain_inner_ep{self._episode_counter}"

        self._state = init_inner_training(
            env_id=self.env_id,
            cfg=self.inner_cfg,
            steps_per_regime=self.steps_per_regime,
            episodes_per_regime=self.episodes_per_regime,
            start_regime=self.start_regime,
            run_name=run_name,
            save_every_updates=9999 if not self.save_checkpoints else 50,
            anneal_lr=self.anneal_lr,
            intrinsic_coef=self.intrinsic_coef_init,
            intrinsic_reward_clip=self.intrinsic_reward_clip,
            imagined_horizon=self.imagined_horizon_init,
            wm_lr=self.wm_lr,
        )

        self._signal_extractor = SignalExtractor()
        self._prev_success_rate = 0.0
        self._prev_mean_return = 0.0

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

        delta_success = success_rate - self._prev_success_rate
        delta_return = mean_return - self._prev_mean_return

        reward = delta_success + self.reward_alpha * delta_return - self.reward_beta * failure_rate

        self._prev_success_rate = success_rate
        self._prev_mean_return = mean_return

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
        """Map Brain action [-1, 1]^4 to HP adjustments and apply to inner state."""
        s = self._state

        # Action[0]: lr scale → multiply by [0.5, 2.0]
        lr_multiplier = self._action_to_multiplier(action[0])
        new_lr = np.clip(
            s.optimizer.param_groups[0]["lr"] * lr_multiplier,
            *self.lr_bounds
        )
        s.optimizer.param_groups[0]["lr"] = new_lr

        # Action[1]: ent_coef scale → multiply by [0.5, 2.0]
        ent_multiplier = self._action_to_multiplier(action[1])
        s.cfg.ent_coef = float(np.clip(
            s.cfg.ent_coef * ent_multiplier,
            *self.ent_coef_bounds
        ))

        # Action[2]: intrinsic_coef scale → multiply by [0.5, 2.0]
        ic_multiplier = self._action_to_multiplier(action[2])
        s.intrinsic_coef = float(np.clip(
            s.intrinsic_coef * ic_multiplier,
            *self.intrinsic_coef_bounds
        ))

        # Action[3]: imagined_horizon delta → {-2, -1, 0, +1, +2}
        horizon_delta = int(np.round(action[3] * 2))  # [-1,1] → [-2,2]
        s.imagined_horizon = int(np.clip(
            s.imagined_horizon + horizon_delta,
            *self.imagined_horizon_bounds
        ))

    @staticmethod
    def _action_to_multiplier(a: float) -> float:
        """Map action ∈ [-1, 1] to multiplier ∈ [0.5, 2.0]."""
        # Linear: -1 → 0.5, 0 → 1.0, 1 → 2.0
        # Using exponential mapping for smoother behavior:
        #   a=-1 → 2^(-1) = 0.5, a=0 → 2^0 = 1.0, a=1 → 2^1 = 2.0
        return float(2.0 ** a)

    def close(self):
        if self._state is not None:
            close_inner_training(self._state)
            self._state = None
