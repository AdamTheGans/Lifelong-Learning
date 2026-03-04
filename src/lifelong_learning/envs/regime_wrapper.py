# src/lifelong_learning/envs/regime_wrapper.py
from __future__ import annotations

import gymnasium as gym
import numpy as np

class RegimeGoalSwapWrapper(gym.Wrapper):
    """
    Non-stationarity wrapper that shifts which goal color gives a positive reward.
    
    Regime 0: Goal 0 is +5, all others are -1
    Regime 1: Goal 1 is +5, all others are -1
    ...
    Regime N: Goal N is +5, all others are -1
    """

    def __init__(
        self,
        env: gym.Env,
        steps_per_regime: int | None = None,
        episodes_per_regime: int | None = None,
        start_regime: int = 0,
        num_regimes: int = 2,
        seed: int = 0,
        shared_step_counter: list | None = None,
    ):
        super().__init__(env)
        self.steps_per_regime = steps_per_regime
        self.episodes_per_regime = episodes_per_regime
        self.start_regime = start_regime
        self.num_regimes = num_regimes
        
        self.regime_id: int = start_regime
        self.cumulative_steps: int = 0
        self.cumulative_episodes: int = 0
        self._shared_step_counter = shared_step_counter

    def _update_regime_deterministic(self):
        # Use shared counter if available, else per-env counter
        step_count = self._shared_step_counter[0] if self._shared_step_counter is not None else self.cumulative_steps
        # Update based on steps
        if self.steps_per_regime:
            cycle = step_count // self.steps_per_regime
            self.regime_id = (self.start_regime + cycle) % self.num_regimes
        # Update based on episodes
        elif self.episodes_per_regime:
            cycle = self.cumulative_episodes // self.episodes_per_regime
            self.regime_id = (self.start_regime + cycle) % self.num_regimes

    def reset(self, **kwargs):
        self._update_regime_deterministic()
        obs, info = self.env.reset(**kwargs)
        info["regime_id"] = self.regime_id
        return obs, info

    def step(self, action):
        self.cumulative_steps += 1
        self._update_regime_deterministic()

        obs, original_reward, terminated, truncated, info = self.env.step(action)
        info["regime_id"] = self.regime_id

        # Success counters (set on every step so log/ keys always have a value)
        info["reached_good_goal"] = 0.0
        info["reached_bad_goal"] = 0.0
        info["timed_out"] = 0.0

        # Base step penalty
        final_reward = -0.01
        
        # Only override reward if the environment actually terminated (reached a goal)
        if terminated and original_reward > 0:
            
            # Use 'hit_goal_index' from MultiGoalEnv if present
            if "hit_goal_index" in info:
                hit_index = info["hit_goal_index"]
                
                if hit_index == self.regime_id:
                    # Hit the correct goal for this regime
                    final_reward += 5.0
                    info["reached_good_goal"] = 1.0
                else:
                    # Hit a wrong goal
                    final_reward += -1.0
                    info["reached_bad_goal"] = 1.0
            else:
                # Fallback for standard MiniGrid environments (like Empty)
                if self.regime_id == 0:
                    final_reward += 5.0
                    info["reached_good_goal"] = 1.0
                else:
                    final_reward += -1.0
                    info["reached_bad_goal"] = 1.0
        elif truncated:
            info["timed_out"] = 1.0

        reward = final_reward

        # IMPORTANT: Count episodes for regime switching
        if terminated or truncated:
            self.cumulative_episodes += 1

        return obs, reward, terminated, truncated, info
