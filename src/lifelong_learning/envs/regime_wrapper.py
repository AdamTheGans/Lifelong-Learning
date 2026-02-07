# src/lifelong_learning/envs/regime_wrapper.py
from __future__ import annotations

import gymnasium as gym
import numpy as np

class RegimeGoalSwapWrapper(gym.Wrapper):
    """
    Non-stationarity wrapper that inverts reward mapping.
    
    Regime 0: Green Goal is +1, Blue Goal is -1
    Regime 1: Green Goal is -1, Blue Goal is +1
    """

    def __init__(
        self,
        env: gym.Env,
        steps_per_regime: int | None = None,
        episodes_per_regime: int | None = None,
        start_regime: int = 0,
        seed: int = 0,
    ):
        super().__init__(env)
        self.steps_per_regime = steps_per_regime
        self.episodes_per_regime = episodes_per_regime
        self.start_regime = start_regime
        
        self.regime_id: int = start_regime
        self.cumulative_steps: int = 0
        self.cumulative_episodes: int = 0

    def _update_regime_deterministic(self):
        # Update based on steps
        if self.steps_per_regime:
            cycle = self.cumulative_steps // self.steps_per_regime
            self.regime_id = (self.start_regime + cycle) % 2
        # Update based on episodes
        elif self.episodes_per_regime:
            cycle = self.cumulative_episodes // self.episodes_per_regime
            self.regime_id = (self.start_regime + cycle) % 2

    def reset(self, **kwargs):
        self._update_regime_deterministic()
        obs, info = self.env.reset(**kwargs)
        info["regime_id"] = self.regime_id
        return obs, info

    def step(self, action):
        self.cumulative_steps += 1
        self._update_regime_deterministic()

        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Base step penalty
        final_reward = -0.01
        
        # Only override reward if the environment actually terminated (reached a goal)
        # We assume original_reward > 0 implies a goal was hit in MiniGrid
        if terminated and original_reward > 0:
            base_env = self.unwrapped
            
            # SAFEGUARD: Check if attributes exist (handles Empty-8x8 vs DualGoal)
            if hasattr(base_env, "green_goal_pos") and hasattr(base_env, "blue_goal_pos"):
                agent_pos = tuple(base_env.agent_pos)
                green_pos = tuple(base_env.green_goal_pos)
                blue_pos = tuple(base_env.blue_goal_pos)
                
                hit_green = (agent_pos == green_pos)
                hit_blue = (agent_pos == blue_pos)
                
                if self.regime_id == 0:
                    # Regime 0: Green Good (+5), Blue Bad (-1)
                    if hit_green: final_reward += 5.0
                    elif hit_blue: final_reward += -1.0
                else:
                    # Regime 1: Green Bad (-1), Blue Good (+5)
                    if hit_green: final_reward += -1.0
                    elif hit_blue: final_reward += 5.0
            else:
                # Fallback for standard MiniGrid environments (like Empty)
                # Just treat the single goal as "Green" (Good in R0, Bad in R1)
                if self.regime_id == 0:
                    final_reward += 5.0
                else:
                    final_reward += -1.0

        reward = final_reward

        # IMPORTANT: Count episodes for regime switching
        if terminated or truncated:
            self.cumulative_episodes += 1
            info["regime_id"] = self.regime_id

        return obs, reward, terminated, truncated, info
