from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401
from collections import deque
import numpy as np

from lifelong_learning.envs.minigrid_obs import MiniGridImageObsWrapper
from lifelong_learning.envs.regime_wrapper import RegimeActionRemapWrapper

class RegimeStatsWrapper(gym.Wrapper):
    """
    Splits episodic returns into 'return_regime_0' and 'return_regime_1'.
    This allows TensorBoard to show performance per regime.
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_return = 0.0
        self.current_length = 0

    def reset(self, **kwargs):
        self.current_return = 0.0
        self.current_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += reward
        self.current_length += 1

        if terminated or truncated:
            # Identify which regime we finished in
            regime = info.get("regime_id", 0)
            
            # Inject custom stats for PPO logger
            # (CleanRL/RecordEpisodeStatistics usually looks at info["episode"])
            # We add a custom key that we hope the logger picks up, 
            # OR we modify info["episode"] directly if we know the structure.
            
            if "episode" not in info:
                info["episode"] = {}
            
            # Add specific regime keys
            info["episode"][f"r_regime_{regime}"] = self.current_return
            info["episode"][f"l_regime_{regime}"] = self.current_length
            
            # Reset trackers
            self.current_return = 0.0
            self.current_length = 0

        return obs, reward, terminated, truncated, info

def make_env(
    env_id: str,
    seed: int,
    *,
    # Deterministic args
    steps_per_regime: int | None = None,
    episodes_per_regime: int | None = None,
    start_regime: int = 0,
    # Random args
    switch_on_reset: bool = False,
    switch_mid_episode: bool = False,
    mid_episode_switch_step_range: tuple[int, int] = (10, 40),
    record_stats: bool = True,
) -> gym.Env:
    
    env = gym.make(env_id)
    env = MiniGridImageObsWrapper(env)

    env = RegimeActionRemapWrapper(
        env,
        steps_per_regime=steps_per_regime,
        episodes_per_regime=episodes_per_regime,
        start_regime=start_regime,
        switch_on_reset=switch_on_reset,
        switch_mid_episode=switch_mid_episode,
        mid_episode_switch_step_range=mid_episode_switch_step_range,
        seed=seed,
    )

    if record_stats:
        # Standard Gym stats (overall return/length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Our custom splitter for regime-specific logging
        env = RegimeStatsWrapper(env)

    return env
