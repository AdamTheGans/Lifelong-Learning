from __future__ import annotations
import collections
import gymnasium as gym
import minigrid 
import numpy as np

import lifelong_learning.envs.dual_goal
from lifelong_learning.envs.minigrid_obs import MiniGridImageObsWrapper
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper

class RollingAvgWrapper(gym.Wrapper):
    """
    Aggregates returns and lengths over a rolling window.
    Only writes to 'info' when a smoothed average is ready.
    This makes TensorBoard graphs much cleaner.
    """
    def __init__(self, env, window_size=50):
        super().__init__(env)
        self.window_size = window_size
        self.return_queue = collections.deque(maxlen=window_size)
        self.length_queue = collections.deque(maxlen=window_size)
        
        self.current_return = 0.0
        self.current_length = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_return += reward
        self.current_length += 1

        if terminated or truncated:
            self.return_queue.append(self.current_return)
            self.length_queue.append(self.current_length)

            # Inject the ROLLING AVERAGE into info
            # We use a special key that your logger can look for, 
            # or simply overwrite "episode" if you prefer.
            if "episode" not in info:
                info["episode"] = {}
            
            # Calculates mean of last N episodes
            avg_r = np.mean(self.return_queue)
            avg_l = np.mean(self.length_queue)

            info["episode"]["r"] = avg_r
            info["episode"]["l"] = avg_l
            
            # Pass regime stats if present
            regime = info.get("regime_id", 0)
            info["episode"][f"r_regime_{regime}"] = self.current_return
            
            self.current_return = 0.0
            self.current_length = 0

        return obs, reward, terminated, truncated, info

def make_env(
    env_id: str,
    seed: int,
    *,
    steps_per_regime: int | None = None,
    episodes_per_regime: int | None = None,
    start_regime: int = 0,
    record_stats: bool = True,
    **kwargs, 
) -> gym.Env:
    
    env = gym.make(env_id)
    
    # 1. Observation: (C, H, W) Integers
    env = MiniGridImageObsWrapper(env)

    # 2. Regime: Swap Rewards
    env = RegimeGoalSwapWrapper(
        env,
        steps_per_regime=steps_per_regime,
        episodes_per_regime=episodes_per_regime,
        start_regime=start_regime,
        seed=seed,
    )

    # 3. Stats: Record & Smooth
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Apply smoothing wrapper ON TOP of RecordEpisodeStatistics
        # This will override the noisy single-episode "r" with the smooth average
        env = RollingAvgWrapper(env, window_size=50)

    return env