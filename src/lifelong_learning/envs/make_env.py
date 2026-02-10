from __future__ import annotations
import collections
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TimeLimit
from lifelong_learning.envs.minigrid_obs import MiniGridImageObsWrapper
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from lifelong_learning.envs.dreamer_compat import DreamerReadyWrapper
from lifelong_learning.envs.wrappers.action_reduce import ActionReduceWrapper

class RollingAvgWrapper(gym.Wrapper):
    """
    Aggregates returns/lengths and writes them to 'episode_avg'.
    """
    def __init__(self, env, window_size=50):
        super().__init__(env)
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

            if "episode_avg" not in info:
                info["episode_avg"] = {}

            info["episode_avg"]["r"] = np.mean(self.return_queue)
            info["episode_avg"]["l"] = np.mean(self.length_queue)
            
            # Pass regime stats explicitly
            regime = info.get("regime_id", 0)
            info["episode_avg"][f"r_regime_{regime}"] = self.current_return
            
            self.current_return = 0.0
            self.current_length = 0

        return obs, reward, terminated, truncated, info

def make_env(env_id: str, seed: int, record_stats: bool = True, dreamer_compatible: bool = False, **kwargs):
    render_mode = "rgb_array" if dreamer_compatible else None
    
    # Initialize environment
    # For DreamerV3: tile_size=8 gives native 64x64 render (8 cells * 8px)
    # avoiding the 256→64 downscale that loses goal color info
    tile_kwargs = {"tile_size": 8} if dreamer_compatible else {}
    env = gym.make(env_id, render_mode=render_mode, **tile_kwargs)
    
    # Disable FOV shading for DreamerV3 — renders full grid at uniform brightness
    if dreamer_compatible:
        env.unwrapped.highlight = False
    
    # 0. Reduce action space for DreamerV3 (Discrete(7) → Discrete(3))
    # Must come BEFORE TimeLimit so all wrappers see reduced space.
    # MiniGrid actions 0=left, 1=right, 2=forward — the only ones needed
    # for navigation. Removing pickup/drop/toggle/done cuts exploration space.
    if dreamer_compatible:
        env = ActionReduceWrapper(env, actions=[0, 1, 2])
    
    # 1. Apply Strict TimeLimit
    # Note: If env already has a TimeLimit, this wraps it. 
    # MiniGrid usually returns truncated=True on timeout.
    env = TimeLimit(env, max_episode_steps=256)
    
    # 2. Image Obs Wrapper (Flatten/Preprocess for CNN)
    if not dreamer_compatible:
        env = MiniGridImageObsWrapper(env)
    
    # 3. Regime Wrapper (Logic for swapping goals)
    env = RegimeGoalSwapWrapper(
        env,
        steps_per_regime=kwargs.get("steps_per_regime"),
        episodes_per_regime=kwargs.get("episodes_per_regime"),
        start_regime=kwargs.get("start_regime", 0),
        seed=seed,
    )

    if dreamer_compatible:
        env = DreamerReadyWrapper(env, oracle_mode=kwargs.get("oracle_mode", False))
    
    # NOTE: record_stats is handled at the VectorEnv level in train.py
    # to ensure it captures returns even after wrappers modify them.

    return env