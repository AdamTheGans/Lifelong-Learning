from __future__ import annotations
import gymnasium as gym
import numpy as np

from minigrid.wrappers import FullyObsWrapper
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from lifelong_learning.envs.wrappers.action_reduce import ActionReduceWrapper
from lifelong_learning.envs.wrappers.one_hot import OneHotPartialObsWrapper

def make_env(env_id: str, seed: int, record_stats: bool = True, **kwargs):
    """
    Factory for creating a fully-wrapped MiniGrid environment.

    Wrapper stack (inner → outer):
        1. FullyObsWrapper — full 8×8 grid instead of partial 7×7 agent view
        2. ActionReduceWrapper — Discrete(7) → Discrete(3) (left/right/forward)
        3. OneHotPartialObsWrapper — symbolic (H,W,3) → one-hot (21,H,W)
        4. RegimeGoalSwapWrapper — non-stationary reward switching
    """
    env = gym.make(env_id, render_mode=None, max_episode_steps=256)

    # Full observability: PPO sees the entire 8×8 grid
    env = FullyObsWrapper(env)

    # Reduce action space: only left/right/forward needed for navigation
    env = ActionReduceWrapper(env, actions=[0, 1, 2])

    # One-Hot encode observations: (H, W, 3) → (21, H, W) float tensor
    env = OneHotPartialObsWrapper(env, dict_mode=False)

    # Regime switching: swaps which goal color is "good" vs "bad"
    env = RegimeGoalSwapWrapper(
        env,
        steps_per_regime=kwargs.get("steps_per_regime"),
        episodes_per_regime=kwargs.get("episodes_per_regime"),
        start_regime=kwargs.get("start_regime", 0),
        seed=seed,
    )

    return env
