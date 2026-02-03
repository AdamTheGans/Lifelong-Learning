from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  # registers MiniGrid envs

from lifelong_learning.envs.minigrid_obs import MiniGridImageObsWrapper
from lifelong_learning.envs.regime_wrapper import RegimeActionRemapWrapper


def make_env(
    env_id: str,
    seed: int,
    *,
    switch_on_reset: bool = False,
    switch_mid_episode: bool = False,
    mid_episode_switch_step_range: tuple[int, int] = (20, 80),
    record_stats: bool = True,
) -> gym.Env:
    """
    Factory that creates the canonical env pipeline used across training/eval.
    """
    env = gym.make(env_id)

    # Convert MiniGrid dict obs -> image tensor CHW float
    env = MiniGridImageObsWrapper(env)

    # Add non-stationarity regime switching (optional)
    env = RegimeActionRemapWrapper(
        env,
        switch_on_reset=switch_on_reset,
        switch_mid_episode=switch_mid_episode,
        mid_episode_switch_step_range=mid_episode_switch_step_range,
        seed=seed,
    )

    # Track episodic returns/lengths in info["episode"] when episodes end
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
