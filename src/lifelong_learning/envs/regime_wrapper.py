from __future__ import annotations

import gymnasium as gym
import numpy as np


class RegimeActionRemapWrapper(gym.Wrapper):
    """
    A simple non-stationarity wrapper for MiniGrid that changes the action mapping
    depending on the current regime.

    Regime 0: identity mapping
    Regime 1: swap LEFT and RIGHT

    Supports:
      - switch_on_reset: pick a regime at each episode reset
      - switch_mid_episode: flip regime once mid-episode at a random step

    It writes:
      info["regime_id"] : current regime id (0 or 1)
      info["switch_step"] : the step at which the mid-episode switch occurs (or None)
    """

    def __init__(
        self,
        env: gym.Env,
        switch_on_reset: bool = False,
        switch_mid_episode: bool = False,
        mid_episode_switch_step_range: tuple[int, int] = (20, 80),
        seed: int = 0,
    ):
        super().__init__(env)
        self.switch_on_reset = switch_on_reset
        self.switch_mid_episode = switch_mid_episode
        self.mid_range = mid_episode_switch_step_range
        self.rng = np.random.default_rng(seed)

        self.regime_id: int = 0
        self.t: int = 0
        self.switch_step: int | None = None

    def _sample_regime(self) -> int:
        return int(self.rng.integers(0, 2))  # 0 or 1

    def _schedule_mid_episode_switch(self) -> None:
        if not self.switch_mid_episode:
            self.switch_step = None
            return
        lo, hi = self.mid_range
        # inclusive/exclusive behavior: integers(lo, hi) chooses lo..hi-1
        self.switch_step = int(self.rng.integers(lo, hi))

    def reset(self, **kwargs):
        self.t = 0
        if self.switch_on_reset:
            self.regime_id = self._sample_regime()
        self._schedule_mid_episode_switch()

        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["regime_id"] = self.regime_id
        info["switch_step"] = self.switch_step
        return obs, info

    def step(self, action):
        self.t += 1

        # Mid-episode switch
        if self.switch_mid_episode and self.switch_step is not None and self.t == self.switch_step:
            self.regime_id = 1 - self.regime_id  # flip 0 <-> 1

        mapped_action = self._map_action(int(action), self.regime_id)
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)

        info = dict(info)
        info["regime_id"] = self.regime_id
        info["switch_step"] = self.switch_step
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _map_action(action: int, regime_id: int) -> int:
        """
        MiniGrid action meanings (common):
          0 left, 1 right, 2 forward, 3 pickup, 4 drop, 5 toggle, 6 done

        We only swap left/right in regime 1.
        """
        if regime_id == 0:
            return action
        if action == 0:
            return 1
        if action == 1:
            return 0
        return action
