from __future__ import annotations

import gymnasium as gym
import numpy as np


class RegimeActionRemapWrapper(gym.Wrapper):
    """
    Non-stationarity wrapper that changes action mapping based on a deterministic schedule.

    Regime 0: Identity
    Regime 1: Swap LEFT (0) and RIGHT (1)

    Scheduling Modes (Priority ordered):
    1. steps_per_regime (int): Switch every N steps (per env).
    2. episodes_per_regime (int): Switch every K episodes (per env).
    3. switch_on_reset (bool): Randomly sample regime every episode.
    4. switch_mid_episode (bool): Flip once randomly mid-episode (legacy mode).
    """

    def __init__(
        self,
        env: gym.Env,
        # Deterministic schedules
        steps_per_regime: int | None = None,
        episodes_per_regime: int | None = None,
        # Random/Legacy schedules
        switch_on_reset: bool = False,
        switch_mid_episode: bool = False,
        mid_episode_switch_step_range: tuple[int, int] = (20, 80),
        # Config
        start_regime: int = 0,
        seed: int = 0,
    ):
        super().__init__(env)
        self.steps_per_regime = steps_per_regime
        self.episodes_per_regime = episodes_per_regime
        self.switch_on_reset = switch_on_reset
        self.switch_mid_episode = switch_mid_episode
        self.mid_range = mid_episode_switch_step_range
        self.start_regime = start_regime
        self.rng = np.random.default_rng(seed)

        # State tracking
        self.regime_id: int = start_regime
        self.current_episode_step: int = 0
        self.switch_step: int | None = None  # For mid-episode legacy mode
        
        # Global counters (persistent across resets)
        self.cumulative_steps: int = 0
        self.cumulative_episodes: int = 0

    def _update_regime_deterministic(self):
        """Calculates current regime based on global counters."""
        if self.steps_per_regime:
            # E.g. 0-999 -> Regime 0, 1000-1999 -> Regime 1
            cycle = self.cumulative_steps // self.steps_per_regime
            self.regime_id = (self.start_regime + cycle) % 2
        
        elif self.episodes_per_regime:
            cycle = self.cumulative_episodes // self.episodes_per_regime
            self.regime_id = (self.start_regime + cycle) % 2

    def reset(self, **kwargs):
        self.current_episode_step = 0
        
        # Update episode-based schedules
        self._update_regime_deterministic()
        
        # Handle random-reset legacy mode
        if self.switch_on_reset and not (self.steps_per_regime or self.episodes_per_regime):
            self.regime_id = int(self.rng.integers(0, 2))

        # Handle mid-episode legacy mode
        self.switch_step = None
        if self.switch_mid_episode:
            lo, hi = self.mid_range
            self.switch_step = int(self.rng.integers(lo, hi))

        obs, info = self.env.reset(**kwargs)
        
        info["regime_id"] = self.regime_id
        return obs, info

    def step(self, action):
        self.current_episode_step += 1
        self.cumulative_steps += 1
        
        # Update step-based schedules immediately
        self._update_regime_deterministic()

        # Mid-episode legacy switch
        if self.switch_mid_episode and self.switch_step is not None:
            if self.current_episode_step == self.switch_step:
                self.regime_id = 1 - self.regime_id

        mapped_action = self._map_action(int(action), self.regime_id)
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)

        if terminated or truncated:
            self.cumulative_episodes += 1
            info["episode_extra"] = {
                "regime_id_end": self.regime_id,
            }

        info["regime_id"] = self.regime_id
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _map_action(action: int, regime_id: int) -> int:
        # Regime 0: Identity
        # Regime 1: Swap Left(0) <-> Right(1)
        if regime_id == 0:
            return action
        if action == 0:
            return 1
        if action == 1:
            return 0
        return action
