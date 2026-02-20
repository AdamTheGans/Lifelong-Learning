"""
ActionReduceWrapper: Maps a smaller Discrete action space to MiniGrid's Discrete(7).

MiniGrid actions:
  0 = left, 1 = right, 2 = forward,
  3 = pickup, 4 = drop, 5 = toggle, 6 = done

For pure navigation (DualGoal), only left/right/forward are needed.
Reducing from 7 to 3 actions dramatically improves exploration efficiency.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class ActionReduceWrapper(gym.ActionWrapper):
    """
    Reduce MiniGrid action space from Discrete(7) to Discrete(N).

    Parameters
    ----------
    env : gym.Env
        A MiniGrid environment with Discrete(7) action space.
    actions : list[int]
        List of original MiniGrid action indices to keep.
        Default: [0, 1, 2] = left, right, forward.
    """

    MINIGRID_ACTION_NAMES = [
        "left", "right", "forward", "pickup", "drop", "toggle", "done"
    ]

    def __init__(self, env: gym.Env, actions: list[int] | None = None):
        super().__init__(env)
        self._actions = actions or [0, 1, 2]  # left, right, forward
        self.action_space = gym.spaces.Discrete(len(self._actions))

        # e.g. [(0, 'left', 0), (1, 'right', 1), (2, 'forward', 2)]

    def action(self, action):
        """Map reduced action index to original MiniGrid action."""
        return self._actions[int(action)]

    def reverse_action(self, action):
        """Map original MiniGrid action back to reduced index (for logging)."""
        if action in self._actions:
            return self._actions.index(action)
        return 0  # fallback
