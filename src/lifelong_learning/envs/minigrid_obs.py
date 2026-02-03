from __future__ import annotations

import gymnasium as gym
import numpy as np


class MiniGridImageObsWrapper(gym.ObservationWrapper):
    """
    MiniGrid observations are typically a dict containing:
      - obs["image"]: (H, W, C) uint8
      - obs["direction"]: int
      - obs["mission"]: str

    For PPO/world-model baselines, we often just want the image.

    This wrapper converts obs["image"] -> float32 CHW in [0,1].
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                "MiniGridImageObsWrapper expects Dict observation_space. "
                f"Got: {type(env.observation_space)}"
            )

        if "image" not in env.observation_space.spaces:
            raise KeyError(
                "MiniGridImageObsWrapper expects key 'image' in observation_space."
            )

        img_space = env.observation_space.spaces["image"]
        if not isinstance(img_space, gym.spaces.Box) or img_space.shape is None:
            raise TypeError("Expected obs['image'] to be a Box with shape.")

        h, w, c = img_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c, h, w),
            dtype=np.float32,
        )

    def observation(self, obs):
        img = obs["image"]
        # HWC uint8 -> float32 [0,1] -> CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img
