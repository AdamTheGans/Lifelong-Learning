from __future__ import annotations

import gymnasium as gym
import numpy as np


class MiniGridImageObsWrapper(gym.ObservationWrapper):
    """
    Extracts the image tensor from MiniGrid dict obs.
    IMPORTANT: Keeps data as categorical integers (float-ready) for One-Hot encoding.
    Does NOT divide by 255.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        img_space = env.observation_space.spaces["image"]
        h, w, c = img_space.shape
        
        # We output (C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=np.float32, # Float32 container, but holds integer values
        )

    def observation(self, obs):
        img = obs["image"]
        # Standard MiniGrid is (H, W, C) -> Transpose to (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        # Do NOT divide by 255.0. 
        # The PPO model needs to see "1" for Green, not "0.0039"
        return img.astype(np.float32)