from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box
import scipy.ndimage

class DreamerReadyWrapper(gym.Wrapper):
    """
    Wrapper to make environment compatible with DreamerV3 requirements.
    
    Features:
    - Forces observation to be a Dict with an 'image' key.
    - 'image' is resized to 64x64 and is uint8 (H, W, 3).
    - Rewards are cast to float32.
    - info['is_terminal'] is explicitly set to terminated status.
    - Optional Oracle mode to inject regime_id into observation.
    """
    def __init__(self, env, size=(64, 64), oracle_mode=False):
        super().__init__(env)
        self._size = size
        self._oracle_mode = oracle_mode
        
        # Calculate new observation space
        obs_spaces = {}
        
        # 1. Filter and keep only safe keys from original space
        if isinstance(env.observation_space, Dict):
            for key, space in env.observation_space.spaces.items():
                # Exclude 'mission' and other text/object keys that might crash Dreamer
                if key in ["mission", "image"]: 
                    continue
                # Keep numeric/safe keys
                obs_spaces[key] = space
            
        # Ensure 'image' key specification
        obs_spaces['image'] = Box(
            low=0, high=255, 
            shape=(size[0], size[1], 3), 
            dtype=np.uint8
        )
        
        if self._oracle_mode:
            # In Oracle mode, provide regime_id as a vector
            # Fixed: Box(0, 1) as requested, though logic implies it could be 0 or 1.
            obs_spaces['regime_id'] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
            
        self.observation_space = Dict(obs_spaces)

    def _resize_image(self, image):
        """Resize image to target size using scipy.ndimage."""
        # Note: We now raise error on None in _get_obs, so image is guaranteed not None here
        h, w = image.shape[:2]
        target_h, target_w = self._size
        
        if h == target_h and w == target_w:
            return image
            
        # Check for channel dim
        if image.ndim == 2:
            image = image[:, :, None] # Add channel dim if missing
            
        # Compute zoom factors
        # scipy.ndimage.zoom expects (depth, height, width) or (h, w, c)
        zoom_factors = (target_h / h, target_w / w, 1.0)
        
        # Resize using bilinear interpolation (order=1)
        resized = scipy.ndimage.zoom(image, zoom_factors, order=1)
        
        return np.clip(resized, 0, 255).astype(np.uint8)

    def _get_obs(self, obs, info=None):
        out_obs = {}
        
        # 2. Strict Key Filtering on Run
        if isinstance(obs, dict):
            for k, v in obs.items():
                # Only pass through keys we explicitly kept in __init__
                if k in self.observation_space.spaces and k != "image" and k != "regime_id":
                    out_obs[k] = v
            
        # Render the environment for the image key
        frame = self.env.render()
        
        # 3. Crash on Null Render
        if frame is None:
            raise RuntimeError("Env returned None render. Dreamer requires valid pixels.")
        
        # Resize and set image
        out_obs['image'] = self._resize_image(frame)
        
        # Inject regime_id if in Oracle mode
        if self._oracle_mode:
            regime = 0.0
            # Try to fetch regime_id from info first
            if info and 'regime_id' in info:
                regime = float(info['regime_id'])
            # Fallback to env attribute
            elif hasattr(self.env, "get_wrapper_attr"):
                try:
                    regime = float(self.env.get_wrapper_attr('regime_id'))
                except AttributeError:
                    pass
            elif hasattr(self.env, "regime_id"):
                 regime = float(self.env.regime_id)
                 
            out_obs['regime_id'] = np.array([regime], dtype=np.float32)
            
        return out_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if info is None: info = {}
        
        # Explicit termination signal
        info['is_terminal'] = False
        
        return self._get_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None: info = {}
        
        # Explicit termination signal (Dreamer distinguishes terminal from time limit via this)
        info['is_terminal'] = terminated
        
        return self._get_obs(obs, info), np.float32(reward), terminated, truncated, info
