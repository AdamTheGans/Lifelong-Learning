from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Box
from PIL import Image

class DreamerReadyWrapper(gym.Wrapper):
    """
    Wrapper to make environment compatible with DreamerV3 requirements.
    
    Features:
    - Forces observation to be a Dict with an 'image' key.
    - 'image' is resized to 64x64 and is uint8 (H, W, 3).
    - Filters out all non-numeric keys from observation.
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
        
        # 1. Filter and keep only safe numeric keys from original space
        if isinstance(env.observation_space, Dict):
            for key, space in env.observation_space.spaces.items():
                if key == "image": 
                    continue # specific handling below
                
                # Check if space is Box or compatible numeric type
                if isinstance(space, Box):
                    obs_spaces[key] = space
                # We strictly exclude Discrete/Text/etc unless converted to Box, 
                # but to be minimal invasive we just drop non-Box for now 
                # or keys that look like strings.
            
        # Ensure 'image' key specification
        obs_spaces['image'] = Box(
            low=0, high=255, 
            shape=(size[0], size[1], 3), 
            dtype=np.uint8
        )
        
        if self._oracle_mode:
            # In Oracle mode, provide regime_id as a vector
            obs_spaces['regime_id'] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        
        # log/ prefixed keys are automatically picked up by DreamerV3's
        # logfn and written to TensorBoard (avg/max/sum per episode)
        obs_spaces['log/regime_id'] = Box(
            low=0, high=1, shape=(), dtype=np.float32
        )
        obs_spaces['log/reached_good_goal'] = Box(
            low=0, high=1, shape=(), dtype=np.float32
        )
        obs_spaces['log/reached_bad_goal'] = Box(
            low=0, high=1, shape=(), dtype=np.float32
        )
        obs_spaces['log/timed_out'] = Box(
            low=0, high=1, shape=(), dtype=np.float32
        )
            
        self.observation_space = Dict(obs_spaces)

    def _resize_image(self, image):
        """Resize image to target size using Pillow."""
        # Check for channel dim and ensure HWC
        if image.ndim == 2:
            image = image[:, :, None]
            
        h, w = image.shape[:2]
        target_h, target_w = self._size
        
        # If already correct size, just handle channels
        if h == target_h and w == target_w:
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            return image.astype(np.uint8)

        # Convert to PIL Image
        # Expects dict, but image is numpy array here from render or obs
        if image.shape[2] == 1:
            pil_img = Image.fromarray(image[:, :, 0], mode='L')
            pil_img = pil_img.convert('RGB')
        else:
            # Assume RGB if 3 channels, or drop extra
            if image.shape[2] > 3:
                image = image[:, :, :3]
            pil_img = Image.fromarray(image.astype(np.uint8), mode='RGB')
            
        # Resize
        resized_pil = pil_img.resize((target_w, target_h), resample=Image.BILINEAR)
        
        # Convert back to numpy
        resized = np.array(resized_pil, dtype=np.uint8)
        
        # Ensure it has 3 channels (PIL RGB does this, but good to be safe)
        if resized.ndim == 2:
            resized = resized[:, :, None]
            
        return resized

    def _get_obs(self, obs, info=None):
        out_obs = {}
        
        # 1. Strict Key Filtering & Numeric Check
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k in self.observation_space.spaces and k != "image" and k != "regime_id":
                    # Check if value is numeric array
                    if isinstance(v, (np.ndarray, np.generic, float, int)):
                         out_obs[k] = v
        
        # 2. Image Source Logic
        # Try to use existing 'image' if valid numeric array
        frame = None
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
            if isinstance(img, np.ndarray) and img.size > 0:
                frame = img
        
        # Fallback to render if no valid image in obs
        if frame is None:
            frame = self.env.render()
        
        # 3. Crash on Null Render or Invalid Frame
        if frame is None:
            raise RuntimeError("Env returned None render and no 'image' in obs. Dreamer requires valid pixels.")
        
        # Resize and set image
        out_obs['image'] = self._resize_image(frame)
        
        # Fetch regime_id from info or env attribute
        regime = 0.0
        if info and 'regime_id' in info:
            regime = float(info['regime_id'])
        elif hasattr(self.env, "get_wrapper_attr"):
            try:
                regime = float(self.env.get_wrapper_attr('regime_id'))
            except AttributeError:
                pass
        elif hasattr(self.env, "regime_id"):
            regime = float(self.env.regime_id)
        
        # Always inject log/ keys for TensorBoard logging
        out_obs['log/regime_id'] = np.float32(regime)
        out_obs['log/reached_good_goal'] = np.float32(
            info.get('reached_good_goal', 0.0) if info else 0.0)
        out_obs['log/reached_bad_goal'] = np.float32(
            info.get('reached_bad_goal', 0.0) if info else 0.0)
        out_obs['log/timed_out'] = np.float32(
            info.get('timed_out', 0.0) if info else 0.0)
        
        # Inject regime_id as observation input if in Oracle mode
        if self._oracle_mode:
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
