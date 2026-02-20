import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class OneHotPartialObsWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert MiniGrid 'image' observation (H, W, 3) into a One-Hot encoded 
    float tensor (20, H, W).
    
    Args:
        env: The environment to wrap
        dict_mode (bool): If True, returns observation as a Dict {'image': ...}.
                          If False, returns just the image tensor (Box).
                          Default True.
    """
    def __init__(self, env, dict_mode=True):
        super().__init__(env)
        self.dict_mode = dict_mode
        
        # Define max classes based on MiniGrid constants
        self.num_objs = 11   # Empty, Wall, Floor, Door, Key, Ball, Box, Goal, Lava, Agent, etc.
        self.num_cols = 6    # Red, Green, Blue, Purple, Yellow, Grey
        self.num_states = 4  # Open, Closed, Locked, plus Direction (0=East, 1=South, 2=West, 3=North)
        
        self.total_channels = self.num_objs + self.num_cols + self.num_states
        
        # Get old shape (H, W, 3)
        h, w, c = env.observation_space['image'].shape
        
        # New shape: (20, H, W)
        new_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(self.total_channels, h, w), 
            dtype=np.float32
        )
        
        if self.dict_mode:
            self.observation_space['image'] = new_space
        else:
            self.observation_space = new_space
        
    def observation(self, obs):
        image = obs['image'] # (H, W, 3)
        
        # Split channels
        objs = image[:, :, 0]
        cols = image[:, :, 1]
        states = image[:, :, 2]
        
        # One-Hot Encode (H, W, N)
        objs = np.clip(objs, 0, self.num_objs - 1)
        objs_hot = np.eye(self.num_objs)[objs]
        
        cols = np.clip(cols, 0, self.num_cols - 1)
        cols_hot = np.eye(self.num_cols)[cols]
        
        states = np.clip(states, 0, self.num_states - 1)
        states_hot = np.eye(self.num_states)[states]
        
        # Concatenate
        combined = np.concatenate([objs_hot, cols_hot, states_hot], axis=-1)
        
        # Permute to (C, H, W)
        combined = combined.transpose(2, 0, 1).astype(np.float32)
        
        if self.dict_mode:
            obs['image'] = combined
            return obs
        else:
            return combined
