
import gymnasium as gym
import numpy as np
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from lifelong_learning.envs.make_env import make_env
from gymnasium.envs.registration import register

# Register the environment for the script
try:
    register(
        id="MiniGrid-DualGoal-8x8-v0",
        entry_point="lifelong_learning.envs.dual_goal:DualGoalEnv",
        kwargs={"size": 8}
    )
except Exception as e:
    print(f"Registration warning: {e}")

def test_dreamer_compat():
    print("Testing Dreamer Compatibility...")
    
    # Create env with dreamer_compatible=True
    env = make_env(
        env_id="MiniGrid-DualGoal-8x8-v0", 
        seed=42, 
        dreamer_compatible=True
    )
    
    print(f"Environment created: {env}")
    print(f"Observation Space: {env.observation_space}")
    
    # RESET
    obs, info = env.reset()
    
    # 1. Check Observation Structure
    if not isinstance(obs, dict):
        print(f"FAILED: Observation must be dict, got {type(obs)}")
        sys.exit(1)
        
    if "image" not in obs:
        print("FAILED: Observation must contain 'image' key")
        sys.exit(1)
        
    img = obs["image"]
    if img.dtype == np.float32 and img.ndim == 1:
        print(f"Observation structure valid (dict, image {img.shape} float32 — symbolic mode).")
    elif img.dtype == np.uint8 and img.shape == (64, 64, 3):
        print("Observation structure valid (dict, image 64x64x3 uint8 — pixel mode).")
    else:
        print(f"FAILED: Unexpected image shape={img.shape} dtype={img.dtype}")
        sys.exit(1)
    
    # 2. Check Info
    if "is_terminal" not in info:
        print("FAILED: Info must contain 'is_terminal'")
        sys.exit(1)
        
    if info["is_terminal"] is not False:
        print(f"FAILED: is_terminal should be False on reset, got {info['is_terminal']}")
        sys.exit(1)
        
    # 3. Step Loop
    print("Running 100 steps...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check reward type
        if not (isinstance(reward, float) or isinstance(reward, np.float32)):
             print(f"FAILED: Reward must be float, got {type(reward)}")
             sys.exit(1)
             
        # Check is_terminal
        if "is_terminal" not in info:
            print("FAILED: Info missing 'is_terminal' during step")
            sys.exit(1)
            
        if info["is_terminal"] != terminated:
            print(f"FAILED: is_terminal {info['is_terminal']} != terminated {terminated}")
            sys.exit(1)
            
        if terminated or truncated:
            obs, info = env.reset()
            # Check reset info
            if info["is_terminal"] is not False:
                print("FAILED: is_terminal should be False on reset")
                sys.exit(1)

    print("SUCCESS: Dreamer Compatibility Test Passed.")

if __name__ == "__main__":
    test_dreamer_compat()
