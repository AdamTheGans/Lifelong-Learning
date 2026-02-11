import sys
import os
import argparse
import numpy as np
import gymnasium as gym

# Add source to path so we can import lifelong_learning
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from lifelong_learning.envs.make_env import make_env
import lifelong_learning.envs # Register envs

def check_env_io(env_id="MiniGrid-DualGoal-8x8-v0", steps=50, oracle=False):
    print(f"Checking {env_id} with dreamer_compatible=True, oracle={oracle}...")
    
    env = make_env(
        env_id=env_id, 
        seed=42, 
        dreamer_compatible=True, 
        oracle_mode=oracle,
        steps_per_regime=10, 
        episodes_per_regime=2
    )
    
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    obs, info = env.reset()
    
    # Check 1: Info is_terminal
    if 'is_terminal' not in info:
        print("FAIL: 'is_terminal' not in reset info")
        sys.exit(1)
        
    for i in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check 2: Obs Structure
        if not isinstance(obs, dict):
            print("FAIL: Obs is not a dict")
            sys.exit(1)
            
        # Check 3: Check Keys & Types
        for k, v in obs.items():
            if not isinstance(k, str):
                print(f"FAIL: Obs key {k} is not string")
                sys.exit(1)
            if not isinstance(v, (np.ndarray, float, int, np.number)):
                print(f"FAIL: Obs value for {k} is not numeric/numpy. Got {type(v)}")
                sys.exit(1)
            if isinstance(v, np.ndarray) and v.dtype.kind in {'U', 'S', 'O'}:
                 print(f"FAIL: Obs value for {k} has unsafe dtype {v.dtype}")
                 sys.exit(1)

        # Check 4: Image validity
        if 'image' not in obs:
            print("FAIL: 'image' missing from obs")
            sys.exit(1)
            
        img = obs['image']
        # Accept either symbolic (float32, 1D) or pixel (uint8, 3D) observations
        if img.dtype == np.float32:
            # Symbolic mode: expect flattened vector
            if img.ndim != 1:
                print(f"FAIL: Symbolic image ndim is {img.ndim}, expected 1")
                sys.exit(1)
        elif img.dtype == np.uint8:
            # Pixel mode: expect (H, W, 3)
            if img.ndim != 3 or img.shape[2] != 3:
                print(f"FAIL: Pixel image shape {img.shape} incorrect. Expected (H, W, 3)")
                sys.exit(1)
        else:
            print(f"FAIL: Image dtype is {img.dtype}, expected float32 or uint8")
            sys.exit(1)
        
        # Check 5: Non-blank
        if img.sum() == 0:
            print(f"FAIL: Image is completely blank (all zeros) at step {i}")
            sys.exit(1)
            
        # Check 6: Reward float32
        if not isinstance(reward, float) and not (isinstance(reward, np.number) and reward.dtype == np.float32):
             # Python float is fine, but if numpy it must be float32
             if isinstance(reward, np.ndarray) and reward.dtype != np.float32:
                 print(f"FAIL: Reward dtype {reward.dtype} not float32")
                 sys.exit(1)

        # Check 7: Info is_terminal
        if 'is_terminal' not in info:
            print(f"FAIL: 'is_terminal' missing from info at step {i}")
            sys.exit(1)
            
        if terminated or truncated:
            obs, info = env.reset()
            
    print("SUCCESS: 50 steps passed with valid Dreamer I/O.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-DualGoal-8x8-v0")
    parser.add_argument("--oracle", type=bool, default=False)
    args = parser.parse_args()
    
    check_env_io(args.env_id, oracle=args.oracle)
