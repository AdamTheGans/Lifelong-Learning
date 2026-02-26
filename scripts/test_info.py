import types
import pprint
import json
import gymnasium as gym
from lifelong_learning.agents.brain.meta_env import MetaEnv

def main():
    try:
        env = gym.vector.SyncVectorEnv([lambda: MetaEnv(env_index=0), lambda: MetaEnv(env_index=1)])
        obs, info = env.reset()
        obs, rew, term, trunc, info = env.step([[0,0,0,0], [0,0,0,0]])
        
        # Write to file
        with open('info_out.txt', 'w') as f:
            pprint.pprint(info, stream=f)
    except Exception as e:
        with open('info_out.txt', 'w') as f:
            f.write(f"Error: {e}")

if __name__ == "__main__":
    main()
