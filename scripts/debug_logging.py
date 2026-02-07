
import gymnasium as gym
import numpy as np
from lifelong_learning.envs.make_env import make_env
from gymnasium.envs.registration import register

try:
    register(
        id="MiniGrid-DualGoal-8x8-v0",
        entry_point="lifelong_learning.envs.dual_goal:DualGoalEnv",
    )
except:
    pass

def debug_logging():
    env_id = "MiniGrid-DualGoal-8x8-v0"
    num_envs = 4
    
    # Create Vector Env similar to train.py
    def make_thunk(i: int):
        def thunk():
            return make_env(
                env_id=env_id,
                seed=0 + i,
                steps_per_regime=500, # fast switch
                record_stats=True,
            )
        return thunk

    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(num_envs)])
    
    # Apply Wrappers manually as per new train.py
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    
    obs, info = envs.reset(seed=0)
    print(f"Reset Info Keys: {list(info.keys())}")
    
    print("Stepping...")
    for step in range(1000):
        # random actions
        actions = envs.action_space.sample()
        next_obs, reward, terminated, truncated, infos = envs.step(actions)
        
        dones = np.logical_or(terminated, truncated)
        if np.any(dones):
            print(f"\nStep {step}: Episode Finished for envs {np.where(dones)[0]}")
            print(f"Infos Keys: {list(infos.keys())}")
            
            if "_episode" in infos:
                print(f"Has '_episode': {infos['_episode']}")
            else:
                print("MISSING '_episode'")
                
            if "episode" in infos:
                 print(f"Has 'episode' key: {infos['episode']}")
            else:
                 print("MISSING 'episode'")
                 
            break
            
    envs.close()

if __name__ == "__main__":
    debug_logging()
