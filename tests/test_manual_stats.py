import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.registration import register

register(
    id="MiniGrid-DualGoal-8x8-v0",
    entry_point="lifelong_learning.envs.dual_goal:DualGoalEnv",
)

def make_test_env(env_id, seed, idx):
    def thunk():
        # Use a standard env
        env = gym.make(env_id, render_mode=None)
        # Force a timeout at 10 steps
        env = TimeLimit(env, max_episode_steps=10) 
        env.action_space.seed(seed + idx)
        return env
    return thunk

def test_manual_stats_logic():
    """
    Verifies that manually accumulating rewards/lengths works
    correctly when the environment truncates (times out).
    """
    num_envs = 4
    env_id = "MiniGrid-DualGoal-8x8-v0"
    
    # 1. Create Vector Env (No Stats Wrapper!)
    envs = gym.vector.SyncVectorEnv([
        make_test_env(env_id, seed=42, idx=i) for i in range(num_envs)
    ])
    
    print(f"\n[Test] Running {num_envs} envs with MANUAL tracking.")
    print(f"[Test] Max steps set to 10. Expecting logs at step 10.")
    
    # 2. Initialize Manual Trackers (Just like in the new train.py)
    running_returns = np.zeros(num_envs)
    running_lengths = np.zeros(num_envs, dtype=int)
    
    obs, _ = envs.reset()
    
    stats_recorded = False

    # Run for 15 steps
    for step in range(15):
        actions = envs.action_space.sample()
        
        # Standard Step
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        
        # Calculate Done
        dones = np.logical_or(terminateds, truncateds)
        
        # [CRITICAL LOGIC] Update trackers BEFORE checking done
        running_returns += rewards
        running_lengths += 1
        
        # Check for completion
        if np.any(dones):
            print(f"Step {step+1}: Encountered done signal.")
            
            # Find which envs finished
            done_indices = np.where(dones)[0]
            
            for i in done_indices:
                # Capture the data
                final_r = running_returns[i]
                final_l = running_lengths[i]
                
                print(f"  -> Env {i} finished. Recorded Return={final_r:.2f}, Length={final_l}")
                
                # Verification:
                # Since we forced max_steps=10, the length MUST be 10.
                if final_l == 10:
                    stats_recorded = True
                else:
                    print(f"  [ERROR] Expected length 10, got {final_l}")

                # Reset trackers for this env
                running_returns[i] = 0
                running_lengths[i] = 0
            
            if stats_recorded:
                print("  [SUCCESS] Manual tracking correctly captured the truncated episode!")
                return

    if not stats_recorded:
        raise RuntimeError("Test finished without capturing any stats.")

if __name__ == "__main__":
    test_manual_stats_logic()