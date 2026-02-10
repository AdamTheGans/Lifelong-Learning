
import gymnasium as gym
import numpy as np
from lifelong_learning.envs.dual_goal import DualGoalEnv
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from minigrid.core.world_object import Goal

def find_valid_start_pos(env, goal_pos):
    """Finds a neighbor of goal_pos that is not a wall."""
    # Try all 4 directions
    # 0: right, 1: down, 2: left, 3: up
    directions = [
        (0, (-1, 0)), # Right (start left)
        (1, (0, -1)), # Down (start above)
        (2, (1, 0)),  # Left (start right)
        (3, (0, 1))   # Up (start below)
    ]
    
    for direction, (dx, dy) in directions:
        start_pos = (goal_pos[0] + dx, goal_pos[1] + dy)
        try:
            cell = env.unwrapped.grid.get(*start_pos)
            # If None or not a wall (can overlap or transparent is usually fine for empty)
            # Actually just check if it's a wall
            if cell is None or cell.type != "wall":
                return start_pos, direction # direction to move to hit goal
        except:
            continue
    return None, None

def test_rewards():
    base_env = DualGoalEnv(size=8, max_steps=100)
    # Wrap with our regime wrapper
    env = RegimeGoalSwapWrapper(base_env, start_regime=0, steps_per_regime=1000)
    
    print("Testing rewards...")
    
    # 1. Test Regime 0: Green=+5, Blue=-1 (plus step cost)
    obs, info = env.reset()
    assert info["regime_id"] == 0, "Should start in regime 0"
    
    green_pos = env.unwrapped.green_goal_pos
    start_pos, direction = find_valid_start_pos(env, green_pos)
    assert start_pos is not None, "Could not find valid start pos for Green"
    
    print(f"Green Pos: {green_pos}, Start Pos: {start_pos}, Move Dir: {direction}")
    
    env.unwrapped.agent_pos = start_pos
    env.unwrapped.agent_dir = direction
    
    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
    print(f"Regime 0 Hit Green Reward: {reward}")
    expected = 5.0 - 0.01
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"
    assert terminated, "Should terminate on goal"

    # 2. Test Regime 0: Hit Blue
    obs, info = env.reset()
    blue_pos = env.unwrapped.blue_goal_pos
    start_pos, direction = find_valid_start_pos(env, blue_pos)
    assert start_pos is not None, "Could not find valid start pos for Blue"
    
    print(f"Blue Pos: {blue_pos}, Start Pos: {start_pos}, Move Dir: {direction}")
    
    env.unwrapped.agent_pos = start_pos
    env.unwrapped.agent_dir = direction
    
    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
    print(f"Regime 0 Hit Blue Reward: {reward}")
    expected = -1.0 - 0.01
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"
    assert terminated
    
    # 3. Test Step Cost (Regime 0)
    obs, info = env.reset()
    env.unwrapped.agent_pos = (1, 1) # Generally safe in 8x8?
    # Ensure (1,1) is not a goal
    while env.unwrapped.agent_pos == env.unwrapped.green_goal_pos or env.unwrapped.agent_pos == env.unwrapped.blue_goal_pos:
         env.unwrapped.agent_pos = (1, 2)
         
    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.left)
    print(f"Step Cost Reward: {reward}")
    assert np.isclose(reward, -0.01), f"Expected -0.01, got {reward}"
    assert not terminated
    
    # 4. Test Regime 1: Green=-1, Blue=+5
    # Force regime switch via cumulative steps
    env.cumulative_steps = 1005
    
    # Hit Green (Bad now)
    obs, info = env.reset()
    # Reset might reset regime based on cumulative episodes? No, steps/episodes.
    # Wrapper reset(): _update_regime_deterministic()
    # But wrapper reset doesn't reset cumulative_steps?
    # Actually checking wrapper code:
    # It preserves cumulative_steps?
    # No, usually cumulative_steps is lifetime.
    # Let's check wrapper source in my head... or view it.
    pass

    # Actually, I need to check if reset clears cumulative_steps.
    # View file showed:
    # self.cumulative_steps = 0 in __init__
    # reset() calls _update_regime_deterministic()
    # It does NOT zero cumulative_steps.
    
    # So if I set it here, it should persist.
    env.cumulative_steps = 1005

    # Hit Green (Bad now)
    obs, info = env.reset()
    assert info["regime_id"] == 1, f"Should be regime 1, got {info['regime_id']}"
    
    # Refresh goal positions!
    green_pos = env.unwrapped.green_goal_pos
    blue_pos = env.unwrapped.blue_goal_pos

    start_pos, direction = find_valid_start_pos(env, green_pos)
    
    # Debug
    print(f"R1 Green Pos: {green_pos}, Start: {start_pos}")

    env.unwrapped.agent_pos = start_pos
    env.unwrapped.agent_dir = direction
    
    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
    print(f"Regime 1 Hit Green Reward: {reward}")
    expected = -1.0 - 0.01
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"
    
    # Hit Blue (Good now)
    obs, info = env.reset()
    env.cumulative_steps = 1005 
    
    # Refresh goal positions again (reset changed them)
    green_pos = env.unwrapped.green_goal_pos
    blue_pos = env.unwrapped.blue_goal_pos
    
    start_pos, direction = find_valid_start_pos(env, blue_pos)
    env.unwrapped.agent_pos = start_pos
    env.unwrapped.agent_dir = direction
    
    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
    print(f"Regime 1 Hit Blue Reward: {reward}")
    expected = 5.0 - 0.01
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"

    print("\nALL REWARD TESTS PASSED!")

if __name__ == "__main__":
    test_rewards()
