import gymnasium as gym
import numpy as np
from lifelong_learning.envs.multi_goal import MultiGoalEnv
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper

def find_valid_start_pos(env, goal_pos):
    """Finds a walkable neighbor of goal_pos and the direction to face it."""
    directions = [
        (0, (-1, 0)), # East  (start one cell left)
        (1, (0, -1)), # South (start one cell above)
        (2, (1, 0)),  # West  (start one cell right)
        (3, (0, 1))   # North (start one cell below)
    ]

    for direction, (dx, dy) in directions:
        start_pos = (goal_pos[0] + dx, goal_pos[1] + dy)
        try:
            cell = env.unwrapped.grid.get(*start_pos)
            if cell is None or cell.type != "wall":
                return start_pos, direction
        except:
            continue
    return None, None

def test_rewards():
    num_regimes = 4
    base_env = MultiGoalEnv(size=8, num_goals=num_regimes, max_steps=100)
    env = RegimeGoalSwapWrapper(base_env, start_regime=0, num_regimes=num_regimes, steps_per_regime=1000)

    print(f"Testing rewards with {num_regimes} goals/regimes...")

    for regime in range(num_regimes):
        print(f"\n--- Testing Regime {regime} ---")
        
        # Advance steps to force correct regime
        env.cumulative_steps = regime * 1005
        
        # Test hitting ALL goals in this specific regime
        for goal_idx in range(num_regimes):
            obs, info = env.reset()
            assert info["regime_id"] == regime, f"Expected regime {regime}, got {info['regime_id']}"
            
            target_goal_pos = env.unwrapped.goal_positions[goal_idx]
            target_color = env.unwrapped.goal_colors[goal_idx]
            
            start_pos, direction = find_valid_start_pos(env, target_goal_pos)
            assert start_pos is not None, f"Could not find start pos for Goal {goal_idx} ({target_color})"
            
            env.unwrapped.agent_pos = start_pos
            env.unwrapped.agent_dir = direction
            
            obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
            
            print(f"Goal index hit: {goal_idx} ({target_color}). Reward: {reward}")
            
            # The currently active regime should yield +5, anything else -1 (minus base step penalty)
            if goal_idx == regime:
                expected = 5.0 - 0.01
            else:
                expected = -1.0 - 0.01
                
            assert np.isclose(reward, expected), f"Expected {expected}, got {reward} for Goal {goal_idx} in Regime {regime}"
            assert terminated, "Should terminate on goal"
            
    # Step cost (no goal hit)
    obs, info = env.reset()
    env.unwrapped.agent_pos = (1, 1)
    # Ensure (1,1) is not a goal
    while env.unwrapped.agent_pos in env.unwrapped.goal_positions:
         env.unwrapped.agent_pos = (1, env.unwrapped.agent_pos[1] + 1)

    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.left)
    print(f"\nStep Cost Reward: {reward}")
    assert np.isclose(reward, -0.01), f"Expected -0.01, got {reward}"
    assert not terminated

    print("\nALL REWARD TESTS PASSED!")

if __name__ == "__main__":
    test_rewards()
