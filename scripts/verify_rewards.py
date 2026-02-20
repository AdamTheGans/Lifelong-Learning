
import gymnasium as gym
import numpy as np
from lifelong_learning.envs.dual_goal import DualGoalEnv
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from minigrid.core.world_object import Goal

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
    base_env = DualGoalEnv(size=8, max_steps=100)
    env = RegimeGoalSwapWrapper(base_env, start_regime=0, steps_per_regime=1000)

    print("Testing rewards...")

    # --- Regime 0: Green = +5, Blue = -1 ---

    obs, info = env.reset()
    assert info["regime_id"] == 0, "Should start in regime 0"

    # Hit Green (good goal in regime 0)
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

    # Hit Blue (bad goal in regime 0)
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

    # Step cost (no goal hit)
    obs, info = env.reset()
    env.unwrapped.agent_pos = (1, 1)
    while env.unwrapped.agent_pos == env.unwrapped.green_goal_pos or env.unwrapped.agent_pos == env.unwrapped.blue_goal_pos:
         env.unwrapped.agent_pos = (1, 2)

    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.left)
    print(f"Step Cost Reward: {reward}")
    assert np.isclose(reward, -0.01), f"Expected -0.01, got {reward}"
    assert not terminated

    # --- Regime 1: Green = -1, Blue = +5 ---

    # Force regime switch by advancing cumulative steps past the threshold
    env.cumulative_steps = 1005

    # Hit Green (bad goal in regime 1)
    obs, info = env.reset()
    assert info["regime_id"] == 1, f"Should be regime 1, got {info['regime_id']}"

    green_pos = env.unwrapped.green_goal_pos
    start_pos, direction = find_valid_start_pos(env, green_pos)
    print(f"R1 Green Pos: {green_pos}, Start: {start_pos}")

    env.unwrapped.agent_pos = start_pos
    env.unwrapped.agent_dir = direction

    obs, reward, terminated, truncated, info = env.step(env.unwrapped.actions.forward)
    print(f"Regime 1 Hit Green Reward: {reward}")
    expected = -1.0 - 0.01
    assert np.isclose(reward, expected), f"Expected {expected}, got {reward}"

    # Hit Blue (good goal in regime 1)
    obs, info = env.reset()
    env.cumulative_steps = 1005

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
