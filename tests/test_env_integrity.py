import unittest
import numpy as np
import gymnasium as gym

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Import our environment factory
from lifelong_learning.envs.make_env import make_env

class BaseEnvIntegrity(object):
    """
    Base class for environment integrity tests.
    Subclasses must implement setUp to create either a Dreamer or PPO env.
    """
    def setUpEnv(self, dreamer_compatible):
        self.dreamer_compatible = dreamer_compatible
        self.env = make_env(
            env_id="MiniGrid-DualGoal-8x8-v0",
            seed=42,
            dreamer_compatible=dreamer_compatible,
            symbolic=True,
            steps_per_regime=1000,
        )
        self.obs_shape = (21, 8, 8) 

    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()

    def get_raw_obs(self, obs):
        """Standardizes observation format between Dreamer (dict) and PPO (tensor)."""
        if self.dreamer_compatible:
            return obs['image']
        return obs

    def decode_observation(self, obs):
        """
        Decodes the observation back to its component grids.
        Handles both flattened (Dreamer) and 3D (PPO) formats.
        """
        flat_obs = self.get_raw_obs(obs)
        
        # 1. Ensure 3D (C, H, W)
        if flat_obs.ndim == 1:
            obs_tensor = flat_obs.reshape(self.obs_shape)
        else:
            obs_tensor = flat_obs
        
        # 2. Decode One-Hot
        # Channels 0-10: Objects (11)
        # Channels 11-16: Colors (6)
        # Channels 17-20: States/Dir (4)
        
        obj_logits = obs_tensor[0:11, :, :]
        col_logits = obs_tensor[11:17, :, :]
        state_logits = obs_tensor[17:21, :, :]
        
        obj_idx = np.argmax(obj_logits, axis=0) # (H, W) or (X, Y)
        col_idx = np.argmax(col_logits, axis=0) # (H, W)
        state_idx = np.argmax(state_logits, axis=0) # (H, W)
        
        return obj_idx, col_idx, state_idx

    def test_data_integrity(self):
        """Data Type & Range Sanity"""
        print(f"\n=== Test: Data Integrity ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        obs, _ = self.env.reset()
        raw_obs = self.get_raw_obs(obs)
        
        self.assertEqual(raw_obs.dtype, np.float32, "Observation must be float32")
        self.assertTrue(np.all(np.logical_or(raw_obs == 0.0, raw_obs == 1.0)), 
                        "Observation must be purely One-Hot (0.0 or 1.0).")
        
        # Check Shape
        if self.dreamer_compatible:
            self.assertEqual(raw_obs.shape, (1344,), "Dreamer obs should be flattened vector of 1344 (21*8*8)")
        else:
            self.assertEqual(raw_obs.shape, (21, 8, 8), "PPO obs should be 3D tensor (21, 8, 8)")
            
        print("PASSED")

    def test_reset_cleanliness(self):
        """Reset Cleanliness"""
        print(f"\n=== Test: Reset Cleanliness ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        self.env.reset()
        self.env.step(self.env.action_space.sample())
        
        obs, _ = self.env.reset()
        obj_idx, _, _ = self.decode_observation(obs)
        
        # Count Goals
        goal_mask = (obj_idx == OBJECT_TO_IDX['goal'])
        num_goals = np.sum(goal_mask)
        self.assertEqual(num_goals, 2, f"Should have exactly 2 goals after reset, found {num_goals}")
        print("PASSED")

    def test_full_compass_sweep(self):
        """Full Compass Sweep (0,1,2,3) verification"""
        print(f"\n=== Test: Full Compass Sweep ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        self.env.reset()
        
        for i in range(4):
            current_dir_gt = self.env.unwrapped.agent_dir
            
            # Action: Right turn (PPO uses standard minigrid, but make_env might wrap it)
            # In make_env: if dreamer_compatible, ActionReduceWrapper is applied.
            # If not, no ActionReduce. 
            # MiniGrid standard: 1 = Right, 0 = Left, 2 = Forward
            step_action = 1 
            obs, _, _, _, _ = self.env.step(step_action)
            
            expected_dir = (current_dir_gt + 1) % 4
            actual_pos = self.env.unwrapped.agent_pos
            actual_dir = self.env.unwrapped.agent_dir
            
            self.assertEqual(actual_dir, expected_dir, "Agent did not rotate correctly in GT.")
            
            _, _, state = self.decode_observation(obs)
            
            x, y = actual_pos
            # Our probe discovered (X, Y) layout in the processed obs
            agent_cell_state = state[x, y] 
            
            self.assertEqual(agent_cell_state, expected_dir, 
                             f"Compass Failure! GT Dir={expected_dir}, Obs Encoded={agent_cell_state}")
            
        print("PASSED")
        
    def test_scenario_collision(self):
        """Collision/Physics Test"""
        print(f"\n=== Test: Collision Physics ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        self.env.reset()
        
        self.env.unwrapped.agent_pos = np.array([1, 1])
        self.env.unwrapped.agent_dir = 2 # West
        
        start_pos = self.env.unwrapped.agent_pos.copy()
        obs, _, _, _, _ = self.env.step(2) # Forward
        
        end_pos = self.env.unwrapped.agent_pos
        self.assertTrue(np.array_equal(start_pos, end_pos), "Agent should have collided!")
        
        _, _, state = self.decode_observation(obs)
        x, y = end_pos
        self.assertEqual(state[x, y], 2, "Agent direction in obs changed after collision!")
        print("PASSED")

    def test_scenario_basic_navigation(self):
        """Scenario: Basic Navigation (Empty path)"""
        print(f"\n=== Test: Basic Navigation ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        self.env.reset()
        
        # Place agent in a clear spot
        self.env.unwrapped.agent_pos = np.array([1, 1])
        self.env.unwrapped.agent_dir = 0 # East
        
        start_pos = self.env.unwrapped.agent_pos.copy()
        
        # Action: Forward
        obs, _, _, _, _ = self.env.step(2)
        
        end_pos = self.env.unwrapped.agent_pos
        
        # Check: Position changed
        expected_pos = start_pos + np.array([1, 0])
        self.assertTrue(np.array_equal(end_pos, expected_pos), "Agent failed to move forward in open space!")
        
        # Check: Obs reflects new position
        obj, col, state = self.decode_observation(obs)
        self.assertEqual(state[2, 1], 0, f"Observation doesn't show agent at new position (2, 1)!")
        print("PASSED")

    def test_color_isolation(self):
        """Color Channel Isolation"""
        print(f"\n=== Test: Color Isolation ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        obs, _ = self.env.reset()
        raw_obs = self.get_raw_obs(obs)
        
        if raw_obs.ndim == 1:
            obs_tensor = raw_obs.reshape(self.obs_shape)
        else:
            obs_tensor = raw_obs
            
        IDX_WALL = OBJECT_TO_IDX['wall'] # 1
        IDX_GREY = COLOR_TO_IDX['grey'] # 5
        IDX_GREEN = COLOR_TO_IDX['green'] # 1
        IDX_BLUE = COLOR_TO_IDX['blue'] # 2
        
        wall_map = obs_tensor[IDX_WALL, :, :]
        grey_map = obs_tensor[11 + IDX_GREY, :, :]
        green_map = obs_tensor[11 + IDX_GREEN, :, :]
        blue_map = obs_tensor[11 + IDX_BLUE, :, :]
        
        self.assertEqual(np.min(grey_map[wall_map == 1]), 1.0, "Found a Wall that is not Grey!")
        self.assertEqual(np.max(green_map[wall_map == 1]), 0.0, "Found a Wall leaking into Green channel!")
        self.assertEqual(np.max(blue_map[wall_map == 1]), 0.0, "Found a Wall leaking into Blue channel!")
        print("PASSED")

    def test_scenario_goals(self):
        """Scenario: Goals correctness and distinctness"""
        print(f"\n=== Test: Goal Correctness ({'Dreamer' if self.dreamer_compatible else 'PPO'}) ===")
        self.env.reset()
        
        goal_pos = self.env.unwrapped.green_goal_pos
        target_pos = np.array([goal_pos[0]-1, goal_pos[1]])
        self.env.unwrapped.agent_pos = target_pos
        self.env.unwrapped.agent_dir = 0 # East -> Facing Goal
        
        obs, reward, terminated, _, _ = self.env.step(2) # Forward
        
        self.assertTrue(terminated, "Hitting Goal did not terminate episode!")
        # 4.99 = 5.0 goal - 0.01 step penalty
        self.assertAlmostEqual(reward, 4.99, places=2)
        print("PASSED")

class TestDreamerEnvIntegrity(BaseEnvIntegrity, unittest.TestCase):
    def setUp(self):
        self.setUpEnv(dreamer_compatible=True)

class TestPPOEnvIntegrity(BaseEnvIntegrity, unittest.TestCase):
    def setUp(self):
        self.setUpEnv(dreamer_compatible=False)

if __name__ == '__main__':
    unittest.main()
