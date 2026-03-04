from __future__ import annotations
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES

class MultiGoalEnv(MiniGridEnv):
    """
    A grid with multiple goals. Supports up to 6 distinct color goals:
    ('red', 'green', 'blue', 'purple', 'yellow', 'grey').
    """

    def __init__(
        self, 
        size=8, 
        num_goals=2,
        agent_start_pos=None, 
        max_steps: int | None = None, 
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.num_goals = min(num_goals, 6) # MiniGrid has 6 distinct colors commonly used
        
        # Colors guaranteed to be distinct in minigrid
        available_colors = ['green', 'blue', 'purple', 'red', 'yellow', 'grey']
        self.goal_colors = available_colors[:self.num_goals]
        
        # Will store the [x, y] positions of the goals 
        # in the same order as self.goal_colors
        self.goal_positions = []
        
        mission_space = MissionSpace(mission_func=lambda: "Go to the correct goal")
        
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True, 
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.goal_positions = []
        
        # Place Goals
        for color in self.goal_colors:
            pos = self.place_obj(Goal(color=color))
            self.goal_positions.append(pos)

        # Place Agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = 0
        else:
            self.place_agent()

    def step(self, action):
        # 1. Run the standard MiniGrid movement logic
        obs, reward, terminated, truncated, info = super().step(action)

        # 2. Check for manual termination (Goal Hit)
        # MiniGrid's default step only checks self.goal_pos (singular)
        # We must check all of our goals regardless of what minigrid base env says.
        for i, goal_pos in enumerate(self.goal_positions):
            if np.array_equal(self.agent_pos, goal_pos):
                terminated = True
                # Trigger positive base reward so the wrapper sees it as a success
                reward = 1.0 - 0.9 * (self.step_count / self.max_steps) 
                info["hit_goal_index"] = i
                break

        return obs, reward, terminated, truncated, info
