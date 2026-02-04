from __future__ import annotations
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

class DualGoalEnv(MiniGridEnv):
    """
    A grid with two goals: Green and Blue.
    """

    def __init__(
        self, 
        size=8, 
        agent_start_pos=None, 
        max_steps: int | None = None, 
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.green_goal_pos = None
        self.blue_goal_pos = None
        
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

        # Place Goals
        self.green_goal_pos = self.place_obj(Goal(color="green"))
        self.blue_goal_pos = self.place_obj(Goal(color="blue"))

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
        # MiniGrid's default step only checks self.goal_pos. We must check ours.
        if not terminated:
            # Check if agent is on Green
            if np.array_equal(self.agent_pos, self.green_goal_pos):
                terminated = True
                reward = 1.0 # Base reward (RegimeWrapper will swap this later if needed)
            
            # Check if agent is on Blue
            elif np.array_equal(self.agent_pos, self.blue_goal_pos):
                terminated = True
                reward = 1.0 # Trigger "Goal Hit" logic (RegimeWrapper will swap this)

        return obs, reward, terminated, truncated, info