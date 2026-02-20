from gymnasium.envs.registration import register
from lifelong_learning.envs.dual_goal import DualGoalEnv
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from lifelong_learning.envs.make_env import make_env

register(
    id='MiniGrid-DualGoal-8x8-v0',
    entry_point='lifelong_learning.envs.dual_goal:DualGoalEnv',
    kwargs={'size': 8}
)

register(
    id='MiniGrid-DualGoal-5x5-v0',
    entry_point='lifelong_learning.envs.dual_goal:DualGoalEnv',
    kwargs={'size': 5}
)