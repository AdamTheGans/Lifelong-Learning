from gymnasium.envs.registration import register
from lifelong_learning.envs.multi_goal import MultiGoalEnv
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from lifelong_learning.envs.make_env import make_env

register(
    id='MiniGrid-MultiGoal-8x8-v0',
    entry_point='lifelong_learning.envs.multi_goal:MultiGoalEnv',
    kwargs={'size': 8}
)

register(
    id='MiniGrid-MultiGoal-5x5-v0',
    entry_point='lifelong_learning.envs.multi_goal:MultiGoalEnv',
    kwargs={'size': 5}
)