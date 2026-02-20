import gymnasium as gym
import numpy as np
from lifelong_learning.envs.make_env import make_env

def make_thunk(i):
    def thunk():
        return make_env("MiniGrid-Empty-8x8-v0", seed=0 + i)
    return thunk

def main():
    envs = gym.vector.SyncVectorEnv([make_thunk(i) for i in range(4)])
    obs, info = envs.reset(seed=0)
    print("expected vec obs shape: (4, 21, 8, 8)")
    print("actual vec obs shape:", obs.shape)
    for _ in range(50):
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        obs, reward, term, trunc, infos = envs.step(actions)
    envs.close()
    print("done")

if __name__ == "__main__":
    main()
