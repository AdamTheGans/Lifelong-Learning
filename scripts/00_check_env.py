import gymnasium as gym
import minigrid  # noqa: F401

def main():
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")  # try "rgb_array" if needed
    obs, info = env.reset(seed=0)
    print("obs keys:", obs.keys() if isinstance(obs, dict) else type(obs))
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()
