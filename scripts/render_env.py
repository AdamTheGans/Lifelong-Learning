import os
import time
import torch
import imageio
import numpy as np
import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper
from lifelong_learning.envs.regime_wrapper import RegimeGoalSwapWrapper
from lifelong_learning.envs.wrappers.action_reduce import ActionReduceWrapper
from lifelong_learning.envs.wrappers.one_hot import OneHotPartialObsWrapper
from lifelong_learning.agents.ppo.network import CNNActorCritic

# Load the trained checkpoint
model_path = "evals/eval_hyperparams_ep1_env0_update2197.pt"

print(f"Loading checkpoint {model_path}")
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

# Make the environment but enable rgb_array rendering
# Note: we need the original gym.make to return the rgb_array so we can't fully use make_env directly
env = gym.make("MiniGrid-MultiGoal-8x8-v0", render_mode="rgb_array", num_goals=3, max_episode_steps=128)
env = FullyObsWrapper(env)
env = ActionReduceWrapper(env, actions=[0, 1, 2])
env = OneHotPartialObsWrapper(env, dict_mode=False)
env = RegimeGoalSwapWrapper(
    env,
    start_regime=0,
    num_regimes=3,
    steps_per_regime=50, # very fast transition for visualization
    seed=42
)

# Extract obs shape to build network
obs_shape = env.observation_space.shape
n_actions = env.action_space.n

model = CNNActorCritic(obs_shape, n_actions)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

frames = []
obs, info = env.reset(seed=42)

# Save the frame
frame = env.render()
frames.append(frame)

print("Simulating agent...")
# Run for 150 steps = 3 regimes at 50 steps each
for step in range(150):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action, _, _, _ = model.act(obs_t)
        
    action_val = action.item()
    obs, reward, terminated, truncated, info = env.step(action_val)
    
    frame = env.render()
    frames.append(frame)
    
    if terminated or truncated:
        obs, info = env.reset()
        frames.append(env.render())

print(f"Saving GIF with {len(frames)} frames...")
os.makedirs("evals", exist_ok=True)
output_path = "evals/agent_3regimes.gif"
imageio.mimsave(output_path, frames, fps=10)
print(f"Saved to {output_path}")

env.close()
