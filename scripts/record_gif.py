import argparse
import os
import torch
import numpy as np
from PIL import Image

from lifelong_learning.envs.make_env import make_env
from lifelong_learning.agents.ppo.network import CNNActorCritic

def main():
    parser = argparse.ArgumentParser(description="Record a GIF of the agent or random/manual actions.")
    parser.add_argument("--env_id", type=str, default="MiniGrid-DualGoal-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained PPO model checkpoint")
    parser.add_argument("--out", type=str, default="gameplay.gif")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--manual", action="store_true", help="Manually enter actions (0: Left, 1: Right, 2: Forward)")
    args = parser.parse_args()

    # Enable rendering for the GIF
    env = make_env(args.env_id, seed=args.seed, render_mode="rgb_array")
    
    agent = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n
        agent = CNNActorCritic(obs_shape, action_dim).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.eval()

    obs, info = env.reset(seed=args.seed)
    frames = []

    # Get first frame
    frame = env.unwrapped.get_frame(highlight=False)
    if frame is not None:
        frames.append(Image.fromarray(frame))

    for step in range(args.max_steps):
        if args.manual:
            print(f"Step {step} - Env info: {info}")
            try:
                action_str = input("Action (0=Left, 1=Right, 2=Forward): ")
                action = int(action_str.strip())
            except ValueError:
                print("Invalid input, defaulting to Forward (2).")
                action = 2
        elif agent:
            with torch.no_grad():
                o_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                regime_id_val = info.get("regime_id", 0)
                regime_id = torch.tensor([regime_id_val], dtype=torch.long).to(device)
                action_tensor, _, _, _ = agent.act(o_tensor, regime_id=regime_id)
                action = action_tensor.item()
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.unwrapped.get_frame(highlight=False)
        if frame is not None:
            frames.append(Image.fromarray(frame))

        if terminated or truncated:
            print(f"Episode ended at step {step} with reward {reward}")
            break

    if frames:
        print(f"Saving {len(frames)} frames to {args.out}...")
        frames[0].save(
            args.out,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            # 300 ms per frame -> ~3.33 FPS
            duration=300, 
            loop=0
        )
        print(f"Successfully saved {args.out}")
    else:
        print("No frames were rendered.")
        
    env.close()

if __name__ == "__main__":
    main()
