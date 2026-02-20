from __future__ import annotations

import argparse
import numpy as np
from lifelong_learning.envs.make_env import make_env

def main():
    parser = argparse.ArgumentParser(description="Sanity check: regime switching behavior")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps_per_regime", type=int, default=None)
    parser.add_argument("--episodes_per_regime", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Checking env with schedule: Steps/Regime={args.steps_per_regime}, Episodes/Regime={args.episodes_per_regime}")

    env = make_env(
        args.env_id,
        seed=args.seed,
        steps_per_regime=args.steps_per_regime,
        episodes_per_regime=args.episodes_per_regime,
    )

    obs, info = env.reset(seed=args.seed)
    regime_prev = info.get("regime_id", 0)

    print(f"Start Regime: {regime_prev}")

    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        regime_now = info.get("regime_id", 0)

        if regime_now != regime_prev:
            print(f"[Step {t+1}] REGIME FLIP: {regime_prev} -> {regime_now}")
            regime_prev = regime_now

        if terminated or truncated:
            obs, info = env.reset()
            regime_now = info.get("regime_id", 0)
            if regime_now != regime_prev:
                print(f"[Reset] REGIME FLIP: {regime_prev} -> {regime_now}")
                regime_prev = regime_now

    env.close()
    print("Done.")

if __name__ == "__main__":
    main()
