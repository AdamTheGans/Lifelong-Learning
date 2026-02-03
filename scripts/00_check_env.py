from __future__ import annotations

import argparse
import numpy as np

from lifelong_learning.envs.make_env import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--switch_on_reset", action="store_true")
    parser.add_argument("--switch_mid_episode", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    env = make_env(
        args.env_id,
        seed=args.seed,
        switch_on_reset=args.switch_on_reset,
        switch_mid_episode=args.switch_mid_episode,
    )

    obs, info = env.reset(seed=args.seed)
    print(f"env_id: {args.env_id}")
    print(f"obs type: {type(obs)}  shape: {getattr(obs, 'shape', None)}  dtype: {getattr(obs, 'dtype', None)}")
    print(f"obs min/max: {np.min(obs):.3f}/{np.max(obs):.3f}")
    print(f"action_space: {env.action_space}")
    print(f"reset info: {info}")

    regime_prev = info.get("regime_id", None)

    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        regime_now = info.get("regime_id", None)
        if regime_now != regime_prev:
            print(f"[t={t+1}] REGIME CHANGED: {regime_prev} -> {regime_now} (switch_step={info.get('switch_step')})")
            regime_prev = regime_now

        if (t + 1) % 10 == 0:
            print(f"[t={t+1}] reward={reward:.3f} terminated={terminated} truncated={truncated} regime={regime_now}")

        if terminated or truncated:
            obs, info = env.reset()
            regime_prev = info.get("regime_id", None)
            print(f"--- episode reset: info={info}")

    env.close()
    print("done.")


if __name__ == "__main__":
    main()
