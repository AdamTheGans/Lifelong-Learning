from __future__ import annotations

import argparse
import numpy as np
from lifelong_learning.envs.make_env import make_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--seed", type=int, default=0)
    
    # New schedule args
    parser.add_argument("--steps_per_regime", type=int, default=None)
    parser.add_argument("--episodes_per_regime", type=int, default=None)
    
    # Legacy args
    parser.add_argument("--switch_on_reset", action="store_true")
    parser.add_argument("--switch_mid_episode", action="store_true")
    
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Checking env with schedule: Steps/Regime={args.steps_per_regime}, Episodes/Regime={args.episodes_per_regime}")

    env = make_env(
        args.env_id,
        seed=args.seed,
        steps_per_regime=args.steps_per_regime,
        episodes_per_regime=args.episodes_per_regime,
        switch_on_reset=args.switch_on_reset,
        switch_mid_episode=args.switch_mid_episode,
    )

    obs, info = env.reset(seed=args.seed)
    regime_prev = info.get("regime_id", 0)

    print(f"Start Regime: {regime_prev}")

    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        regime_now = info.get("regime_id", 0)
        
        # Check for change
        if regime_now != regime_prev:
            print(f"[Step {t+1}] REGIME FLIP: {regime_prev} -> {regime_now}")
            regime_prev = regime_now

        if terminated or truncated:
            # Verify logging
            ep_info = info.get("episode", {})
            print(f"--- Episode Done. Info keys: {list(ep_info.keys())}")
            if f"r_regime_{regime_now}" in ep_info:
                print(f"    Logged Regime {regime_now} return: {ep_info[f'r_regime_{regime_now}']:.2f}")
            
            obs, info = env.reset()
            # Note: Regime might change on reset if using episodes_per_regime
            regime_now = info.get("regime_id", 0)
            if regime_now != regime_prev:
                print(f"[Reset] REGIME FLIP: {regime_prev} -> {regime_now}")
                regime_prev = regime_now

    env.close()
    print("Done.")

if __name__ == "__main__":
    main()
