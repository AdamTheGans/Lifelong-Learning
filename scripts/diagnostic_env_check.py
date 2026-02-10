"""
Diagnostic script for DreamerV3 + MiniGrid DualGoal pipeline.

Checks:
  A) Observation quality: image shape, dtype, min/max/mean, saves PNG
  B) Termination semantics: terminated vs truncated vs is_terminal
  C) Action space: prints action_space.n and mapping
  D) Reward distribution over random episodes

Usage:
  python scripts/diagnostic_env_check.py
"""
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import lifelong_learning.envs  # Register envs
from lifelong_learning.envs.make_env import make_env


def run_diagnostics(env_id="MiniGrid-DualGoal-8x8-v0", num_episodes=50, seed=42):
    print("=" * 60)
    print("  DreamerV3 Pipeline Diagnostic")
    print("=" * 60)

    # Create env exactly as training does
    env = make_env(
        env_id=env_id,
        seed=seed,
        dreamer_compatible=True,
        oracle_mode=False,
        steps_per_regime=None,
        episodes_per_regime=None,
    )

    # --- A) Action Space ---
    print(f"\n{'─'*40}")
    print("A) ACTION SPACE")
    print(f"{'─'*40}")
    print(f"  action_space: {env.action_space}")
    print(f"  action_space.n: {env.action_space.n}")

    # --- B) Observation Quality (first reset) ---
    print(f"\n{'─'*40}")
    print("B) OBSERVATION QUALITY")
    print(f"{'─'*40}")

    obs, info = env.reset()

    assert isinstance(obs, dict), f"FAIL: obs is {type(obs)}, expected dict"
    assert "image" in obs, f"FAIL: 'image' not in obs. Keys: {list(obs.keys())}"

    img = obs["image"]
    print(f"  obs keys: {sorted(obs.keys())}")
    print(f"  image shape: {img.shape}")
    print(f"  image dtype: {img.dtype}")
    print(f"  image min:   {img.min()}")
    print(f"  image max:   {img.max()}")
    print(f"  image mean:  {img.mean():.1f}")

    if img.max() <= 15:
        print("  ⚠️  WARNING: image max is very low — possible symbolic/dark frames!")
    elif img.max() >= 200:
        print("  ✅ Image appears to be valid RGB (max >= 200)")

    if img.mean() < 5:
        print("  ⚠️  WARNING: image mean is near 0 — likely black frames!")
    else:
        print(f"  ✅ Image mean is {img.mean():.1f} — looks reasonable")

    # Save frame as PNG
    try:
        from PIL import Image
        save_dir = "logdir"
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, "diagnostic_frame_0.png")
        Image.fromarray(img).save(png_path)
        print(f"  📸 Saved frame to: {png_path}")
    except ImportError:
        print("  (PIL not available — skipping PNG save)")

    # Check a few steps for variation
    prev_mean = img.mean()
    changes = 0
    for i in range(10):
        action = env.action_space.sample()
        obs2, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs2, _ = env.reset()
        new_mean = obs2["image"].mean()
        if abs(new_mean - prev_mean) > 0.1:
            changes += 1
        prev_mean = new_mean

    print(f"  Image mean changed in {changes}/10 steps: "
          f"{'✅ Dynamic' if changes >= 2 else '⚠️  Possibly static'}")

    # --- C) Termination Semantics ---
    print(f"\n{'─'*40}")
    print("C) TERMINATION SEMANTICS")
    print(f"{'─'*40}")

    n_terminated = 0
    n_truncated = 0
    n_is_terminal_true = 0
    n_is_terminal_false_when_last = 0
    n_violations = 0
    rewards = []
    good_goals = 0
    bad_goals = 0
    timeouts = 0
    episode_count = 0
    ep_reward = 0.0
    ep_len = 0

    obs, info = env.reset()

    while episode_count < num_episodes:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_len += 1

        if terminated or truncated:
            if terminated:
                n_terminated += 1
            if truncated:
                n_truncated += 1

            is_term = info.get("is_terminal", None)
            if is_term is True:
                n_is_terminal_true += 1
            elif is_term is False:
                n_is_terminal_false_when_last += 1

            # Violation check: truncated should NEVER have is_terminal=True
            if truncated and not terminated and is_term is True:
                n_violations += 1
                print(f"  ⚠️  VIOLATION at ep {episode_count}: "
                      f"truncated=True but is_terminal=True!")

            # Track outcomes
            if info.get("reached_good_goal", 0) > 0:
                good_goals += 1
            elif info.get("reached_bad_goal", 0) > 0:
                bad_goals += 1
            elif info.get("timed_out", 0) > 0:
                timeouts += 1

            rewards.append(ep_reward)
            episode_count += 1
            ep_reward = 0.0
            ep_len = 0
            obs, info = env.reset()

    print(f"  Episodes completed: {episode_count}")
    print(f"  Terminated:      {n_terminated}")
    print(f"  Truncated:       {n_truncated}")
    print(f"  is_terminal=True when last:  {n_is_terminal_true}")
    print(f"  is_terminal=False when last: {n_is_terminal_false_when_last}")
    print(f"  Violations (truncated + is_terminal=True): {n_violations}")
    if n_violations == 0:
        print("  ✅ No termination semantic violations")
    else:
        print("  ❌ VIOLATIONS FOUND — timeouts are being treated as terminal!")

    # --- D) Reward & Outcome Distribution ---
    print(f"\n{'─'*40}")
    print("D) REWARD & OUTCOME DISTRIBUTION (random policy)")
    print(f"{'─'*40}")

    rewards = np.array(rewards)
    print(f"  Avg episode reward: {rewards.mean():.2f} (std {rewards.std():.2f})")
    print(f"  Min/Max reward:     {rewards.min():.2f} / {rewards.max():.2f}")
    print(f"  Good goal reached:  {good_goals}/{episode_count} "
          f"({100*good_goals/episode_count:.0f}%)")
    print(f"  Bad goal reached:   {bad_goals}/{episode_count} "
          f"({100*bad_goals/episode_count:.0f}%)")
    print(f"  Timed out:          {timeouts}/{episode_count} "
          f"({100*timeouts/episode_count:.0f}%)")

    # With 3 actions on 8x8, random policy should reach SOME goals
    if good_goals + bad_goals == 0:
        print("  ⚠️  WARNING: random policy never reached a goal in "
              f"{episode_count} episodes!")
    else:
        print(f"  ✅ Random policy reaches goals "
              f"{100*(good_goals+bad_goals)/episode_count:.0f}% of the time")

    env.close()
    print(f"\n{'='*60}")
    print("  Diagnostic Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_diagnostics()
