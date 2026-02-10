"""
Interactive training run analyzer.
Works both as a standalone script and in Colab/Jupyter notebooks.
Usage:
  python scripts/analyze_runs.py               # interactive, scans ./logdir
  python scripts/analyze_runs.py path/to/run   # direct path to a run folder
"""
import json
import os
import sys
import numpy as np

try:
    import matplotlib
    # Use non-interactive backend if not in notebook
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def find_runs(logdir="logdir"):
    """Find all run directories that contain a metrics.jsonl file."""
    runs = []
    if not os.path.isdir(logdir):
        return runs
    for name in sorted(os.listdir(logdir)):
        path = os.path.join(logdir, name, "metrics.jsonl")
        if os.path.isfile(path):
            runs.append((name, path))
    # Also check root-level directories (e.g. baseline_new/)
    for name in sorted(os.listdir(".")):
        if name == logdir:
            continue
        path = os.path.join(name, "metrics.jsonl")
        if os.path.isfile(path):
            runs.append((name, path))
    return runs


def load_metrics(jsonl_path):
    """Load all metrics from a JSONL file."""
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_series(entries, key):
    """Extract (steps, values) for a given metric key."""
    steps, vals = [], []
    for e in entries:
        if key in e and e[key] is not None:
            steps.append(e.get("step", 0))
            vals.append(float(e[key]))
    return np.array(steps), np.array(vals)


def smooth(values, weight=0.8):
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values, dtype=float)
    if len(values) == 0:
        return smoothed
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def print_summary(entries):
    """Print a text summary of key metrics."""
    ep_steps, ep_scores = extract_series(entries, "episode/score")
    _, ep_lengths = extract_series(entries, "episode/length")

    if len(ep_scores) == 0:
        print("  No episode data found.")
        return

    n = len(ep_scores)
    q1 = ep_scores[: n // 4]
    q4 = ep_scores[3 * n // 4 :]

    print(f"  Total log entries:    {len(entries)}")
    print(f"  Episode entries:      {n}")
    print(f"  Step range:           {ep_steps[0]:,} → {ep_steps[-1]:,}")
    print(f"  Score (first 25%):    {np.mean(q1):+.2f}  (std {np.std(q1):.2f})")
    print(f"  Score (last 25%):     {np.mean(q4):+.2f}  (std {np.std(q4):.2f})")
    if len(ep_lengths) > 0:
        l1 = ep_lengths[: n // 4]
        l4 = ep_lengths[3 * n // 4 :]
        print(f"  Length (first 25%):   {np.mean(l1):.0f}")
        print(f"  Length (last 25%):    {np.mean(l4):.0f}")

    # Epstats
    for key, label in [
        ("epstats/log/reached_good_goal/max", "Good goal rate"),
        ("epstats/log/reached_bad_goal/max", "Bad goal rate"),
        ("epstats/log/timed_out/max", "Timeout rate"),
    ]:
        s, v = extract_series(entries, key)
        if len(v) > 2:
            last = v[-min(5, len(v)) :]
            print(f"  {label} (last 5):  {np.mean(last):.0%}")

    # Losses
    for key, label in [
        ("train/loss/image", "Image loss"),
        ("train/loss/rew", "Reward loss"),
        ("train/loss/dyn", "Dynamics loss"),
    ]:
        s, v = extract_series(entries, key)
        if len(v) > 1:
            print(f"  {label}:           {v[0]:.0f} → {v[-1]:.1f}")


def plot_run(entries, run_name):
    """Create a multi-panel figure with key training metrics."""
    if not HAS_MPL:
        print("  matplotlib not installed — skipping graphs.")
        print("  Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Training Analysis: {run_name}", fontsize=14, fontweight="bold")
    alpha_raw = 0.15  # transparency for raw data
    smoothing = 0.85

    # --- Panel 1: Episode Score ---
    ax = axes[0, 0]
    s, v = extract_series(entries, "episode/score")
    if len(v) > 0:
        ax.scatter(s, v, alpha=alpha_raw, s=3, c="steelblue", label="raw")
        ax.plot(s, smooth(v, smoothing), c="navy", lw=2, label="smoothed")
        ax.axhline(y=0, color="gray", ls="--", lw=0.5)
    ax.set_title("Episode Score")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)

    # --- Panel 2: Episode Length ---
    ax = axes[0, 1]
    s, v = extract_series(entries, "episode/length")
    if len(v) > 0:
        ax.scatter(s, v, alpha=alpha_raw, s=3, c="coral", label="raw")
        ax.plot(s, smooth(v, smoothing), c="darkred", lw=2, label="smoothed")
        ax.axhline(y=256, color="gray", ls="--", lw=0.5, label="timeout (256)")
    ax.set_title("Episode Length")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Length")
    ax.legend(fontsize=8)

    # --- Panel 3: Goal Rates ---
    ax = axes[0, 2]
    for key, color, label in [
        ("epstats/log/reached_good_goal/max", "green", "Good goal"),
        ("epstats/log/reached_bad_goal/max", "red", "Bad goal"),
        ("epstats/log/timed_out/max", "gray", "Timeout"),
    ]:
        s, v = extract_series(entries, key)
        if len(v) > 0:
            ax.plot(s, smooth(v, 0.7), c=color, lw=2, label=label)
            ax.scatter(s, v, alpha=alpha_raw, s=5, c=color)
    ax.set_title("Goal Rates (per episode)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # --- Panel 4: Regime ID ---
    ax = axes[1, 0]
    s, v = extract_series(entries, "epstats/log/regime_id/avg")
    if len(v) > 0:
        ax.plot(s, v, c="purple", lw=2)
        ax.fill_between(s, 0, v, alpha=0.2, color="purple")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Regime ID")
    else:
        ax.set_title("Regime ID (no data)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Regime")

    # --- Panel 5: World Model Losses ---
    ax = axes[1, 1]
    for key, color, label in [
        ("train/loss/image", "blue", "Image"),
        ("train/loss/rew", "orange", "Reward"),
        ("train/loss/dyn", "green", "Dynamics"),
    ]:
        s, v = extract_series(entries, key)
        if len(v) > 0:
            ax.plot(s, v, c=color, lw=1.5, label=label)
    ax.set_title("World Model Losses")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # --- Panel 6: Gradient Norm ---
    ax = axes[1, 2]
    s, v = extract_series(entries, "train/opt/grad_norm")
    if len(v) > 0:
        ax.plot(s, v, c="teal", lw=1.5)
        ax.set_title("Gradient Norm")
    else:
        ax.set_title("Gradient Norm (no data)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Norm")
    ax.set_yscale("log")

    plt.tight_layout()

    # Save to file and show
    out_path = f"analysis_{run_name.replace('/', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Saved figure to: {out_path}")
    plt.show()


def main():
    # Direct path mode
    if len(sys.argv) > 1:
        path = sys.argv[1]
        jsonl = os.path.join(path, "metrics.jsonl") if os.path.isdir(path) else path
        if not os.path.isfile(jsonl):
            print(f"Error: {jsonl} not found.")
            return
        entries = load_metrics(jsonl)
        name = os.path.basename(os.path.dirname(jsonl) if jsonl.endswith("metrics.jsonl") else path)
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print_summary(entries)
        plot_run(entries, name)
        return

    # Interactive mode
    runs = find_runs("logdir")
    if not runs:
        print("No runs found in ./logdir or current directory.")
        print("Usage: python scripts/analyze_runs.py [path/to/run]")
        return

    print("\n╔══════════════════════════════════════════╗")
    print("║       Training Run Analyzer              ║")
    print("╚══════════════════════════════════════════╝")
    print("\nAvailable runs:\n")
    for i, (name, _) in enumerate(runs, 1):
        print(f"  [{i}] {name}")
    print(f"  [0] Analyze ALL runs")
    print()

    try:
        choice = int(input("Enter the number of the run to analyze: "))
    except (ValueError, EOFError):
        print("Invalid input.")
        return

    if choice == 0:
        selected = runs
    elif 1 <= choice <= len(runs):
        selected = [runs[choice - 1]]
    else:
        print("Invalid selection.")
        return

    for name, path in selected:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        entries = load_metrics(path)
        print_summary(entries)
        plot_run(entries, name)


if __name__ == "__main__":
    main()
