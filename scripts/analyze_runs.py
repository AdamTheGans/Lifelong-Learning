"""
Interactive training run analyzer.
Merges functionality from visualize_regimes.py for enhanced plotting.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    # Use non-interactive backend if not in notebook
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Try imports for TensorBoard reading
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def load_data_from_logdir(logdir, tags=None):
    """Loads scalars for specified tags from TensorBoard event files."""
    if not TENSORBOARD_AVAILABLE:
        print("Error: 'tensorboard' package not installed/found. Cannot read event logs.")
        return {}
        
    print(f"Loading logs from {logdir}...")
    
    # Initialize accumulator (load all scalars)
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    # Check available tags
    available_tags = event_acc.Tags()['scalars']
    
    data = {}
    
    # If no tags specified, load all relevant ones
    if not tags:
        tags = available_tags
        
    for tag in tags:
        if tag not in available_tags:
            continue
            
        events = event_acc.Scalars(tag)
        # Convert to list of [timestamp, step, value]
        raw_data = [[e.wall_time, e.step, e.value] for e in events]
        data[tag] = pd.DataFrame(raw_data, columns=['timestamp', 'step', 'value'])
        
    return data


def detect_switches(data):
    """
    Detects regime switches by analyzing r_regime_0 and r_regime_1 logs.
    Returns a list of steps where switches probably occurred.
    """
    if "charts/r_regime_0" not in data or "charts/r_regime_1" not in data:
        return None

    df0 = data["charts/r_regime_0"].copy()
    df1 = data["charts/r_regime_1"].copy()
    
    df0["regime"] = 0
    df1["regime"] = 1
    
    # Combine and sort by step
    combined = pd.concat([df0, df1]).sort_values("step")
    
    if combined.empty:
        return None

    # Calculate rolling mean of regime label (0 or 1)
    window_size = 20
    combined["smoothed_regime"] = combined["regime"].rolling(window=window_size, center=True).mean()
    
    # Find crossings of 0.5
    combined["pred_regime"] = (combined["smoothed_regime"] > 0.5).astype(int)
    combined["switch"] = combined["pred_regime"].diff().abs()
    
    # Filter for actual switches (value == 1)
    switch_points = combined[combined["switch"] == 1]
    switches = switch_points["step"].tolist()
    
    # Clean up close duplicates
    if not switches:
        return None
        
    cleaned_switches = []
    if switches:
        cleaned_switches.append(switches[0])
        for s in switches[1:]:
            if s - cleaned_switches[-1] > 5000: # Minimum regime length heuristic
                cleaned_switches.append(s)
                
    return cleaned_switches


def smooth_data(data, window_size=10):
    """Applies a moving average smoothing."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def find_runs(logdir="logdir"):
    """Find all run directories that contain tensorboard events."""
    runs = []
    if not os.path.exists(logdir):
        # Try checking current directory for runs folder or similar
        if os.path.exists("runs"):
            logdir = "runs"
        else:
            return runs # Nothing found

    # Walk through directory to find event files
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if "events.out.tfevents" in file:
                # We found a run directory
                rel_path = os.path.relpath(root, start=os.getcwd())
                runs.append((os.path.basename(rel_path), rel_path))
                break # One event file is enough to mark the dir
                
    # Sort
    runs.sort(key=lambda x: x[0])
    return runs


def plot_single_graph(df_main, df_wm, title, ylabel, main_color, wm_color, regime_boundaries, max_step, do_smoothing, output_path=None, ax=None):
    """
    Plots a single graph with main metric on left axis and WM loss on right axis.
    Optionally saves to file if output_path is provided, otherwise plots on provided ax.
    Uses robust zero-alignment logic from visualize_regimes.py.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
        return_fig = True
    else:
        return_fig = False
    
    if output_path is None:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(title, fontsize=16, pad=20)
        
    ax.grid(True, linestyle='--', alpha=0.5)

    # 1. Plot Left Axis (Main Metric)
    if do_smoothing and len(df_main) > 50:
        smooth_window = 20
        # Ensure we don't crash if len < window
        vals_smooth = df_main['value'].rolling(window=min(smooth_window, len(df_main)), min_periods=1).mean()
        line1, = ax.plot(df_main['step'], vals_smooth, color=main_color, alpha=0.9, linewidth=1.5, label=f"{ylabel} (Smoothed)")
        ax.plot(df_main['step'], df_main['value'], color=main_color, alpha=0.15, linewidth=0.5) 
    else:
        line1, = ax.plot(df_main['step'], df_main['value'], color=main_color, alpha=0.9, linewidth=1.5, label=ylabel)
    
    ax.set_ylabel(ylabel, color=main_color, fontsize=12)
    ax.set_xlabel("Global Steps", fontsize=12)
    ax.tick_params(axis='y', labelcolor=main_color)

    # 2. Plot Right Axis (World Model Loss)
    ax2 = ax.twinx()
    if df_wm is not None and not df_wm.empty:
        line2, = ax2.plot(df_wm['step'], df_wm['value'], color=wm_color, alpha=0.7, linewidth=1.0, label="WM Loss (Raw)")
        ax2.set_ylabel("WM Loss (Raw)", color=wm_color, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=wm_color)
    else:
        # Fallback if no WM data
        ax2.set_ylabel("WM Loss (No Data)", color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')

    # 3. Align Zero Lines logic (Complex version)
    try:
        def get_range(df):
            if df is None or df.empty: return 0, 1
            vmin = df['value'].min()
            vmax = df['value'].max()
            margin = (vmax - vmin) * 0.05 if vmax != vmin else 0.1
            return vmin - margin, vmax + margin

        y1_min, y1_max = get_range(df_main)
        y2_min, y2_max = get_range(df_wm)

        up1 = max(y1_max, 0)
        down1 = max(-y1_min, 0)
        up2 = max(y2_max, 0)
        down2 = max(-y2_min, 0)

        f1 = down1 / (up1 + down1) if (up1 + down1) > 0 else 0
        f2 = down2 / (up2 + down2) if (up2 + down2) > 0 else 0
        target_f = max(f1, f2)
        
        def adjust_limits(up, down, f_target):
            if f_target >= 1.0: return -down, 0
            if f_target <= 0.0: return 0, up
            new_down = up * f_target / (1.0 - f_target)
            if new_down >= down:
                return -new_down, up
            return -down, down * (1.0 - f_target) / f_target # simplified backup

        lim1 = adjust_limits(up1, down1, target_f)
        lim2 = adjust_limits(up2, down2, target_f)
        
        ax.set_ylim(lim1)
        ax2.set_ylim(lim2)
    except Exception as e:
        print(f"Warning: Zero alignment failed ({e}), using default scales.")
        
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3, zorder=1)

    # 4. Regime Shading
    regime_colors = ['#e6f5ff', '#fff5e6'] # Light blue, Light orange
    y_min, y_max = ax.get_ylim()

    if regime_boundaries:
        for i in range(len(regime_boundaries) - 1):
            start = regime_boundaries[i]
            end = regime_boundaries[i+1]
            if start >= max_step: break
            if end > max_step: end = max_step
            
            regime_idx = i
            color = regime_colors[regime_idx % 2]
            
            ax.axvspan(start, end, color=color, alpha=0.5, zorder=0)
            
            # Label
            if end - start > (max_step * 0.05): 
                ax.text(start + (end-start)/2, y_max, f"Regime {regime_idx % 2}", 
                         ha='center', va='bottom', fontsize=8, fontweight='bold', color='gray')

    if return_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"  📊 Saved separate plot to: {output_path}")
        else:
            plt.show()


def plot_run(logdir, run_name):
    """Create a multi-panel figure with key training metrics."""
    if not HAS_MPL:
        print("  matplotlib not installed — skipping graphs.")
        return

    # 1. Load Data with TensorBoard tags
    tag_map = {
        "score": "charts/episodic_return",
        "length": "charts/episodic_length",
        "success": "charts/success_rate",
        "failure": "charts/failure_rate",
        "timeout": "charts/timeout_rate",
        "wm_state": "world_model/loss_state",
        "wm_rew": "world_model/loss_reward",
        "wm_total": "world_model/loss_total",
        "intrinsic": "ppo/intrinsic_reward_mean",
        "intrinsic_ratio": "ppo/intrinsic_reward_ratio",
        "regime0": "charts/r_regime_0",
        "regime1": "charts/r_regime_1",
        "step_rew": "charts/reward_step_mean"
    }
    
    data = load_data_from_logdir(logdir, list(tag_map.values()))
    
    if not data:
        print(f"No data found for {run_name}")
        return

    # Detect switches
    switches = detect_switches(data)
    max_step = 0
    for k, df in data.items():
        if not df.empty:
            max_step = max(max_step, df['step'].max())
            
    boundaries = []
    if switches:
        boundaries = [0] + switches + [max_step]
        boundaries = sorted(list(set(boundaries)))
        print(f"  Detected {len(switches)} regime switches.")
    else:
        # Just 0 to max
        boundaries = [0, max_step]

    # --- Generate Separate Regime Plots ---
    # Ensure graphs directory exists
    os.makedirs("graphs", exist_ok=True)

    # 1. Step Reward vs Surprise
    df_step_rew = data.get(tag_map["step_rew"])
    df_wm_rew_loss = data.get(tag_map["wm_rew"])
    
    if df_step_rew is not None:
        out_file = os.path.join("graphs", f"regime_analysis_step_reward_{run_name.replace(os.path.sep, '_')}.png")
        plot_single_graph(
            df_step_rew, df_wm_rew_loss,
            "Regime Analysis: Step Reward vs Surprise", "Mean Step Reward",
            'tab:green', 'tab:red',
            boundaries, max_step, False, output_path=out_file
        )

    # 2. Episodic Return vs Surprise
    df_eps_ret = data.get(tag_map["score"])
    if df_eps_ret is not None:
        out_file = os.path.join("graphs", f"regime_analysis_episodic_return_{run_name.replace(os.path.sep, '_')}.png")
        plot_single_graph(
            df_eps_ret, df_wm_rew_loss,
            "Regime Analysis: Episodic Return vs Surprise", "Episodic Return",
            'tab:blue', 'tab:red',
            boundaries, max_step, True, output_path=out_file
        )

    # 3. Success Rate vs Surprise
    df_success = data.get(tag_map["success"])
    if df_success is not None:
        out_file = os.path.join("graphs", f"regime_analysis_success_rate_{run_name.replace(os.path.sep, '_')}.png")
        plot_single_graph(
            df_success, df_wm_rew_loss,
            "Regime Analysis: Success Rate vs Surprise", "Success Rate",
            'tab:green', 'tab:red',
            boundaries, max_step, True, output_path=out_file
        )


    # --- Generate Main Summary Plot (6 panels) ---
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Training Analysis: {run_name}", fontsize=16, fontweight="bold")
    
    # Panel 1: Episode Score
    ax1 = fig.add_subplot(2, 3, 1)
    if tag_map["score"] in data:
        df = data[tag_map["score"]]
        ax1.scatter(df['step'], df['value'], alpha=0.15, s=3, c="steelblue", label="raw")
        smooth_val = df['value'].rolling(window=20, min_periods=1).mean()
        ax1.plot(df['step'], smooth_val, c="navy", lw=2, label="smoothed")
        ax1.axhline(0, color="gray", ls="--", lw=0.5)
        ax1.set_title("Episodic Return")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No Score Data", ha='center')
        
    # Panel 2: Episode Length
    ax2 = fig.add_subplot(2, 3, 2)
    if tag_map["length"] in data:
        df = data[tag_map["length"]]
        ax2.scatter(df['step'], df['value'], alpha=0.15, s=3, c="coral", label="raw")
        smooth_val = df['value'].rolling(window=20, min_periods=1).mean()
        ax2.plot(df['step'], smooth_val, c="darkred", lw=2, label="smoothed")
        ax2.axhline(256, color="gray", ls="--", lw=0.5)
        ax2.set_title("Episode Length")
    else:
        ax2.text(0.5, 0.5, "No Length Data", ha='center')
        
    # Panel 3: Goal Rates
    ax3 = fig.add_subplot(2, 3, 3)
    has_rates = False
    for key, color, label in [
        (tag_map["success"], "green", "Success"),
        (tag_map["failure"], "red", "Failure"),
        (tag_map["timeout"], "gray", "Timeout"),
    ]:
        if key in data:
            df = data[key]
            smooth_val = df['value'].rolling(window=20, min_periods=1).mean()
            ax3.plot(df['step'], smooth_val, c=color, lw=2, label=label)
            has_rates = True
            
    ax3.set_title("Goal Rates")
    ax3.set_ylim(-0.05, 1.05)
    if has_rates:
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No Goal Rate Data\n(Metrics missing in old runs)", ha='center', va='center', color='gray')
    
    # Panel 4: Intrinsic Reward
    ax4 = fig.add_subplot(2, 3, 4)
    if tag_map["intrinsic"] in data:
        df = data[tag_map["intrinsic"]]
        smooth_val = df['value'].rolling(window=20, min_periods=1).mean()
        ax4.plot(df['step'], smooth_val, c="orange", lw=1.5, label="Mean Intrinsic")
        
    if tag_map["intrinsic_ratio"] in data:
        df = data[tag_map["intrinsic_ratio"]]
        smooth_val = df['value'].rolling(window=20, min_periods=1).mean()
        ax4.plot(df['step'], smooth_val, c="purple", lw=1.5, label="Intrinsic Ratio")
        
    ax4.set_title("Intrinsic Rewards")
    ax4.legend(fontsize=8)
    
    # Panel 5: World Model Losses
    ax5 = fig.add_subplot(2, 3, 5)
    for key, color, label in [
        (tag_map["wm_state"], "blue", "State"),
        (tag_map["wm_rew"], "orange", "Reward"),
        (tag_map["wm_total"], "green", "Total"),
    ]:
        if key in data:
            df = data[key]
            ax5.plot(df['step'], df['value'], c=color, lw=1.5, label=label)
    ax5.set_yscale('log')
    ax5.set_title("World Model Losses (Log)")
    ax5.legend(fontsize=8)

    # Panel 6: Episodic Return vs Surprise (Regime Analysis) — Duplicate of separate plot
    ax6 = fig.add_subplot(2, 3, 6)
    if df_eps_ret is not None:
        plot_single_graph(
            df_eps_ret, df_wm_rew_loss, 
            "Regime Analysis: Episodic Return vs Surprise",
            "Episodic Return",
            'tab:blue', 'tab:red',
            boundaries, max_step, True, 
            ax=ax6
        )
    else:
        ax6.text(0.5, 0.5, "No Episodic Return Data", ha='center', va='center')

    plt.tight_layout()
    
    # Save Main Figure
    out_path = os.path.join("graphs", f"analysis_{run_name.replace(os.path.sep, '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Saved summary figure to: {out_path}")
    
    # Only try to show if we have an interactive backend
    if matplotlib.get_backend().lower() not in ["agg", "cairo", "ps", "pdf", "svg"]:
        try:
            plt.show()
        except Exception:
            pass
    else:
        # Close the figure explicitly
        plt.close(fig)


def main():
    # Direct path mode
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            return
            
        name = os.path.basename(os.path.normpath(path))
        print(f"\n{'='*60}")
        print(f"  Analyzing: {name}")
        print(f"{'='*60}")
        plot_run(path, name)
        return

    # Interactive mode
    runs = find_runs()
    if not runs:
        print("No runs found in ./logdir or ./runs or current directory.")
        print("Usage: python scripts/analyze_runs.py [path/to/run]")
        return

    print("\n╔══════════════════════════════════════════╗")
    print("║       Training Run Analyzer              ║")
    print("╚══════════════════════════════════════════╝")
    print("\nAvailable runs:\n")
    for i, (name, path) in enumerate(runs, 1):
        print(f"  [{i}] {name}  ({path})")
    print(f"  [0] Analyze ALL runs")
    print()

    try:
        choice_str = input("Enter the number of the run to analyze: ")
        choice = int(choice_str)
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
        print(f"  Analyzing: {name}")
        print(f"{'='*60}")
        plot_run(path, name)


if __name__ == "__main__":
    main()
