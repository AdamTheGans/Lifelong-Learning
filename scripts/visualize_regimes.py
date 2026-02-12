import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# Try imports for TensorBoard reading
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

def smooth_data(data, window_size=10):
    """Applies a moving average smoothing."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def load_data_from_json(json_path):
    """Loads data from a TensorBoard-exported JSON file."""
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} points from {json_path}")
    return pd.DataFrame(raw_data, columns=['timestamp', 'step', 'value'])

def load_data_from_logdir(logdir, tags):
    """Loads scalars for specified tags from TensorBoard event files."""
    if not TENSORBOARD_AVAILABLE:
        print("Error: 'tensorboard' package not installed/found. Cannot read event logs.")
        print("Please install it or use --ppo_json / --wm_json arguments.")
        sys.exit(1)
        
    print(f"Loading logs from {logdir}...")
    
    # Initialize accumulator (load all scalars)
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    # Check available tags
    available_tags = event_acc.Tags()['scalars']
    print(f"Found {len(available_tags)} scalar tags.")
    
    data = {}
    for tag in tags:
        if tag not in available_tags:
            print(f"Warning: Tag '{tag}' not found in logs.")
            # Try fuzzy match or skip
            continue
            
        events = event_acc.Scalars(tag)
        # Convert to list of [timestamp, step, value]
        # Event(wall_time, step, value)
        raw_data = [[e.wall_time, e.step, e.value] for e in events]
        data[tag] = pd.DataFrame(raw_data, columns=['timestamp', 'step', 'value'])
        print(f"Extracted {len(raw_data)} points for '{tag}'")
        
    return data

def detect_switches(data):
    """
    Detects regime switches by analyzing r_regime_0 and r_regime_1 logs.
    Returns a list of steps where switches probably occurred.
    """
    if "charts/r_regime_0" not in data or "charts/r_regime_1" not in data:
        print("Warning: Regime charts not found. Cannot auto-detect switches.")
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
    # 0 means strongly Regime 0, 1 means strongly Regime 1
    # We use a window to smooth out any overlaps/noise
    window_size = 20
    combined["smoothed_regime"] = combined["regime"].rolling(window=window_size, center=True).mean()
    
    # Find crossings of 0.5
    # A switch happens when we go from <0.5 to >0.5 or vice versa
    combined["pred_regime"] = (combined["smoothed_regime"] > 0.5).astype(int)
    combined["switch"] = combined["pred_regime"].diff().abs()
    
    # Filter for actual switches (value == 1)
    switch_points = combined[combined["switch"] == 1]
    
    switches = switch_points["step"].tolist()
    
    # Clean up close duplicates (if the rolling avg dithers)
    if not switches:
        return None
        
    cleaned_switches = []
    if switches:
        cleaned_switches.append(switches[0])
        for s in switches[1:]:
            if s - cleaned_switches[-1] > 20000: # Minimum regime length heuristic
                cleaned_switches.append(s)
                
    return cleaned_switches

def plot_single_graph(df_main, df_wm, title, ylabel, main_color, wm_color, regime_boundaries, max_step, do_smoothing, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)

    # 1. Plot Left Axis (Main Metric)
    if do_smoothing and len(df_main) > 50:
        smooth_window = 20
        vals_smooth = df_main['value'].rolling(window=smooth_window, min_periods=1).mean()
        line1, = ax.plot(df_main['step'], vals_smooth, color=main_color, alpha=0.9, linewidth=1.5, label=f"{ylabel} (Smoothed)")
        ax.plot(df_main['step'], df_main['value'], color=main_color, alpha=0.15, linewidth=0.5) 
    else:
        line1, = ax.plot(df_main['step'], df_main['value'], color=main_color, alpha=0.9, linewidth=1.5, label=ylabel)
    
    ax.set_ylabel(ylabel, color=main_color, fontsize=12)
    ax.set_xlabel("Global Steps", fontsize=12)
    ax.tick_params(axis='y', labelcolor=main_color)

    # 2. Plot Right Axis (World Model Loss)
    ax2 = ax.twinx()
    line2, = ax2.plot(df_wm['step'], df_wm['value'], color=wm_color, alpha=0.7, linewidth=1.0, label="WM Loss (Raw)")
    ax2.set_ylabel("WM Loss (Raw)", color=wm_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=wm_color)

    # 3. Align Zero Lines
    # Calculate data ranges with padding
    def get_range(df):
        vmin = df['value'].min()
        vmax = df['value'].max()
        # Add 5% padding
        margin = (vmax - vmin) * 0.05 if vmax != vmin else 0.1
        return vmin - margin, vmax + margin

    y1_min, y1_max = get_range(df_main)
    y2_min, y2_max = get_range(df_wm)

    # Distances from zero
    up1 = max(y1_max, 0)
    down1 = max(-y1_min, 0)
    up2 = max(y2_max, 0) # Loss is usually positive, but handle robustly
    down2 = max(-y2_min, 0)

    # Calculate fraction of height devoted to "below zero"
    # Handling divide by zero if range is 0 (unlikely)
    f1 = down1 / (up1 + down1) if (up1 + down1) > 0 else 0
    f2 = down2 / (up2 + down2) if (up2 + down2) > 0 else 0
    
    # We must satisfy the maximum "downward" requirement to avoid cutting off data
    # (i.e., we push the zero line 'up' to the highest required relative position)
    target_f = max(f1, f2)
    
    # Recalculate limits based on target fraction
    # new_down = new_up * (target_f / (1 - target_f))
    # We keep the 'up' limit fixed (tight to data) and expand 'down', OR vice versa?
    # If target_f > current_f, we need MORE 'down' space relative to 'up'.
    # So we should expand 'down'.
    
    def adjust_limits(up, down, f_target):
        if f_target >= 1.0: return -down, 0 # All negative
        if f_target <= 0.0: return 0, up    # All positive
        
        # We need down_new / (up + down_new) = f_target
        # down_new * (1 - f_target) = up * f_target
        # down_new = up * f_target / (1 - f_target)
        
        # Check if calculating from 'up' covers 'down'
        new_down = up * f_target / (1.0 - f_target)
        if new_down >= down:
            return -new_down, up
        else:
            # If not, it means we need to expand 'up' to satisfy the ratio with fixed 'down'
            # This happens if we picked f_target from the OTHER axis which had a bigger 'down' ratio,
            # but THIS axis has a huge 'down' magnitude? No, wait.
            # f_target = max(f1, f2). So f_target >= f_this.
            # f_this = down / (up + down).
            # f_target >= down / (up + down)
            # f_target * (up + down) >= down
            # f_target * up >= down * (1 - f_target)
            # up * f_target / (1 - f_target) >= down.
            # So calculating 'new_down' from 'up' is ALWAYS sufficient to cover original 'down'.
            return -new_down, up

    lim1 = adjust_limits(up1, down1, target_f)
    lim2 = adjust_limits(up2, down2, target_f)
    
    ax.set_ylim(lim1)
    ax2.set_ylim(lim2)
    
    # Add Zero Line for clarity
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3, zorder=1)

    # 4. Regime Shading
    regime_colors = ['#e6f5ff', '#fff5e6']
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
            if end - start > 10000: 
                mid_point = start + (end - start) / 2
                ax.text(mid_point, y_max, f"Regime {regime_idx % 2}", 
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='gray')

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")

def plot_regime_analysis(logdir, ppo_json, wm_json, global_regime_steps, output_dir="graphs"):
    df_ppo_step = None
    df_ppo_eps = None
    df_wm = None
    detected_switches = None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    if logdir:
        TAG_PPO_STEP = "charts/reward_step_mean"
        TAG_PPO_EPS = "charts/episodic_return"
        TAG_WM = "world_model/loss_reward"
        TAG_REGIME0 = "charts/r_regime_0"
        TAG_REGIME1 = "charts/r_regime_1"
        
        tags_to_load = [TAG_PPO_STEP, TAG_PPO_EPS, TAG_WM, TAG_REGIME0, TAG_REGIME1]
        data = load_data_from_logdir(logdir, tags_to_load)
        
        if TAG_PPO_STEP in data:
            df_ppo_step = data[TAG_PPO_STEP]
        if TAG_PPO_EPS in data:
            df_ppo_eps = data[TAG_PPO_EPS]
        if TAG_WM in data:
            df_wm = data[TAG_WM]
            
        detected_switches = detect_switches(data)
        if detected_switches:
            print(f"Auto-detected regime switches at steps: {detected_switches}")
        
    else:
        # Fallback to JSON
        if not ppo_json or not wm_json:
            print("Error: Must provide either --logdir OR both --ppo_json and --wm_json")
            return
        print(f"Loading PPO data from {ppo_json}...")
        df_ppo_step = load_data_from_json(ppo_json)
        print(f"Loading World Model data from {wm_json}...")
        df_wm = load_data_from_json(wm_json)
    
    if df_ppo_step is None or df_wm is None:
        print("Error: Required data not found.")
        return

    max_step = max(df_ppo_step['step'].max(), df_wm['step'].max())
    
    # Define Boundaries
    boundaries = []
    if detected_switches:
        print(f"Using auto-detected regime boundaries.")
        boundaries = [0] + detected_switches + [max_step]
        boundaries = sorted(list(set(boundaries)))
    else:
        print(f"Using fixed regime intervals: {global_regime_steps}")
        accumulated_steps = 0
        while accumulated_steps < max_step:
            boundaries.append(accumulated_steps)
            accumulated_steps += global_regime_steps
        boundaries.append(max_step)

    # Plot 1: Step Mean Reward
    output_path_step = os.path.join(output_dir, "regime_analysis_step_reward.png")
    plot_single_graph(df_ppo_step, df_wm, 
                 "Step Mean Reward vs. Surprise", 
                 "Step Mean Reward", 'tab:green', 'tab:red', 
                 boundaries, max_step, False, output_path_step)

    # Plot 2: Episodic Return
    output_path_eps = os.path.join(output_dir, "regime_analysis_episodic_return.png")
    if df_ppo_eps is not None:
        plot_single_graph(df_ppo_eps, df_wm, 
                     "Episodic Return (Smoothed) vs. Surprise", 
                     "Episodic Return", 'tab:blue', 'tab:red', 
                     boundaries, max_step, True, output_path_eps)
    else:
        print("Warning: Episodic Return data not found. Skipping second plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PPO vs World Model with Regime Shading")
    parser.add_argument("--logdir", type=str, help="Path to TensorBoard log directory (e.g., runs/exp_...)")
    parser.add_argument("--ppo_json", type=str, help="Path to PPO reward JSON (optional fallback)")
    parser.add_argument("--wm_json", type=str, help="Path to World Model loss JSON (optional fallback)")
    parser.add_argument("--global_regime_steps", type=int, default=720000, help="Global steps per regime switch")
    parser.add_argument("--output_dir", type=str, default="graphs", help="Directory to save graphs in")
    
    args = parser.parse_args()
    
    plot_regime_analysis(args.logdir, args.ppo_json, args.wm_json, args.global_regime_steps, args.output_dir)
