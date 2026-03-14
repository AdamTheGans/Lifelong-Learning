import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def exponential_moving_average(data, alpha=0.05):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def plot_metric(data_list, metric_name, save_path, regime_data, step_interval=50000, alpha=0.05):
    if not data_list:
        print(f"Skipping {metric_name}, no data.")
        return

    steps, values = zip(*data_list)
    steps = np.array(steps)
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(40, 10), dpi=300)
    
    # Draw regime colored boxes
    # regime_data is a list of (step, regime_id)
    if regime_data:
        r_steps, r_vals = zip(*regime_data)
        
        # Find switch points
        switch_indices = [0]
        for i in range(1, len(r_vals)):
            if r_vals[i] != r_vals[i-1]:
                switch_indices.append(i)
        switch_indices.append(len(r_vals) - 1)
        # Use slightly bolder, darker pastel colors
        colors = ['#b3d9ff', '#ffcca3', '#b3ffb3', '#d9b3ff', '#ffffb3']

        for i in range(len(switch_indices) - 1):
            start_idx = switch_indices[i]
            end_idx = switch_indices[i+1]
            
            start_step = r_steps[start_idx]
            end_step = r_steps[end_idx]
            regime_id = int(r_vals[start_idx])
            
            color = colors[regime_id % len(colors)]
            
            # The span covers the entire y-axis
            ax.axvspan(start_step, end_step, color=color, alpha=0.5, zorder=0)
            
            # Add text label for the regime at the top
            mid_step = (start_step + end_step) / 2
            ax.text(mid_step, 0.95, f'Regime {regime_id}', 
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=20, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

    # Plot lines
    ax.plot(steps, values, color='lightblue', alpha=0.3, linewidth=0.5, label='Raw', zorder=1)
    smoothed = exponential_moving_average(values, alpha=alpha)
    ax.plot(steps, smoothed, color='blue', alpha=0.9, linewidth=1.5, label=f'EMA (alpha={alpha})', zorder=2)
    
    # Specific logic for success rate: time to 95%
    if "success_rate" in metric_name.lower() and regime_data:
        r_steps, r_vals = zip(*regime_data)
        
        # Find switch points
        switch_steps = [0]
        for i in range(1, len(r_vals)):
            if r_vals[i] != r_vals[i-1]:
                switch_steps.append(r_steps[i])
        
        # Analyze each regime segment
        for i in range(len(switch_steps)):
            start_step = switch_steps[i]
            end_step = switch_steps[i+1] if i + 1 < len(switch_steps) else steps[-1]
            
            # Find the first point in this regime where success rate reaches >= 0.95
            reached_95_step = None
            for s, v in zip(steps, values):
                if s >= start_step and s <= end_step:
                    # Only start looking for 95% success rate 10k steps AFTER the regime switch
                    if v >= 0.95 and s >= (start_step + 10000):
                        reached_95_step = s
                        break
            
            if reached_95_step is not None:
                steps_to_95 = reached_95_step - start_step
                
                # All annotations aligned horizontally
                y_pos = 0.5
                
                # Draw a horizontal line or simply put text
                ax.axvline(x=reached_95_step, color='green', linestyle='--', alpha=0.7, zorder=1)
                ax.text(reached_95_step, y_pos, f"Regime {i}\nreached\n95% in\n{int(steps_to_95)}\nsteps", 
                        color='green', fontsize=12, fontweight='bold', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='green', boxstyle='round,pad=0.3'))
            else:
                # Did not reach 95% — check for 80%
                reached_80_step = None
                for s, v in zip(steps, values):
                    if s >= start_step and s <= end_step:
                        if v >= 0.80 and s >= (start_step + 10000):
                            reached_80_step = s
                            break
                
                if reached_80_step is not None:
                    steps_to_80 = reached_80_step - start_step
                    mid_step = reached_80_step
                    y_pos = 0.5
                    
                    ax.axvline(x=reached_80_step, color='orange', linestyle='--', alpha=0.7, zorder=1)
                    ax.text(mid_step, y_pos, f"Regime {i}\nreached\n80% in\n{int(steps_to_80)}\nsteps", 
                            color='orange', fontsize=12, fontweight='bold', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange', boxstyle='round,pad=0.3'))
                else:
                    # Did not reach even 80%
                    mid_step = start_step + (end_step - start_step) / 2
                    y_pos = 0.5
                    
                    ax.text(mid_step, y_pos, f"Regime {i}:\nfailed\nto reach\n80%", 
                            color='red', fontsize=12, fontweight='bold', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'))
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step_interval))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.set_title(f"High-Scale: {metric_name}", fontsize=24)
    ax.set_xlabel("Steps", fontsize=18)
    ax.set_ylabel(metric_name.split('/')[-1].replace('_', ' ').title(), fontsize=18)
    
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    
    # ensure y axis handles 0 to 1 for success rate nicely
    if "success_rate" in metric_name.lower():
        ax.set_ylim(-0.05, 1.05)
        
    # Show average success rate at the top of the graph
    if "success_rate" in metric_name.lower():
        avg_sr = float(np.mean(values))
        ax.text(0.98, 0.98, f"Avg Success Rate: {avg_sr:.4f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=20, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'),
                zorder=10)

    ax.legend(fontsize=16, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved highly scaled graph for {metric_name} to {save_path}")

def generate_high_scale_plots(folder, interval=50000, smoothing=0.05):
    json_files = glob.glob(os.path.join(folder, "*_data.json"))
    if not json_files:
        print(f"Error: No *_data.json found in {folder}")
        return
    
    for json_file in json_files:
        print(f"Loading data from {json_file}...")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON {json_file}: {e}")
            continue

        target_keys = {
            "success_rate": [k for k in data.keys() if "success_rate" in k.lower()],
            "episode_return": [k for k in data.keys() if "episodic_return" in k.lower()], # explicitly look for episodic_return instead of just return/reward
            "loss": [k for k in data.keys() if "loss_total" in k.lower() or "total_loss" in k.lower()], # Explicitly target just total loss
        }

        keys_to_plot = set()
        for concept, keys in target_keys.items():
            for k in keys:
                keys_to_plot.add(k)
        
        print(f"Keys to plot: {keys_to_plot}")
        
        base_name = os.path.basename(json_file).replace("_data.json", "")
        
        regime_data = data.get("charts/regime_id", [])
        
        for k in keys_to_plot:
            safe_name = k.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(folder, f"{base_name}_{safe_name}_highres.png")
            plot_metric(data[k], k, save_path, regime_data, step_interval=interval, alpha=smoothing)

def main():
    parser = argparse.ArgumentParser(description="Create high scale graphs from evals")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing *_data.json")
    parser.add_argument("--interval", type=int, default=50000, help="Interval for x-axis ticks")
    parser.add_argument("--smoothing", type=float, default=0.05, help="EMA alpha for smoothing (0-1)")
    args = parser.parse_args()

    generate_high_scale_plots(folder=args.folder, interval=args.interval, smoothing=args.smoothing)

if __name__ == "__main__":
    main()
