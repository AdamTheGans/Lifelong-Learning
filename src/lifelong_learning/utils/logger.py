import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class DataLogger:
    run_name: str
    log_dir: str = "runs"
    data: dict = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.full_dir = os.path.join(self.log_dir, f"{self.run_name}_{ts}")
        os.makedirs(self.full_dir, exist_ok=True)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.data[tag].append((step, float(value)))

    def plot(self, save_dir: str, title: str = "Training Metrics") -> None:
        if not self.data:
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Dump raw data to JSON for independent manual viewing
        json_path = os.path.join(save_dir, f"{self.run_name}_data.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to save raw JSON data: {e}")
        
        # Determine grid size based on number of tracked metrics
        tags = list(self.data.keys())
        n_metrics = len(tags)
        cols = 3
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(title, fontsize=16)
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        # Auto-detect regime switch points from charts/regime_id
        regime_switch_steps = []
        regime_data = self.data.get("charts/regime_id", [])
        if len(regime_data) > 1:
            for i in range(1, len(regime_data)):
                prev_val = regime_data[i - 1][1]
                curr_val = regime_data[i][1]
                if round(prev_val) != round(curr_val):
                    regime_switch_steps.append(regime_data[i][0])

        for i, tag in enumerate(tags):
            ax = axes[i]
            points = self.data[tag]
            if not points:
                continue
            
            steps, values = zip(*points)
            
            # Simple moving average for smoothing if we have enough points
            values_np = np.array(values)
            window = max(1, len(values_np) // 20)
            if window > 1 and len(values_np) >= window:
                smoothed = np.convolve(values_np, np.ones(window)/window, mode='valid')
                # Pad start to match length
                pad = len(values_np) - len(smoothed)
                smoothed = np.pad(smoothed, (pad, 0), mode='edge')
                ax.plot(steps, smoothed, color='blue', alpha=0.8, label="Smoothed")
                ax.plot(steps, values, color='lightblue', alpha=0.3)
            else:
                ax.plot(steps, values, color='blue')

            # Draw regime switch lines
            if regime_switch_steps:
                for rs_step in regime_switch_steps:
                    ax.axvline(x=rs_step, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                
            ax.set_title(tag)
            ax.set_xlabel('Steps' if 'step' in tag.lower() else 'Updates/Episodes')
            ax.grid(True, alpha=0.3)
            
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{self.run_name}_charts.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save separate unsmoothed success rate graph
        if "charts/success_rate" in self.data and len(self.data["charts/success_rate"]) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            points = self.data["charts/success_rate"]
            steps, values = zip(*points)
            
            ax.plot(steps, values, color='green', alpha=0.9, linewidth=1.5, label="Raw Success Rate")
            
            # Draw regime switch lines
            if regime_switch_steps:
                for rs_step in regime_switch_steps:
                    ax.axvline(x=rs_step, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            
            ax.set_title("Unsmoothed Success Rate vs Steps", fontsize=14)
            ax.set_xlabel('Steps', fontsize=12)
            ax.set_ylabel('Success Rate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            sr_save_path = os.path.join(save_dir, f"{self.run_name}_success_rate.png")
            plt.savefig(sr_save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
    def close(self) -> None:
        # Compatibility method, does nothing now
        pass
