import os
import json
import time
import numpy as np
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
        
    def close(self) -> None:
        # Compatibility method, does nothing now
        pass
