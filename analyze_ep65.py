import json
import numpy as np

json_path = "evals/eval_brain_3_regimes_ep65_20260304-202238/eval_brain_3_regimes_ep63_20260304-202243/eval_brain_3_regimes_ep63_data.json"

print("Loading data...")
with open(json_path, 'r') as f:
    data = json.load(f)

output = []

def analyze_key(key):
    if key not in data:
        output.append(f"{key} not found")
        return
    vals = [item[1] for item in data[key]]
    steps = [item[0] for item in data[key]]
    if len(vals) == 0:
        return
    output.append(f"\n{key}:")
    output.append(f"  Min: {np.min(vals):.4f}")
    output.append(f"  Max: {np.max(vals):.4f}")
    output.append(f"  Mean: {np.mean(vals):.4f}")
    n = len(vals)
    output.append(f"  At 25% (step {steps[n//4]}): {vals[n//4]:.4f}")
    output.append(f"  At 50% (step {steps[n//2]}): {vals[n//2]:.4f}")
    output.append(f"  At 75% (step {steps[3*n//4]}): {vals[3*n//4]:.4f}")
    output.append(f"  At 100% (step {steps[-1]}): {vals[-1]:.4f}")
    
    # Also grab values at exactly regime switch points (every 225000 * 8 = 1.8M? No, 225,000 steps per regime)
    # The total steps is 4.5M, so 20 regimes. 225,000 inner steps per regime.
    # We can just sample every 10% to see a trend.
    trend = []
    for i in range(0, 101, 10):
        idx = min(int((i / 100.0) * n), n-1)
        trend.append(f"{vals[idx]:.4f}")
    output.append(f"  Trend (0% to 100% in 10% increments): {', '.join(trend)}")

analyze_key("charts/learning_rate")
analyze_key("brain/ent_coef")
analyze_key("brain/intrinsic_coef")
analyze_key("brain/imagined_horizon")

with open("analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Done. Saved to analysis_summary.txt")
