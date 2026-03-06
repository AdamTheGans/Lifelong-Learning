import json, glob, os
import numpy as np

base_dir = r'c:\Users\elija\OneDrive\Desktop\175Project\Lifelong-Learning\runs\brain_3_regimes_8x8_run_2_20260304-032507\episode_3'
env_dirs = glob.glob(os.path.join(base_dir, 'ep3_env*'))

def extract_mean_over_last_n(data, key, n=100):
    if key in data:
        vals = [v[1] for v in data[key][-n:]]
        return np.mean(vals)
    return 0.0

for env_dir in env_dirs:
    json_path = glob.glob(os.path.join(env_dir, '*data.json'))
    if not json_path:
        continue
    with open(json_path[0], 'r') as f:
        data = json.load(f)
    
    succ = extract_mean_over_last_n(data, 'charts/success_rate', 500)
    lr = extract_mean_over_last_n(data, 'brain/lr', 500)
    ent = extract_mean_over_last_n(data, 'brain/ent_coef', 500)
    replay = extract_mean_over_last_n(data, 'brain/replay_ratio', 500)
    intr = extract_mean_over_last_n(data, 'brain/intrinsic_coef', 500)
    horizon = extract_mean_over_last_n(data, 'brain/imagined_horizon', 500)
    
    env_num = os.path.basename(env_dir).split('_')[1]
    print(f'{env_num}: Final Success Rate: {succ:.2f} | Brain LR: {lr:.5f} | Ent: {ent:.4f} | Replay: {replay:.3f} | Intr: {intr:.4f} | Horizon: {horizon:.1f}')
