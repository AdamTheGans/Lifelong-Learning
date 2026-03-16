import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def exponential_moving_average(data, alpha=0.05):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def plot_metric(data_list, metric_name, save_path, regime_data, step_interval=50000, alpha=0.05):
    if not data_list:
        print(f'Skipping {metric_name}, no data.')
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
            if r_vals[i] != r_vals[i - 1]:
                switch_indices.append(i)
        switch_indices.append(len(r_vals) - 1)
        colors = ['#b3d9ff', '#ffcca3', '#b3ffb3', '#d9b3ff', '#ffffb3']

        for i in range(len(switch_indices) - 1):
            start_idx = switch_indices[i]
            end_idx = switch_indices[i + 1]

            start_step = r_steps[start_idx]
            end_step = r_steps[end_idx]
            regime_id = int(r_vals[start_idx])

            color = colors[regime_id % len(colors)]
            ax.axvspan(start_step, end_step, color=color, alpha=0.5, zorder=0)

            mid_step = (start_step + end_step) / 2
            ax.text(
                mid_step,
                0.95,
                f'Regime {regime_id}',
                transform=ax.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=20,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3),
            )

    ax.plot(steps, values, color='lightblue', alpha=0.3, linewidth=0.5, label='Raw', zorder=1)
    smoothed = exponential_moving_average(values, alpha=alpha)
    ax.plot(steps, smoothed, color='blue', alpha=0.9, linewidth=1.5, label=f'EMA (alpha={alpha})', zorder=2)

    if 'success_rate' in metric_name.lower() and regime_data:
        r_steps, r_vals = zip(*regime_data)

        switch_steps = [0]
        for i in range(1, len(r_vals)):
            if r_vals[i] != r_vals[i - 1]:
                switch_steps.append(r_steps[i])

        for i in range(len(switch_steps)):
            start_step = switch_steps[i]
            end_step = switch_steps[i + 1] if i + 1 < len(switch_steps) else steps[-1]

            reached_95_step = None
            for s, v in zip(steps, values):
                if s >= start_step and s <= end_step and v >= 0.95 and s >= (start_step + 10000):
                    reached_95_step = s
                    break

            if reached_95_step is not None:
                steps_to_95 = reached_95_step - start_step
                y_pos = 0.5
                ax.axvline(x=reached_95_step, color='green', linestyle='--', alpha=0.7, zorder=1)
                ax.text(
                    reached_95_step,
                    y_pos,
                    f'Regime {i}\nreached\n95% in\n{int(steps_to_95)}\nsteps',
                    color='green',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='green', boxstyle='round,pad=0.3'),
                )
            else:
                reached_80_step = None
                for s, v in zip(steps, values):
                    if s >= start_step and s <= end_step and v >= 0.80 and s >= (start_step + 10000):
                        reached_80_step = s
                        break

                if reached_80_step is not None:
                    steps_to_80 = reached_80_step - start_step
                    y_pos = 0.5
                    ax.axvline(x=reached_80_step, color='orange', linestyle='--', alpha=0.7, zorder=1)
                    ax.text(
                        reached_80_step,
                        y_pos,
                        f'Regime {i}\nreached\n80% in\n{int(steps_to_80)}\nsteps',
                        color='orange',
                        fontsize=12,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange', boxstyle='round,pad=0.3'),
                    )
                else:
                    mid_step = start_step + (end_step - start_step) / 2
                    y_pos = 0.5
                    ax.text(
                        mid_step,
                        y_pos,
                        f'Regime {i}:\nfailed\nto reach\n80%',
                        color='red',
                        fontsize=12,
                        fontweight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'),
                    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(step_interval))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_title(f'High-Scale: {metric_name}', fontsize=24)
    ax.set_xlabel('Steps', fontsize=18)
    ax.set_ylabel(metric_name.split('/')[-1].replace('_', ' ').title(), fontsize=18)

    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=14)

    if 'success_rate' in metric_name.lower():
        ax.set_ylim(-0.05, 1.05)
        avg_sr = float(np.mean(values))
        ax.text(
            0.98,
            0.98,
            f'Avg Success Rate: {avg_sr:.4f}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=20,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'),
            zorder=10,
        )

    ax.legend(fontsize=16, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved highly scaled graph for {metric_name} to {save_path}')


def get_series(data, key):
    points = data.get(key, [])
    if not points:
        return None
    steps, values = zip(*points)
    return np.asarray(steps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def align_series_to_steps(data, key, target_steps):
    series = get_series(data, key)
    if series is None or target_steps.size == 0:
        return None

    steps, values = series
    indices = np.searchsorted(steps, target_steps, side='right') - 1
    aligned = np.full(target_steps.shape[0], np.nan, dtype=np.float64)
    valid = indices >= 0
    aligned[valid] = values[indices[valid]]
    return aligned


def build_matrix(data, prefix, count, target_steps):
    rows = []
    for i in range(count):
        aligned = align_series_to_steps(data, f'{prefix}{i}', target_steps)
        if aligned is None:
            return None
        rows.append(aligned)
    return np.vstack(rows)


def fill_forward(values):
    filled = np.asarray(values, dtype=np.float64).copy()
    last_valid = np.nan
    for idx, value in enumerate(filled):
        if np.isfinite(value):
            last_valid = value
        elif np.isfinite(last_valid):
            filled[idx] = last_valid
    return filled


def draw_regime_spans(ax, regime_values):
    if regime_values.size == 0:
        return

    filled = fill_forward(regime_values)
    if not np.isfinite(filled).any():
        return

    colors = ['#b3d9ff', '#ffcca3', '#b3ffb3', '#d9b3ff', '#ffffb3']
    start = 0
    for idx in range(1, filled.size + 1):
        if idx == filled.size or filled[idx] != filled[start]:
            regime_id = int(filled[start])
            ax.axvspan(start - 0.5, idx - 0.5, color=colors[regime_id % len(colors)], alpha=0.25, zorder=0)
            start = idx


def get_regime_switch_positions(regime_values):
    if regime_values is None or regime_values.size == 0:
        return np.array([], dtype=np.float64)

    filled = fill_forward(regime_values)
    finite_indices = np.flatnonzero(np.isfinite(filled))
    if finite_indices.size == 0:
        return np.array([], dtype=np.float64)

    switch_positions = []
    last_regime = filled[finite_indices[0]]
    for idx in range(finite_indices[0] + 1, filled.size):
        if not np.isfinite(filled[idx]):
            continue
        if filled[idx] != last_regime:
            switch_positions.append(idx - 0.5)
            last_regime = filled[idx]

    return np.asarray(switch_positions, dtype=np.float64)


def draw_regime_switch_lines(ax, regime_values):
    for position in get_regime_switch_positions(regime_values):
        ax.axvline(position, color='black', linestyle='--', linewidth=1.0, alpha=0.45, zorder=0)


def apply_decision_ticks(ax, decision_steps, show_xlabel=False):
    if decision_steps.size == 0:
        return

    tick_count = min(6, decision_steps.size)
    tick_positions = np.linspace(0, decision_steps.size - 1, num=tick_count, dtype=int)
    tick_positions = np.unique(tick_positions)
    tick_labels = [f'{int(decision_steps[idx])}' for idx in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    if show_xlabel:
        ax.set_xlabel('Inner Global Step at Brain Decision')


def has_finite(values):
    return values is not None and np.isfinite(values).any()


def plot_neuromodulation_dashboard(data, save_path):
    decision_steps_series = get_series(data, 'brain_context/context_0')
    if decision_steps_series is None:
        print('Skipping neuromodulation dashboard, no brain context logs found.')
        return

    decision_steps = decision_steps_series[0]
    if decision_steps.size == 0:
        print('Skipping neuromodulation dashboard, no decision steps found.')
        return

    context_matrix = build_matrix(data, 'brain_context/context_', 8, decision_steps)
    channel_matrix = build_matrix(data, 'brain_neuromod/channel_mean_', 64, decision_steps)
    if context_matrix is None or channel_matrix is None:
        print('Skipping neuromodulation dashboard, missing context or channel logs.')
        return

    regime_values = align_series_to_steps(data, 'charts/regime_id', decision_steps)
    success_rate = align_series_to_steps(data, 'charts/success_rate', decision_steps)
    episodic_return = align_series_to_steps(data, 'charts/episodic_return', decision_steps)
    reward_step_mean = align_series_to_steps(data, 'charts/reward_step_mean', decision_steps)
    policy_kl = align_series_to_steps(data, 'brain_neuromod/policy_kl_vs_unmasked', decision_steps)
    entropy_delta = align_series_to_steps(data, 'brain_neuromod/entropy_delta_vs_unmasked', decision_steps)
    value_delta = align_series_to_steps(data, 'brain_neuromod/value_delta_abs_vs_unmasked', decision_steps)

    x = np.arange(decision_steps.size)
    fig, axes = plt.subplots(4, 1, figsize=(20, 18), dpi=200, sharex=True)

    ax = axes[0]
    if regime_values is not None:
        draw_regime_spans(ax, regime_values)
        draw_regime_switch_lines(ax, regime_values)
    if has_finite(success_rate):
        ax.plot(x, success_rate, color='tab:green', linewidth=2.0, label='Success Rate')
    if has_finite(reward_step_mean):
        ax.plot(x, reward_step_mean, color='tab:blue', linewidth=1.5, alpha=0.8, label='Reward Step Mean')
    ax.set_ylabel('Learning Signal')
    ax.set_title('Learning Progress and Regime Context')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if has_finite(episodic_return):
        ax_return = ax.twinx()
        ax_return.plot(x, episodic_return, color='tab:orange', linewidth=1.2, alpha=0.7, label='Episodic Return')
        ax_return.set_ylabel('Episodic Return')
        lines2, labels2 = ax_return.get_legend_handles_labels()
        legend_handles += lines2
        legend_labels += labels2
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper left')

    ax = axes[1]
    if regime_values is not None:
        draw_regime_switch_lines(ax, regime_values)
    context_im = ax.imshow(context_matrix, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax.set_ylabel('Context Dim')
    ax.set_yticks(np.arange(8))
    ax.set_title('Brain Context Code')
    fig.colorbar(context_im, ax=ax, pad=0.01, label='Code Value')

    ax = axes[2]
    if regime_values is not None:
        draw_regime_switch_lines(ax, regime_values)
    channel_im = ax.imshow(channel_matrix, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_ylabel('Feature Channel')
    ax.set_yticks(np.arange(0, 64, 8))
    ax.set_title('Decoded Neuromodulation Mask (Channel Mean)')
    fig.colorbar(channel_im, ax=ax, pad=0.01, label='Mask Strength')

    ax = axes[3]
    if regime_values is not None:
        draw_regime_switch_lines(ax, regime_values)
    if has_finite(policy_kl):
        ax.plot(x, policy_kl, color='tab:red', linewidth=1.8, label='Policy KL vs Unmasked')
    if has_finite(entropy_delta):
        ax.plot(x, entropy_delta, color='tab:purple', linewidth=1.5, label='Entropy Delta vs Unmasked')
    if has_finite(value_delta):
        ax.plot(x, value_delta, color='tab:brown', linewidth=1.5, label='Abs Value Delta vs Unmasked')
    if not ax.get_legend_handles_labels()[0]:
        ax.text(0.5, 0.5, 'No neuromodulation effect metrics logged', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.legend(loc='upper left')
    ax.set_ylabel('Effect Size')
    ax.set_title('Neuromodulation Effect on Policy and Value')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    apply_decision_ticks(ax, decision_steps, show_xlabel=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved neuromodulation dashboard to {save_path}')


def generate_high_scale_plots(folder, interval=50000, smoothing=0.05):
    json_files = glob.glob(os.path.join(folder, '*_data.json'))
    if not json_files:
        print(f'Error: No *_data.json found in {folder}')
        return

    for json_file in json_files:
        print(f'Loading data from {json_file}...')
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f'Error reading JSON {json_file}: {exc}')
            continue

        target_keys = {
            'success_rate': [k for k in data.keys() if 'success_rate' in k.lower()],
            'episode_return': [k for k in data.keys() if 'episodic_return' in k.lower()],
            'loss': [k for k in data.keys() if 'loss_total' in k.lower() or 'total_loss' in k.lower()],
        }

        keys_to_plot = set()
        for keys in target_keys.values():
            keys_to_plot.update(keys)

        print(f'Keys to plot: {keys_to_plot}')

        base_name = os.path.basename(json_file).replace('_data.json', '')
        regime_data = data.get('charts/regime_id', [])

        neuromod_save_path = os.path.join(folder, f'{base_name}_neuromodulation_dashboard.png')
        plot_neuromodulation_dashboard(data, neuromod_save_path)

        for key in keys_to_plot:
            safe_name = key.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(folder, f'{base_name}_{safe_name}_highres.png')
            plot_metric(data[key], key, save_path, regime_data, step_interval=interval, alpha=smoothing)


def main():
    parser = argparse.ArgumentParser(description='Create high scale graphs from evals')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing *_data.json')
    parser.add_argument('--interval', type=int, default=50000, help='Interval for x-axis ticks')
    parser.add_argument('--smoothing', type=float, default=0.05, help='EMA alpha for smoothing (0-1)')
    args = parser.parse_args()

    generate_high_scale_plots(folder=args.folder, interval=args.interval, smoothing=args.smoothing)


if __name__ == '__main__':
    main()
