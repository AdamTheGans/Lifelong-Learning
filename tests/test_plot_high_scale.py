import json
import numpy as np
from pathlib import Path

import scripts.plot_high_scale as plot_high_scale


def test_generate_high_scale_plots_writes_neuromodulation_dashboard(tmp_path):
    json_path = Path(tmp_path) / 'demo_data.json'
    steps = [0, 100, 200, 300]
    data = {
        'charts/regime_id': [[0, 0], [150, 1], [300, 1]],
        'charts/success_rate': [[step, 0.25 + 0.2 * idx] for idx, step in enumerate(steps)],
        'charts/episodic_return': [[step, float(idx)] for idx, step in enumerate(steps)],
        'charts/reward_step_mean': [[step, 0.1 * (idx + 1)] for idx, step in enumerate(steps)],
        'brain_neuromod/policy_kl_vs_unmasked': [[step, 0.01 * (idx + 1)] for idx, step in enumerate(steps)],
        'brain_neuromod/entropy_delta_vs_unmasked': [[step, -0.02 * idx] for idx, step in enumerate(steps)],
        'brain_neuromod/value_delta_abs_vs_unmasked': [[step, 0.03 * (idx + 1)] for idx, step in enumerate(steps)],
    }

    for dim in range(8):
        data[f'brain_context/context_{dim}'] = [[step, (-1.0 + 0.25 * dim) + 0.05 * idx] for idx, step in enumerate(steps)]

    for channel in range(64):
        data[f'brain_neuromod/channel_mean_{channel}'] = [[step, min(1.0, 0.01 * channel + 0.02 * idx)] for idx, step in enumerate(steps)]

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle)

    plot_high_scale.generate_high_scale_plots(str(tmp_path), interval=100, smoothing=0.1)

    dashboard_path = Path(tmp_path) / 'demo_neuromodulation_dashboard.png'
    assert dashboard_path.exists()


def test_get_regime_switch_positions_returns_decision_boundaries():
    regime_values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)

    positions = plot_high_scale.get_regime_switch_positions(regime_values)

    assert np.allclose(positions, np.array([1.5, 3.5], dtype=np.float64))
