import torch
import pytest
from lifelong_learning.agents.ppo.mowm import MixtureOfWorldModels


def test_mowm_spawning_logic():
    # 1. Initialize the Brain
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32)
    assert len(mowm.models) == 1
    assert mowm.active_regime_id == 0

    # Fast forward past global grace period so we can test spawns
    global_step = mowm.global_grace_period + 1

    # 2. Simulate standard learning (EMA drops to ~0.1)
    for _ in range(1000):
        mowm.update_ema(0.1, steps_added=10)
    
    # After 1000 steps with alpha=0.05, EMA should be very close to 0.1
    assert abs(mowm.ema_losses[0] - 0.1) < 1e-4
    assert len(mowm.models) == 1
    
    # Fill the EMA history with the baseline so slope = 0.0
    for _ in range(10):
        mowm.ema_history[0].append(0.1)
    
    # 3. Test a normal fluctuation (e.g. loss jumping to 0.3)
    # Threshold is 0.35 (EMA 0.1 * 3.5). 0.3 < 0.35 (Should NOT spawn)
    # Must fill the surprise window to trigger a check
    for _ in range(mowm.surprise_window_size):
        did_spawn = mowm.check_and_spawn(raw_losses=[0.3], best_regime_id=0, global_step=global_step)
        global_step += 1
    
    assert not did_spawn
    assert len(mowm.models) == 1

    # 4. Simulate a sudden Regime Switch (Loss spikes to 3.0)
    # 3.0 > 0.5 (Should spawn!)
    
    # We must clear the queue manually because the previous test filled it with 0.4s.
    mowm.surprise_window.clear()
    
    for _ in range(mowm.surprise_window_size):
        did_spawn = mowm.check_and_spawn(raw_losses=[3.0], best_regime_id=0, global_step=global_step) 
        global_step += 1
    
    assert did_spawn
    assert mowm.active_regime_id == 1
    assert len(mowm.models) == 2
    
    # Ensure EMA resets correctly on spawn to the active model's smoothed loss
    assert mowm.ema_losses[1] == 3.0
    
    # Ensure newborn grace period is activated
    assert mowm.force_active_until >= global_step + mowm.newborn_grace_period - 1
