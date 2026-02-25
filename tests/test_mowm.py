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


def test_mowm_rescue_routing_logic():
    from lifelong_learning.agents.ppo.world_model import SimpleWorldModel
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32)
    # Initialize second model correctly
    mowm.models.append(SimpleWorldModel((21, 8, 8), 4, 32))
    
    # Configure so Model 1 is active, Model 0 is inactive veteran
    mowm.active_regime_id = 1
    mowm.ema_losses.append(0.1)
    mowm.routing_emas.append(0.1)
    mowm.fast_routing_emas.append(0.1)
    mowm.has_mastered.append(True)
    mowm.steps_under_threshold.append(mowm.mastery_buffer_steps)
    mowm.spawn_steps.append(1000)
    if hasattr(mowm, 'timeout_triggered'):
        mowm.timeout_triggered.append(False)
    mowm.force_active_until = 0

    global_step = mowm.global_grace_period + mowm.newborn_grace_period + 1

    # Simulate active model failing (loss = 2.0), while Model 0 is perfect (loss = 0.1)
    # This should NOT trigger a spawn because Model 0 rescues it.
    for _ in range(mowm.surprise_window_size):
        did_spawn = mowm.check_and_spawn(raw_losses=[0.1, 2.0], best_regime_id=1, global_step=global_step)
        global_step += 1
    
    assert not did_spawn
    assert len(mowm.models) == 2

    mowm.surprise_window.clear()
    
    # Simulate active model failing (loss = 2.0), and Model 0 is also terrible (loss = 0.8)
    # Both > 0.3. This SHOULD spawn Model 2.
    for _ in range(mowm.surprise_window_size):
        did_spawn = mowm.check_and_spawn(raw_losses=[0.8, 2.0], best_regime_id=1, global_step=global_step)
        global_step += 1

    assert did_spawn
    assert len(mowm.models) == 3
