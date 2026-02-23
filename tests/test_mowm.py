import torch
import pytest
from lifelong_learning.agents.ppo.mowm import MixtureOfWorldModels


def test_mowm_spawning_logic():
    # 1. Initialize the Brain
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32)
    assert len(mowm.models) == 1
    assert mowm.active_regime_id == 0

    # 2. Simulate standard learning (Loss drops to ~0.1)
    for _ in range(1000):
        mowm.update_ema(0.1)
    
    # After 1000 steps with alpha=0.01, EMA should be very close to 0.1
    assert abs(mowm.ema_loss - 0.1) < 1e-4
    assert len(mowm.models) == 1
    
    # 3. Test a normal fluctuation (e.g. loss jumping to 0.4)
    # Threshold is 5.0, so 0.4 / 0.1 = 4.0 < 5.0 (Should NOT spawn)
    did_spawn = mowm.check_and_spawn(lowest_loss=0.4)
    assert not did_spawn
    assert len(mowm.models) == 1

    # 4. Simulate a sudden Regime Switch (Loss spikes to 1.0)
    # 1.0 / 0.1 = 10.0 > 5.0 (Should spawn!)
    did_spawn = mowm.check_and_spawn(lowest_loss=1.0) 
    
    assert did_spawn
    assert mowm.active_regime_id == 1
    assert len(mowm.models) == 2
    
    # Ensure EMA resets correctly on spawn
    assert mowm.ema_loss == 1.0
