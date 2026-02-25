import torch
import pytest
from lifelong_learning.agents.ppo.mowm import MixtureOfWorldModels


def test_mowm_spawning_logic():
    # 1. Initialize the Brain
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)
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
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)
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


def test_safe_state_rollback():
    """Verify that rollback restores model weights exactly to the snapshot."""
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)
    model = mowm.models[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Drive EMA below mastery threshold so refresh considers the model "stable"
    for _ in range(1000):
        mowm.update_ema(0.05, steps_added=10)
    assert mowm.ema_losses[0] < mowm.mastery_loss_threshold

    # Capture the safe state (global_step past grace period)
    global_step = mowm.force_active_until + 1
    mowm.refresh_safe_state(0, model, optimizer, global_step=global_step)
    assert mowm.safe_state_dicts[0] is not None

    # Save a copy of the clean weights for comparison
    clean_weights = {k: v.clone() for k, v in model.state_dict().items()}
    saved_ema = mowm.safe_ema_losses[0]

    # Corrupt the model with random noise (simulating training on wrong regime)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 10.0)
    # Also pollute the EMA
    mowm.ema_losses[0] = 999.0

    # Verify corruption happened
    for k in clean_weights:
        assert not torch.allclose(model.state_dict()[k], clean_weights[k]), f"Weight {k} was not corrupted"

    # Rollback
    did_rollback = mowm.rollback_safe_state(0, model, optimizer)
    assert did_rollback

    # Verify exact weight restoration
    for k in clean_weights:
        assert torch.allclose(model.state_dict()[k], clean_weights[k]), f"Weight {k} not restored correctly"

    # Verify EMA was also restored
    assert mowm.ema_losses[0] == saved_ema


def test_safe_state_not_set_during_grace_period():
    """Verify that safe state is NOT captured while in the newborn grace period."""
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)
    model = mowm.models[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Drive EMA low so it passes the mastery check
    for _ in range(1000):
        mowm.update_ema(0.05, steps_added=10)

    # Set force_active_until far in the future (simulating newborn grace)
    mowm.force_active_until = 999999
    mowm.refresh_safe_state(0, model, optimizer, global_step=100)

    # Should still be None because we're in the grace period
    assert mowm.safe_state_dicts[0] is None


def test_spawn_appends_none_safe_state():
    """Verify that spawning a new model initializes its safe state slots to None."""
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)

    # Fast forward past global grace period
    global_step = mowm.global_grace_period + 1

    # Drive EMA to a low stable value
    for _ in range(1000):
        mowm.update_ema(0.1, steps_added=10)

    # Fill surprise window with high losses to trigger spawn
    mowm.surprise_window.clear()
    for _ in range(mowm.surprise_window_size):
        did_spawn = mowm.check_and_spawn(raw_losses=[3.0], best_regime_id=0, global_step=global_step)
        global_step += 1

    assert did_spawn
    assert len(mowm.models) == 2

    # Verify safe state lists grew but the new model's slots are None
    assert len(mowm.safe_state_dicts) == 2
    assert len(mowm.safe_optimizer_states) == 2
    assert len(mowm.safe_ema_losses) == 2
    assert mowm.safe_state_dicts[1] is None
    assert mowm.safe_optimizer_states[1] is None
    assert mowm.safe_ema_losses[1] is None

