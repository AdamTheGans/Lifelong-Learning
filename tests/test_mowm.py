import torch
import pytest
from lifelong_learning.agents.ppo.mowm import MixtureOfWorldModels


def test_mowm_epoch_boundary_spawning():
    """Verify that check_epoch_transition triggers a spawn when the active model is surprised
    and no veteran can rescue."""
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)
    assert len(mowm.models) == 1
    assert mowm.active_regime_id == 0

    # Fast forward past global grace period
    global_step = mowm.global_grace_period + 1

    # Simulate standard learning (EMA drops to ~0.1)
    for _ in range(1000):
        mowm.update_ema(0.1, steps_added=10)
    
    assert abs(mowm.ema_losses[0] - 0.1) < 1e-4
    assert len(mowm.models) == 1
    
    # Normal fluctuation: epoch loss 0.3, threshold = 0.1 * 3.5 = 0.35
    # 0.3 < 0.35 → should stay
    action, target = mowm.check_epoch_transition(
        epoch_avg_loss=0.3, eval_losses=[0.3], global_step=global_step
    )
    assert action == "stay"
    assert len(mowm.models) == 1

    # Regime switch: epoch loss 3.0 >> threshold 0.35 → should spawn
    # eval_losses shows only one model, and it's bad → no veteran to rescue
    action, target = mowm.check_epoch_transition(
        epoch_avg_loss=3.0, eval_losses=[3.0], global_step=global_step
    )
    assert action == "spawn"

    # Execute the spawn
    new_id = mowm.spawn_new_model(global_step, seed_ema=3.0)
    assert new_id == 1
    assert mowm.active_regime_id == 1
    assert len(mowm.models) == 2
    assert mowm.ema_losses[1] == 3.0
    assert mowm.force_active_until >= global_step + mowm.newborn_grace_period - 1


def test_mowm_epoch_boundary_veteran_rescue():
    """Verify that check_epoch_transition returns 'switch' when a veteran can handle it,
    and 'spawn' when no veteran is viable."""
    from lifelong_learning.agents.ppo.world_model import SimpleWorldModel
    mowm = MixtureOfWorldModels(obs_shape=(21, 8, 8), n_actions=4, hidden_dim=32, max_regimes=3)
    # Add a second model
    mowm.models.append(SimpleWorldModel((21, 8, 8), 4, 32))
    
    mowm.active_regime_id = 1
    mowm.ema_losses.append(0.1)
    mowm.has_mastered.append(True)
    mowm.steps_under_threshold.append(mowm.mastery_buffer_steps)
    mowm.spawn_steps.append(1000)
    if hasattr(mowm, 'timeout_triggered'):
        mowm.timeout_triggered.append(False)
    mowm.safe_state_dicts.append(None)
    mowm.safe_optimizer_states.append(None)
    mowm.safe_ema_losses.append(None)
    mowm.safe_state_steps.append(None)
    mowm.force_active_until = 0

    global_step = mowm.global_grace_period + mowm.newborn_grace_period + 1

    # Active model (1) is surprised, but Model 0 has low loss → switch
    action, target = mowm.check_epoch_transition(
        epoch_avg_loss=2.0, eval_losses=[0.1, 2.0], global_step=global_step
    )
    assert action == "switch"
    assert target == 0

    # Active model (1) is surprised, and Model 0 is also terrible → spawn
    action, target = mowm.check_epoch_transition(
        epoch_avg_loss=2.0, eval_losses=[0.8, 2.0], global_step=global_step
    )
    assert action == "spawn"


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

    # Spawn via the new spawn_new_model method
    new_id = mowm.spawn_new_model(global_step=50000, seed_ema=1.5)

    assert new_id == 1
    assert len(mowm.models) == 2
    assert len(mowm.safe_state_dicts) == 2
    assert len(mowm.safe_optimizer_states) == 2
    assert len(mowm.safe_ema_losses) == 2
    assert len(mowm.safe_state_steps) == 2
    assert mowm.safe_state_dicts[1] is None
    assert mowm.safe_optimizer_states[1] is None
    assert mowm.safe_ema_losses[1] is None
    assert mowm.safe_state_steps[1] is None
