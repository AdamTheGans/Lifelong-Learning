# Lifelong-Learning

> Preventing catastrophic forgetting with world models, planning, and context routing — a continual RL research project.

See [PLAN.md](PLAN.md) for our high-level roadmap.

## Project Structure

```
Lifelong-Learning/
├── src/lifelong_learning/
│   ├── agents/ppo/
│   │   ├── train.py            # Core training loop (Dyna-PPO logic)
│   │   ├── ppo.py              # PPO loss and update function
│   │   ├── network.py          # Actor-Critic network architecture
│   │   ├── world_model.py      # Simple World Model (predicts state/reward)
│   │   └── buffers.py          # Rollout buffer with GAE
│   ├── envs/
│   │   ├── dual_goal.py        # Custom MiniGrid DualGoal environment
│   │   ├── regime_wrapper.py   # Wrapper for non-stationary reward regimes
│   │   ├── make_env.py         # Factory function with wrapper stack
│   │   └── wrappers/
│   │       ├── action_reduce.py # Action space from Discrete(7) -> Discrete(3)
│   │       └── one_hot.py      # Image (H,W,3) -> OneHot (21,H,W)
│   └── utils/
│       ├── logger.py           # TensorBoard logging utility
│       └── seeding.py          # Deterministic seeding helper
├── PLAN.md                     # Research roadmap and architecture docs
├── README.md                   # Project overview and instructions
├── requirements.txt            # Project dependencies
├── pyproject.toml              # Build system configuration
├── scripts/                    # Entry points for training and analysis
└── tests/                      # Unit tests
    ├── test_dyna_logic.py      # Tests for World Model and intrinsic reward
    ├── test_env_integrity.py   # Tests for env physics and rules
    └── test_manual_stats.py    # Tests for logging logic
```

---

## Part 1: Dyna-PPO Baseline

The active training pipeline uses **Dyna-PPO**, which augments a standard PPO agent with a `SimpleWorldModel` sidekick. The World Model provides:
1. **Intrinsic Curiosity**: Reward augmentation based on prediction error (surprise).
2. **Dreaming**: Generates imagined trajectories to train the policy on latent/predicted transitions (Dyna-style).

### 1.1 Install (use a venv if you wish)

```bash
pip install -r requirements.txt
pip install -e .
```

### 1.2 Sanity Checks

```bash
# Verify environment + rewards are correct
python scripts/verify_rewards.py

# Verify vectorized envs work
python scripts/check_vec_env.py

# Pytest tests
pytest tests/

# Ensure no errors with short smoke test
python scripts/train_ppo.py --mode passive --total_timesteps 5000 --run_name smoke_test
```

### 1.3 Train Normal PPO

```bash
# Stationary (no regime switching)
python scripts/train_ppo.py --mode passive --total_timesteps 1000000 --run_name ppo_stationary

# Regime switching (slow)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --mode passive --total_timesteps 2500000 --steps_per_regime 27500 --run_name ppo_regime_switch_slow

# Regime switching (fast)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --mode passive --total_timesteps 1500000 --steps_per_regime 12500 --run_name ppo_regime_switch_fast
```

### 1.4 Train Dyna-PPO

```bash
# Stationary (no regime switching)
python scripts/train_ppo.py --mode dyna --total_timesteps 1000000 --run_name dyna_stationary

# Regime switching (slow)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --mode dyna --total_timesteps 2500000 --steps_per_regime 27500 --run_name dyna_regime_switch_slow

# Regime switching (fast)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --mode dyna --total_timesteps 1500000 --steps_per_regime 12500 --run_name dyna_regime_switch_fast
```

### 1.5 View Results

```bash
# View results in browser
tensorboard --logdir runs

# Create detailed analysis plots
python scripts/analyze_runs.py
```

### 1.6 Resume Training

To resume training from a checkpoint:

```bash
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 1500000 --run_name ppo_stationary --resume_path checkpoints/ppo_stationary_update170.pt
```
*Note: Learning rate annealing will reset unless you manually adjust timesteps, but for fine-tuning/continuation, this is usually acceptable.*
