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
│   │   ├── network.py          # Context-aware Actor-Critic network (accepts regime_id)
│   │   ├── mowm.py             # Mixture of World Models manager (handles regime switching)
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

The active training pipeline uses **Dyna-PPO**, which augments a standard PPO agent with a **Mixture of World Models (MoWM)** sidekick. The MoWM provides:
1. **Context Routing**: Dynamically detects regime changes via surprise (prediction error) spikes and spawns or routes to specialized world models.
2. **Intrinsic Curiosity**: Reward augmentation based on prediction error.
3. **Dreaming**: Generates imagined trajectories across different regimes to train the policy on latent/predicted transitions, preventing catastrophic forgetting.

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
python scripts/train_ppo.py --mode passive --total_timesteps 800000 --run_name ppo_stationary

# Regime switching
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --mode passive --total_timesteps 800000 --steps_per_regime 18500 --run_name ppo_regime_switch
```

### 1.4 Train Dyna-PPO

```bash
# Stationary (no regime switching)
python scripts/train_ppo.py --mode dyna --max_regimes 1 --no_save_buffer --total_timesteps 800000 --run_name dyna_stationary

# Regime switching
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --mode dyna --max_regimes 1 --no_save_buffer --total_timesteps 800000 --steps_per_regime 18500 --run_name dyna_regime_switch
```

### 1.5 View Results

```bash
# View results in browser
tensorboard --logdir runs

# Create detailed analysis plots
python scripts/analyze_runs.py
```



# Part 2: Run the new MoWM code

```bash
# Stationary (no regime switching)
python scripts/train_ppo.py --mode dyna --total_timesteps 800000 --run_name mowm_stationary

# Regime switching
python scripts/train_ppo.py --mode dyna --total_timesteps 800000 --steps_per_regime 18500 --run_name mowm_regime_switch
```


# Part 3: Run all tests with 4 regimes

```bash
# PPO 4 regimes
python scripts/train_ppo.py --mode passive --total_timesteps 2250000 --steps_per_regime 18500 --total_env_regimes 4 --run_name ppo_4_regimes

# Dyna-PPO 4 regimes
python scripts/train_ppo.py --mode dyna --max_regimes 1 --no_save_buffer --total_timesteps 2250000 --steps_per_regime 18500 --total_env_regimes 4 --run_name dyna_4_regimes

# MoWM 4 regimes
python scripts/train_ppo.py --mode dyna --total_timesteps 2250000 --steps_per_regime 18500 --total_env_regimes 4 --run_name mowm_4_regimes
```
