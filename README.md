# Lifelong-Learning

> Preventing catastrophic forgetting with world models, planning, and context routing — a continual RL research project.

See [PLAN.md](PLAN.md) for our high-level roadmap.

## Project Structure

```
Lifelong-Learning/
├── src/lifelong_learning/
│   ├── envs/                     # Environment definitions
│   │   ├── dual_goal.py          # MiniGrid-DualGoal-8x8-v0 (two goals, +5 / -1)
│   │   ├── regime_wrapper.py     # Non-stationary reward switching
│   │   ├── make_env.py           # Env factory (FullyObsWrapper + ActionReduce)
│   │   └── wrappers/
│   │       ├── action_reduce.py  # Discrete(7) → Discrete(3) optimization
│   │       └── one_hot.py        # Unified Observation Wrapper (One-Hot Encoding)
│   ├── agents/ppo/               # PPO agent implementation
│   └── utils/                    # Logging, seeding utilities
├── scripts/
│   ├── check_env.py              # Env sanity check (regime switching)
│   ├── check_vec_env.py          # VectorEnv sanity check
│   ├── verify_rewards.py         # Reward correctness check
│   ├── train_ppo.py              # PPO training script (Dyna-PPO)
│   └── analyze_runs.py           # Training run analyzer + plots
├── tests/
├── requirements.txt              # Dependencies
├── pyproject.toml
└── PLAN.md                       # Research roadmap
```

---

## Part 1: Dyna-PPO Baseline

The active training pipeline uses **Dyna-PPO**, which augments a standard PPO agent with a `SimpleWorldModel` sidekick. The World Model provides:
1. **Intrinsic Curiosity**: Reward augmentation based on prediction error (surprise).
2. **Dreaming**: Generates imagined trajectories to train the policy on latent/predicted transitions (Dyna-style).

### 1.1 Install

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
```

### 1.3 Train PPO

```bash
# Stationary (no regime switching)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --run_name baseline_stationary --no-anneal_lr

# Dyna Mode (Active Dreaming + Curiosity)
python scripts/train_ppo.py --mode dyna --total_timesteps 500000 --run_name dyna_run

# Passive Mode (Standard PPO Baseline)
python scripts/train_ppo.py --mode passive --total_timesteps 500000 --run_name passive_run
```

### 1.4 View Results

```bash
tensorboard --logdir runs
```

### 1.5 Resume Training

To resume training from a checkpoint:

```bash
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --run_name baseline_stationary --resume_path checkpoints/baseline_stationary_update170.pt --no-anneal_lr
```
*Note: Learning rate annealing will reset unless you manually adjust timesteps, but for fine-tuning/continuation, this is usually acceptable.*
