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

---

## Part 2: Meta-RL Hyperparameter Controller ("The Brain")

A second RL agent that learns to adjust the Dyna-PPO agent's hyperparameters (learning rate, entropy coefficient, intrinsic curiosity coefficient, imagined dream horizon) in real-time to maximize recovery speed after regime switches.

### 2.1 Smoke Test

```bash
python scripts/train_brain.py \
    --inner_total_timesteps 50000 \
    --inner_steps_per_regime 5000 \
    --brain_episodes 2 \
    --decision_interval 5 \
    --run_name brain_smoke_test
```

### 2.2 Full Training

```bash
# Train Brain with regime switching
python scripts/train_brain.py \
    --inner_total_timesteps 250000 \
    --inner_steps_per_regime 15000 \
    --brain_episodes 50 \
    --decision_interval 10 \
    --run_name brain_full_run

# View Brain + inner agent tensorboard logs
tensorboard --logdir runs
```

### 2.3 Recommended Robust Training Command

Use this command for a full-scale Meta-RL training run that prioritizes robust generalization and recovery from catastrophic forgetting, utilizing the new Episodic Memory system:

```bash
python scripts/train_brain.py \
    --brain_episodes 65 \
    --inner_total_timesteps 450000 \
    --brain_num_envs 2 \
    --pretrain_episodes 1 \
    --inner_steps_per_regime 8000 \
    --episodic_memory_capacity 10000 \
    --run_name brain_episodic_run_1
```

Train a Brain with 4 regimes on the 8x8 environment:

```powershell
.\myenv\Scripts\python.exe scripts\train_brain.py --env_id MiniGrid-MultiGoal-8x8-v0 --num_regimes 4 --brain_episodes 65 --inner_total_timesteps 450000 --brain_num_envs 2 --pretrain_episodes 1 --inner_steps_per_regime 18500 --episodic_memory_capacity 10000 --run_name brain_4_regimes_8x8
```

### 2.4 Evaluation

Evaluate a trained Brain checkpoint on a fresh inner agent (inference-only):

```powershell
.\myenv\Scripts\python.exe scripts\eval_brain.py --brain_checkpoint runs\brain_episodic_run_1_20260226-005533\brain_model.pt --total_timesteps 1150000 --steps_per_regime 18500 --episodic_memory_capacity 10000 --run_name eval_brain_run1
```

Evaluate with 4 regimes on the 8x8 environment:

```powershell
.\myenv\Scripts\python.exe scripts\eval_brain.py --env_id MiniGrid-MultiGoal-8x8-v0 --num_regimes 4 --steps_per_regime 18500 --brain_checkpoint runs\brain_episodic_run_1_20260226-005533\brain_model.pt --total_timesteps 1150000 --run_name eval_brain_4_regimes
```

### 2.5 Resume Training

To resume training the Brain from a checkpoint:

```powershell
.\myenv\Scripts\python.exe scripts\train_brain.py --resume_path runs\brain_episodic_run_1_20260226-005533\brain_model.pt --brain_episodes 100
```
*Note: This will restore the model weights, optimizer state, bypass initial imitation learning, and pick up exactly at the episode you left off.*

---

## Part 3: Configuration & Hyperparameters

Understanding the key CLI flags for `train_brain.py` and `eval_brain.py`:

### Environment Complexity
* `--env_id MiniGrid-DualGoal-5x5-v0` : Default. A compact 5x5 grid (8x8 with walls). Good for fast prototyping.
* `--env_id MiniGrid-DualGoal-8x8-v0` : A larger 8x8 grid (11x11 with walls). Significantly harder navigation task; usually requires higher `inner_total_timesteps` to master.

### Meta-RL Training (The Brain)
* `--brain_episodes` : Number of complete inner agent training runs.
* `--brain_num_envs` : Number of parallel meta-environments. Higher values (e.g., 4 or 8) give the Brain smoother gradients and more stable learning, but use more RAM/VRAM. Start with 2 or 4.
* `--pretrain_episodes` : Number of initial episodes where the Brain uses Imitation Learning (behavioral cloning) on a hardcoded "explore vs exploit" heuristic before switching to PPO. Highly recommended to keep at 1–3 to seed the Brain with a good starting policy.
* `--brain_lr` : Learning rate for the Brain's PPO optimizer.
* `--brain_ent_coef` : Entropy coefficient for the Brain's PPO optimizer.

### Inner Agent Forgetting & Adaptation
* `--inner_total_timesteps` : Total steps the inner agent trains for per Brain episode.
* `--inner_steps_per_regime` : How often the environment flips the goal rewards (Regime 0 ↔ Regime 1). Shorter intervals force faster adaptation.
* `--decision_interval` : How often the Brain adjusts hyperparameters (in terms of inner PPO updates). Default is 10 (roughly every 20,000 inner steps).
* `--episodic_memory_capacity` : Size of the ring buffer for past experiences. A capacity of 10,000 holds roughly the last 5 PPO updates. Useful for rehearsing old knowledge to prevent catastrophic forgetting.

