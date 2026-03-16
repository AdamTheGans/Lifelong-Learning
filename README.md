# Lifelong-Learning

> Preventing catastrophic forgetting with world models, planning, and context routing — a continual RL research project.

See [PLAN.md](PLAN.md) for our high-level roadmap.

## Project Structure

```
Lifelong-Learning/
|- src/lifelong_learning/
|  |- agents/ppo/
|  |  |- train.py             # Inner Dyna-PPO update loop
|  |  |- ppo.py               # PPO loss + optional anchor penalty
|  |  |- network.py           # Inner actor-critic with neuromodulation gate
|  |  |- world_model.py       # Predicts next state and reward
|  |  |- episodic_memory.py   # Replay buffer across regime switches
|  |  `- buffers.py           # Rollout buffer with GAE
|  |- agents/brain/
|  |  |- meta_env.py          # Wraps a full inner run as a Gym env
|  |  |- meta_agent.py        # PPO brain policy/value network
|  |  `- signals.py           # Builds the Brain's 19-dim observation
|  |- envs/
|  |  |- dual_goal.py         # Legacy 2-goal MiniGrid task
|  |  |- multi_goal.py        # N-goal MiniGrid task used by the Brain
|  |  |- regime_wrapper.py    # Switches which goal is rewarded
|  |  |- make_env.py          # Factory with wrapper stack
|  |  `- wrappers/
|  |     |- action_reduce.py  # MiniGrid Discrete(7) -> Discrete(3)
|  |     `- one_hot.py        # (H, W, 3) -> (21, H, W) one-hot obs
|  `- utils/
|     |- logger.py            # JSON/PNG run logger
|     `- seeding.py           # Deterministic seeding helper
|- scripts/                   # Train/eval/analyze entry points
|- runs/                      # Timestamped training outputs
|- evals/                     # Timestamped evaluation outputs
`- tests/                     # Regression tests
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
# Each run writes JSON scalars and PNG charts into runs/<run_name>_<timestamp>/
python scripts/analyze_runs.py runs/<run_folder>

# Directly inspect exported JSON/PNG artifacts
#   <run_folder>/<run_name>_data.json
#   <run_folder>/<run_name>_charts.png
```

### 1.6 Resume Training

To resume training from a checkpoint:

```bash
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 1500000 --run_name ppo_stationary --resume_path checkpoints/ppo_stationary_update170.pt
```
*Note: Learning rate annealing will reset unless you manually adjust timesteps, but for fine-tuning/continuation, this is usually acceptable.*

### 1.7 Testing Continual Learning Levers Manually

You can manually test the new Replay Prioritization and Policy Anchoring Weight levers in Dyna-PPO without the Brain agent:

```bash
python scripts/train_ppo.py \
    --env_id MiniGrid-MultiGoal-8x8-v0 \
    --mode dyna \
    --total_timesteps 1500000 \
    --steps_per_regime 18500 \
    --anchoring_weight 0.5 \
    --replay_ratio 0.25 \
    --replay_prioritization 1.0 \
    --run_name ppo_with_levers
```

```powershell
.\myenv\Scripts\python.exe scripts\train_ppo.py --env_id MiniGrid-MultiGoal-8x8-v0 --mode dyna --total_timesteps 1500000 --steps_per_regime 18500 --anchoring_weight 0.5 --replay_ratio 0.25 --replay_prioritization 1.0 --run_name ppo_with_levers_ps
```

---

## Part 2: Meta-RL Hyperparameter Controller ("The Brain")

A second RL agent that learns to adjust the inner Dyna-PPO agent online. In the current implementation it controls learning rate, entropy coefficient, intrinsic curiosity coefficient, imagined horizon, replay ratio, replay prioritization, anchoring weight, and an 8-D neuromodulation context code.

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

# Brain runs write charts and JSON summaries into runs/<run_name>_<timestamp>/
python scripts/analyze_runs.py runs/<brain_run_folder>
```

### 2.3 Recommended Robust Training Command

Use this command for a full-scale Meta-RL training run. The current Brain policy consumes a **19-signal observation** and outputs a **15-dimensional continuous action**: 7 scalar learning levers plus an 8-D context code for neuromodulation.

```bash
python scripts/train_brain.py \
    --brain_episodes 65 \
    --inner_total_timesteps 450000 \
    --brain_num_envs 2 \
    --brain_vectorization async \
    --pretrain_episodes 1 \
    --inner_steps_per_regime 8000 \
    --episodic_memory_capacity 10000 \
    --run_name brain_episodic_run_1
```

Train a Brain with 4 regimes on the 8x8 environment:

```powershell
.\myenv\Scripts\python.exe scripts\train_brain.py --env_id MiniGrid-MultiGoal-8x8-v0 --num_regimes 4 --brain_episodes 65 --inner_total_timesteps 450000 --brain_num_envs 2 --brain_vectorization async --pretrain_episodes 1 --inner_steps_per_regime 18500 --episodic_memory_capacity 10000 --run_name brain_4_regimes_8x8
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

### 2.5 Evaluate Pretrain Checkpoints

Evaluate inner agent hyperparameters dynamically learned during pretraining:

```powershell
$env:PYTHONPATH="src"; .\myenv\Scripts\python.exe scripts\eval_inner_hyperparams.py --checkpoint_path runs\brain_3_regimes_8x8_run_2_20260305-004341\pretrain_1\inner_checkpoints\ep1_env0_update842.pt --env_id MiniGrid-MultiGoal-8x8-v0 --total_timesteps 450000 --num_regimes 3 --steps_per_regime 22500
```
*Note: This script automatically reads `lr`, `ent_coef`, `intrinsic_coef`, and `imagined_horizon` from the checkpoint and associated JSON logs.*

### 2.6 Resume Training

To resume training the Brain from a checkpoint:

```powershell
.\myenv\Scripts\python.exe scripts\train_brain.py --resume_path runs\brain_episodic_run_1_20260226-005533\brain_model.pt --brain_episodes 100
```
*Note: This will restore the model weights, optimizer state, bypass initial imitation learning, and pick up exactly at the episode you left off.*

---

.\myenv\Scripts\python.exe scripts\train_brain.py --resume_path runs/brain_episodic_run_9_20260227-140426/brain_checkpoints/brain_ep10.pt  --brain_episodes 30 --brain_num_envs 6 --inner_total_timesteps 1725000 --inner_steps_per_regime 18500 --run_name brain_4_regimes_8x8_2 --env_id MiniGrid-MultiGoal-8x8-v0 --num_regimes 3

## Part 3: Configuration & Hyperparameters

Understanding the key CLI flags for `train_brain.py` and `eval_brain.py`:

### Environment Complexity
* `--env_id MiniGrid-MultiGoal-5x5-v0` : Current Brain default. A compact multi-goal grid for faster prototyping.
* `--env_id MiniGrid-MultiGoal-8x8-v0` : Larger and harder. Usually needs more `inner_total_timesteps` to adapt cleanly across regime switches.
* `--env_id MiniGrid-DualGoal-5x5-v0` / `MiniGrid-DualGoal-8x8-v0` : Legacy two-goal variants that are still supported for baseline comparisons and tests.

### Meta-RL Training (The Brain)
* `--brain_episodes` : Number of complete inner agent training runs.
* `--brain_num_envs` : Number of parallel meta-environments. Higher values (e.g., 4 or 8) give the Brain smoother gradients and more stable learning, but use more RAM/VRAM. Start with 2 or 4.
* `--brain_vectorization` : Outer Brain env batching mode. `async` is the current default and runs MetaEnvs in subprocesses for real parallel inner training. `sync` keeps everything in one process, which is simpler for debugging but serializes meta-env stepping.
* `--pretrain_episodes` : Number of initial episodes where the Brain uses Imitation Learning (behavioral cloning) on a hardcoded "explore vs exploit" heuristic before switching to PPO. Highly recommended to keep at 1–3 to seed the Brain with a good starting policy.
* `--pretrain_mode` : Pretrain heuristic style. `basic` (default) uses a binary explore/exploit split at 50% success rate. `recovery` uses a multi-tier, surprise-reactive heuristic with 5 graduated tiers and immediate explore response to regime-switch surprise spikes.
* `--brain_lr` : Learning rate for the Brain's PPO optimizer.
* `--brain_ent_coef` : Entropy coefficient for the Brain's PPO optimizer.
* `--reward_mode` : Brain reward function. `auc` (default) rewards absolute success rate. `recovery` uses a hybrid reward that heavily incentivizes fast recovery after regime switches (Δ success bonus + urgency penalty below 80%).

### Inner Agent Forgetting & Adaptation
* `--inner_total_timesteps` : Total steps the inner agent trains for per Brain episode.
* `--inner_steps_per_regime` : How often the environment flips the goal rewards (Regime 0 ↔ Regime 1). Shorter intervals force faster adaptation.
* `--decision_interval` : How often the Brain adjusts hyperparameters (in terms of inner PPO updates). Default is 10 (roughly every 20,000 inner steps).
* `--episodic_memory_capacity` : Size of the ring buffer for past experiences. A capacity of 10,000 holds roughly the last 5 PPO updates. Useful for rehearsing old knowledge to prevent catastrophic forgetting.

