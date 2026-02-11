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
│   │   ├── dreamer_compat.py     # DreamerReadyWrapper (symbolic/pixel obs, log/ keys)
│   │   ├── make_env.py           # Env factory (FullyObsWrapper + PPO/Dreamer routing)
│   │   ├── minigrid_obs.py       # Image obs wrapper (Legacy)
│   │   └── wrappers/
│   │       ├── action_reduce.py  # Discrete(7) → Discrete(3) for Dreamer
│   │       └── one_hot.py        # Unified Observation Wrapper (One-Hot Encoding)
│   ├── agents/ppo/               # PPO agent implementation
│   └── utils/                    # Logging, seeding utilities
├── scripts/
│   ├── check_env.py              # Env sanity check (regime switching)
│   ├── check_vec_env.py          # VectorEnv sanity check
│   ├── verify_rewards.py         # Reward correctness check
│   ├── train_ppo.py              # PPO training script
│   ├── train_dreamerv3_minigrid.py   # DreamerV3 training script
│   ├── check_dreamer_env_io.py       # Dreamer I/O contract check
│   ├── diagnostic_env_check.py       # Full pipeline diagnostic
│   └── analyze_runs.py               # Training run analyzer + plots
├── third_party/dreamerv3/        # Vendored DreamerV3 (git submodule)
├── tests/
├── requirements.txt              # PPO dependencies
├── pyproject.toml
└── PLAN.md                       # Research roadmap
```

---

## Part 1: PPO Baseline

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

# Slow regime switching (15k steps per regime)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --steps_per_regime 15000 --run_name exp_slow_switch --no-anneal_lr

# Fast regime switching (3.5k steps per regime)
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --steps_per_regime 3500 --run_name exp_fast_switch --no-anneal_lr
```

### 1.4 View PPO Results

```bash
tensorboard --logdir runs
```

### 1.5 Resume Training

To resume training from a checkpoint (e.g., if a run didn't converge):

```bash
python scripts/train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --run_name baseline_stationary --resume_path checkpoints/baseline_stationary_update170.pt --no-anneal_lr
```
*Note: Learning rate annealing will reset unless you manually adjust timesteps, but for fine-tuning/continuation, this is usually acceptable.*

---

## Part 2: DreamerV3

### 2.1 Install

<details>
<summary><b>Linux / Colab (GPU) — Recommended</b></summary>

```bash
# 1. Pull DreamerV3 submodule
git submodule update --init --recursive

# 2. Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install project + Dreamer deps
pip install -U pip setuptools wheel
pip install -e .
pip install -r third_party/dreamerv3/requirements.txt

# 4. Verify GPU
python -c "import jax; print(jax.devices())"  # Should show GPU
```

</details>

<details>
<summary><b>Windows (CPU only)</b></summary>

> **Note:** Native Windows NVIDIA GPU is not supported by JAX. Use WSL2 for GPU support.

```powershell
.\.venv\Scripts\Activate.ps1

# Pull submodule
git submodule update --init --recursive

# Set PYTHONPATH
$env:PYTHONPATH="$PWD\third_party\dreamerv3;$env:PYTHONPATH"

# Install Dreamer deps (excluding JAX/NVIDIA)
Get-Content third_party/dreamerv3/requirements.txt | Where-Object { $_ -notmatch "jax|nvidia" } | Set-Content dreamerv3_requirements.txt
pip install -U -r dreamerv3_requirements.txt

# Install CPU-only JAX for the correct version
pip install -U "jax[cpu]==0.4.33"

# Verify
python -c "import dreamerv3, embodied; import jax; print('OK'); print('devices:', jax.devices())"
```

</details>

### 2.2 Sanity Checks

```bash
# Full pipeline diagnostic (obs quality, action space, termination, reward distribution)
python scripts/diagnostic_env_check.py

# Dreamer I/O contract check (50-step smoke test)
python scripts/check_dreamer_env_io.py
```

### 2.3 Train DreamerV3

All tuned defaults are built into the script. No flags needed for a standard run.

```bash
# Smoke test (~minutes, good for validating setup)
python scripts/train_dreamerv3_minigrid.py --run.steps 10000

# If running on CPU, add `--jax.platform cpu`

# Full stationary run (100k steps, ~30-60min on Colab GPU)
python scripts/train_dreamerv3_minigrid.py

# Add regime switching (after stationary baseline is solid)
python scripts/train_dreamerv3_minigrid.py --env.steps_per_regime 15000

# Revert to pixel-based observations (64×64 RGB)
python scripts/train_dreamerv3_minigrid.py --env.symbolic False
```

**Observation modes:**
- **Symbolic (default):** unified One-Hot float tensor `(20, 8, 8)` → MLP encoder.
    - **Unified Strategy:** Both PPO and DreamerV3 now see the exact same mathematically correct representation.
    - **Encoding:** Object Type (11), Color (6), State (3) are one-hot encoded and concatenated.
    - DreamerV3 receives this as a flattened vector `(1280,)`. PPO receives it as a `(20, 8, 8)` tensor.
    - *Note: DreamerV3 automatically detects 1D inputs as vectors (MLP) and 3D inputs as images (CNN). By flattening to `(1280,)`, we ensure it uses the correct MLP encoder without applying image augmentations.*
- **Pixel:** `(64, 64, 3)` uint8 RGB from `env.render()` → CNN encoder. Set `--env.symbolic False`.

Both PPO and DreamerV3 use `FullyObsWrapper` for full 8×8 grid observability (no partial view).

**Built-in defaults** (overridable via CLI):
| Setting | Value | Rationale |
|---|---|---|
| `--configs` | `size12m` | 12M-param model (256 units) |
| `--run.steps` | `100000` | Sanity run length |
| `--run.envs` | `4` | Parallel envs for data diversity |
| `--run.train_ratio` | `64` | Gradient steps per env step |
| `--agent.imag_length` | `64` | Imagination horizon (256-step episodes) |
| `--batch_size` | `16` | Replay batch size |
| `--env.symbolic` | `True` | Symbolic obs (MLP) vs pixel obs (CNN) |

A resolved config is automatically saved to `logdir/<run>/config_resolved.json`.

### 2.4 Analyze Results

```bash
# Interactive analyzer with plots
python scripts/analyze_runs.py

# Or point directly at a run
python scripts/analyze_runs.py logdir/<run_folder>

# TensorBoard
tensorboard --logdir logdir
```
