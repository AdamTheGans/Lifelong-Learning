# Lifelong-Learning

Our goal is to prevent catastrophic forgetting, enabling lifelong learning in RL.

We will experiment with continual / non-stationary RL with world models + planning + context routing.

## Roadmap
1) PPO baseline on MiniGrid (stationary + non-stationary regimes)
2) Single world model + MPC planner baseline
3) Multi-context (multi-head) WM + surprise routing
4) WM adapters + hypernetwork-conditioned modulation (final method)


## Directions to Run PPO
0) clear pycache: `find . -type d -name "__pycache__" -exec rm -r {} +`
1) `pip install -r requirements.txt`
2) `pip install -e .`
3) Option sanity checks: 
    a. (OUTDATED RIGHT NOW) env with regime switch: `python scripts/00_check_env.py --switch_mid_episode --steps 120`
    b. env vector: `python scripts/00_check_vec_env.py`
    c. correct rewards: `python scripts/00_verify_rewards.py` 
4) Train PPO:
    a. Stationary: `python scripts/01_train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --run_name baseline_stationary --no-anneal_lr`
    b. Slow switch: `python scripts/01_train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --steps_per_regime 15000 --run_name exp_slow_switch --no-anneal_lr`
    c. Fast switches: `python scripts/01_train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --steps_per_regime 3500 --run_name exp_fast_switch --no-anneal_lr`
5) View results: `tensorboard --logdir runs`

## Directions to Run Dreamer

**Install (Windows CPU):**
1. `pip install -U "jax[cpu]==0.4.33"`
2. `mkdir third_party`
3. `git submodule add https://github.com/danijar/dreamerv3 third_party/dreamerv3`
4. `$env:PYTHONPATH="$PWD\third_party\dreamerv3;$env:PYTHONPATH"`
5. **Install Deps:**
   ```powershell
   Get-Content third_party/dreamerv3/requirements.txt | Where-Object { $_ -notmatch "jax|nvidia" } | Set-Content dreamerv3_requirements.txt
   pip install -U -r dreamerv3_requirements.txt
   ```
6. Sanity check: `python -c "import dreamerv3, embodied; import jax; print('imports ok'); print('embodied:', embodied.__file__); print('devices:', jax.devices())"`
   - *Note: Native Windows NVIDIA GPU is not supported by JAX. Use WSL2 for GPU support.*

**Install (Linux GPU):**
1. Check `nvidia-smi` to ensure drivers are 525 or higher.
2. `git submodule update --init --recursive` (Ensure dreamerv3 is pulled)
3. `python3.12 -m venv .venv`
4. `source .venv/bin/activate`
5. `pip install -U pip setuptools wheel`
6. `pip install -e .`
7. `pip install -r third_party/dreamerv3/requirements.txt` (Installs JAX+CUDA and other Dreamer deps)
8. Sanity check: `python -c "import jax; print(jax.devices())"` (Should show GPU)

**Run:**

1. **Sanity Checks:**
   - I/O Check: `python scripts/check_dreamer_env_io.py`
   - Script Check (CPU/Safe): `python scripts/train_dreamerv3_minigrid.py --jax.platform cpu --check_env_io True` 
     - *On Linux w/ GPU, you can omit `--jax.platform cpu` to check if CUDA initializes correctly.*

2. **Train:**

   **Windows (CPU):**
   ```powershell
   python scripts/train_dreamerv3_minigrid.py --jax.platform cpu --run.steps 50000 --run.train_ratio 32
   ```

   **Linux (GPU):**
   ```bash
   # Omit --jax.platform cpu to use default (CUDA)
   python scripts/train_dreamerv3_minigrid.py --run.steps 50000 --run.train_ratio 32
   ```
