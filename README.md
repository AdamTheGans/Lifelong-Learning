# Lifelong-Learning

Our goal is to prevent catastrophic forgetting, enabling lifelong learning in RL.

We will experiment with continual / non-stationary RL with world models + planning + context routing.

## Roadmap
1) PPO baseline on MiniGrid (stationary + non-stationary regimes)
2) Single world model + MPC planner baseline
3) Multi-context (multi-head) WM + surprise routing
4) WM adapters + hypernetwork-conditioned modulation (final method)


## Directions to Run (so far)
1) `pip install -r requirements.txt`
2) `pip install -e .`
3) Run the sanity check, env with regime switch: `python scripts/00_check_env.py --switch_mid_episode --steps 120` and this check `python scripts/00_check_vec_env.py`
4) Test with PPO:
    a. Stationary: `python scripts/01_train_ppo.py --env_id MiniGrid-Empty-8x8-v0 --total_timesteps 1000000 --run_name baseline_stationary`
    b. Slow switch (~200k): `python scripts/01_train_ppo.py --env_id MiniGrid-Empty-8x8-v0 --total_timesteps 1000000 --steps_per_regime 25000 --run_name exp_slow_switch`
    c. Fast switches (~40k): `python scripts/01_train_ppo.py --env_id MiniGrid-Empty-8x8-v0 --total_timesteps 1000000 --steps_per_regime 5000 --run_name exp_fast_switch`
    d. View results: `tensorboard --logdir runs`
  