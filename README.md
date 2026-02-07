# Lifelong-Learning

Our goal is to prevent catastrophic forgetting, enabling lifelong learning in RL.

We will experiment with continual / non-stationary RL with world models + planning + context routing.

## Roadmap
1) PPO baseline on MiniGrid (stationary + non-stationary regimes)
2) Single world model + MPC planner baseline
3) Multi-context (multi-head) WM + surprise routing
4) WM adapters + hypernetwork-conditioned modulation (final method)


## Directions to Run (so far)
0) clear pycache: `find . -type d -name "__pycache__" -exec rm -r {} +`
1) `pip install -r requirements.txt`
2) `pip install -e .`
3) Option sanity checks: 
    a. (OUTDATED RIGHT NOW) env with regime switch: `python scripts/00_check_env.py --switch_mid_episode --steps 120`
    b. env vector: `python scripts/00_check_vec_env.py`
    c. correct rewards: `python scripts/00_verify_rewards.py` 
4) Test with PPO:
    a. Stationary: `python scripts/01_train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --run_name baseline_stationary --no-anneal_lr`
    b. Slow switch: `python scripts/01_train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --steps_per_regime 15000 --run_name exp_slow_switch --no-anneal_lr`
    c. Fast switches: `python scripts/01_train_ppo.py --env_id MiniGrid-DualGoal-8x8-v0 --total_timesteps 750000 --steps_per_regime 3500 --run_name exp_fast_switch --no-anneal_lr`
    d. View results: `tensorboard --logdir runs`
  