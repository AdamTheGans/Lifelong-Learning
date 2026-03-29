# Lifelong Learning

Continual reinforcement learning experiments focused on catastrophic forgetting in regime-shifting MiniGrid tasks.

This project studies how an agent can adapt to changing reward regimes without erasing previously learned behavior. The codebase combines an inner Dyna-PPO learner with a higher-level meta-controller ("the Brain") that adjusts learning dynamics online.

Relevant reading:
- [Combating Catastrophic Forgetting](./Combating%20Catastrophic%20Forgetting.pdf)
- [Project roadmap](./PLAN.md)
- [Reference bibliography](./references.bib)

## Overview

The core research question is simple: how do we make an RL agent recover quickly after an environment regime switch while still preserving useful behavior from earlier regimes?

The current implementation has two layers:

- `Dyna-PPO` inner agent: a PPO learner augmented with a world model, intrinsic curiosity, imagined rollouts, episodic replay, and optional policy anchoring.
- `Brain` meta-agent: a PPO controller that watches training signals from the inner agent and adjusts hyperparameters such as learning rate, entropy coefficient, curiosity strength, imagined horizon, replay ratio, replay prioritization, anchoring weight, and an 8D context code for neuromodulation.

## What Is In This Repo

- Custom continual-learning MiniGrid environments with regime switching.
- A trainable Dyna-PPO baseline for passive and world-model-assisted learning.
- A meta-RL controller that treats inner training as its environment.
- Analysis scripts for turning run logs into summary plots.
- Regression tests covering environments, signals, CLI entry points, and plotting utilities.

## Repository Layout

```text
Lifelong-Learning/
|- src/lifelong_learning/
|  |- agents/ppo/          # Inner Dyna-PPO learner, replay, world model, networks
|  |- agents/brain/        # Meta-agent, signal extraction, MetaEnv wrapper
|  |- envs/                # Multi-goal / dual-goal environments and wrappers
|  `- utils/               # Logging and reproducibility helpers
|- scripts/                # Train, evaluate, verify, and analyze entry points
|- tests/                  # Automated checks
|- runs/                   # Training outputs written by experiments
|- evals/                  # Evaluation outputs written by experiments
|- checkpoints/            # Saved checkpoints
|- PLAN.md                 # Project roadmap
`- Combating Catastrophic Forgetting.pdf
```

## Core Ideas

### 1. Regime-shifting continual learning

The environments can flip which goal is rewarded over time. That makes forgetting visible: an agent that adapts too aggressively to the new regime can lose competence on earlier ones.

### 2. Dyna-style inner learning

The inner agent extends PPO with:

- intrinsic curiosity from world-model prediction error
- imagined rollouts for Dyna-style policy updates
- episodic replay for rehearsal across regime switches
- replay prioritization toward older regimes
- policy anchoring to reduce destructive drift

### 3. Meta-control with the Brain

The Brain sees a 19-signal observation summarizing inner training and outputs a 15-dimensional continuous action:

- 7 scalar learning levers
- 8 context dimensions used for neuromodulation

In practice, this turns inner-agent training itself into an RL problem.

## Setup

### Requirements

- Python 3.10+
- PyTorch
- Gymnasium
- MiniGrid

Install with:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Validation

Run the basic checks before longer experiments:

```bash
python scripts/verify_rewards.py
python scripts/check_vec_env.py
pytest tests/
```

## Common Workflows

### Train a passive PPO baseline

```bash
python scripts/train_ppo.py \
  --mode passive \
  --env_id MiniGrid-MultiGoal-8x8-v0 \
  --total_timesteps 300000 \
  --steps_per_regime 18500 \
  --run_name passive_baseline
```

### Train the Dyna-PPO continual-learning agent

```bash
python scripts/train_ppo.py \
  --mode dyna \
  --env_id MiniGrid-MultiGoal-8x8-v0 \
  --total_timesteps 300000 \
  --steps_per_regime 18500 \
  --replay_ratio 0.25 \
  --replay_prioritization 0.5 \
  --anchoring_weight 0.1 \
  --run_name dyna_continual
```

### Smoke test the Brain

```bash
python scripts/train_brain.py \
  --env_id MiniGrid-MultiGoal-5x5-v0 \
  --inner_total_timesteps 50000 \
  --inner_steps_per_regime 5000 \
  --brain_episodes 2 \
  --brain_num_envs 2 \
  --decision_interval 5 \
  --run_name brain_smoke
```

### Train the Brain on a larger continual-learning run

```bash
python scripts/train_brain.py \
  --env_id MiniGrid-MultiGoal-8x8-v0 \
  --num_regimes 3 \
  --inner_total_timesteps 450000 \
  --inner_steps_per_regime 18500 \
  --brain_episodes 65 \
  --brain_num_envs 2 \
  --brain_vectorization async \
  --pretrain_episodes 1 \
  --episodic_memory_capacity 10000 \
  --run_name brain_multiregime
```

### Evaluate a trained Brain checkpoint

```bash
python scripts/eval_brain.py \
  --brain_checkpoint runs/<brain_run>/brain_model.pt \
  --env_id MiniGrid-MultiGoal-8x8-v0 \
  --num_regimes 3 \
  --total_timesteps 1150000 \
  --steps_per_regime 18500 \
  --run_name eval_brain_multiregime
```

### Analyze results

```bash
python scripts/analyze_runs.py runs/<run_folder>
```

## Useful Scripts

- `scripts/train_ppo.py`: trains the inner PPO or Dyna-PPO agent
- `scripts/train_brain.py`: trains the meta-controller
- `scripts/eval_brain.py`: runs inference-only evaluation with a trained Brain
- `scripts/eval_inner_hyperparams.py`: evaluates hyperparameters extracted from a checkpoint
- `scripts/analyze_runs.py`: summarizes run logs and generates charts
- `scripts/plot_high_scale.py`: produces higher-resolution plots for a run folder

## Testing

The test suite covers:

- environment integrity and reward behavior
- signal extraction and Brain observation handling
- network and training CLI regressions
- plotting and analysis helpers

Run all tests with:

```bash
pytest tests/
```

## Project Status

This repository is an active research workspace rather than a finished library. The current code is best read as an experimental platform for:

- catastrophic forgetting under reward regime switches
- replay- and anchoring-based continual-learning interventions
- meta-RL control of training dynamics

## Citation and References

If you are reviewing the background for this project, start with the local paper linked here:

- [Combating Catastrophic Forgetting](./Combating%20Catastrophic%20Forgetting.pdf)

Supporting references used during development live in [references.bib](./references.bib).
