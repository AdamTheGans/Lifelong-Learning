# Lifelong-Learning (World Models + Continual RL)

## What we’re trying to solve
Deep RL agents (and world models) often **catastrophically forget** older skills/dynamics when the environment’s rules change over time (non-stationary / continual RL). We want an agent that can **adapt quickly to new regimes** while **retaining** (and rapidly reusing) competence in old regimes when they return.

## Why it matters
- Real-world environments are non-stationary (rules, dynamics, rewards shift).
- “Train once” agents are brittle; continual adaptation is a core ingredient for robust autonomy and (eventually) AGI-like behavior.
- World models are attractive for data efficiency, but the **world model itself can forget**, breaking planning.

## The Environment: Dual-Goal MiniGrid
We use a custom **8x8 MiniGrid** environment with two colored squares (Goals).
- **Rules**: Stepping on the *correct* color yields **+5 reward**. Stepping on the *wrong* color yields **-1 reward**.
- **Time Penalty**: -0.01 per step.
- **Max Steps**: 256 per episode.
- **Actions**: Discrete(3) — Turn Left, Turn Right, Move Forward.
- **Regime Switching**: The "correct" color swaps periodically (e.g., every 15k steps), causing the agent's policy to become outdated.

## Current Architecture: Dyna-PPO with Simple World Model

> **Note on DreamerV3**: We initially attempted to use DreamerV3 (a powerful, JAX-based world model), but found it to be overkill for this simple grid environment. It was computationally heavy and difficult to tune for our specific symbolic observation needs. We pivoted to a custom, lightweight **Simple World Model** integrated directly into PPO.

### 1. Simple World Model
A lightweight, CNN-based model designed for Symbolic (One-Hot) observations.
- **Input**:
    - **State**: One-Hot Tensor (21, 8, 8) processed by a 3-layer CNN (mirrors Actor-Critic encoder).
    - **Action**: Learned Embedding (dim=32), concatenated after CNN (late fusion).
- **Network**: CNN → Flatten → Concat [CNN features, Action Emb] → 2-layer MLP → Heads.
- **Outputs**:
    - **Next State**: Logits for Cross-Entropy Loss (predicts next One-Hot grid, 21 channels).
    - **Reward**: Scalar prediction.
- **Key Feature**: `discretize_state` — forcing predicted states back into crisp 0/1 One-Hot vectors to prevent "blurry dreams."

### 2. Dyna-PPO Loop (`train_ppo.py`)
Training occurs in three phases per update:
1.  **Phase A: Real Experience**
    - Agent interacts with real env.
    - **Intrinsic Curiosity**: Adds `prediction_error * intrinsic_coef` to reward.
2.  **Phase B: World Model Training**
    - Supervised learning on real trajectories to minimize State (CE) and Reward (MSE) error.
3.  **Phase C: Imagination (Dreaming)**
    - Seeding: Start from real states in replay buffer.
    - Dreaming: WM acts as simulator for `imagined_horizon` steps.
    - Learning: PPO updates policy/value on these *dreamed* trajectories (gradients blocked to WM).

## Research Roadmap

### 1. Baseline: Standard PPO
- Verify task learnability and measure forgetting curves under regime switches.
- **Status**: [x] Working.

### 2. Baseline: Dyna-PPO (Single Model)
- Establish performance of the single-model Dyna approach.
- Measure how quickly the WM adapts vs. how quickly it forgets old regimes.
- **Status**: [x] Working (Refined & Audited).

### 3. Future Direction A: Multi-Head World Model
- **Idea**: Detect surprise spikes (high prediction error) and spawn/switch to new network heads.
- **Mechanism**:
    - Keep a shared Trunk.
    - Branch multiple Heads for dynamics/reward.
    - Route based on prediction error (Surprise).

### 4. Future Direction B: Regime-Specific World Models (The "Library" Approach)
- **Idea**: Explicitly learn distinct World Models for different regimes and "route" the PPO agent to the correct one.
- **Mechanism**:
    - Usage: PPO receives a `REGIME_ID` input (latent or explicit).
    - Training: Maintain a library of WMs. When a regime switch is detected (high surprise), spin up a new WM or retrieve a matching old one.
    - Dreaming: PPO practices on *all* known WMs in the library, preventing catastrophic forgetting of the policy.

### 5. Meta-RL Hyperparameter Controller ("The Brain")
- **Idea**: A second RL agent (the "Brain") observes training signals from the inner Dyna-PPO (surprise, success rate, losses, etc.) and learns to dynamically adjust hyperparameters (lr, entropy coef, intrinsic curiosity coef, imagined horizon) to maximize recovery speed after regime switches.
- **Architecture**:
    - The inner training loop is wrapped as a Gymnasium environment (`MetaEnv`).
    - Brain observes a 15-dim signal vector every N inner updates.
    - Brain outputs relative HP adjustments (multiplicative scaling) as continuous actions.
    - Brain is trained via PPO on the meta-MDP with reward = Δ success_rate + α·Δ return − β·failure_rate.
- **Key Files**: `src/lifelong_learning/agents/brain/`, `scripts/train_brain.py`
- **Status**: [x] Implemented (awaiting experimental validation).

## Metrics & Evaluation
- **Recovery usage**: Steps to reach optimal performance after a regime switch.
- **Retained performance**: Zero-shot performance when returning to a known regime.
- **Routing accuracy**: How quickly the system identifies the active regime.
