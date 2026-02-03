# Lifelong-Learning (World Models + Continual RL)

## What we’re trying to solve
Deep RL agents (and world models) often **catastrophically forget** older skills/dynamics when the environment’s rules change over time (non-stationary / continual RL). We want an agent that can **adapt quickly to new regimes** while **retaining** (and rapidly reusing) competence in old regimes when they return.

## Why it matters
- Real-world environments are non-stationary (rules, dynamics, rewards shift).
- “Train once” agents are brittle; continual adaptation is a core ingredient for robust autonomy and (eventually) AGI-like behavior.
- World models are attractive for data efficiency, but the **world model itself can forget**, breaking planning.

## High-level approach
We’ll build a staged system in MiniGrid:
1) **PPO baselines** to verify tasks are learnable and to measure forgetting under regime switches.
2) **World Model + Planner (MPC)** baseline (single model) to establish model-based performance and failure modes under non-stationarity.
3) **Multi-context World Model + routing**:
   - maintain multiple context-specific dynamics (e.g., multi-head transitions),
   - use a **surprise / prediction-error router** to select the best context online,
   - plan using the selected context model.
4) **Adapters / Hypernetwork-conditioned modulation (final method)**:
   - keep a shared world model trunk (stable),
   - add small **context-specific adapters** inside dynamics (plastic),
   - optionally generate adapter parameters via a tiny **hypernetwork** from a stored context embedding (“re-indexing”),
   - aim for fast recovery + reduced forgetting with low parameter overhead.

## Evaluation (paper-shaped)
We’ll measure:
- **Recovery time** after a regime switch (steps to regain performance)
- **Forgetting / backward transfer** on old regimes after learning new ones
- **Routing accuracy & switch delay** (when regimes are discrete and known for logging)
- Comparisons: PPO vs single-WM planning vs multi-head vs adapters (+ replay/distill baselines)

## Deliverables
- Reproducible training scripts + logging
- Non-stationary MiniGrid regime wrapper(s)
- Clear plots showing: switch → surprise spike → routing → rapid recovery
- A concise write-up suitable for a workshop-style paper/report
