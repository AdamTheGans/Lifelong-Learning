# Architecture Spec: Context-Aware Meta-RL for Continual Learning in MiniGrid

## 1. The Problem: Catastrophic Forgetting

We are training a PPO agent in a MiniGrid environment that undergoes discrete "Regime Shifts" (e.g., the actions swap, or the reward targets invert). Standard RL agents suffer from catastrophic forgetting; when the physics change, they overwrite their network weights and completely forget how to solve the previous regimes.

## 2. The Solution: Memory-Augmented Meta-RL

Instead of forcing the PPO agent to blindly guess the current rules, we are building a "World Model" (a CNN + GRU). The World Model reads the last 30 steps of gameplay and summarizes them into a 256-D hidden state ($h_t$). This $h_t$ vector acts as a "Context Badge" that tells the PPO agent exactly which regime it is currently in.

To prevent the World Model itself from forgetting older regimes, we maintain a Long-Term Memory Buffer of historical 30-step sequences. We train the World Model on a forced 50/50 mixture of current gameplay and historical memory.

## 3. Core Components & Logic

### A. The Long-Term Memory Buffer

This is a custom Replay Buffer designed to store sequences, not individual steps.

* **Storage Unit:** A "Chunk" of 30 consecutive transitions `(State, Action, Reward, Done)`.
* **Save Trigger:** A chunk is saved if the World Model's prediction error (surprise) exceeds an EMA threshold.
* **Rate Limits:** * Maximum 1 save per 1,000 environment steps (prevents flooding the buffer during a highly volatile regime).
* Minimum 1 save per 5,000 environment steps (forces routine saves even if the agent is perfectly comfortable).


* **Eviction Policy:** Uniform Random. When the buffer hits maximum capacity, randomly overwrite an existing chunk. This mimics Reservoir Sampling and ensures an unbiased history.

### B. The World Model (Short-Term Memory / Predictor)

This network predicts the next state and reward to build an understanding of the environment's physics.

* **Input Dimensions:** * CNN Feature Extractor Output: `4096-D`
* Action Embedding: `32-D`
* Reward: `1-D`


* **Architecture:** The concatenated input `(4129-D)` passes into a GRU.
* **GRU Hidden State ($h_t$):** `256-D`.
* **Output / Transition MLP:** `256-D` (from GRU) -> `256-D` -> Output Heads (Next Frame Prediction, Next Reward Prediction).

### C. The PPO Agent (The Decision Maker)

The PPO network remains largely standard, with one critical modification to its input.

* **Standard Input:** `(B, 21, 8, 8)` representing the current MiniGrid state.
* **The Context Injection:** The PPO's internal feature extractor will process the `(21, 8, 8)` grid into a flat vector. We must **concatenate** the GRU's current `256-D` hidden state ($h_t$) to this flat vector *before* passing it to the Actor and Critic heads.

## 4. Critical Implementation Rules (DO NOT IGNORE)

1. **The Gradient Wall (`.detach()`):** * **Rule:** PPO must NEVER update the World Model's weights.
* **Implementation:** When passing the GRU's hidden state $h_t$ to the PPO agent, you must call `.detach()`.
* **Why:** The World Model's only job is System Identification (understanding physics). If PPO gradients flow backward into the GRU, the GRU will try to optimize for reward instead of truth, corrupting the memory system.


2. **The 50/50 Training Batch Split:**
* **Rule:** When updating the World Model, the training batch must be strictly balanced.
* **Implementation:** For a batch size of $N$, sample $N/2$ chunks from the live, online rollout buffer, and sample $N/2$ chunks uniformly from the Long-Term Memory Buffer.
* **Why:** This is the exact mechanism that prevents the World Model from catastrophically forgetting old rules.


3. **Episode Boundary Masking:**
* **Rule:** The World Model cannot predict a randomized board reset.
* **Implementation:** When calculating the Mean Squared Error (MSE) loss for the World Model's next-state prediction, multiply the loss by `(1 - done)`.
* **Why:** If the transition is a terminal step (`done = True`), the loss becomes 0. This stops the network from blowing up its gradients trying to predict a purely random initialization. Keep the GRU internal state rolling, but ignore the loss on that specific boundary step.
