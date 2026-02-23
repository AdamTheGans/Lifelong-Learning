import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from lifelong_learning.agents.ppo.world_model import SimpleWorldModel

class MixtureOfWorldModels(nn.Module):
    """
    Manager class for a mixture of SimpleWorldModels to support Continuous Learning.
    Dynamically spawns new world models when the prediction error (surprise) for a transition
    exceeds a dynamically tracked threshold (EMA).
    """

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Initialize with a single world model
        initial_model = SimpleWorldModel(obs_shape, n_actions, hidden_dim)
        self.models = nn.ModuleList([initial_model])
        
        # Status variables
        self.active_regime_id = 0
        self.ema_losses = [1.0]
        self.ema_history = [collections.deque([1.0], maxlen=10)]
        
        # Hyperparameters
        self.ema_alpha = 0.05
        self.surprise_threshold = 5.0
        
        # MoWM Routing Fixes
        self.global_grace_period = 20000     # No spawns before this step
        self.newborn_grace_period = 10000    # Force active regime after spawn
        self.surprise_window_size = 100      # Smooth surprise across 100 transitions
        self.switch_penalty = 1.2            # Hysteresis stickiness penalty
        self.ema_epsilon = 1e-5              # Denominator safety avoids div by zero
        self.max_ema_growth = 0.10           # Trend-Aware Spawning: Max relative growth
        
        # State tracking
        self.force_active_until = 0
        self.surprise_window = collections.deque(maxlen=self.surprise_window_size)

    def update_ema(self, current_loss: float):
        """
        Updates the exponential moving average of the loss.
        Note: This should ONLY be called externally during the training loop of the active world model,
        and not during inference or spawn checks, to keep the baseline stable during surprise spikes.
        """
        self.ema_losses[self.active_regime_id] = (self.ema_alpha * current_loss) + ((1 - self.ema_alpha) * self.ema_losses[self.active_regime_id])
        self.ema_history[self.active_regime_id].append(self.ema_losses[self.active_regime_id])

    def infer_regime(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, global_step: int
    ) -> tuple[int, float]:
        """
        Evaluates a transition across all models and returns the ID of the best fitting regime
        and its corresponding lowest loss. Handles newborn stickiness to force training.
        """
        # Newborn stickiness bypass: if in grace period, stick with active model
        if global_step < self.force_active_until:
            best_regime_id = self.active_regime_id
            best_raw_loss = float('inf')  # Value doesn't matter, we bypass routing and spawning
        else:
            best_regime_id = 0
            lowest_relative_loss = float('inf')
            best_raw_loss = 0.0

        # Convert next_state (B, C, H, W) float to class indices (B, H, W) for CE Loss
        # Assuming next_state is one-hot or normalized probabilities
        next_state_indices = torch.argmax(next_state, dim=1)

        for i, model in enumerate(self.models):
            with torch.no_grad():
                next_obs_pred, pred_reward = model(state, action)
                # Compute cross-entropy loss for state
                state_loss = F.cross_entropy(next_obs_pred, next_state_indices).item()
                # Compute MSE loss for reward
                reward_loss = F.mse_loss(pred_reward, reward).item()
                # Total surprise
                loss = state_loss + reward_loss
                
            # Only find the lowest loss if we aren't bypassing via newborn stickiness
            if global_step >= self.force_active_until:
                relative_loss = loss / max(self.ema_losses[i], self.ema_epsilon)
                if i != self.active_regime_id:
                    relative_loss *= self.switch_penalty

                if relative_loss < lowest_relative_loss:
                    lowest_relative_loss = relative_loss
                    best_regime_id = i
                    best_raw_loss = loss
                
        # If bypassing, calculate just the active model's loss to return
        if global_step < self.force_active_until:
            with torch.no_grad():
                next_obs_pred, pred_reward = self.models[self.active_regime_id](state, action)
                state_loss = F.cross_entropy(next_obs_pred, next_state_indices).item()
                reward_loss = F.mse_loss(pred_reward, reward).item()
                best_raw_loss = state_loss + reward_loss
                
        return best_regime_id, best_raw_loss

    def check_and_spawn(self, lowest_loss: float, best_regime_id: int, global_step: int) -> bool:
        """
        Checks if the lowest available loss constitutes a surprise based on the dynamic EMA.
        Includes safeguards for global grace period, newborn grace period, and sequence smoothing.
        """
        # Global Step 0 Grace Guard & Newborn Stickiness Guard
        if global_step < self.global_grace_period or global_step < self.force_active_until:
            self.surprise_window.clear()
            return False
            
        # Surprise Smoothing
        self.surprise_window.append(lowest_loss)
        
        # Wait until window is full before making spawn decisions
        if len(self.surprise_window) < self.surprise_window_size:
            return False
            
        smoothed_loss = sum(self.surprise_window) / len(self.surprise_window)

        # False Spawn Guard: Policy Shift Check (Trend-Aware Spawning)
        # If the EMA baseline is already rising rapidly, the agent is exploring and we should block spawns.
        history = self.ema_history[best_regime_id]
        if len(history) == history.maxlen:
            relative_growth = (history[-1] - history[0]) / max(history[0], self.ema_epsilon)
            if relative_growth > self.max_ema_growth:
                self.surprise_window.clear()
                return False

        ratio = smoothed_loss / max(self.ema_losses[best_regime_id], self.ema_epsilon)
        
        if ratio > self.surprise_threshold:
            # Instantiate a new world model
            new_model = SimpleWorldModel(self.obs_shape, self.n_actions, self.hidden_dim)
            
            # CRITICAL: Prevent PyTorch Device Trap by placing the new model on the same device
            device = next(self.models[0].parameters()).device
            new_model = new_model.to(device)
            
            # Append it to the mixture of experts
            self.models.append(new_model)
            
            # Update internal tracking variables
            self.active_regime_id = len(self.models) - 1
            self.ema_losses.append(smoothed_loss)  # Set EMA for the new regime baseline
            self.ema_history.append(collections.deque([smoothed_loss], maxlen=10))
            self.force_active_until = global_step + self.newborn_grace_period
            self.surprise_window.clear()
            
            return True
        return False

    def forward(self, state: torch.Tensor, action: torch.Tensor, regime_id: int):
        """
        Routes the forward pass to the specific world model determined by the given regime_id.
        """
        return self.models[regime_id](state, action)
