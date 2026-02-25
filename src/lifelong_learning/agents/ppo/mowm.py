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
        self.has_mastered = [False]
        self.steps_under_threshold = [0]
        self.routing_emas = [1.0]
        self.fast_routing_emas = [1.0]
        self.spawn_steps = [0]
        self.timeout_triggered = [False]

        # Hyperparameters
        self.ema_alpha = 0.05
        self.anomaly_floor = 0.05  # Absolute minimum for the EMA used in dynamic threshold calculation
        self.anomaly_multiplier = 3.5  # Multiplier for the dynamic threshold
        
        # MoWM Routing Fixes
        self.global_grace_period = 20000     # No spawns before this step
        self.newborn_grace_period = 10000    # Force active regime after spawn
        self.surprise_window_size = 100      # Smooth surprise across 100 transitions
        self.routing_hysteresis = 0.85       # Hysteresis stickiness multiplier (challenger < active * 0.85)
        self.ema_epsilon = 1e-5              # Denominator safety avoids div by zero
        self.max_ema_growth = 0.10           # Trend-Aware Spawning: Max relative growth
        self.mastery_loss_threshold = 0.20   # Mastery Prerequisite: Loss threshold to accrue mastery steps
        self.mastery_buffer_steps = 5000     # Mastery Prerequisite: Continuous steps required below threshold
        self.max_lockin_steps = 250000       # Maximum steps to keep the mastery shield up
        
        # State tracking
        self.force_active_until = 0
        self.surprise_window = collections.deque(maxlen=self.surprise_window_size)

    def update_ema(self, current_loss: float, steps_added: int = 0, global_step: int = -1):
        """
        Updates the exponential moving average of the loss.
        Note: This should ONLY be called externally during the training loop of the active world model,
        and not during inference or spawn checks, to keep the baseline stable during surprise spikes.
        """
        idx = self.active_regime_id
        self.ema_losses[idx] = (self.ema_alpha * current_loss) + ((1 - self.ema_alpha) * self.ema_losses[idx])

        if self.ema_losses[idx] < self.mastery_loss_threshold:
            self.steps_under_threshold[idx] += steps_added
            if self.steps_under_threshold[idx] >= self.mastery_buffer_steps:
                if not self.has_mastered[idx]:
                    self.has_mastered[idx] = True
                    step_str = f" at {global_step} steps" if global_step >= 0 else ""
                    print(f"\n[MoWM] World Model {idx} reached mastery{step_str}! Shield dropped.")
        else:
            self.steps_under_threshold[idx] = 0

    def infer_regime(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, global_step: int
    ) -> tuple[int, float, list[float]]:
        """
        Evaluates a transition across all models and returns the ID of the best fitting regime,
        its corresponding lowest loss, and a list of all raw losses for the collective ignorance check.
        Handles newborn stickiness and mastery lock-in to force training.
        """
        # Convert next_state (B, C, H, W) float to class indices (B, H, W) for CE Loss
        # Assuming next_state is one-hot or normalized probabilities
        next_state_indices = torch.argmax(next_state, dim=1)

        raw_losses = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                next_obs_pred, pred_reward = model(state, action)
                
                # Compute unreduced losses to find the extreme anomaly in the batch
                state_loss = F.cross_entropy(next_obs_pred, next_state_indices, reduction='none')
                state_loss_per_batch = state_loss.mean(dim=[1, 2])
                
                reward_loss = F.mse_loss(pred_reward, reward, reduction='none')
                
                # Mean surprise across all environments in this specific transition batch
                loss_batch = state_loss_per_batch + reward_loss
                loss = loss_batch.mean().item() 
                raw_losses.append(loss)
                
            # Update routing EMAs unconditionally for all regimes to maintain a parallel inference track
            self.routing_emas[i] = (self.ema_alpha * loss) + ((1 - self.ema_alpha) * self.routing_emas[i])
            self.fast_routing_emas[i] = (0.2 * loss) + (0.8 * self.fast_routing_emas[i])

        # Newborn stickiness bypass: if in grace period, stick with active model
        # Mastery lock-in bypass: if the active regime has not mastered the environment, it cannot be unseated by veterans.
        elapsed_steps = global_step - self.spawn_steps[self.active_regime_id]
        
        # Timeout warning (ensuring backwards compatibility of attribute)
        if not hasattr(self, 'timeout_triggered'):
            self.timeout_triggered = [False] * len(self.models)
        
        if not self.has_mastered[self.active_regime_id] and elapsed_steps >= self.max_lockin_steps:
            if not self.timeout_triggered[self.active_regime_id]:
                self.timeout_triggered[self.active_regime_id] = True
                print(f"\n[MoWM] World Model {self.active_regime_id} Mastery Shield TIMEOUT at {global_step} steps! Shield forced down.")

        is_locked_in = (global_step < self.force_active_until) or (not self.has_mastered[self.active_regime_id] and elapsed_steps < self.max_lockin_steps)

        if is_locked_in:
            best_regime_id = self.active_regime_id
            best_raw_loss = raw_losses[self.active_regime_id]
        else:
            best_regime_id = self.active_regime_id
            best_raw_loss = raw_losses[self.active_regime_id]
            
            dynamic_threshold = max(self.ema_losses[self.active_regime_id], self.anomaly_floor) * self.anomaly_multiplier
            catastrophic_ceiling = dynamic_threshold
            
            active_fast_loss = self.fast_routing_emas[self.active_regime_id]
            
            # Argmin Routing Reminder: Rapid Catastrophic Takeover
            if active_fast_loss > catastrophic_ceiling:
                best_candidate = min(range(len(self.models)), key=lambda i: self.fast_routing_emas[i])
                if self.fast_routing_emas[best_candidate] <= catastrophic_ceiling:
                    best_regime_id = best_candidate
                    best_raw_loss = raw_losses[best_candidate]
                    print(f"\n[MoWM] Emergency Routing Switch! Active model {self.active_regime_id} failing ({active_fast_loss:.2f} > {catastrophic_ceiling:.2f}). "
                          f"Routing to Model {best_candidate} ({self.fast_routing_emas[best_candidate]:.2f})")
            else:
                # Standard Hysteresis Routing: Smooth Trend Takeover
                lowest_routing_ema = self.routing_emas[self.active_regime_id]
                for i in range(len(self.models)):
                    if i != self.active_regime_id:
                        if self.routing_emas[i] < (self.routing_emas[self.active_regime_id] * self.routing_hysteresis):
                            if self.routing_emas[i] < lowest_routing_ema:
                                lowest_routing_ema = self.routing_emas[i]
                                best_regime_id = i
                                best_raw_loss = raw_losses[i]
                
        return best_regime_id, best_raw_loss, raw_losses

    def check_and_spawn(self, raw_losses: list[float], best_regime_id: int, global_step: int) -> bool:
        """
        Checks if the entire collective of models represents a high surprise.
        Only spawns a new model if EVERY existing model exceeds the dynamic EMA surprise threshold.
        Includes safeguards for global grace period, newborn grace period, and sequence smoothing.
        """
        # Global Step 0 Grace Guard & Newborn Stickiness Guard
        if global_step < self.global_grace_period or global_step < self.force_active_until:
            self.surprise_window.clear()
            return False
            
        # Collective Surprise Smoothing
        self.surprise_window.append(raw_losses)
        
        # Wait until window is full before making spawn decisions
        if len(self.surprise_window) < self.surprise_window_size:
            return False
            
        # Routing Transition Guard
        # If the router officially decides to switch to a veteran model (hysteresis broken),
        # block spawning and clear the window so old regime losses don't pollute the new evaluation.
        if best_regime_id != self.active_regime_id:
            self.surprise_window.clear()
            return False
            
        # Calculate smoothed loss for EVERY model
        smoothed_losses = [sum(losses[i] for losses in self.surprise_window) / len(self.surprise_window) for i in range(len(self.models))]


        # Collective Ignorance Check
        # Are there any models in the collective that are NOT surprised?
        all_surprised = True
        
        dynamic_threshold = max(self.ema_losses[self.active_regime_id], self.anomaly_floor) * self.anomaly_multiplier
        
        for i in range(len(self.models)):
            # Active model: sluggish 100-step SMA prevents false spawns from micro-fluctuations.
            # Inactive models: extremely responsive fast EMA rapidly blocks false spawns when a veteran wakes up.
            metric = smoothed_losses[i] if i == self.active_regime_id else self.fast_routing_emas[i]
            
            if metric <= dynamic_threshold:
                all_surprised = False
                break
        
        if all_surprised:
            # Instantiate a new world model
            new_model = SimpleWorldModel(self.obs_shape, self.n_actions, self.hidden_dim)
            
            # CRITICAL: Prevent PyTorch Device Trap by placing the new model on the same device
            device = next(self.models[0].parameters()).device
            new_model = new_model.to(device)
            
            # Append it to the mixture of experts
            self.models.append(new_model)
            
            # Use the active model's smoothed loss as the seed for the new baseline
            new_baseline = smoothed_losses[self.active_regime_id]

            # Update internal tracking variables
            self.active_regime_id = len(self.models) - 1
            self.ema_losses.append(new_baseline)  # Set EMA for the new regime baseline
            self.routing_emas.append(new_baseline) # Seed routing EMA
            self.fast_routing_emas.append(new_baseline) # Seed fast routing EMA
            self.has_mastered.append(False)
            self.steps_under_threshold.append(0)
            self.spawn_steps.append(global_step)
            if hasattr(self, 'timeout_triggered'):
                self.timeout_triggered.append(False)
            self.force_active_until = global_step + self.newborn_grace_period
            self.surprise_window.clear()
            
            return True
        return False

    def forward(self, state: torch.Tensor, action: torch.Tensor, regime_id: int):
        """
        Routes the forward pass to the specific world model determined by the given regime_id.
        """
        return self.models[regime_id](state, action)
