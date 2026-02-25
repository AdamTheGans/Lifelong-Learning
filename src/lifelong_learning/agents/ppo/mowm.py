import copy

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

    def __init__(self, obs_shape: tuple[int, int, int], n_actions: int, hidden_dim: int = 256, max_regimes: int = 2):
        super().__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.max_regimes = max_regimes

        # Initialize with a single world model
        initial_model = SimpleWorldModel(obs_shape, n_actions, hidden_dim)
        self.models = nn.ModuleList([initial_model])
        
        # Status variables
        self.active_regime_id = 0
        self.ema_losses = [1.0]
        self.has_mastered = [False]
        self.steps_under_threshold = [0]
        self.spawn_steps = [0]
        self.timeout_triggered = [False]

        # Hyperparameters
        self.ema_alpha = 0.05
        self.anomaly_floor = 0.05  # Absolute minimum for the EMA used in dynamic threshold calculation
        self.anomaly_multiplier = 3.5  # Multiplier for the dynamic threshold
        
        # MoWM Routing
        self.absolute_spawn_threshold = 0.3  # Absolute upper ceiling for rescue model viability
        self.global_grace_period = 20000     # No spawns before this step
        self.newborn_grace_period = 10000    # Force active regime after spawn
        self.mastery_loss_threshold = 0.20   # Mastery Prerequisite: Loss threshold to accrue mastery steps
        self.mastery_buffer_steps = 5000     # Mastery Prerequisite: Continuous steps required below threshold
        self.max_lockin_steps = 250000       # Maximum steps to keep the mastery shield up
        
        # Safe State Rollback tracking (per-model)
        self.safe_state_dicts = [None]       # CPU-offloaded weight snapshots
        self.safe_optimizer_states = [None]   # Optimizer momentum snapshots
        self.safe_ema_losses = [None]         # EMA loss at time of snapshot
        self.safe_state_steps = [None]        # Global step when snapshot was taken

        # State tracking
        self.force_active_until = 0

    def refresh_safe_state(self, regime_id: int, model: nn.Module, optimizer, global_step: int = 0):
        """
        Snapshot the model's weights if it is currently stable.

        Stability requires:
            1. EMA loss is below the mastery threshold (model is well-trained)
            2. We are past the newborn grace period (not a brand-new model)

        Snapshots are stored on CPU to avoid doubling GPU VRAM usage.
        """
        is_stable = (
            self.ema_losses[regime_id] < self.mastery_loss_threshold
            and global_step >= self.force_active_until
        )
        if not is_stable:
            return

        # Deep-copy and offload to CPU to prevent VRAM doubling
        self.safe_state_dicts[regime_id] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.safe_optimizer_states[regime_id] = copy.deepcopy(optimizer.state_dict())
        self.safe_ema_losses[regime_id] = self.ema_losses[regime_id]
        self.safe_state_steps[regime_id] = global_step

    def rollback_safe_state(self, regime_id: int, model: nn.Module, optimizer, global_step: int = -1) -> bool:
        """
        Restore a world model's weights to its last safe checkpoint.

        Called at the exact moment the Router transitions away from this model,
        undoing any weight corruption accumulated during the detection lag period.

        Returns True if rollback occurred, False if no safe state was available.
        """
        if self.safe_state_dicts[regime_id] is None:
            print(f"[MoWM] Rollback skipped for Model {regime_id}: no safe state available (newborn).")
            return False

        model.load_state_dict(self.safe_state_dicts[regime_id])
        optimizer.load_state_dict(self.safe_optimizer_states[regime_id])
        self.ema_losses[regime_id] = self.safe_ema_losses[regime_id]
        saved_step = self.safe_state_steps[regime_id]
        steps_ago = f" from step {saved_step} ({global_step - saved_step} steps ago)" if saved_step is not None and global_step >= 0 else ""
        print(f"[MoWM] Rolled back Model {regime_id} to safe state{steps_ago} (EMA: {self.safe_ema_losses[regime_id]:.4f}).")
        return True

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

    def evaluate_all_models(
        self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor
    ) -> list[float]:
        """
        Evaluate all world models on a batch and return per-model raw losses.
        Used at epoch boundaries to determine which model best fits the current data.
        """
        next_state_indices = torch.argmax(next_state, dim=1)

        raw_losses = []
        for i, model in enumerate(self.models):
            with torch.no_grad():
                next_obs_pred, pred_reward = model(state, action)
                state_loss = F.cross_entropy(next_obs_pred, next_state_indices, reduction='none')
                state_loss_per_batch = state_loss.mean(dim=[1, 2])
                reward_loss = F.mse_loss(pred_reward, reward, reduction='none')
                loss = (state_loss_per_batch + reward_loss).mean().item()
                raw_losses.append(loss)

        return raw_losses

    def check_epoch_transition(
        self, epoch_avg_loss: float, eval_losses: list[float], global_step: int
    ) -> tuple[str, int]:
        """
        Epoch-boundary routing decision. Replaces the old per-step Rescue Routing.

        Compares the active model's epoch average loss against the EMA-based dynamic
        threshold. If surprised, finds the best alternative model or signals a spawn.

        Returns:
            (action, target_id) where action is "stay", "switch", or "spawn".
            target_id is the model to switch to (for "switch"), -1 otherwise.
        """
        # Grace period guards
        if global_step < self.global_grace_period or global_step < self.force_active_until:
            return ("stay", self.active_regime_id)

        # Mastery lock-in: don't route away from a non-mastered model (unless timeout)
        elapsed_steps = global_step - self.spawn_steps[self.active_regime_id]

        if not hasattr(self, 'timeout_triggered'):
            self.timeout_triggered = [False] * len(self.models)

        if not self.has_mastered[self.active_regime_id] and elapsed_steps >= self.max_lockin_steps:
            if not self.timeout_triggered[self.active_regime_id]:
                self.timeout_triggered[self.active_regime_id] = True
                print(f"\n[MoWM] World Model {self.active_regime_id} Mastery Shield TIMEOUT at {global_step} steps! Shield forced down.")

        is_locked_in = (
            not self.has_mastered[self.active_regime_id]
            and elapsed_steps < self.max_lockin_steps
        )
        if is_locked_in:
            return ("stay", self.active_regime_id)

        # Is the active model surprised?
        dynamic_threshold = max(self.ema_losses[self.active_regime_id], self.anomaly_floor) * self.anomaly_multiplier
        if epoch_avg_loss <= dynamic_threshold:
            return ("stay", self.active_regime_id)

        print(f"\n[MoWM] Surprise detected! Model {self.active_regime_id} epoch loss ({epoch_avg_loss:.4f}) > threshold ({dynamic_threshold:.4f}).")

        # Active model is surprised. Find the best alternative.
        best_candidate = min(range(len(self.models)), key=lambda i: eval_losses[i])

        if best_candidate != self.active_regime_id and eval_losses[best_candidate] < self.absolute_spawn_threshold:
            # A veteran can handle the current regime
            return ("switch", best_candidate)

        # No good veteran — try to spawn
        if len(self.models) >= self.max_regimes:
            print(f"[MoWM] Hard Cap Reached! Cannot spawn Model {len(self.models)}.")
            return ("stay", self.active_regime_id)

        return ("spawn", -1)

    def spawn_new_model(self, global_step: int, seed_ema: float) -> int:
        """
        Instantiate a new world model and set it as active.

        Returns the new model's regime ID.
        """
        new_model = SimpleWorldModel(self.obs_shape, self.n_actions, self.hidden_dim)
        device = next(self.models[0].parameters()).device
        new_model = new_model.to(device)
        self.models.append(new_model)

        new_id = len(self.models) - 1
        self.active_regime_id = new_id
        self.ema_losses.append(seed_ema)
        self.has_mastered.append(False)
        self.steps_under_threshold.append(0)
        self.spawn_steps.append(global_step)
        if hasattr(self, 'timeout_triggered'):
            self.timeout_triggered.append(False)
        self.safe_state_dicts.append(None)
        self.safe_optimizer_states.append(None)
        self.safe_ema_losses.append(None)
        self.safe_state_steps.append(None)
        self.force_active_until = global_step + self.newborn_grace_period

        print(f"\n[MoWM] Spawned new World Model {new_id} at step {global_step}.")
        return new_id

    def forward(self, state: torch.Tensor, action: torch.Tensor, regime_id: int):
        """
        Routes the forward pass to the specific world model determined by the given regime_id.
        """
        return self.models[regime_id](state, action)
