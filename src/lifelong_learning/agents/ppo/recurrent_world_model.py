import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerWorldModel(nn.Module):
    """
    In-Context Causal Transformer World Model for predicting environments with regime shifts.
    
    Architecture:
        - CNN Extractor: Processes (C, H, W) One-Hot grids into a 4096-D flat vector, projected to d_model.
        - Action/Reward/Done Embeddings: Extends discrete actions, scalar rewards, and boolean dones into d_model.
        - Transformer Core: Unrolls sequences of (CNN + Action + Reward + Done) to track context over time.
        - Output Heads: Predicts next state logits and next reward scalar from the Transformer states.
    """
    def __init__(self, obs_shape: tuple[int, int, int] = (21, 8, 8), n_actions: int = 3, hidden_dim: int = 256, max_seq_len: int = 30):
        super().__init__()
        self.obs_shape = obs_shape
        self.c, self.h, self.w = obs_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.flat_obs_dim = self.c * self.h * self.w
        
        # Borrowed CNN architecture from SimpleWorldModel
        self.cnn = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Dynamically calculate CNN output size (expected 4096 for 8x8 with 64 channels)
        with torch.no_grad():
            dummy = torch.zeros(1, self.c, self.h, self.w)
            self.cnn_out_dim = self.cnn(dummy).shape[1]
            
        # Project CNN features to a compact latent space
        self.cnn_proj = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Embeddings for previous action, reward, and done
        # We add 1 to n_actions for a dummy/padding action if needed, but we can just use 0 as dummy since it's an embedding
        self.act_emb = nn.Embedding(n_actions + 1, hidden_dim) # +1 for dummy action at t=0
        self.rew_emb = nn.Linear(1, hidden_dim)
        self.don_emb = nn.Embedding(2, hidden_dim) # 0 or 1
        
        # Positional Encoding
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        
        # Output Heads
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_obs_dim)
        )
        
        # Categorical Reward Head
        self.num_reward_bins = 256
        self.reward_min = -2.0
        self.reward_max = 5.0
        self.register_buffer("reward_bins", torch.linspace(self.reward_min, self.reward_max, self.num_reward_bins))
        
        self.next_reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_reward_bins)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward_sequence(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Unrolls the Transformer over an entire sequence of transitions for training.
        
        Args:
            states:  [B, S, C, H, W] tensor of observations
            actions: [B, S] tensor of actions
            rewards: [B, S] tensor of rewards
            dones:   [B, S] tensor of dones
            padding_mask: [B, S] boolean tensor, True where padded (ignored)
            
        Returns:
            next_state_preds:  [B, S, C, H, W] predicted logits
            next_reward_preds: [B, S] predicted rewards
            out:               [B, S, hidden_dim] contextual embeddings
        """
        B, S, C, H, W = states.shape
        
        # Flatten time and batch dim to pass gracefully through CNN
        states_flat = states.contiguous().view(B * S, C, H, W)
        cnn_features = self.cnn(states_flat)              # [B*S, 4096]
        cnn_features = self.cnn_proj(cnn_features)        # [B*S, 256]
        cnn_features = cnn_features.view(B, S, self.hidden_dim) # [B, S, 256]
        
        # CRITICAL FIX: The model must not see a_t, r_t, d_t when predicting step t.
        # We shift them to be a_{t-1}, r_{t-1}, d_{t-1}, padding the first step with dummies.
        # We use n_actions as the dummy action index.
        dummy_actions = torch.full((B, 1), self.n_actions, dtype=torch.long, device=actions.device)
        prev_actions = torch.cat([dummy_actions, actions[:, :-1]], dim=1)
        
        prev_rewards = torch.cat([torch.zeros(B, 1, device=rewards.device), rewards[:, :-1]], dim=1)
        prev_dones = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dones.device), dones[:, :-1].long()], dim=1)
        
        act_feat = self.act_emb(prev_actions)
        rew_feat = self.rew_emb(prev_rewards.unsqueeze(-1))
        don_feat = self.don_emb(prev_dones)
        
        # Combine features
        tokens = cnn_features + act_feat + rew_feat + don_feat
        
        # Positional Encoding
        positions = torch.arange(S, device=states.device).unsqueeze(0).expand(B, S)
        pos_feat = self.pos_emb(positions)
        
        tokens = tokens + pos_feat
        
        # Causal Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=states.device, dtype=torch.bool)
        
        # Transformer
        out = self.transformer(
            src=tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True
        ) # [B, S, 256]
        
        # Flatten again for independent step-wise predictions
        out_flat = out.contiguous().view(B * S, -1)
        
        # Heads
        next_states_flat = self.next_state_head(out_flat) # [B*S, C*H*W]
        next_state_preds = next_states_flat.view(B, S, C, H, W)
        
        next_rewards_flat = self.next_reward_head(out_flat) # [B*S, num_bins]
        next_reward_preds = next_rewards_flat.view(B, S, self.num_reward_bins) # Return logits
        
        return next_state_preds, next_reward_preds, out

    def step(self, states_window: torch.Tensor, actions_window: torch.Tensor, rewards_window: torch.Tensor, dones_window: torch.Tensor, padding_mask: torch.Tensor):
        """
        Live PPO Inference stepping using a rolling window.
        
        Args:
            states_window:  [B, S, C, H, W]
            actions_window: [B, S] (These are PREVIOUS actions, already shifted by the rolling window)
            rewards_window: [B, S] (PREVIOUS rewards)
            dones_window:   [B, S] (PREVIOUS dones)
            padding_mask:   [B, S] boolean tensor, True where padded (ignored)
            
        Returns:
            next_state_pred:  [B, C, H, W]
            next_reward_pred: [B]
            context_seq:      [B, S, 256] full sequence of contexts
        """
        B, S, C, H, W = states_window.shape
        
        # Embeddings
        states_flat = states_window.contiguous().view(B * S, C, H, W)
        cnn_feat = self.cnn(states_flat)
        cnn_feat = self.cnn_proj(cnn_feat).view(B, S, self.hidden_dim)
        
        act_feat = self.act_emb(actions_window)
        rew_feat = self.rew_emb(rewards_window.unsqueeze(-1))
        don_feat = self.don_emb(dones_window.long())
        
        # Combine
        tokens = cnn_feat + act_feat + rew_feat + don_feat
        
        # Positional Encoding
        positions = torch.arange(S, device=states_window.device).unsqueeze(0).expand(B, S)
        pos_feat = self.pos_emb(positions)
        
        tokens = tokens + pos_feat
        
        # Causal Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=states_window.device, dtype=torch.bool)
        
        out = self.transformer(
            src=tokens,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True
        ) # [B, S, hidden_dim]
        
        # We only need the predictions for the last step
        last_out = out[:, -1, :] # [B, hidden_dim]
        
        next_state_pred = self.next_state_head(last_out).view(-1, self.c, self.h, self.w)
        
        next_reward_logits = self.next_reward_head(last_out)
        next_reward_probs = F.softmax(next_reward_logits, dim=-1)
        next_reward_pred = (next_reward_probs * self.reward_bins).sum(dim=-1)
        
        return next_state_pred, next_reward_pred, out

    def autoregressive_dream_step(self, states_window, actions_window, rewards_window, dones_window, padding_mask):
        """
        Scaffold for dreaming. In the future, this will take the window, predict the next state/reward,
        and the caller will append the prediction to the window for the next step.
        """
        return self.step(states_window, actions_window, rewards_window, dones_window, padding_mask)

    def generate_dream_trajectories(self, policy_net, start_states: torch.Tensor, start_h_t: torch.Tensor, horizon: int) -> list[dict]:
        """
        Temporarily disabled or requires refactoring to use rolling windows.
        """
        raise NotImplementedError("Dreaming with Transformer requires rolling window refactoring.")

    def compute_loss(self, states, actions, rewards, next_states, target_rewards, dones, padding_mask=None):
        """Standard loss wrapper"""
        loss, _, _ = self.compute_loss_detailed(states, actions, rewards, next_states, target_rewards, dones, padding_mask)
        return loss

    def compute_loss_detailed(self, states, actions, rewards, next_states, target_rewards, dones, padding_mask=None):
        """
        Calculates loss handling randomized episode bounds masking, returning detailed components.
        """
        # 1. Forward pass
        next_state_preds, next_reward_preds, _ = self.forward_sequence(states, actions, rewards, dones, padding_mask)
        
        # --- DIAGNOSTIC 2: SEQUENCE ALIGNMENT ---
        if np.random.rand() < 0.005:
            print(f"\n[DIAGNOSTIC 2] Alignment Check:")
            print(f"  states shape: {states.shape}, actions shape: {actions.shape}")
            print(f"  next_reward_preds shape: {next_reward_preds.shape}, target_rewards shape: {target_rewards.shape}")
            print(f"  Temporal flow: next_reward_preds[:, t] is predicted from state[:, t], action[:, t], and reward[:, t-1].")
            print(f"  It is evaluated against target_rewards[:, t].")

        # Convert reward logits to expected scalar values for diagnostic
        next_reward_probs = F.softmax(next_reward_preds, dim=-1)
        next_reward_scalar_preds = (next_reward_probs * self.reward_bins).sum(dim=-1)

        # --- DIAGNOSTIC 3: OUTPUT VS TARGET VALUES ---
        # Find terminal steps
        success_mask = target_rewards > 4.0
        failure_mask = target_rewards < -0.5
        
        if success_mask.any() or failure_mask.any():
            if np.random.rand() < 0.05: # Print occasionally when terminal is found
                print(f"\n[DIAGNOSTIC 3] Terminal State Found!")
                if success_mask.any():
                    print(f"  Success Preds:    {next_reward_scalar_preds[success_mask][:5].detach().cpu().numpy()}")
                    print(f"  Success Targets:  {target_rewards[success_mask][:5].detach().cpu().numpy()}")
                if failure_mask.any():
                    print(f"  Failure Preds:    {next_reward_scalar_preds[failure_mask][:5].detach().cpu().numpy()}")
                    print(f"  Failure Targets:  {target_rewards[failure_mask][:5].detach().cpu().numpy()}")
                
                # Also print some non-terminal for comparison (around -0.01)
                non_term_mask = (target_rewards > -0.5) & (target_rewards < 1.0)
                if non_term_mask.any():
                    print(f"  Non-Term Preds:   {next_reward_scalar_preds[non_term_mask][:5].detach().cpu().numpy()}")
                    print(f"  Non-Term Targets: {target_rewards[non_term_mask][:5].detach().cpu().numpy()}")
        
        B, S, C, H, W = states.shape
        
        # 2. State loss (Cross-Entropy). Converting one-hot (C channel) to hard class indices.
        target_state_idx = next_states.argmax(dim=2)              # [B, S, H, W]
        
        preds_flat = next_state_preds.view(B * S, C, H, W)
        targets_flat = target_state_idx.view(B * S, H, W)
        
        # Calculate cross entropy per pixel and mean them across spatial dims (yielding [B * S])
        ce_loss = F.cross_entropy(preds_flat, targets_flat, reduction='none') 
        state_loss_per_step = ce_loss.mean(dim=(1, 2)).view(B, S) # [B, S]
        
        # 3. Reward loss (Categorical Cross-Entropy)
        # Convert continuous target_rewards to nearest bin indices
        clamped_targets = torch.clamp(target_rewards, self.reward_min, self.reward_max)
        bin_width = (self.reward_max - self.reward_min) / (self.num_reward_bins - 1)
        target_indices = torch.round((clamped_targets - self.reward_min) / bin_width).long() # [B, S]
        
        preds_flat_rew = next_reward_preds.view(B * S, self.num_reward_bins)
        targets_flat_rew = target_indices.view(B * S)
        
        ce_loss_rew = F.cross_entropy(preds_flat_rew, targets_flat_rew, reduction='none')
        reward_loss_per_step = ce_loss_rew.view(B, S) # [B, S]
        
        # 4. Boundary Masking
        # Mask out invalid transitions that bridge the end of one episode and the start of a new one.
        # If dones[:, t-1] is true, the sequence crossed an episode boundary, so step t is invalid.
        valid_mask = torch.cat([torch.ones(B, 1, device=dones.device), 1.0 - dones[:, :-1].float()], dim=1) # [B, S]
        
        if padding_mask is not None:
            valid_mask = valid_mask * (~padding_mask).float()
        
        # We also mask the state loss for the step where dones[:, t] is true, because next_state is a random reset.
        state_mask = valid_mask * (1.0 - dones.float())
        
        # Reward loss is masked by valid_mask, but NOT by dones[:, t], because we need to learn terminal rewards!
        reward_mask = valid_mask
        
        # Dynamic Loss Balancing for Rewards
        if padding_mask is not None:
            valid_dones = dones[~padding_mask]
        else:
            valid_dones = dones
            
        num_non_terminal = (valid_dones == 0.0).sum()
        num_terminal = (valid_dones == 1.0).sum().clamp(min=1.0)
        dynamic_weight = num_non_terminal / num_terminal
        
        # We want to value predicting terminal rewards 20x higher than other steps
        reward_weights = torch.where(dones == 1.0, dynamic_weight * 20.0, torch.ones_like(dones))
        
        # Apply masks and weights
        masked_state = (state_loss_per_step * state_mask).sum() / state_mask.sum().clamp(min=1.0)
        
        # For reward, we use a weighted mean over the valid elements
        valid_reward_weights = reward_weights * reward_mask
        masked_reward = (reward_loss_per_step * valid_reward_weights).sum() / valid_reward_weights.sum().clamp(min=1.0)
        
        masked_total = masked_state + masked_reward
        
        return masked_total, masked_state, masked_reward


if __name__ == "__main__":
    print("--- Testing TransformerWorldModel Component ---")
    
    # Instantiate the model
    model = TransformerWorldModel()
    
    # 1. Dummy batch dimensions mimicking SequenceMemoryBuffer sample space
    B, S, C, H, W = 64, 30, 21, 8, 8
    
    # Mimicking real data with appropriate shapes
    states = torch.zeros((B, S, C, H, W))
    states[:, :, 0, :, :] = 1.0  # Synthetic one-hot active channel
    
    actions = torch.randint(0, 3, (B, S))
    rewards = torch.randn((B, S))
    
    # The 'next' targets from environment
    next_states = torch.zeros((B, S, C, H, W))
    next_states[:, :, 1, :, :] = 1.0
    target_rewards = torch.randn((B, S))
    dones = torch.randint(0, 2, (B, S)).float()
    padding_mask = torch.zeros((B, S), dtype=torch.bool)
    
    # 2. Verify sequence shapes 
    print("Evaluating forward_sequence processing...")
    s_preds, r_preds, out = model.forward_sequence(states, actions, rewards, dones, padding_mask)
    assert list(s_preds.shape) == [B, S, C, H, W]
    assert list(r_preds.shape) == [B, S, model.num_reward_bins]
    assert list(out.shape) == [B, S, 256]
    print("  -> forward_sequence shapes match specifications.")

    # 3. Verify singular stepwise inference shapes
    print("Evaluating PPO live step integration processing...")
    # Initialize rolling windows
    states_window = torch.zeros((B, S, C, H, W))
    actions_window = torch.zeros((B, S), dtype=torch.long)
    rewards_window = torch.zeros((B, S))
    dones_window = torch.zeros((B, S), dtype=torch.long)
    
    s_pred_live, r_pred_live, out_live = model.step(states_window, actions_window, rewards_window, dones_window, padding_mask)
    assert list(s_pred_live.shape) == [B, C, H, W]
    assert list(r_pred_live.shape) == [B]
    assert list(out_live.shape) == [B, S, 256]
    print("  -> live step mechanism shapes match specifications.")
    
    # 4. Calculate Loss & Masking Validation
    print("Computing Loss and running backpropagation...")
    loss = model.compute_loss(states, actions, rewards, next_states, target_rewards, dones, padding_mask)
    
    loss.backward()
    
    # 5. Gradient Assertions
    assert model.cnn[0].weight.grad is not None, "CNN extractor layer 0 did not receive gradients."
    assert model.next_state_head[0].weight.grad is not None, "MLP Output head did not receive gradients."
    
    print("\nVerification Successful: Gradients successfully backpropagated through sequential Transformer back into the CNN Extractor!")
