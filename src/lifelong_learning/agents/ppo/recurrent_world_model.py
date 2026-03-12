import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RecurrentWorldModel(nn.Module):
    """
    Recurrent World Model for predicting environments with regime shifts.
    
    Architecture:
        - CNN Extractor: Processes (C, H, W) One-Hot grids into a 4096-D flat vector.
        - Action Embedding: Extends discrete actions into a 32-D space.
        - GRU Core: Unrolls sequences of (CNN + Action + Reward) to track context over time.
        - Output Heads: Predicts next state logits and next reward scalar from the GRU states.
    """
    def __init__(self, obs_shape: tuple[int, int, int] = (21, 8, 8), n_actions: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.obs_shape = obs_shape
        self.c, self.h, self.w = obs_shape
        self.n_actions = n_actions
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
            
        # Action embedding
        self.action_emb = nn.Embedding(n_actions, 32)
        
        # Input size to GRU: CNN Output (4096) + Action (32) + Reward (1) = 4129
        self.gru_input_dim = self.cnn_out_dim + 32 + 1
        
        # GRU Core
        self.gru = nn.GRU(input_size=self.gru_input_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Output Heads (from 256-D GRU output)
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_obs_dim)
        )
        
        self.next_reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward_sequence(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        """
        Unrolls the GRU over an entire sequence of transitions for training.
        
        Args:
            states:  [B, S, C, H, W] tensor of observations
            actions: [B, S] tensor of actions
            rewards: [B, S] tensor of rewards
            
        Returns:
            next_state_preds:  [B, S, C, H, W] predicted logits
            next_reward_preds: [B, S] predicted rewards
        """
        B, S, C, H, W = states.shape
        
        # Flatten time and batch dim to pass gracefully through CNN
        states_flat = states.contiguous().view(B * S, C, H, W)
        cnn_features = self.cnn(states_flat)              # [B*S, 4096]
        cnn_features = cnn_features.view(B, S, -1)        # [B, S, 4096]
        
        act_emb = self.action_emb(actions)                # [B, S, 32]
        
        # CRITICAL FIX: The model must not see r_t when predicting r_t.
        # We shift the rewards to be r_{t-1}, padding the first step with 0.
        prev_rewards = torch.cat([torch.zeros(B, 1, device=rewards.device), rewards[:, :-1]], dim=1)
        rew_emb = prev_rewards.unsqueeze(-1)              # [B, S, 1]
        
        # Concatenate features 
        rnn_input = torch.cat([cnn_features, act_emb, rew_emb], dim=-1) # [B, S, 4129]
        
        # Unroll GRU
        gru_out, _ = self.gru(rnn_input)                  # [B, S, 256]
        
        # Flatten again for independent step-wise predictions
        gru_out_flat = gru_out.contiguous().view(B * S, -1)
        
        # Heads
        next_states_flat = self.next_state_head(gru_out_flat) # [B*S, C*H*W]
        next_state_preds = next_states_flat.view(B, S, C, H, W)
        
        next_rewards_flat = self.next_reward_head(gru_out_flat) # [B*S, 1]
        next_reward_preds = next_rewards_flat.view(B, S)
        
        return next_state_preds, next_reward_preds

    def step(self, state: torch.Tensor, action: torch.Tensor, prev_reward: torch.Tensor, hidden_state: torch.Tensor):
        """
        Live PPO Inference stepping.
        
        Args:
            state:        [B, C, H, W]
            action:       [B]
            prev_reward:  [B] (The reward from the previous step)
            hidden_state: [1, B, 256] GRU recurrent state
            
        Returns:
            next_state_pred:  [B, C, H, W]
            next_reward_pred: [B]
            new_hidden_state: [1, B, 256] GRU recurrent state
        """
        # Process inputs
        cnn_feat = self.cnn(state)                           # [B, 4096]
        act_emb = self.action_emb(action)                    # [B, 32]
        rew_emb = prev_reward.unsqueeze(-1)                  # [B, 1]
        
        # Add sequence dimension of size 1
        rnn_input = torch.cat([cnn_feat, act_emb, rew_emb], dim=-1).unsqueeze(1) # [B, 1, 4129]
        
        gru_out, new_hidden_state = self.gru(rnn_input, hidden_state) # gru_out: [B, 1, 256]
        
        # Remove sequence dimension for heads
        gru_out_squeeze = gru_out.squeeze(1)                 # [B, 256]
        
        # Forward through heads
        next_state_pred = self.next_state_head(gru_out_squeeze).view(-1, self.c, self.h, self.w)
        next_reward_pred = self.next_reward_head(gru_out_squeeze).squeeze(-1)
        
        return next_state_pred, next_reward_pred, new_hidden_state

    def generate_dream_trajectories(self, policy_net, start_states: torch.Tensor, start_h_t: torch.Tensor, horizon: int) -> list[dict]:
        """
        Generates simulated trajectories using the World Model for PPO Generative Replay.
        
        Args:
            policy_net: Main ContextAwarePPONetwork
            start_states: [B, C, H, W] initial discrete grid states (seeded from buffer)
            start_h_t: [1, B, 256] initial GRU hidden contexts (seeded from buffer)
            horizon: Number of steps to roll forward
            
        Returns:
            List of dictionaries containing transition data for the imagined trajectory.
        """
        trajectories = []
        curr_state = start_states
        curr_h_t = start_h_t
        
        # In a dream, we start with a dummy previous reward of 0
        curr_prev_reward = torch.zeros(start_states.shape[0], dtype=torch.float32, device=start_states.device)
        
        for _ in range(horizon):
            with torch.no_grad():
                # 1. PPO decides action based on current state & WM context badge
                action, logprob, entropy, value = policy_net.get_action_and_value(curr_state, curr_h_t.squeeze(0))
                
                # 2. WM predicts next frame and next reward
                next_state_logits, next_reward_pred, next_h_t = self.step(curr_state, action, curr_prev_reward, curr_h_t)
                
                # 3. Strict Discretization (Guardrail against compounding blurriness)
                # Argument max over channel dim C, then re-encode into One-Hot float tensor
                max_indices = torch.argmax(next_state_logits, dim=1)           # [B, H, W]
                one_hot = torch.nn.functional.one_hot(max_indices, num_classes=self.c)  # [B, H, W, C]
                next_state_discrete = one_hot.permute(0, 3, 1, 2).float()      # [B, C, H, W]
                
                # Assume dreams don't terminate early to keep batch dimensions clean
                dones = torch.zeros_like(action, dtype=torch.float32)
                
                # 5. Store imagined transition
                trajectories.append({
                    "state": curr_state,
                    "action": action,
                    "logprob": logprob,
                    "reward": next_reward_pred,
                    "done": dones,
                    "value": value,
                    "next_state": next_state_discrete,
                    "h_t": curr_h_t.squeeze(0) # Store context used to make decision
                })
                
                curr_state = next_state_discrete
                curr_h_t = next_h_t
                curr_prev_reward = next_reward_pred # The predicted reward becomes the prev_reward for the next step
                
        return trajectories

    def compute_loss(self, states, actions, rewards, next_states, next_rewards, dones):
        """Standard loss wrapper"""
        loss, _, _ = self.compute_loss_detailed(states, actions, rewards, next_states, next_rewards, dones)
        return loss

    def compute_loss_detailed(self, states, actions, rewards, next_states, next_rewards, dones):
        """
        Calculates loss handling randomized episode bounds masking, returning detailed components.
        
        Args:
            states:       [B, S, C, H, W]
            actions:      [B, S]
            rewards:      [B, S]
            next_states:  [B, S, C, H, W] real next states 
            next_rewards: [B, S] real next rewards
            dones:        [B, S] 0 or 1 done flags
            
        Returns:
            Mean total loss, Mean state loss, Mean reward loss
        """
        # 1. Forward pass
        next_state_preds, next_reward_preds = self.forward_sequence(states, actions, rewards)
        
        B, S, C, H, W = states.shape
        
        # 2. State loss (Cross-Entropy). Converting one-hot (C channel) to hard class indices.
        target_state_idx = next_states.argmax(dim=2)              # [B, S, H, W]
        
        preds_flat = next_state_preds.view(B * S, C, H, W)
        targets_flat = target_state_idx.view(B * S, H, W)
        
        # Calculate cross entropy per pixel and mean them across spatial dims (yielding [B * S])
        ce_loss = F.cross_entropy(preds_flat, targets_flat, reduction='none') 
        state_loss_per_step = ce_loss.mean(dim=(1, 2)).view(B, S) # [B, S]
        
        # 3. Reward loss (Mean Squared Error)
        reward_loss_per_step = F.mse_loss(next_reward_preds, next_rewards, reduction='none') # [B, S]
        
        # total per-step loss
        total_loss_per_step = state_loss_per_step + reward_loss_per_step
        
        # 4. Boundary Masking
        # Mask out steps strictly landing on reset frames (1.0 = valid, 0.0 = done)
        mask = 1.0 - dones.float()
        
        masked_total = (total_loss_per_step * mask).mean()
        masked_state = (state_loss_per_step * mask).mean()
        masked_reward = (reward_loss_per_step * mask).mean()
        
        return masked_total, masked_state, masked_reward


if __name__ == "__main__":
    print("--- Testing RecurrentWorldModel Component ---")
    
    # Instantiate the model
    model = RecurrentWorldModel()
    
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
    next_rewards = torch.randn((B, S))
    dones = torch.randint(0, 2, (B, S)).float()
    
    # 2. Verify sequence shapes 
    print("Evaluating forward_sequence processing...")
    s_preds, r_preds = model.forward_sequence(states, actions, rewards)
    assert list(s_preds.shape) == [B, S, C, H, W]
    assert list(r_preds.shape) == [B, S]
    print("  -> forward_sequence shapes match specifications.")

    # 3. Verify singular stepwise inference shapes
    print("Evaluating PPO live step integration processing...")
    # Initialize zero hidden batch
    h0 = torch.zeros(1, B, 256)
    prev_r = torch.zeros(B)
    
    s_pred_live, r_pred_live, h1 = model.step(states[:, 0], actions[:, 0], prev_r, h0)
    assert list(s_pred_live.shape) == [B, C, H, W]
    assert list(r_pred_live.shape) == [B]
    assert list(h1.shape) == [1, B, 256]
    print("  -> live step mechanism shapes match specifications.")
    
    # 4. Calculate Loss & Masking Validation
    print("Computing Loss and running backpropagation...")
    loss = model.compute_loss(states, actions, rewards, next_states, next_rewards, dones)
    
    loss.backward()
    
    # 5. Gradient Assertions
    assert model.gru.weight_ih_l0.grad is not None, "GRU input-hidden weights did not receive gradients."
    assert model.gru.weight_hh_l0.grad is not None, "GRU hidden-hidden weights did not receive gradients."
    assert model.cnn[0].weight.grad is not None, "CNN extractor layer 0 did not receive gradients."
    assert model.next_state_head[0].weight.grad is not None, "MLP Output head did not receive gradients."
    
    print("\n✅ Verification Successful: Gradients successfully backpropagated through sequential GRU back into the CNN Extractor!")
