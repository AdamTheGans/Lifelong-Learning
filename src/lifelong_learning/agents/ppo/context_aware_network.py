import torch
import torch.nn as nn
import numpy as np

class ContextAwarePPONetwork(nn.Module):
    """
    Context-Aware PPO Actor-Critic Network.
    
    This architecture integrates a standalone world model's recurrent state (Context Badge) 
    into the PPO decision-making process. The system receives a standard visual state 
    along with a 256-D hidden state vector from the World Model.
    
    Architecture:
        - CNN Extractor: A standalone 3-layer CNN processes the (21, 8, 8) input into a flat vector.
        - The Detach Mechanism: Context vector is explicitly severed from the computation graph.
        - Concatenation: Joins the CNN extraction and the detached World Model Context.
        - Actor Head: 2-layer MLP generating Policy Logits.
        - Critic Head: 2-layer MLP generating value scalar.
    """
    def __init__(self, obs_shape: tuple[int, int, int] = (21, 8, 8), context_dim: int = 256, n_actions: int = 3):
        super().__init__()
        self.obs_shape = obs_shape
        self.c, self.h, self.w = obs_shape
        self.context_dim = context_dim
        self.n_actions = n_actions

        # Standard CNN feature extractor processing exactly (C, H, W) -> Flat vector
        self.encoder = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate exact linear dimension post-flattening
        with torch.no_grad():
            dummy = torch.zeros(1, self.c, self.h, self.w)
            cnn_flat_size = self.encoder(dummy).shape[1]
            
        # The fused representation dimensions
        fused_dim = cnn_flat_size + self.context_dim

        # Actor head (Policy)
        self.actor_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

        # Critic head (Value function)
        self.critic_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Standard Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Orthogonal init with role-specific gains for output layers."""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Actor output: initialize policy as near-uniform
        if isinstance(self.actor_head[-1], nn.Linear):
            nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
            if self.actor_head[-1].bias is not None:
                nn.init.zeros_(self.actor_head[-1].bias)

        # Critic output
        if isinstance(self.critic_head[-1], nn.Linear):
            nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
            if self.critic_head[-1].bias is not None:
                nn.init.zeros_(self.critic_head[-1].bias)

    def forward(self, state: torch.Tensor, context_ht: torch.Tensor):
        """
        Forward pass producing Actor logits and Critic value.
        
        Args:
            state:      (B, 21, 8, 8) Tensor of exact visual grids
            context_ht: (B, 256) Internal GRU historical summary from WM
        """
        # --- THE BULLETPROOF VEST (STOP-GRADIENT) ---
        # Critical rule: PPO must NEVER backpropagate into the world model's GRU. 
        # The world model solely exists for system identification, not reward hacking.
        context_ht = context_ht.detach()
        # --------------------------------------------
        
        # 1. Process standard visual grid state
        cnn_features = self.encoder(state)
        
        # 2. Inject Context Badge via concatenation
        fused_features = torch.cat([cnn_features, context_ht], dim=-1)
        
        # 3. Decision Making
        logits = self.actor_head(fused_features)
        value = self.critic_head(fused_features).squeeze(-1)
        
        return logits, value

    def get_action_and_value(self, state: torch.Tensor, context_ht: torch.Tensor, action: torch.Tensor | None = None):
        """
        Higher-level functional interface for PPO rollout loops.
        """
        logits, value = self.forward(state, context_ht)
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
            
        return action, dist.log_prob(action), dist.entropy(), value


if __name__ == "__main__":
    print("--- Testing ContextAwarePPONetwork Component ---")
    
    # 1. Instantiation
    model = ContextAwarePPONetwork()
    
    # 2. Validating the "Bulletproof Vest" Detachment Safety
    print("Verifying the strict stop-gradient (.detach()) mechanism...")
    
    B, C, H, W = 64, 21, 8, 8
    context_dim = 256
    
    dummy_state = torch.randn((B, C, H, W))
    
    # Create the context tensor and formally declare it requires gradients
    # This simulates it coming directly from a live recurrent computational graph
    dummy_context = torch.randn((B, context_dim), requires_grad=True)
    
    # 3. Forward Pass & Loss Hook
    # We call the full PPO interface method
    action, logprob, entropy, value = model.get_action_and_value(dummy_state, dummy_context)
    
    # Dummy Loss function: Summing the critic output simulates a generic PPO Objective function backward signal
    dummy_loss = value.sum()
    dummy_loss.backward()
    
    # 4. Strict Assertion validation
    # If the bulletproof vest failed, dummy_context would receive explicit gradients generated from PPO objective.
    assert dummy_context.grad is None, "FATAL ERROR: The .detach() mechanism failed. Gradients from PPO leaked into the context_ht variable!"
    
    # If we made it here, the system correctly blocked gradients
    assert model.encoder[0].weight.grad is not None, "Network itself did not receive valid PPO gradients"
    
    print("\nVerification Successful: Context inputs strictly block gradient flow into the World Model.")
    print(f"   -> Forward pass returned {action.shape} shape actions and {value.shape} value.")
