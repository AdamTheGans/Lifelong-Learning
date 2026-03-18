import torch
import torch.nn.functional as F

class ContextProber:
    """
    Evaluates the World Model's context representations (h_t) to detect 
    representational drift and catastrophic forgetting.
    """
    def __init__(self, world_model, ppo_net, eval_interval=10000, baseline_step=2000000):
        self.world_model = world_model
        self.ppo_net = ppo_net
        self.eval_interval = eval_interval
        self.baseline_step = baseline_step
        
        self.template_sequence = None
        self.baseline_h_t = None
        self.last_eval_step = 0

    def _cache_template(self, memory_buffer):
        """Finds and caches a successful sequence from the memory buffer to use as a template."""
        if len(memory_buffer.buffers['success']) > 0:
            # Take the first successful chunk as our fixed template
            chunk = memory_buffer.buffers['success'][0]
            
            # Move to the same device as the model
            device = next(self.world_model.parameters()).device
            
            # Add batch dimension [1, S, ...]
            self.template_sequence = {
                'state': chunk['state'].unsqueeze(0).to(device),
                'action': chunk['action'].unsqueeze(0).to(device),
                'reward': chunk['reward'].unsqueeze(0).to(device),
                'done': chunk['done'].unsqueeze(0).to(device)
            }

    def evaluate(self, global_step, memory_buffer):
        """Runs the context probing evaluation."""
        # Only evaluate at specified intervals
        if global_step - self.last_eval_step < self.eval_interval:
            return {}
            
        self.last_eval_step = global_step
        
        # Try to cache template if we haven't already
        if self.template_sequence is None:
            self._cache_template(memory_buffer)
            
        # If still no template (buffer empty or no successes), skip evaluation
        if self.template_sequence is None:
            return {}
            
        metrics = {}
        device = next(self.world_model.parameters()).device
        
        # Create Probe 0 (Reward = +5.0) and Probe 1 (Reward = -1.0)
        probe0_rewards = self.template_sequence['reward'].clone()
        probe1_rewards = self.template_sequence['reward'].clone()
        
        # Modify the final reward in the sequence
        probe0_rewards[:, -1] = 5.0
        probe1_rewards[:, -1] = -1.0
        
        # Padding mask (all False since it's a full sequence)
        B, S = probe0_rewards.shape
        padding_mask = torch.zeros((B, S), dtype=torch.bool, device=device)
        
        with torch.no_grad():
            # Step B: Context Extraction & Tracking
            h_t_seq_0 = self.world_model.get_context(
                self.template_sequence['state'],
                self.template_sequence['action'],
                probe0_rewards,
                self.template_sequence['done'],
                padding_mask
            )
            
            h_t_seq_1 = self.world_model.get_context(
                self.template_sequence['state'],
                self.template_sequence['action'],
                probe1_rewards,
                self.template_sequence['done'],
                padding_mask
            )
            
            # Extract final context vectors
            h_t_regime0 = h_t_seq_0[:, -1, :]  # [1, hidden_dim]
            h_t_regime1 = h_t_seq_1[:, -1, :]  # [1, hidden_dim]
            
            # Compute Cosine Similarity between regimes
            regime_sim = F.cosine_similarity(h_t_regime0, h_t_regime1, dim=-1).item()
            metrics['probe/regime_cosine_sim'] = regime_sim
            
            # Track Baseline Cosine Similarity
            if global_step >= self.baseline_step:
                if self.baseline_h_t is None:
                    self.baseline_h_t = h_t_regime0.clone()
                
                baseline_sim = F.cosine_similarity(h_t_regime0, self.baseline_h_t, dim=-1).item()
                metrics['probe/baseline_cosine_sim'] = baseline_sim
                
            # Step C: Policy Reaction
            # Extract the final state
            final_state = self.template_sequence['state'][:, -1, ...]  # [1, C, H, W]
            
            # Get policy logits for both contexts
            logits_0, _ = self.ppo_net(final_state, h_t_regime0)
            logits_1, _ = self.ppo_net(final_state, h_t_regime1)
            
            # Compute KL Divergence between action distributions
            # KL(P || Q) = sum P * log(P / Q)
            # F.kl_div expects input in log-space and target in prob-space
            log_probs_0 = F.log_softmax(logits_0, dim=-1)
            probs_1 = F.softmax(logits_1, dim=-1)
            
            # kl_div computes mean over batch by default, which is fine since batch=1
            kl_div = F.kl_div(log_probs_0, probs_1, reduction='batchmean').item()
            metrics['probe/policy_kl_divergence'] = kl_div
            
        return metrics
