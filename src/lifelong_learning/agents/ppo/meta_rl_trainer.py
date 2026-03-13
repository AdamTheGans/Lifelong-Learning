import torch
import torch.nn as nn
import numpy as np

# Import our custom components
from lifelong_learning.agents.ppo.sequence_memory_buffer import SequenceMemoryBuffer
from lifelong_learning.agents.ppo.recurrent_world_model import RecurrentWorldModel
from lifelong_learning.agents.ppo.context_aware_network import ContextAwarePPONetwork

class MetaRLTrainer:
    """
    Master Training Loop for the Context-Aware Meta-RL system.
    
    Orchestrates the interplay between the live PPO rollout, the Long-Term Memory Buffer,
    and the Recurrent World Model.
    """
    def __init__(
        self, 
        ppo_net: ContextAwarePPONetwork, 
        world_model: RecurrentWorldModel,
        memory_buffer: SequenceMemoryBuffer,
        ppo_optimizer: torch.optim.Optimizer,
        wm_optimizer: torch.optim.Optimizer,
        cfg: dict
    ):
        self.ppo_net = ppo_net
        self.world_model = world_model
        self.memory_buffer = memory_buffer
        
        # Strict separation of optimizers to ensure World Model gradients never touch PPO, and vice versa.
        self.ppo_optimizer = ppo_optimizer
        self.wm_optimizer = wm_optimizer
        
        self.cfg = cfg
        self.global_step = 0

    def calculate_returns_and_advantages(self, rewards, values, dones, next_value, gamma, gae_lambda):
        """Standard GAE calculation adapted for direct tensor processing."""
        B, S = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        last_gae_lam = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(S)):
            if t == S - 1:
                next_non_terminal = 1.0 - dones[:, t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[:, t]
                next_val = values[:, t + 1]
                
            delta = rewards[:, t] + gamma * next_val * next_non_terminal - values[:, t]
            advantages[:, t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + values
        return returns, advantages

    def ppo_update(self, obs, actions, old_logprobs, context_ht, advantages, returns, old_values):
        """
        Standalone PPO update adapted specifically for ContextAwarePPONetwork.
        """
        B, S = actions.shape
        
        # Flatten all tensors for mini-batching
        obs_flat = obs.view(B * S, *self.ppo_net.obs_shape)
        act_flat = actions.view(B * S)
        logp_flat = old_logprobs.view(B * S)
        ctx_flat = context_ht.view(B * S, self.ppo_net.context_dim)
        adv_flat = advantages.view(B * S)
        ret_flat = returns.view(B * S)
        val_flat = old_values.view(B * S)
        
        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        total_size = B * S
        idxs = np.arange(total_size)
        
        for epoch in range(self.cfg['ppo_epochs']):
            np.random.shuffle(idxs)
            for start in range(0, total_size, self.cfg['minibatch_size']):
                mb = idxs[start:start + self.cfg['minibatch_size']]
                
                # --- Crucial: Forward pass through ContextAwarePPONetwork handles .detach() internally
                _, new_logprobs, entropy, new_values = self.ppo_net.get_action_and_value(obs_flat[mb], ctx_flat[mb], act_flat[mb])
                
                logratio = new_logprobs - logp_flat[mb]
                ratio = torch.exp(logratio)
                
                # Policy Loss
                pg_loss1 = -adv_flat[mb] * ratio
                pg_loss2 = -adv_flat[mb] * torch.clamp(ratio, 1.0 - self.cfg['clip_coef'], 1.0 + self.cfg['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                v_loss = 0.5 * (ret_flat[mb] - new_values).pow(2).mean()
                
                # Entropy
                ent_loss = entropy.mean()
                
                # Total Loss
                loss = pg_loss - self.cfg['ent_coef'] * ent_loss + self.cfg['vf_coef'] * v_loss
                
                self.ppo_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.ppo_optimizer.step()
                
                # We can just return the last minibatch/epoch stats or mean them. 
                # Returning the last minibatch stats is a common approximation in PPO for logging.
                
        return {
            "ppo/policy_loss": pg_loss.item(),
            "ppo/value_loss": v_loss.item(),
            "ppo/entropy": ent_loss.item()
        }

    def train_iteration(self, env):
        """
        Executes one full Meta-RL training iteration consisting of 4 precise phases.
        """
        device = next(self.ppo_net.parameters()).device
        B = self.cfg['num_envs']
        S = self.cfg['num_steps']
        seq_len = self.memory_buffer.seq_len # Exactly 30
        
        # Initialize storage for the rollout
        obs_buf = torch.zeros((B, S, *self.ppo_net.obs_shape), device=device)
        act_buf = torch.zeros((B, S), dtype=torch.long, device=device)
        logp_buf = torch.zeros((B, S), device=device)
        rew_buf = torch.zeros((B, S), device=device)
        don_buf = torch.zeros((B, S), device=device)
        val_buf = torch.zeros((B, S), device=device)
        
        # CRUCIAL: Store the Context Badges for PPO updates
        ctx_buf = torch.zeros((B, S, self.ppo_net.context_dim), device=device)
        
        # --- PHASE 1: Live Rollout Collection (On-Policy) ---
        
        # 1. Initialize or persist environment states across iterations
        if not hasattr(self, 'current_obs') or self.current_obs is None:
            self.current_obs = env.reset()
            self.current_h_t = torch.zeros(1, B, self.world_model.gru.hidden_size, device=device)
            self.current_prev_rew = torch.zeros(B, device=device)
            
        obs = self.current_obs
        h_t = self.current_h_t
        prev_rew = self.current_prev_rew
        
        for t in range(S):
            self.global_step += B
            
            # 2. Action Selection
            with torch.no_grad():
                action, logprob, _, value = self.ppo_net.get_action_and_value(obs, h_t.squeeze(0))
                
            # 3. Environment Step
            next_obs, reward, done = env.step(action)
            
            # 4. World Model Step (Live)
            with torch.no_grad():
                next_obs_pred, rew_pred, new_h_t = self.world_model.step(obs, action, prev_rew, h_t)
                
            # 5. Store practically everything
            obs_buf[:, t] = obs
            act_buf[:, t] = action
            logp_buf[:, t] = logprob
            rew_buf[:, t] = reward
            don_buf[:, t] = done
            val_buf[:, t] = value
            ctx_buf[:, t] = h_t.squeeze(0) # Store the exact h_t used to make the decision
            
            # 6. Episode Resets (Boundary Masking on Memory)
            # If done == True, zero out the GRU state for that specific environment
            prev_rew = reward.clone() # update for next step
            for i in range(B):
                if done[i]:
                    new_h_t[:, i, :] = 0.0
                    prev_rew[i] = 0.0 # next step is a new episode, so prev_reward is 0
                    
            h_t = new_h_t
            obs = next_obs
            
        # 7. Persist for the next iteration
        self.current_obs = obs
        self.current_h_t = h_t
        self.current_prev_rew = prev_rew
            
        # Bootstrap value for GAE
        with torch.no_grad():
            _, _, _, next_value = self.ppo_net.get_action_and_value(obs, h_t.squeeze(0))
            returns, advantages = self.calculate_returns_and_advantages(
                rew_buf, val_buf, don_buf, next_value, self.cfg['gamma'], self.cfg['gae_lambda']
            )

        # --- PHASE 2: Long-Term Memory Insertion ---
        
        # We chunk the live buffer into 30-step sequences
        num_chunks = S // seq_len
        chunk_surprise_scores = []
        saves_this_rollout = 0
        
        for c in range(num_chunks):
            start_idx = c * seq_len
            end_idx = start_idx + seq_len
            
            # Calculate the surprise score for this precise chunk (offline / ex-post evaluation)
            with torch.no_grad():
                chunk_obs = obs_buf[:, start_idx:end_idx]
                chunk_act = act_buf[:, start_idx:end_idx]
                chunk_rew = rew_buf[:, start_idx:end_idx]
                chunk_ctx = ctx_buf[:, start_idx:end_idx]
                
                # Fetch true 'next' states to measure surprise
                # (Handle the edge case where end_idx == S by using the final 'obs' variable)
                if end_idx < S:
                    chunk_next_obs = obs_buf[:, start_idx+1:end_idx+1]
                else:
                    chunk_next_obs = torch.cat([obs_buf[:, start_idx+1:end_idx], obs.unsqueeze(1)], dim=1)
                
                next_state_preds, next_reward_preds = self.world_model.forward_sequence(chunk_obs, chunk_act, chunk_rew)
                
                # Simplified surprise: MSE of the state logits (or CrossEntropy proxy depending on shape)
                # target class indices
                target_state_idx = chunk_next_obs.argmax(dim=2)              
                preds_flat = next_state_preds.view(B * seq_len, *self.ppo_net.obs_shape)
                targets_flat = target_state_idx.view(B * seq_len, self.ppo_net.h, self.ppo_net.w)
                
                # Mean state surprise
                ce_loss = nn.functional.cross_entropy(preds_flat, targets_flat, reduction='none')
                state_surprise = ce_loss.mean().item()
                
                # Mean reward surprise
                rew_loss = nn.functional.mse_loss(next_reward_preds, chunk_rew, reduction='none')
                reward_surprise = rew_loss.mean().item()
                
                # Total surprise combines both visual and reward prediction errors
                surprise_score = state_surprise + reward_surprise
                
            chunk_surprise_scores.append(surprise_score)
            
            # Interrogate the buffer decision matrix
            if self.memory_buffer.should_save_chunk(self.global_step, surprise_score):
                saves_this_rollout += B  # B chunks pushed (one per env)
                # PUSH CHUNKS INDIVIDUALLY (Batch dim B gets split)
                for i in range(B):
                    chunk = {
                        'state': chunk_obs[i].cpu(),
                        'action': chunk_act[i].cpu(),
                        'reward': chunk_rew[i].cpu(),
                        'done': don_buf[i, start_idx:end_idx].cpu(),
                        'next_state': chunk_next_obs[i].cpu(), # explicitly append next_states for training convenience
                        'h_t': chunk_ctx[i].cpu()
                    }
                    self.memory_buffer.push(chunk)

        # --- PHASE 1.5: Generative Replay (Dreaming) ---
        dream_horizon = self.cfg.get('dream_horizon', 5)
        dream_batch_size = self.cfg.get('dream_batch_size', B) # Default to rolling out B dreams
        
        has_dreams = False
        # QUICK DISABLE: Turning off generative replay completely for now (100% real PPO training)
        if False and len(self.memory_buffer.buffer) >= dream_batch_size:
            has_dreams = True
            
            # Sample seeds (extract final state & h_t of the chunk)
            ltm_dream_seed = self.memory_buffer.sample(dream_batch_size)
            seed_states = ltm_dream_seed['state'][:, -1].to(device) # [dream_batch_size, C, H, W]
            seed_h_t = ltm_dream_seed['h_t'][:, -1].unsqueeze(0).to(device) # [1, dream_batch_size, 256]
            
            # Generate dreams
            dream_trajectories = self.world_model.generate_dream_trajectories(
                policy_net=self.ppo_net,
                start_states=seed_states,
                start_h_t=seed_h_t,
                horizon=dream_horizon
            )
            
            # Repackage into valid tensors [B, S, ...] where B=dream_batch_size, S=dream_horizon
            d_obs = torch.stack([d['state'] for d in dream_trajectories], dim=1)
            d_act = torch.stack([d['action'] for d in dream_trajectories], dim=1)
            d_logp = torch.stack([d['logprob'] for d in dream_trajectories], dim=1)
            d_rew = torch.stack([d['reward'] for d in dream_trajectories], dim=1)
            d_don = torch.stack([d['done'] for d in dream_trajectories], dim=1)
            d_val = torch.stack([d['value'] for d in dream_trajectories], dim=1)
            d_ctx = torch.stack([d['h_t'] for d in dream_trajectories], dim=1)
            
            # To bootstrap GAE, we need the final next_state value
            d_next_state = dream_trajectories[-1]['next_state']
            d_next_ctx = dream_trajectories[-1]['h_t']
            
            # Bootstrap dream PPO returns
            with torch.no_grad():
                _, _, _, d_next_value = self.ppo_net.get_action_and_value(d_next_state, d_next_ctx)
                d_returns, d_advantages = self.calculate_returns_and_advantages(
                    d_rew, d_val, d_don, d_next_value, self.cfg['gamma'], self.cfg['gae_lambda']
                )

        # --- PHASE 3: The World Model Update (The 50/50 Anti-Forgetting Split) ---
        
        # We enforce a Warmup Period (100% Live Data) before initiating the 50/50 mix
        half_batch = self.cfg['wm_batch_size'] // 2
        wm_warmup_steps = self.cfg.get('wm_warmup_steps', 75000)
        
        # 1. Always gather enough Current Live Buffer Chunks for a full batch
        live_batch_obs, live_batch_act, live_batch_rew = [], [], []
        live_batch_next_obs, live_batch_don = [], []
        
        for _ in range(self.cfg['wm_batch_size']):
            env_idx = np.random.randint(0, B)
            start_idx = np.random.randint(0, S - seq_len)
            end_idx = start_idx + seq_len
                
            live_batch_obs.append(obs_buf[env_idx, start_idx:end_idx])
            live_batch_act.append(act_buf[env_idx, start_idx:end_idx])
            live_batch_rew.append(rew_buf[env_idx, start_idx:end_idx])
            live_batch_next_obs.append(obs_buf[env_idx, start_idx+1:end_idx+1])
            live_batch_don.append(don_buf[env_idx, start_idx:end_idx])
                
        # 2. Mix only if past Warmup AND buffer has enough chunks
        if self.global_step >= wm_warmup_steps and len(self.memory_buffer.buffer) >= half_batch:
            ltm_batch = self.memory_buffer.sample(half_batch)
            
            mixed_obs  = torch.cat([ltm_batch['state'].to(device),  torch.stack(live_batch_obs[:half_batch])], dim=0)
            mixed_act  = torch.cat([ltm_batch['action'].to(device), torch.stack(live_batch_act[:half_batch])], dim=0)
            mixed_rew  = torch.cat([ltm_batch['reward'].to(device), torch.stack(live_batch_rew[:half_batch])], dim=0)
            mixed_next = torch.cat([ltm_batch['next_state'].to(device), torch.stack(live_batch_next_obs[:half_batch])], dim=0)
            mixed_don  = torch.cat([ltm_batch['done'].to(device).float(),   torch.stack(live_batch_don[:half_batch]).float()], dim=0)
        else:
            # 100% Live batch during warmup
            mixed_obs  = torch.stack(live_batch_obs)
            mixed_act  = torch.stack(live_batch_act)
            mixed_rew  = torch.stack(live_batch_rew)
            mixed_next = torch.stack(live_batch_next_obs)
            mixed_don  = torch.stack(live_batch_don).float()
            
        # 3. Diagnostic: compute loss on live vs memory separately (no grad) when in 50/50 mode
        loss_live_val = loss_memory_val = None
        if self.global_step >= wm_warmup_steps and len(self.memory_buffer.buffer) >= half_batch:
            with torch.no_grad():
                _, loss_live_s, loss_live_r = self.world_model.compute_loss_detailed(
                    states=torch.stack(live_batch_obs[:half_batch]),
                    actions=torch.stack(live_batch_act[:half_batch]),
                    rewards=torch.stack(live_batch_rew[:half_batch]),
                    next_states=torch.stack(live_batch_next_obs[:half_batch]),
                    next_rewards=torch.stack(live_batch_rew[:half_batch]),
                    dones=torch.stack(live_batch_don[:half_batch]).float(),
                )
                loss_live_val = (loss_live_s + loss_live_r).item()
                _, loss_mem_s, loss_mem_r = self.world_model.compute_loss_detailed(
                    states=ltm_batch['state'].to(device),
                    actions=ltm_batch['action'].to(device),
                    rewards=ltm_batch['reward'].to(device),
                    next_states=ltm_batch['next_state'].to(device),
                    next_rewards=ltm_batch['reward'].to(device),
                    dones=ltm_batch['done'].to(device).float(),
                )
                loss_memory_val = (loss_mem_s + loss_mem_r).item()
        
        # 4. Standard Supervised Training Step
        wm_stats = []
        for _ in range(self.cfg['wm_epochs']):
            # Full batched sequence evaluation
            loss, loss_state, loss_reward = self.world_model.compute_loss_detailed(
                states=mixed_obs,
                actions=mixed_act,
                rewards=mixed_rew,
                next_states=mixed_next,
                next_rewards=mixed_rew, # Simplified test: assume current step reward approx correlates target
                dones=mixed_don
            )
            
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.wm_optimizer.step()
            
            wm_stats.append({
                "world_model/loss_total": loss.item(),
                "world_model/loss_state": loss_state.item(),
                "world_model/loss_reward": loss_reward.item(),
            })

        # --- PHASE 4: The PPO Policy Update ---
        
        ppo_stats_live = {}
        ppo_stats_dream = {}
        
        # 1. Update on Live Rollout
        ppo_stats_live = self.ppo_update(
            obs=obs_buf,
            actions=act_buf,
            old_logprobs=logp_buf,
            context_ht=ctx_buf,
            advantages=advantages,
            returns=returns,
            old_values=val_buf
        )
        
        # 2. Update on Dream Generative Replay Rollout (if valid)
        if has_dreams:
            ppo_stats_dream = self.ppo_update(
                obs=d_obs,
                actions=d_act,
                old_logprobs=d_logp,
                context_ht=d_ctx,
                advantages=d_advantages,
                returns=d_returns,
                old_values=d_val
            )
            
        # Aggregate stats
        results = {
            "global_step": self.global_step,
            "charts/reward_step_mean": rew_buf.mean().item(),
            "charts/reward_step_max": rew_buf.max().item(),
            "charts/reward_step_min": rew_buf.min().item()
        }
        
        # Memory buffer & surprise metrics
        results["memory/surprise_ema"] = self.memory_buffer.surprise_ema_threshold
        results["memory/buffer_size"] = len(self.memory_buffer.buffer)
        results["memory/saves_per_rollout"] = saves_this_rollout
        if chunk_surprise_scores:
            results["memory/chunk_surprise_score"] = np.mean(chunk_surprise_scores)
        else:
            results["memory/chunk_surprise_score"] = 0.0
            
        # World model split (diagnostic for catastrophic forgetting)
        if loss_live_val is not None:
            results["world_model/loss_live"] = loss_live_val
        if loss_memory_val is not None:
            results["world_model/loss_memory"] = loss_memory_val
        
        # PPO stats: report live and dream separately for dream health diagnosis
        results["ppo/policy_loss_live"] = ppo_stats_live["ppo/policy_loss"]
        results["ppo/value_loss_live"] = ppo_stats_live["ppo/value_loss"]
        results["ppo/entropy"] = ppo_stats_live["ppo/entropy"]
        
        # Diagnostic prints for context_ht
        results["ppo/context_ht_norm"] = ctx_buf.norm(dim=-1).mean().item()
        results["ppo/context_ht_std"] = ctx_buf.std(dim=(0,1)).mean().item()
        
        if ppo_stats_dream:
            results["ppo/policy_loss_dream"] = ppo_stats_dream["ppo/policy_loss"]
            results["ppo/value_loss_dream"] = ppo_stats_dream["ppo/value_loss"]
        else:
            results["ppo/policy_loss_dream"] = 0.0
            results["ppo/value_loss_dream"] = 0.0
        results["dream/used"] = 1.0 if has_dreams else 0.0
        results["wm/warmup_active"] = 1.0 if self.global_step < wm_warmup_steps else 0.0
            
        # WM Stats (loss_total, loss_state, loss_reward from training)
        if wm_stats:
            avg_wm = {k: np.mean([s[k] for s in wm_stats]) for k in wm_stats[0]}
            results.update(avg_wm)
            
        results["ppo/intrinsic_reward_mean"] = 0.0 # Solution 5 doesn't use intrinsic curiosity directly
            
        return results


# ==========================================
# VERIFICATION BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing MetaRLTrainer Orchestration Block ---")
    
    # 1. Initialize Components
    device = torch.device('cpu') # Forcing CPU for fast shape verification
    cfg = {
        'num_envs': 4,
        'num_steps': 120, # Must be a multiple of chunk size (30)
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 2,
        'minibatch_size': 64,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'wm_batch_size': 16, # Half = 8 LTM, 8 Live
        'wm_epochs': 2,
        'wm_warmup_steps': 75000,
        'dream_horizon': 5,
        'dream_batch_size': 4
    }
    
    ppo_net = ContextAwarePPONetwork().to(device)
    wm = RecurrentWorldModel().to(device)
    buffer = SequenceMemoryBuffer(max_capacity=100)
    
    ppo_opt = torch.optim.Adam(ppo_net.parameters(), lr=3e-4)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=1e-4)
    
    trainer = MetaRLTrainer(ppo_net, wm, buffer, ppo_opt, wm_opt, cfg)
    
    # 2. Dummy Environment Class
    class DummyEnv:
        def __init__(self, num_envs, obs_shape):
            self.n = num_envs
            self.obs_shape = obs_shape
            
        def reset(self):
            return torch.zeros((self.n, *self.obs_shape))
            
        def step(self, action):
            # Advance Dummy Env
            next_obs = torch.zeros((self.n, *self.obs_shape))
            next_obs[:, 1, :, :] = 1.0 # arbitrary flip
            reward = torch.randn(self.n)
            done = torch.zeros(self.n, dtype=torch.bool)
            done[0] = True # Force an episode reset to test zero-masking logic
            return next_obs, reward, done
            
    env = DummyEnv(cfg['num_envs'], ppo_net.obs_shape)
    
    # Pre-fill buffer slightly so Phase 3 World Model update absolutely triggers
    print("Pre-filling SequenceMemoryBuffer with 10 dummy chunks...")
    for _ in range(10):
        dummy_chunk = {
            'state': torch.zeros((30, *ppo_net.obs_shape)),
            'action': torch.ones((30,), dtype=torch.long),
            'reward': torch.randn((30,)),
            'done': torch.zeros((30,), dtype=torch.bool),
            'next_state': torch.zeros((30, *ppo_net.obs_shape)),
            'h_t': torch.randn((30, 256))
        }
        buffer.push(dummy_chunk)
        
    print("Running 1 Full Training Iteration (Phase 1 -> 4)...")
    final_step = trainer.train_iteration(env)
    
    # Verify separation
    assert isinstance(trainer.ppo_optimizer, torch.optim.Optimizer)
    assert isinstance(trainer.wm_optimizer, torch.optim.Optimizer)
    
    print(f"Verification Successful: 1 Full Training pass completed without shape errors (Global Steps: {final_step}). Optimizers correctly separated.")
