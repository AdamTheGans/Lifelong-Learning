import torch
import numpy as np
import random

class DiagnosticValidationBuffer:
    """
    A read-only buffer that stores a fixed set of "golden" sequences from early in training.
    Used exclusively to evaluate the World Model's convergence and track catastrophic forgetting.
    """
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        self.regime_0_success = []
        self.regime_0_failure = []
        self.regime_1_success = []
        self.regime_1_failure = []
        
        self.max_per_category = 20
        
    def push_if_needed(self, chunk: dict, global_step: int, regime_switch_step: int):
        """
        Evaluates a chunk and saves it if we still need golden sequences for the current regime.
        """
        rewards = chunk['reward']
        if isinstance(rewards, torch.Tensor):
            max_r = rewards.max().item()
            min_r = rewards.min().item()
        else:
            max_r = np.max(rewards)
            min_r = np.min(rewards)
            
        is_success = max_r > 4.0
        is_failure = min_r < -0.5
        
        if not (is_success or is_failure):
            return # We only care about terminal sequences for validation
            
        # Determine regime
        if global_step < regime_switch_step:
            # Regime 0
            if is_success and len(self.regime_0_success) < self.max_per_category:
                self.regime_0_success.append(chunk)
            elif is_failure and len(self.regime_0_failure) < self.max_per_category:
                self.regime_0_failure.append(chunk)
        else:
            # Regime 1
            if is_success and len(self.regime_1_success) < self.max_per_category:
                self.regime_1_success.append(chunk)
            elif is_failure and len(self.regime_1_failure) < self.max_per_category:
                self.regime_1_failure.append(chunk)
                
    def get_all_batches(self) -> dict:
        """
        Returns a dictionary of batched tensors for each category that has data.
        """
        batches = {}
        
        categories = {
            'regime0_success': self.regime_0_success,
            'regime0_failure': self.regime_0_failure,
            'regime1_success': self.regime_1_success,
            'regime1_failure': self.regime_1_failure
        }
        
        for name, chunks in categories.items():
            if len(chunks) > 0:
                batch = {key: [] for key in chunks[0].keys()}
                for chunk in chunks:
                    for key in batch.keys():
                        batch[key].append(chunk[key])
                        
                stacked_batch = {}
                for key in batch.keys():
                    tensor_list = [torch.as_tensor(item) for item in batch[key]]
                    stacked_batch[key] = torch.stack(tensor_list, dim=0)
                
                batches[name] = stacked_batch
                
        return batches

class SequenceMemoryBuffer:
    """
    A Long-Term Memory Buffer designed to store sequences of transitions to
    prevent catastrophic forgetting in a continual learning world model.
    
    This buffer stores 'chunks' representing exactly 30 consecutive transitions:
    (state, action, reward, done, h_t). When the buffer is full, it evicts chunks using
    Uniform Random Eviction to mimic reservoir sampling and maintain an unbiased 
    distribution of historical regimes.
    """
    def __init__(self, max_capacity: int = 2000, seq_len: int = 30, surprise_ema_alpha: float = 0.05):
        """
        Initialize the SequenceMemoryBuffer with Stratified Storage.
        
        Args:
            max_capacity (int): The maximum number of chunks the buffer can hold.
            seq_len (int): The number of consecutive transitions in a chunk (default: 30).
            surprise_ema_alpha (float): The decay rate for the Exponential Moving Average 
                                        tracker of the surprise score.
        """
        self.max_capacity = max_capacity
        self.seq_len = seq_len
        self.surprise_ema_alpha = surprise_ema_alpha
        
        # Stratified Buffer storage
        # We want to save triple the amount of failure endings
        # So we allocate 60% of capacity to failures, 20% to success, 20% to neutral
        self.max_failure = int(max_capacity * 0.6)
        self.max_success = int(max_capacity * 0.2)
        self.max_neutral = max_capacity - self.max_success - self.max_failure
        
        self.buffers = {
            'success': [],
            'failure': [],
            'neutral': []
        }
        
        # Reservoir Sampling Counters (N)
        self.seen_counts = {
            'success': 0,
            'failure': 0,
            'neutral': 0
        }
        
        # Tracking for the rate limits
        self.last_save_step = 0
        
        # EMA Tracker for the surprise threshold
        self.surprise_ema_threshold = 0.0
        self.ema_initialized = False

    def __len__(self):
        return sum(len(b) for b in self.buffers.values())

    @property
    def buffer(self):
        # For backward compatibility where buffer.buffer is accessed directly
        return self.buffers['success'] + self.buffers['failure'] + self.buffers['neutral']

    def update_surprise_ema(self, surprise_score: float):
        """
        Updates the Exponential Moving Average (EMA) of the surprise score.
        """
        if not self.ema_initialized:
            self.surprise_ema_threshold = surprise_score
            self.ema_initialized = True
        else:
            self.surprise_ema_threshold = (
                self.surprise_ema_alpha * surprise_score + 
                (1.0 - self.surprise_ema_alpha) * self.surprise_ema_threshold
            )

    def should_save_chunk(self, current_step: int, surprise_score: float) -> bool:
        """
        Evaluates whether a chunk should be saved based on the current step and the
        most recent surprise score (prediction error), handling rate-limiting and 
        surprise spikes.
        
        Rule 1: If current_step - last_save_step < 1000, return False (Max 1 save per 1k steps).
        Rule 2: If current_step - last_save_step >= 5000, return True (Min 1 save per 5k steps).
        Rule 3: If between 1k and 5k, return True ONLY IF surprise_score > EMA threshold.
        
        Args:
            current_step (int): The current numerical environment step.
            surprise_score (float): The calculated prediction error of the World Model.
            
        Returns:
            bool: True if the chunk should be saved, False otherwise.
        """
        # Always update the EMA threshold with the incoming surprise score
        self.update_surprise_ema(surprise_score)
        
        steps_since_last = current_step - self.last_save_step
        
        # Rule 1: Maximum 1 save per 500 steps (prevent flooding during high volatility)
        if steps_since_last < 500:
            return False
            
        # Rule 2: Minimum 1 save per 2500 steps (force save even if agent is comfortable)
        if steps_since_last >= 2500:
            self.last_save_step = current_step
            return True
            
        # Rule 3: Steps are between 1k and 5k. Save ONLY IF surprise > EMA threshold.
        # This catches regime changes (sudden spikes in prediction error).
        if surprise_score > self.surprise_ema_threshold:
            self.last_save_step = current_step
            return True
            
        return False

    def push(self, chunk: dict):
        """
        Pushes a new chunk of transitions into the stratified buffer.
        
        If the specific sub-buffer is at max capacity, it triggers Uniform Random Eviction.
        
        Args:
            chunk (dict): Dictionary with keys 'state', 'action', 'reward', 'done', 'h_t' (and optionally 'next_state').
                          Values should be arrays or tensors of length `self.seq_len`.
        """
        # Validate that the chunk matches our required length (simplifies batching later)
        for key, value in chunk.items():
            if len(value) != self.seq_len:
                raise ValueError(f"Chunk key '{key}' has length {len(value)}, expected {self.seq_len}")

        # Note 2: Ensure that every saved sequence contains a terminal state
        dones = chunk['done']
        if isinstance(dones, torch.Tensor):
            has_terminal = dones.any().item()
        else:
            has_terminal = np.any(dones)
            
        if not has_terminal:
            return  # Reject chunks without any terminal states to ensure valid contexts

        # Categorize the chunk based on rewards
        rewards = chunk['reward']
        if isinstance(rewards, torch.Tensor):
            max_r = rewards.max().item()
            min_r = rewards.min().item()
        else:
            max_r = np.max(rewards)
            min_r = np.min(rewards)

        if min_r < -0.5:
            category = 'failure'
            max_cap = self.max_failure
        elif max_r > 4.0:
            category = 'success'
            max_cap = self.max_success
        else:
            category = 'neutral'
            max_cap = self.max_neutral

        target_buffer = self.buffers[category]
        
        # Increment the total number of items ever seen for this category (N)
        self.seen_counts[category] += 1
        n = self.seen_counts[category]

        if len(target_buffer) < max_cap:
            # Buffer has room, append normally
            target_buffer.append(chunk)
        else:
            # True Reservoir Sampling Algorithm (Algorithm R)
            # Generate a random number R between 1 and N
            r = random.randint(1, n)
            
            # If R is less than or equal to the buffer capacity, overwrite the sequence at index R - 1.
            # Otherwise, discard the new sequence.
            if r <= max_cap:
                target_buffer[r - 1] = chunk

    def sample(self, batch_size: int) -> dict:
        """
        Samples a stratified batch of chunks and formats them as PyTorch tensors.
        Ensures an equal pull from Success, Failure, and Neutral categories if available.
        
        Args:
            batch_size (int): The number of chunks to sample.
            
        Returns:
            dict: Stacked chunks where each value is a tensor of shape [batch_size, 30, ...].
        """
        total_available = sum(len(b) for b in self.buffers.values())
        if total_available < batch_size:
            raise ValueError(f"Cannot sample {batch_size} chunks, buffer only has {total_available}.")
            
        sampled_chunks = []
        
        # Try to sample proportionally to our new 60/20/20 split
        # If batch_size is 8 (for 50/50 split of 16):
        # 60% of 8 = 4.8 -> 5 failures
        # 20% of 8 = 1.6 -> 2 successes (rounded up to fill)
        # 20% of 8 = 1.6 -> 1 neutral
        categories = ['failure', 'success', 'neutral']
        
        req_failure = int(batch_size * 0.6)
        req_success = int(batch_size * 0.2)
        req_neutral = batch_size - req_failure - req_success
        
        requests = {
            'failure': req_failure,
            'success': req_success,
            'neutral': req_neutral
        }
        
        # Adjust requests if some buffers don't have enough
        for _ in range(2): # Two passes to distribute shortfall
            for cat in categories:
                avail = len(self.buffers[cat])
                if requests[cat] > avail:
                    shortfall = requests[cat] - avail
                    requests[cat] = avail
                    # Distribute shortfall to others
                    others = [c for c in categories if c != cat and len(self.buffers[c]) > requests[c]]
                    if others:
                        for c in others:
                            add = shortfall // len(others) + (1 if shortfall % len(others) > 0 else 0)
                            can_add = min(add, len(self.buffers[c]) - requests[c])
                            requests[c] += can_add
                            shortfall -= can_add
                            if shortfall <= 0: break

        # Actually sample
        for cat in categories:
            if requests[cat] > 0:
                sampled_indices = random.sample(range(len(self.buffers[cat])), requests[cat])
                sampled_chunks.extend([self.buffers[cat][i] for i in sampled_indices])
                
        # Shuffle the combined batch so categories aren't contiguous
        random.shuffle(sampled_chunks)
        
        # Dynamically initialize batch lists based on the keys present in the first chunk
        batch = {key: [] for key in sampled_chunks[0].keys()}
        
        # Extract the fields from our sampled chunks
        for chunk in sampled_chunks:
            for key in batch.keys():
                batch[key].append(chunk[key])
            
        # Stack the lists of tensors into properly formatted 2D+ PyTorch tensors
        # Resulting shape: [batch_size, seq_len, ...]
        stacked_batch = {}
        for key in batch.keys():
            # torch.as_tensor handles cases where input might already be tensors or numpy arrays
            tensor_list = [torch.as_tensor(item) for item in batch[key]]
            stacked_batch[key] = torch.stack(tensor_list, dim=0)
            
        return stacked_batch


if __name__ == "__main__":
    print("--- Testing SequenceMemoryBuffer Component isolated script ---\n")
    
    # 1. Instantiate the buffer
    max_cap = 2000
    seq_length = 30
    buffer = SequenceMemoryBuffer(max_capacity=max_cap, seq_len=seq_length)
    print(f"Instantiated buffer (Capacity: {max_cap}, Seq Length: {seq_length})")
    
    # 2. Simulate step loop and test rules / EMA Tracker
    print("Simulating environment chunks being generated every 30 steps...")
    
    # We will simulate 300,000 env steps to fill the buffer enough to sample a 64-batch
    TOTAL_STEPS = 300_000
    CHUNK_INTERVAL = 30 # A new chunk is formulated every 30 steps
    
    # Mock parameters derived from SPEC
    # Example state: (21 channels, 8 H, 8 W) one-hot grid representation
    state_shape = (21, 8, 8)
    
    saved_chunks_count = 0
    for current_step in range(CHUNK_INTERVAL, TOTAL_STEPS + 1, CHUNK_INTERVAL):
        # Generate a dummy surprise score
        # Normally low (0.1 - 0.5), but spike every ~20,000 steps to simulate a shifting regime
        if current_step % 20000 < (CHUNK_INTERVAL * 5): # Sustained spike over a few chunks
            surprise_score = random.uniform(5.0, 10.0)
        else:
            surprise_score = random.uniform(0.1, 0.5)
            
        if buffer.should_save_chunk(current_step, surprise_score):
            # Create a dummy chunk using PyTorch tensors
            dummy_state = torch.randn((seq_length, *state_shape))
            dummy_action = torch.randint(0, 3, (seq_length,))
            dummy_reward = torch.randn((seq_length,))
            dummy_done = torch.randint(0, 2, (seq_length,), dtype=torch.bool)
            dummy_h_t = torch.randn((seq_length, 256))
            
            chunk = {
                'state': dummy_state,
                'action': dummy_action,
                'reward': dummy_reward,
                'done': dummy_done,
                'h_t': dummy_h_t
            }
            
            # Push into the long-term memory buffer
            buffer.push(chunk)
            saved_chunks_count += 1
            
            # Print a few logs periodically
            if saved_chunks_count % 20 == 0:
                print(f"  [Step {current_step:6d}] Chunk saved! Surprise: {surprise_score:5.2f} (EMA: {buffer.surprise_ema_threshold:5.2f}). Buffer size: {len(buffer.buffer)}")

    print(f"\nSimulation complete. Total chunks saved: {len(buffer)}")
    print(f"  Success: {len(buffer.buffers['success'])}")
    print(f"  Failure: {len(buffer.buffers['failure'])}")
    print(f"  Neutral: {len(buffer.buffers['neutral'])}")
    
    # 3. Sample a batch of 64
    batch_size = 64
    if len(buffer) >= batch_size:
        print(f"\nSampling a batch of {batch_size} chunks...")
        sampled_batch = buffer.sample(batch_size)
        
        # 4. Verification script: Print final tensor shapes
        print("\n--- Shape Verification ---")
        for key, tensor in sampled_batch.items():
            print(f"{key:>7s} Tensor : {list(tensor.shape)}")
            
        # Assertions to ensure shapes perfectly match the specification guarantees
        assert list(sampled_batch['state'].shape) == [batch_size, seq_length, *state_shape]
        assert list(sampled_batch['action'].shape) == [batch_size, seq_length]
        assert list(sampled_batch['reward'].shape) == [batch_size, seq_length]
        assert list(sampled_batch['done'].shape) == [batch_size, seq_length]
        assert list(sampled_batch['h_t'].shape) == [batch_size, seq_length, 256]
        
        print("\n✅ Verification Successful: All tensor shapes perfectly align with [batch_size, 30, ...].")
    else:
        print("\nBuffer has less than batch_size limit.")
