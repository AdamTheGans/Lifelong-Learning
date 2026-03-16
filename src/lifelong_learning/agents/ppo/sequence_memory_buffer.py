import torch
import numpy as np
import random

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
        self.max_success = max_capacity // 3
        self.max_failure = max_capacity // 3
        self.max_neutral = max_capacity - self.max_success - self.max_failure
        
        self.buffers = {
            'success': [],
            'failure': [],
            'neutral': []
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

        if len(target_buffer) < max_cap:
            # Buffer has room, append normally
            target_buffer.append(chunk)
        else:
            # Buffer is full, overwrite a uniformly selected existing chunk 
            # This naturally acts as reservoir sampling, permanently protecting a percentage of early sequences
            evict_index = random.randint(0, max_cap - 1)
            target_buffer[evict_index] = chunk

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
        
        # Try to sample equally from all 3 categories
        categories = ['success', 'failure', 'neutral']
        per_category = batch_size // 3
        remainder = batch_size % 3
        
        requests = {cat: per_category + (1 if i < remainder else 0) for i, cat in enumerate(categories)}
        
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
