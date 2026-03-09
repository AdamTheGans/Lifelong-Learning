"""
Signal extraction for the Brain meta-agent.

Converts raw training stats from the inner Dyna-PPO loop into a
normalized observation vector that the Brain can learn from.
"""
from __future__ import annotations

import numpy as np
from collections import deque


# Indices into the 15-dim observation vector
SIGNAL_NAMES = [
    "mean_episodic_return",
    "success_rate",
    "failure_rate",
    "mean_surprise",
    "surprise_delta",
    "wm_loss_state",
    "wm_loss_reward",
    "policy_entropy",
    "policy_loss",
    "value_loss",
    "current_lr",
    "current_ent_coef",
    "current_intrinsic_coef",
    "current_imagined_horizon",
    "steps_since_surprise_spike",
    "current_replay_ratio",
    "episodic_memory_fullness",
]

NUM_SIGNALS = len(SIGNAL_NAMES)


class RunningNormalizer:
    """Welford's online mean/std tracker for signal normalization."""

    def __init__(self, size: int, clip: float = 10.0):
        self.size = size
        self.clip = clip
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float32)
        self.var = np.ones(size, dtype=np.float32)
        self._M2 = np.zeros(size, dtype=np.float32)

    def update(self, x: np.ndarray):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._M2 += delta * delta2
        if self.count > 1:
            self.var = self._M2 / (self.count - 1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        normed = (x - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


class SignalExtractor:
    """
    Extracts and normalizes training signals for the Brain meta-agent.

    Maintains a history of surprise values for spike detection and
    delta computation.
    """

    def __init__(self, surprise_spike_threshold: float = 2.0, history_len: int = 50, max_inner_lr: float = 0.003, min_inner_lr: float = 1e-4, min_ent_coef: float = 0.001, max_ent_coef: float = 0.2, min_intrinsic_coef: float = 0.001, max_intrinsic_coef: float = 1.0):
        self.normalizer = RunningNormalizer(NUM_SIGNALS)
        self.surprise_history = deque(maxlen=history_len)
        self.spike_threshold = surprise_spike_threshold
        self.steps_since_spike = 0
        self.prev_surprise = 0.0
        self.max_inner_lr = max_inner_lr
        self.min_inner_lr = min_inner_lr
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.min_intrinsic_coef = min_intrinsic_coef
        self.max_intrinsic_coef = max_intrinsic_coef

    def extract(self, stats: dict) -> np.ndarray:
        """
        Convert a stats dict (from run_inner_update) into a 15-dim
        normalized observation vector.

        Args:
            stats: dict returned by run_inner_update()

        Returns:
            np.ndarray of shape (NUM_SIGNALS,), normalized
        """
        surprise = stats.get("mean_surprise", 0.0)
        surprise_delta = surprise - self.prev_surprise
        self.prev_surprise = surprise

        # Spike detection
        self.surprise_history.append(surprise)
        if self._detect_spike(surprise):
            self.steps_since_spike = 0
        else:
            self.steps_since_spike += 1

        raw = np.array([
            stats.get("mean_episodic_return", 0.0),
            stats.get("success_rate", 0.0),
            stats.get("failure_rate", 0.0),
            surprise,
            surprise_delta,
            stats.get("wm_loss_state", 0.0),
            stats.get("wm_loss_reward", 0.0),
            stats.get("policy_entropy", 0.0),
            stats.get("policy_loss", 0.0),
            stats.get("value_loss", 0.0),
            stats.get("current_lr", 0.0),
            stats.get("current_ent_coef", 0.0),
            stats.get("current_intrinsic_coef", 0.0),
            stats.get("current_imagined_horizon", 0.0),
            float(self.steps_since_spike),
            stats.get("current_replay_ratio", 0.0),
            stats.get("episodic_memory_fullness", 0.0),
        ], dtype=np.float32)

        self.normalizer.update(raw)
        normed = self.normalizer.normalize(raw)

        # Override explicitly bounded signals with static normalization to preserve absolute magnitude awareness.
        # This prevents the Brain from losing track of absolute HP values when they saturate.
        
        # 1: success_rate -> [0, 1] mapped to [-1, 1]
        normed[1] = stats.get("success_rate", 0.0) * 2.0 - 1.0
        # 2: failure_rate -> [0, 1] mapped to [-1, 1]
        normed[2] = stats.get("failure_rate", 0.0) * 2.0 - 1.0
        
        # 10: current_lr -> log scale [min_inner_lr, max_inner_lr] mapped to [-1, 1]
        lr = stats.get("current_lr", 1e-4)
        normed[10] = (np.log(max(lr, 1e-8)) - np.log(self.min_inner_lr)) / (np.log(self.max_inner_lr) - np.log(self.min_inner_lr)) * 2.0 - 1.0
        normed[10] = np.clip(normed[10], -1.0, 1.0)
        
        # 11: current_ent_coef -> log scale [min_ent_coef, max_ent_coef] mapped to [-1, 1]
        ent = stats.get("current_ent_coef", 0.01)
        normed[11] = (np.log(max(ent, 1e-8)) - np.log(self.min_ent_coef)) / (np.log(self.max_ent_coef) - np.log(self.min_ent_coef)) * 2.0 - 1.0
        normed[11] = np.clip(normed[11], -1.0, 1.0)
        
        # 12: current_intrinsic_coef -> log scale [min_intrinsic_coef, max_intrinsic_coef] mapped to [-1, 1]
        ic = stats.get("current_intrinsic_coef", 0.015)
        normed[12] = (np.log(max(ic, 1e-8)) - np.log(self.min_intrinsic_coef)) / (np.log(self.max_intrinsic_coef) - np.log(self.min_intrinsic_coef)) * 2.0 - 1.0
        normed[12] = np.clip(normed[12], -1.0, 1.0)
        
        # 13: current_imagined_horizon -> linear [1, 30] mapped to [-1, 1]
        hor = stats.get("current_imagined_horizon", 10.0)
        normed[13] = (hor - 1.0) / (30.0 - 1.0) * 2.0 - 1.0
        normed[13] = np.clip(normed[13], -1.0, 1.0)
        
        # 15: current_replay_ratio -> linear [0.0, 0.5] mapped to [-1, 1]
        rr = stats.get("current_replay_ratio", 0.0)
        normed[15] = (rr - 0.0) / (0.5 - 0.0) * 2.0 - 1.0
        normed[15] = np.clip(normed[15], -1.0, 1.0)
        
        # 16: episodic_memory_fullness -> [0, 1] mapped to [-1, 1]
        full = stats.get("episodic_memory_fullness", 0.0)
        normed[16] = full * 2.0 - 1.0
        normed[16] = np.clip(normed[16], -1.0, 1.0)

        return normed

    def _detect_spike(self, surprise: float) -> bool:
        """
        Detect a surprise spike indicating a likely regime change.

        A spike is detected when the current surprise exceeds
        the running mean by more than `spike_threshold` standard deviations.
        """
        if len(self.surprise_history) < 5:
            return False

        history = np.array(self.surprise_history)
        mean = history.mean()
        std = history.std() + 1e-8
        z_score = (surprise - mean) / std

        return z_score > self.spike_threshold

    def reset(self):
        """Reset state for a new inner training episode."""
        self.surprise_history.clear()
        self.steps_since_spike = 0
        self.prev_surprise = 0.0
        # Keep normalizer stats across episodes for stability
