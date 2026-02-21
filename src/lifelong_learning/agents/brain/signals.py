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

    def __init__(self, surprise_spike_threshold: float = 2.0, history_len: int = 50):
        self.normalizer = RunningNormalizer(NUM_SIGNALS)
        self.surprise_history = deque(maxlen=history_len)
        self.spike_threshold = surprise_spike_threshold
        self.steps_since_spike = 0
        self.prev_surprise = 0.0

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
        ], dtype=np.float32)

        self.normalizer.update(raw)
        return self.normalizer.normalize(raw)

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
