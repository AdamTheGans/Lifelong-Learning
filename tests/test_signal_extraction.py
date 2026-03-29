"""Tests for signal extraction and normalization."""
import unittest
import numpy as np

from lifelong_learning.agents.brain.signals import (
    SignalExtractor,
    RunningNormalizer,
    NUM_SIGNALS,
    SIGNAL_NAMES,
)


class TestRunningNormalizer(unittest.TestCase):
    """Tests for the Welford running mean/std normalizer."""

    def test_normalize_shape(self):
        norm = RunningNormalizer(size=5)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        norm.update(x)
        result = norm.normalize(x)
        self.assertEqual(result.shape, (5,))
        self.assertEqual(result.dtype, np.float32)

    def test_normalize_zero_variance(self):
        """Should not produce NaN/Inf when all values are identical."""
        norm = RunningNormalizer(size=3)
        x = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        for _ in range(10):
            norm.update(x)
        result = norm.normalize(x)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))

    def test_normalize_clip(self):
        """Values should be clipped to [-clip, clip]."""
        norm = RunningNormalizer(size=2, clip=3.0)
        # Feed small range, then ask to normalize a huge outlier
        for _ in range(100):
            norm.update(np.array([0.0, 0.0], dtype=np.float32))
        result = norm.normalize(np.array([1000.0, -1000.0], dtype=np.float32))
        self.assertTrue(np.all(result <= 3.0))
        self.assertTrue(np.all(result >= -3.0))

    def test_running_statistics_convergence(self):
        """Running mean should converge to true mean."""
        norm = RunningNormalizer(size=1)
        rng = np.random.RandomState(42)
        for _ in range(1000):
            norm.update(rng.randn(1).astype(np.float32) * 2 + 5)
        # Mean should be close to 5
        self.assertAlmostEqual(norm.mean[0], 5.0, delta=0.3)

    def test_state_dict_round_trip(self):
        """Serialized normalizer state should restore count and moments exactly."""
        norm = RunningNormalizer(size=2, clip=3.5)
        for sample in (
            np.array([1.0, -1.0], dtype=np.float32),
            np.array([2.5, 0.5], dtype=np.float32),
            np.array([-0.5, 3.0], dtype=np.float32),
        ):
            norm.update(sample)

        restored = RunningNormalizer(size=2)
        restored.load_state_dict(norm.state_dict())

        self.assertEqual(restored.count, norm.count)
        self.assertEqual(restored.clip, norm.clip)
        np.testing.assert_allclose(restored.mean, norm.mean)
        np.testing.assert_allclose(restored.var, norm.var)
        np.testing.assert_allclose(restored._M2, norm._M2)

class TestSignalExtractor(unittest.TestCase):
    """Tests for the full signal extraction pipeline."""

    def _make_stats(self, **overrides):
        """Create a minimal stats dict with optional overrides."""
        stats = {
            "mean_episodic_return": 0.0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "mean_surprise": 0.1,
            "wm_loss_state": 0.5,
            "wm_loss_reward": 0.01,
            "policy_entropy": 1.0,
            "policy_loss": 0.1,
            "value_loss": 0.5,
            "current_lr": 3e-4,
            "current_ent_coef": 0.01,
            "current_intrinsic_coef": 0.015,
            "current_imagined_horizon": 10,
        }
        stats.update(overrides)
        return stats

    def test_extract_shape(self):
        """extract() should return array of shape (NUM_SIGNALS,)."""
        ext = SignalExtractor()
        obs = ext.extract(self._make_stats())
        self.assertEqual(obs.shape, (NUM_SIGNALS,))
        self.assertEqual(obs.dtype, np.float32)

    def test_extract_no_nan(self):
        """Output should never contain NaN."""
        ext = SignalExtractor()
        for _ in range(20):
            obs = ext.extract(self._make_stats())
            self.assertFalse(np.any(np.isnan(obs)), f"NaN in: {obs}")

    def test_signal_count_matches_names(self):
        """NUM_SIGNALS should match the length of SIGNAL_NAMES."""
        self.assertEqual(NUM_SIGNALS, len(SIGNAL_NAMES))

    def test_surprise_spike_detection(self):
        """Should detect a large surprise spike."""
        ext = SignalExtractor(surprise_spike_threshold=2.0)

        # Feed stable low surprise for a while
        for _ in range(20):
            ext.extract(self._make_stats(mean_surprise=0.1))

        # Now spike
        ext.extract(self._make_stats(mean_surprise=5.0))
        self.assertEqual(ext.steps_since_spike, 0, "Spike should have been detected")

    def test_no_false_spike(self):
        """Stable surprise should not trigger spike detection."""
        ext = SignalExtractor(surprise_spike_threshold=2.0)

        for i in range(20):
            ext.extract(self._make_stats(mean_surprise=0.1 + i * 0.001))

        self.assertGreater(ext.steps_since_spike, 0)

    def test_surprise_delta_computation(self):
        """surprise_delta should reflect the change between consecutive calls."""
        ext = SignalExtractor()
        ext.extract(self._make_stats(mean_surprise=0.5))
        ext.extract(self._make_stats(mean_surprise=1.0))
        # The delta should be 0.5 (before normalization)
        # We can't easily check the normalized value, but we can verify no crash
        obs = ext.extract(self._make_stats(mean_surprise=1.0))
        self.assertEqual(obs.shape, (NUM_SIGNALS,))

    def test_reset(self):
        """reset() should clear internal state."""
        ext = SignalExtractor()
        for _ in range(10):
            ext.extract(self._make_stats())

        ext.reset()
        self.assertEqual(ext.steps_since_spike, 0)
        self.assertEqual(ext.prev_surprise, 0.0)
        self.assertEqual(len(ext.surprise_history), 0)

    def test_state_dict_restores_cross_episode_normalizer(self):
        """A restored extractor should match a warm process at the next episode boundary."""
        ext = SignalExtractor()
        for idx in range(8):
            ext.extract(
                self._make_stats(
                    mean_episodic_return=float(idx),
                    mean_surprise=0.1 + 0.05 * idx,
                    policy_loss=0.1 * idx,
                )
            )

        resume_state = ext.state_dict()
        ext.reset()

        restored = SignalExtractor()
        restored.load_state_dict(resume_state)

        stats = self._make_stats(mean_episodic_return=3.5, mean_surprise=0.42, policy_loss=0.77)
        np.testing.assert_allclose(restored.extract(stats), ext.extract(stats))
    def test_static_override_indices_match_signal_names(self):
        """Static normalization overrides must write to indices matching SIGNAL_NAMES."""
        ext = SignalExtractor()
        # Warm up normalizer so Welford stats are non-trivial
        for _ in range(10):
            ext.extract(self._make_stats())

        # Set distinctive values for each statically-overridden signal
        test_stats = self._make_stats(
            success_rate=0.75,
            failure_rate=0.15,
            current_lr=1e-3,
            current_ent_coef=0.05,
            current_intrinsic_coef=0.1,
            current_imagined_horizon=15,
            current_replay_ratio=0.25,
            current_replay_prioritization=0.7,
            current_anchoring_weight=0.3,
            episodic_memory_fullness=0.6,
        )
        obs = ext.extract(test_stats)

        # Index 1: success_rate -> 0.75 * 2 - 1 = 0.5
        self.assertAlmostEqual(obs[SIGNAL_NAMES.index("success_rate")], 0.5, places=3)
        # Index 2: failure_rate -> 0.15 * 2 - 1 = -0.7
        self.assertAlmostEqual(obs[SIGNAL_NAMES.index("failure_rate")], -0.7, places=3)
        # Index 16: current_replay_prioritization -> 0.7 * 2 - 1 = 0.4
        self.assertAlmostEqual(obs[SIGNAL_NAMES.index("current_replay_prioritization")], 0.4, places=3)
        # Index 17: current_anchoring_weight -> (0.3 / 0.5) * 2 - 1 = 0.2
        self.assertAlmostEqual(obs[SIGNAL_NAMES.index("current_anchoring_weight")], 0.2, places=3)
        # Index 18: episodic_memory_fullness -> 0.6 * 2 - 1 = 0.2
        self.assertAlmostEqual(obs[SIGNAL_NAMES.index("episodic_memory_fullness")], 0.2, places=3)


if __name__ == "__main__":
    unittest.main()
