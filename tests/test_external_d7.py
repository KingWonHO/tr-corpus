"""Synthetic checks of the D7 audit helpers (no raw Gardner files needed)."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "trbench"), os.path.join(HERE, "..", "trbench", "adapters")]
import external_d7 as X  # noqa: E402


class RtdLoss(unittest.TestCase):
    def test_terminal_stuck_run_is_loss(self):
        t = np.arange(100.0)
        T = np.linspace(25, 80, 100)
        T[80:] = 883.0
        self.assertEqual(X.rtd_lost_at(t, T), 80.0)

    def test_short_terminal_repeat_is_not_loss(self):
        t = np.arange(100.0)
        T = np.linspace(25, 80, 100)
        T[95:] = T[95]
        self.assertIsNone(X.rtd_lost_at(t, T))

    def test_terminal_nan_is_loss(self):
        t = np.arange(50.0)
        T = np.linspace(25, 60, 50)
        T[40:] = np.nan
        self.assertEqual(X.rtd_lost_at(t, T), 40.0)


class H2Onset(unittest.TestCase):
    def test_threshold_floor_and_persistence(self):
        t = np.arange(2000.0)
        h = np.zeros_like(t)
        h[900:903] = 50.0            # 3 s spike: shorter than the 5 s hold
        h[1200:] = 12.0              # sustained above the 10 ppm floor
        on, m, sigma, thr = X.h2_onset(t, h)
        self.assertEqual(on, 1200.0)
        self.assertEqual(thr, 10.0)

    def test_noisy_baseline_raises_threshold(self):
        rng = np.random.default_rng(0)
        t = np.arange(2000.0)
        h = rng.normal(0, 4.0, len(t))
        h[1500:] += 15.0             # 15 ppm step is inside 5 sigma of a 4 ppm baseline
        on, _, sigma, thr = X.h2_onset(t, h)
        self.assertGreater(thr, 15.0)
        self.assertGreater(sigma, 3.0)


if __name__ == "__main__":
    unittest.main()
