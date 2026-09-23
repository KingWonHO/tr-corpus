"""Regression tests for physical guards on event-reference labels."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trbench"))
import common as C  # noqa: E402


class PressureOnsetTests(unittest.TestCase):
    def test_in_cell_pressure_ignores_impossible_post_vent_rail(self):
        t = np.arange(0.0, 201.0)
        p = np.full(t.shape, 10.0)
        p[20:51] = np.linspace(10.0, 1000.0, 31)
        p[51:54] = [500.0, 100.0, 10.0]
        p[150:] = -2500.0

        onset = C.onset_L1(t, p, t_trigger=5.0, site="in_cell")

        self.assertGreaterEqual(onset, 50.0)
        self.assertLessEqual(onset, 53.0)

    def test_chamber_pressure_is_not_subject_to_gauge_floor(self):
        t = np.arange(0.0, 101.0)
        p = np.full(t.shape, -200.0)
        p[50:] += 100.0

        # Chamber records may use an arbitrary offset; the in-cell vacuum
        # bound must not be applied to them.
        self.assertIsNotNone(C.onset_L1(t, p, site="chamber"))


class VoltageOnsetTests(unittest.TestCase):
    @staticmethod
    def dropout_then_terminal_collapse():
        t = np.arange(0.0, 201.0)
        v = np.full(t.shape, 4.0)
        v[20:90] = 0.1       # longer than either persistence requirement
        v[90:150] = 4.0      # full recovery proves that it was a dropout
        v[150:] = 0.0        # irreversible electrical collapse
        return t, v

    def test_l3_skips_long_dropout_that_later_recovers(self):
        t, v = self.dropout_then_terminal_collapse()
        self.assertEqual(C.onset_L3(t, v, t_trigger=5.0), 150.0)

    def test_isc_skips_long_dropout_that_later_recovers(self):
        t, v = self.dropout_then_terminal_collapse()
        self.assertEqual(C.onset_isc(t, v, t_trigger=5.0), 150.0)

    def test_reversible_nonzero_l3_excursion_keeps_first_crossing(self):
        t = np.arange(0.0, 101.0)
        v = np.full(t.shape, 4.0)
        v[20:35] = 2.5

        # A physical voltage excursion may recover.  It is still the first
        # sustained 20% crossing and does not match the near-zero dropout guard.
        self.assertEqual(C.onset_L3(t, v, t_trigger=5.0), 20.0)

    def test_reversible_25mv_excursion_keeps_published_first_crossing(self):
        t = np.arange(0.0, 101.0)
        v = np.full(t.shape, 4.0)
        v[20:35] = 3.95

        self.assertEqual(C.onset_isc(t, v, t_trigger=5.0), 20.0)

    def test_terminal_gradual_decline_keeps_first_threshold_crossing(self):
        t = np.arange(0.0, 201.0)
        v = np.full(t.shape, 4.0)
        v[100:] = np.linspace(3.99, 0.0, 101)

        expected_l3 = float(t[np.flatnonzero(v < 3.2)[0]])
        expected_isc = float(t[np.flatnonzero(v < 3.975)[0]])
        self.assertEqual(C.onset_L3(t, v, t_trigger=5.0), expected_l3)
        self.assertEqual(C.onset_isc(t, v, t_trigger=5.0), expected_isc)


if __name__ == "__main__":
    unittest.main()
