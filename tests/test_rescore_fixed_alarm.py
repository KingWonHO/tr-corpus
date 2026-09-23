"""Regression tests for event-reference-only fixed-alarm re-scoring."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trbench"))
import rescore_fixed_alarm as R  # noqa: E402


class FixedAlarmRescoreTests(unittest.TestCase):
    @staticmethod
    def registry(l3=100.0):
        return pd.DataFrame([{
            "dataset_id": "d", "experiment_id": "e",
            "t_onset_L2_1.0": 90.0, "t_onset_L2_0.5": 80.0,
            "t_onset_L2_2.0": 95.0, "t_onset_L1": 92.0,
            "t_onset_L3": l3, "t_isc": 70.0,
        }])

    @staticmethod
    def rows():
        rows = []
        old = {"L2@1.0": 90.0, "L2@0.5": 80.0, "L2@2.0": 95.0,
               "L1 venting": 92.0, "L3 voltage": 40.0, "ISC 25mV": 70.0}
        for label, onset in old.items():
            detected = 50.0 < onset
            rows.append({
                "model": "xgboost", "panel": "M1", "held": "d/e",
                "label": label, "seed": 0, "tau": 0.9,
                "alarm_time": 50.0, "t_trigger": 0.0,
                "pre_trigger": False, "onset": onset,
                "lead": onset - 50.0 if detected else np.nan,
                "detected": detected, "far_check": 0.1, "risk_max": 0.95,
            })
        return pd.DataFrame(rows)

    def test_relabel_can_flip_status_without_moving_alarm_or_threshold(self):
        before = self.rows()
        after, audit = R.rescore(before, self.registry(l3=100.0))
        row = after[after.label == "L3 voltage"].iloc[0]

        self.assertTrue(row.detected)
        self.assertEqual(row.lead, 50.0)
        self.assertEqual(row.alarm_time, 50.0)
        self.assertEqual(row.tau, 0.9)
        self.assertEqual(row.far_check, 0.1)
        self.assertEqual(len(audit), 1)

    def test_missing_reference_drops_only_that_label_row(self):
        before = self.rows()
        after, _ = R.rescore(before, self.registry(l3=np.nan))

        self.assertNotIn("L3 voltage", set(after.label))
        self.assertEqual(len(after), len(before) - 1)
        self.assertEqual(after.alarm_time.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
