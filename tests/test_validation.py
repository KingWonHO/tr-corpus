import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trbench"))
import survival as SV
from run_validation import negative_records, rate_summary, representation_mask
import make_windows as MW
import evalv2 as V


class CalibrationTests(unittest.TestCase):
    def test_ties_do_not_exceed_budget(self):
        peaks = np.array([.9, .8, .8, .7, .6, .5, .4, .3, .2, .1])
        e = np.arange(len(peaks)).astype(str)
        for budget in (0., .1, .2, .5, 1.):
            tau = SV.calibrate_threshold(peaks, e, np.ones(len(e), bool), budget)
            self.assertLessEqual(np.mean(peaks >= tau), budget)
            if budget < 1:
                self.assertGreater(np.mean(peaks >= np.nextafter(tau, -np.inf)), budget)

    def test_positive_scores_cannot_change_threshold(self):
        e = np.array(["n1", "n2", "positive"])
        mask = np.array([True, True, False])
        a = SV.calibrate_threshold([.2, .4, .1], e, mask)
        b = SV.calibrate_threshold([.2, .4, 999.], e, mask)
        self.assertEqual(a, b)

    def test_float32_tied_scores_respect_float64_threshold(self):
        risk = np.array([.8, .8, .8], dtype=np.float32)
        exp = np.array(["n1", "n2", "n3"])
        tau = SV.calibrate_threshold(risk, exp, np.ones(3, bool), .1)
        self.assertIsNone(SV.first_alarm([0, 1, 2], risk, tau))
        self.assertEqual(SV.false_alarm_rate(risk, exp, np.ones(3, bool), tau), 0.)

    def test_short_record_is_excluded_not_counted_silent(self):
        d = dict(experiment=np.array(["short"]*3 + ["long"]*4),
                 t_end=np.array([0, 100, 200, 0, 100, 200, 300.]))
        r = np.array([0, 0, 0, 0, 0, 0, 1.])
        nr = negative_records(d, np.arange(7), r, .5, 300)
        s = rate_summary(nr)
        self.assertEqual((s["n_eligible"], s["n_excluded"], s["far"]), (1, 1, 1.))

    def test_head_and_tail_are_distinct_prespecified_windows(self):
        d = dict(experiment=np.array(["a"]*5), t_end=np.array([0, 100, 200, 300, 400.]))
        r = np.array([1, 0, 0, 0, 0.])
        h = negative_records(d, np.arange(5), r, .5, 300, "head")
        t = negative_records(d, np.arange(5), r, .5, 300, "tail")
        self.assertTrue(h.iloc[0].fired)
        self.assertFalse(t.iloc[0].fired)
        self.assertTrue(np.isnan(t.iloc[0].first_alarm_s))
        self.assertEqual(t.iloc[0].followup_s, 300)

    def test_hidden_sensor_age_cannot_enter_m1(self):
        f = ["T_surface_mid", "age__T_surface_mid", "gas_H2", "age__gas_H2"]
        self.assertEqual(MW.panel_mask(f, "M1").tolist(), [True, True, False, False])

    def test_common_surface_controls_exclude_channel_identity_and_clocks(self):
        f = ["T_surface_mid", "T_surface_max", "T_surface_mean",
             "T_ambient", "age__T_surface_max"]
        self.assertEqual(representation_mask(f, "surface_max_only").tolist(),
                         [False, True, False, False, False])
        self.assertEqual(representation_mask(f, "surface_mean_only").tolist(),
                         [False, False, True, False, False])


class SplitAndAlarmTests(unittest.TestCase):
    def setUp(self):
        keys = ["src/train_pos", "src/train_neg", "src/val_pos", "src/cal_neg",
                "src/test_pos", "src/check_neg", "ds12_arc/target"]
        self.d = dict(experiment=np.array(keys), y_tr=np.array([1, 0, 1, 0, 1, 0, 1]),
                      t_end=np.full(7, 50.), features=["T_surface_mid"])
        self.reg = pd.DataFrame(dict(dataset_id=[k.split("/")[0] for k in keys],
                     experiment_id=[k.split("/")[1] for k in keys],
                     split=["train", "train", "val", "val", "test", "test", "test_zeroshot"],
                     t_onset_L2_1_0=[100., np.nan, 100., np.nan, 100., np.nan, 100.]))
        self.reg["t_onset_L2_1.0"] = self.reg.pop("t_onset_L2_1_0")
        self.reg["t_onset_L2_0.5"] = [80., np.nan, 80., np.nan, 80., np.nan, 80.]
        self.meta = pd.DataFrame(dict(key=keys, t_onset=self.reg["t_onset_L2_1.0"],
                                     t_trigger=np.zeros(7)))
        self.panel = pd.DataFrame({"file": ["timeseries/" + k + ".parquet" for k in keys]})

    def test_external_train_uses_only_saved_train_and_loeo_excludes_held(self):
        with patch.object(MW, "panel_experiments", return_value=self.panel):
            plan = V.Plan(self.d, self.meta, self.reg, ["M1"])
        self.assertEqual(set(plan.ds[plan.frozen_train]), {"src/train_pos", "src/train_neg"})
        self.assertIn("src/test_pos", plan.ds[plan.pool])
        self.assertNotIn("src/test_pos", plan.ds[plan.train_idx("src/test_pos")])
        self.assertEqual(set(plan.ds[plan.cal_neg]), {"src/cal_neg"})
        self.assertEqual(set(plan.ds[plan.test_neg]), {"src/check_neg"})

    def test_changing_label_cannot_change_fitted_alarm_or_threshold(self):
        calls = []
        def run_fold(d, cols, model, train_idx, eval_idx, seed):
            calls.append((set(d["experiment"][train_idx]), d["experiment"][eval_idx[0]][0]))
            return [np.array([.9]), np.array([.2]), np.array([.1])]
        with patch.object(MW, "panel_experiments", return_value=self.panel), \
             patch.object(V.R, "run_fold", side_effect=run_fold):
            result = V.crossfit_loeo(self.d, self.meta, self.reg, ["test"], ["M1"], verbose=False)
        self.assertEqual(len(calls), 3)
        for train_keys, held in calls:
            self.assertNotIn(held, train_keys)
            self.assertFalse(train_keys & {"src/cal_neg", "src/check_neg", "ds12_arc/target"})
        for _, rows in result.groupby("held"):
            self.assertEqual(rows.tau.nunique(), 1)
            self.assertEqual(rows.alarm_time.nunique(), 1)
            self.assertEqual(set(rows.lead), {30., 50.})

    def test_unobserved_tail_cannot_enter_calibration_or_test_cost(self):
        for key in ("experiment", "y_tr", "t_end"):
            self.d[key] = np.r_[self.d[key], self.d[key][[3, 5]]]
        self.d["label_observed"] = np.r_[np.ones(7, bool), False, False]
        with patch.object(MW, "panel_experiments", return_value=self.panel):
            plan = V.Plan(self.d, self.meta, self.reg, ["M1"])
        self.assertEqual(plan.cal_neg.tolist(), [3])
        self.assertEqual(plan.test_neg.tolist(), [5])


if __name__ == "__main__":
    unittest.main()
