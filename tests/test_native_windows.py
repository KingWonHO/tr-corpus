"""Focused invariance tests for the native causal window builder."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trbench"))

import native_windows as N  # noqa: E402


class CausalChannelTests(unittest.TestCase):
    def test_trailing_bin_is_right_closed_and_left_open(self):
        t = np.array([0.0, 0.25, 1.0, 1.25, 2.0])
        v = np.array([100.0, 2.0, 4.0, 8.0, 16.0])
        grid = np.array([1.0, 2.0])

        out, mask, age = N.causal_channel(t, v, grid)

        # g=1 uses (0,1]: t=0 is excluded; g=2 uses (1,2]: t=1 is excluded.
        np.testing.assert_allclose(out, [3.0, 12.0])
        np.testing.assert_array_equal(mask, [True, True])
        np.testing.assert_allclose(age, [0.0, 0.0])

    def test_empty_bin_holds_last_native_value_and_reports_age(self):
        out, mask, age = N.causal_channel(
            [2.0, 5.0], [20.0, 50.0], [0.0, 2.0, 3.0, 4.0, 5.0, 7.0])

        np.testing.assert_allclose(
            out, [np.nan, 20.0, 20.0, 20.0, 50.0, 50.0], equal_nan=True)
        np.testing.assert_array_equal(mask, [False, True, True, True, True, True])
        np.testing.assert_allclose(
            age, [np.nan, 0.0, 1.0, 2.0, 0.0, 2.0], equal_nan=True)

    def test_future_value_and_schedule_changes_cannot_change_prefix(self):
        grid_short = np.arange(0.0, 6.0)
        base_t = np.array([0.0, 0.2, 0.8, 2.4, 5.0, 7.0])
        base_v = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        a = N.causal_channel(base_t, base_v, grid_short)

        # After t=5 both the future values and sampling schedule are unrelated
        # to the base record.  A full-record median-rate switch would fail this.
        changed_t = np.r_[base_t[:5], 5.1, 5.2, 5.3, 20.0]
        changed_v = np.r_[base_v[:5], -1000.0, 9000.0, 3.0, 1e9]
        b = N.causal_channel(changed_t, changed_v, grid_short)

        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, equal_nan=True)

    def test_nonfinite_and_duplicate_samples_do_not_leak(self):
        out, mask, age = N.causal_channel(
            [0.0, 1.0, 1.0, 2.0, np.nan, 4.0],
            [1.0, 2.0, 999.0, np.nan, 8.0, 4.0],
            [1.0, 2.0, 3.0, 4.0])
        # The first duplicate matches common._sorted_unique semantics.
        np.testing.assert_allclose(out, [2.0, 2.0, 2.0, 4.0])
        np.testing.assert_array_equal(mask, [True] * 4)
        np.testing.assert_allclose(age, [0.0, 1.0, 2.0, 0.0])


class AggregateTests(unittest.TestCase):
    def test_surface_aggregates_use_only_causal_members(self):
        values = {
            "T_surface_neg": np.array([20.0, 25.0, 30.0]),
            "T_surface_x_1": np.array([22.0, 21.0, 40.0]),
        }
        masks = {
            "T_surface_neg": np.array([1, 1, 1], bool),
            "T_surface_x_1": np.array([0, 1, 1], bool),
        }
        ages = {
            "T_surface_neg": np.array([0.0, 0.0, 1.0]),
            "T_surface_x_1": np.array([np.nan, 3.0, 2.0]),
        }

        N.causal_surface_aggregates(values, masks, ages)

        np.testing.assert_allclose(values["T_surface_max"], [20.0, 25.0, 40.0])
        np.testing.assert_allclose(values["T_surface_mean"], [20.0, 23.0, 35.0])
        np.testing.assert_allclose(ages["T_surface_max"], [0.0, 3.0, 2.0])
        np.testing.assert_allclose(ages["T_surface_mean"], [0.0, 3.0, 2.0])
        np.testing.assert_array_equal(masks["T_surface_max"], [True] * 3)


class RecordAndStorageTests(unittest.TestCase):
    def test_record_keeps_full_tail_and_separate_channel_ages(self):
        raw = SimpleNamespace(
            dataset_id="synthetic", experiment_id="one",
            series={
                "T_surface_mid": ([100.0, 102.0, 110.0], [20.0, 22.0, 40.0]),
                "V_cell": ([100.0, 110.0], [4.2, 3.0]),
            })

        grid, x, mask, origin, channels, digest = N.causal_record(raw)

        self.assertEqual(origin, 100.0)
        np.testing.assert_allclose(grid, np.arange(11.0))
        self.assertEqual(x.shape, (11, len(N.OUTPUT_FEATURES)))
        self.assertIn("T_surface_mid", channels)
        self.assertEqual(len(digest), 64)
        t = N.OUTPUT_FEATURES.index("T_surface_mid")
        t_age = N.OUTPUT_FEATURES.index("age__T_surface_mid")
        v_age = N.OUTPUT_FEATURES.index("age__V_cell")
        self.assertEqual(x[-1, t], 40.0)       # the post-onset/full tail remains
        self.assertEqual(x[5, t_age], 3.0)    # last T observation was at t=2
        self.assertEqual(x[5, v_age], 5.0)    # voltage has its own clock
        self.assertTrue(mask[:, t_age].all())

    def test_record_prefix_keeps_a_single_first_observation(self):
        clock = ([0.0, 5.0], [4.2, 4.1])
        prefix = SimpleNamespace(
            dataset_id="synthetic", experiment_id="prefix",
            series={"V_cell": clock, "T_surface_mid": ([0.0], [20.0])})
        full = SimpleNamespace(
            dataset_id="synthetic", experiment_id="full",
            series={"V_cell": clock,
                    "T_surface_mid": ([0.0, 4.0], [20.0, 1000.0])})

        gp, xp, mp, *_ = N.causal_record(prefix)
        gf, xf, mf, *_ = N.causal_record(full)
        temp = N.OUTPUT_FEATURES.index("T_surface_mid")
        age = N.OUTPUT_FEATURES.index("age__T_surface_mid")

        np.testing.assert_allclose(gp, gf)
        np.testing.assert_allclose(xp[:4, [temp, age]], xf[:4, [temp, age]])
        np.testing.assert_array_equal(mp[:4, [temp, age]], mf[:4, [temp, age]])

    def test_raw_tail_is_retained_but_not_extended_as_negative_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            buf = N._allocate_split(stage, "test", 8, 2, 30, 30)
            raw = SimpleNamespace(
                dataset_id="synthetic", experiment_id="tail record",
                t_trigger=0.0)
            row = {
                "record_duration_s": 2.0, "t_trigger": 0.0,
                "t_onset_L2": np.nan, "trigger": "synthetic",
                "split": "test", "crop_start_s": np.nan,
                "duration_s": 2.0,
            }
            grid = np.arange(5.0)
            f = np.ones((5, len(N.OUTPUT_FEATURES)), np.float32)
            m = np.ones_like(f, np.uint8)

            N._write_record(
                buf, row, raw, grid, f, m, 0.0, ("V_cell",), "0" * 64,
                window_s=2, stride_s=1, chunk_windows=2)

            self.assertEqual(buf["cursor"], 4)  # all of the raw tail is stored
            np.testing.assert_array_equal(
                buf["arrays"]["label_observed"][:4], [1, 1, 0, 0])
            np.testing.assert_allclose(
                buf["arrays"]["y_time"][:4], [1.0, 0.0, -1.0, -2.0])
            N._close_buffers({"test": buf})

    def test_memmap_split_round_trip_is_load_windows_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stage = out / "stage"
            stage.mkdir()
            buf = N._allocate_split(stage, "test", 2, 2, 20, 20)
            arrays = buf["arrays"]
            arrays["X"][:2] = 1.25
            arrays["mask"][:2] = 1
            arrays["y_time"][:2] = [4.0, 3.0]
            arrays["y_event"][:2] = 1
            arrays["t_end"][:2] = [1.0, 2.0]
            arrays["trigger"][:2] = "synthetic"
            arrays["y_tr"][:2] = 1
            arrays["experiment"][:2] = "synthetic/one"
            buf["cursor"] = 2
            buf["meta"].append({"key": "synthetic/one"})

            summary = N._save_split(out, stage, "test", buf, 2, 1.0)

            self.assertEqual(summary["n_windows"], 2)
            with np.load(out / "test.npz", allow_pickle=True) as z:
                required = {"X", "mask", "y_time", "y_event", "y_tr",
                            "label_observed", "t_end", "experiment",
                            "trigger", "features"}
                self.assertTrue(required.issubset(z.files))
                self.assertEqual(z["X"].shape,
                                 (2, 2, len(N.OUTPUT_FEATURES)))
                np.testing.assert_allclose(z["X"], 1.25)


if __name__ == "__main__":
    unittest.main()
