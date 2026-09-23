from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trbench"))
import summarize_v2 as S  # noqa: E402


def _synthetic_rows(include_seed: bool = False) -> pd.DataFrame:
    held = [f"ds{1 if i < 3 else 2:02d}/e{i + 1}" for i in range(6)]
    l2 = {
        "M1": [True, True, False, False, True, False],
        "M2": [True, False, True, False, True, True],
    }
    l2_lead = {
        "M1": [10.0, 12.0, np.nan, np.nan, 30.0, np.nan],
        "M2": [20.0, np.nan, 15.0, np.nan, 25.0, 8.0],
    }
    other = {
        "L1 venting": {
            "M1": [True, False, False, False, True, False],
            "M2": [True, False, True, True, True, True],
        },
        "L3 voltage": {
            "M1": [True, True, True, False, True, False],
            "M2": [False, False, True, False, True, True],
        },
    }
    rows = []
    for panel in ["M1", "M2"]:
        for i, cell in enumerate(held):
            common = {
                "model": "xgboost",
                "panel": panel,
                "held": cell,
                "tau": 0.5,
            }
            if include_seed:
                common["seed"] = 0
            rows.append(
                {
                    **common,
                    "label": S.DEFAULT_L2,
                    "detected": l2[panel][i],
                    "lead": l2_lead[panel][i],
                }
            )
            for label, values in other.items():
                detected = values[panel][i]
                rows.append(
                    {
                        **common,
                        "label": label,
                        "detected": detected,
                        "lead": float(40 + i) if detected else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _write_registry(path: Path) -> None:
    pd.DataFrame(
        {
            "dataset_id": ["ds01"] * 3 + ["ds02"] * 3,
            "experiment_id": [f"e{i + 1}" for i in range(6)],
            "trigger": ["heat"] * 3 + ["nail"] * 3,
        }
    ).to_csv(path, index=False)


class SummarizeV2Tests(unittest.TestCase):
    def test_wilson_and_exact_mcnemar(self) -> None:
        low, high = S.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.236593, places=6)
        self.assertAlmostEqual(high, 0.763407, places=6)
        self.assertAlmostEqual(S.exact_mcnemar_p(0, 6), 0.03125)
        self.assertEqual(S.exact_mcnemar_p(2, 2), 1.0)
        self.assertEqual(S.exact_mcnemar_p(0, 0), 1.0)

    def test_load_v2_deduplicates_identical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            frame = _synthetic_rows()
            first, second = tmp / "first.csv", tmp / "second.csv"
            frame.to_csv(first, index=False)
            frame.iloc[:4].to_csv(second, index=False)
            loaded, accounting = S.load_v2([first, second])
            self.assertEqual(len(loaded), len(frame))
            self.assertEqual(accounting["input_rows"], len(frame) + 4)
            self.assertEqual(accounting["identical_duplicate_keys"], 4)
            self.assertEqual(accounting["identical_duplicate_rows_removed"], 4)
            self.assertEqual(accounting["unique_held_experiments"], 6)

    def test_load_v2_rejects_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            frame = _synthetic_rows()
            first, second = tmp / "first.csv", tmp / "second.csv"
            frame.to_csv(first, index=False)
            conflict = frame.iloc[[0]].copy()
            conflict.loc[:, "detected"] = False
            conflict.loc[:, "lead"] = np.nan
            conflict.to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting duplicate.*detected"):
                S.load_v2([first, second])

    def test_load_v2_treats_missing_result_as_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            frame = _synthetic_rows()
            first, second = tmp / "first.csv", tmp / "second.csv"
            frame.to_csv(first, index=False)
            conflict = frame.iloc[[0]].copy()
            conflict.loc[:, "tau"] = np.nan
            conflict.to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting duplicate.*tau"):
                S.load_v2([first, second])

    def test_seed_is_a_key_and_summary_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            seed0 = _synthetic_rows(include_seed=True)
            seed1 = seed0.copy()
            seed1["seed"] = 1
            source = tmp / "seeds.csv"
            pd.concat([seed0, seed1], ignore_index=True).to_csv(source, index=False)
            loaded, accounting = S.load_v2([source])
            l2 = S.summarize_l2_detection(loaded)
            self.assertEqual(len(loaded), 2 * len(seed0))
            self.assertEqual(accounting["model_seeds"], [0, 1])
            self.assertEqual(set(l2["seed"]), {0, 1})
            self.assertEqual(len(l2.query("scope == 'overall'")), 4)

    def test_mixing_seeded_and_unseeded_inputs_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            seeded, old = tmp / "seeded.csv", tmp / "old.csv"
            _synthetic_rows(include_seed=True).to_csv(seeded, index=False)
            _synthetic_rows().to_csv(old, index=False)
            with self.assertRaisesRegex(ValueError, "with and without a seed"):
                S.load_v2([seeded, old])

    def test_panel_pair_statistics_resample_held_experiments(self) -> None:
        result = S.summarize_panel_pairs(
            _synthetic_rows(), n_bootstrap=2000, seed=17
        )
        row = result.query("scope == 'overall' and panel == 'M2'").iloc[0]
        self.assertEqual(row["n_unique_cells"], 6)
        self.assertEqual(row["n_paired_oof_evaluations"], 12)
        self.assertEqual(row["baseline_detected"], 3)
        self.assertEqual(row["panel_detected"], 4)
        self.assertEqual(row["rescue"], 2)
        self.assertEqual(row["lost"], 1)
        self.assertEqual(row["mcnemar_exact_p"], 1.0)
        self.assertAlmostEqual(row["detection_rate_difference"], 1 / 6)
        self.assertLessEqual(row["detection_diff_ci_low"], 1 / 6)
        self.assertGreaterEqual(row["detection_diff_ci_high"], 1 / 6)
        self.assertEqual(row["n_detected_by_both"], 2)
        self.assertAlmostEqual(row["conditional_lead_delta_median_s"], 2.5)
        self.assertLessEqual(row["conditional_lead_delta_ci_low_s"], 2.5)
        self.assertGreaterEqual(row["conditional_lead_delta_ci_high_s"], 2.5)
        self.assertEqual(row["bootstrap_unit"], "held experiment")

    def test_missing_baseline_panel_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "baseline panel"):
            S.summarize_panel_pairs(
                _synthetic_rows(), baseline_panel="DOES_NOT_EXIST", n_bootstrap=10
            )

    def test_label_sensitivity_uses_common_model_panel_cohorts(self) -> None:
        source = _synthetic_rows()
        pairwise, all_labels = S.summarize_label_sensitivity(source)
        accounting = S.summarize_common_cohort_accounting(source).iloc[0]
        pair = pairwise.query(
            "scope == 'overall' and panel == 'M1' "
            "and comparison_label == 'L1 venting'"
        ).iloc[0]
        all_row = all_labels.query("scope == 'overall' and panel == 'M1'").iloc[0]
        self.assertEqual(pair["n_unique_cells"], 6)
        self.assertEqual(pair["n_label_evaluations"], 12)
        self.assertEqual(pair["n_flip"], 1)
        self.assertEqual(pair["l2_only_detected"], 1)
        self.assertEqual(pair["comparison_only_detected"], 0)
        self.assertEqual(all_row["n_unique_cells"], 6)
        self.assertEqual(all_row["n_labels"], 3)
        self.assertEqual(all_row["n_label_evaluations"], 18)
        self.assertEqual(all_row["n_cells_with_any_detection_flip"], 2)
        self.assertEqual(accounting["n_unique_cells"], 6)
        self.assertEqual(accounting["n_model_panel_seed_cell_evaluations"], 12)
        self.assertEqual(accounting["n_label_evaluations"], 36)

    def test_analysis_adds_registry_strata_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            source = tmp / "v2.csv"
            registry = tmp / "experiments.csv"
            out = tmp / "summary"
            _synthetic_rows().to_csv(source, index=False)
            _write_registry(registry)
            tables, manifest = S.analyze(
                [source],
                registry=registry,
                strata=["dataset", "trigger"],
                n_bootstrap=250,
                seed=9,
            )
            S.write_analysis(out, tables, manifest)
            l2 = tables["l2_detection"]
            m1 = l2.query("model == 'xgboost' and panel == 'M1'")
            self.assertEqual(set(m1["scope"]), {"overall", "stratified"})
            self.assertEqual(
                set(m1.loc[m1["scope"] == "stratified", "dataset"]),
                {"ds01", "ds02"},
            )
            self.assertEqual(manifest["accounting"]["unique_held_experiments"], 6)
            self.assertEqual(
                manifest["accounting"]["model_panel_held_evaluations"], 12
            )
            expected = {
                "l2_detection.csv",
                "panel_pairs.csv",
                "label_sensitivity.csv",
                "all_label_cohort.csv",
                "cohort_accounting.csv",
                "summary_manifest.json",
                "README.md",
            }
            self.assertEqual({path.name for path in out.iterdir()}, expected)
            saved = json.loads(
                (out / "summary_manifest.json").read_text(encoding="utf-8")
            )
            caveats = " ".join(saved["caveats"]).lower()
            self.assertIn("held experiment", caveats)
            self.assertIn("does not establish equivalence", caveats)
            self.assertIn("model-fitting", caveats)

    def test_bootstrap_is_reproducible_and_group_stable(self) -> None:
        frame = _synthetic_rows()
        first = S.summarize_panel_pairs(frame, n_bootstrap=500, seed=123)
        second = S.summarize_panel_pairs(
            frame.sample(frac=1, random_state=44), n_bootstrap=500, seed=123
        )
        columns = [
            "detection_diff_ci_low",
            "detection_diff_ci_high",
            "conditional_lead_delta_ci_low_s",
            "conditional_lead_delta_ci_high_s",
        ]
        self.assertTrue(np.allclose(first[columns], second[columns], equal_nan=True))

    def test_analysis_requires_registry_for_stratification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "v2.csv"
            _synthetic_rows().to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "registry"):
                S.analyze([source], strata=["trigger"])

    def test_one_experiment_bootstrap_has_no_degenerate_ci(self) -> None:
        point, low, high = S.bootstrap_interval([0.5], "mean", 100, 0.95, 1)
        self.assertEqual(point, 0.5)
        self.assertTrue(math.isnan(low))
        self.assertTrue(math.isnan(high))


if __name__ == "__main__":
    unittest.main()
