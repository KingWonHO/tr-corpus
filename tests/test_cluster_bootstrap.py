import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trbench"))
import cluster_bootstrap as CB


class ClusterBootstrapTests(unittest.TestCase):
    def test_singleton_clusters_reduce_to_experiment_bootstrap(self):
        rng = np.random.default_rng(0)
        k = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        lo, hi, g = CB.cluster_ratio_ci(k, np.ones(10), np.arange(10), rng, n_boot=4000)
        self.assertEqual(g, 10)
        self.assertLessEqual(lo, 0.3)
        self.assertGreaterEqual(hi, 0.3)

    def test_replicates_widen_the_interval(self):
        # same 12 outcomes, once as independent cells and once as 3 designs x 4 replicates
        k = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], float)
        a = CB.cluster_ratio_ci(k, np.ones(12), np.arange(12), np.random.default_rng(1), n_boot=4000)
        b = CB.cluster_ratio_ci(k, np.ones(12), np.repeat([0, 1, 2], 4), np.random.default_rng(1), n_boot=4000)
        self.assertGreater(b[1] - b[0], a[1] - a[0])
        self.assertEqual(b[2], 3)

    def test_constant_outcome_has_degenerate_interval(self):
        lo, hi, _ = CB.cluster_ratio_ci(np.zeros(8), np.ones(8), np.repeat([0, 1], 4),
                                        np.random.default_rng(2), n_boot=500)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_cell_model_rule_groups_replicates(self):
        row = dict(dataset_id="ds12_arc", experiment_id="21700__LFP_SOC30_M2", cell_type="LFP_SOC30_M2",
                   chemistry="LFP", capacity_ah=3.0, cell_source=np.nan)
        other = dict(row, experiment_id="21700__LFP_SOC0_M1", cell_type="LFP_SOC0_M1")
        self.assertEqual(CB.cell_model(row), CB.cell_model(other))
        lco = dict(dataset_id="ds09_mech", experiment_id="OE-LCO-6270mAh-40SOC", cell_type="x",
                   chemistry="LCO", capacity_ah=6.27, cell_source=np.nan)
        self.assertEqual(CB.cell_model(lco), "D5:LCO 6.4 Ah")


if __name__ == "__main__":
    unittest.main()
