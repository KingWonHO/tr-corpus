import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trbench"))
import gas_control


class GasControlTests(unittest.TestCase):
    def test_only_gas_changes_with_identical_times_and_other_features(self):
        d = dict(window_dir="W60_native", features=["T_surface_mid", "gas_total_mol"],
                 X=np.full((8, 3, 2), 99., dtype=np.float32),
                 mask=np.ones((8, 3, 2), dtype=np.uint8),
                 experiment=np.array(["ds01_bak/TS0330" + c for c in "ABCDEFGH"]),
                 t_end=np.full(8, 2.))
        before = {k: v.copy() for k, v in d.items() if isinstance(v, np.ndarray)}
        frame = pd.DataFrame({"time": [0, 3000], "n": [0, 3000]})
        with patch.object(gas_control.pd, "read_excel", return_value=frame):
            gas_control.apply(d)
        np.testing.assert_array_equal(d["X"][:, :, 0], before["X"][:, :, 0])
        np.testing.assert_array_equal(d["X"][:, :, 1], np.tile([0, 1, 2], (8, 1)))
        for key in ("mask", "experiment", "t_end"):
            np.testing.assert_array_equal(d[key], before[key])

    def test_clock_features_rejected_to_keep_control_interpretable(self):
        d = dict(window_dir="W60_native", features=["gas_total_mol", "age__gas_total_mol"])
        with self.assertRaisesRegex(ValueError, "without-age"):
            gas_control.apply(d)


if __name__ == "__main__":
    unittest.main()
