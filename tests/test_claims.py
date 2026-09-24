"""Statistical helpers and verdict order of trbench/claims.py (protocol 12, section 5)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trbench"))
import claims as CL  # noqa: E402


class Bounds(unittest.TestCase):
    def test_zero_alarm_upper_matches_protocol(self):
        self.assertAlmostEqual(CL.zero_alarm_upper(25), 0.11293, places=5)
        self.assertAlmostEqual(CL.zero_alarm_upper(26), 0.10883, places=5)

    def test_n_min(self):
        self.assertEqual(CL.n_min(0.10), 29)
        self.assertLessEqual(CL.zero_alarm_upper(29), 0.10)
        self.assertGreater(CL.zero_alarm_upper(28), 0.10)

    def test_clopper_pearson_zero_equals_closed_form(self):
        self.assertAlmostEqual(CL.cp_upper(0, 40), CL.zero_alarm_upper(40), places=10)
        self.assertEqual(CL.cp_upper(5, 5), 1.0)


class Verdict(unittest.TestCase):
    """Amended rule (2026-09-24): symmetric upper/lower one-sided bounds."""

    def test_order(self):
        self.assertEqual(CL.verdict(0, 0), "not assessable")
        self.assertTrue(CL.verdict(0, 26).startswith("inconclusive (observed <="))
        self.assertTrue(CL.verdict(5, 21).startswith("inconclusive (observed >"))
        self.assertEqual(CL.verdict(21, 21), "budget exceeded")      # lower bound 0.87
        self.assertEqual(CL.verdict(10, 50), "budget exceeded")
        self.assertEqual(CL.verdict(0, 60), "supported")
        self.assertTrue(CL.verdict(4, 50).startswith("inconclusive (observed <="))
        # a small pool can exceed the budget but never support it
        self.assertEqual(CL.verdict(6, 21), "budget exceeded")
        self.assertTrue(CL.verdict(0, 28).startswith("inconclusive"))
        self.assertGreater(CL.cp_lower(21, 21), 0.86)

    def test_original_rule_is_retained(self):
        self.assertTrue(CL.verdict_original(0, 26).startswith("insufficient evidence (observed <="))
        self.assertEqual(CL.verdict_original(10, 50), "observed fail")
        self.assertEqual(CL.verdict_original(0, 60), "supported")
        self.assertEqual(CL.verdict_original(4, 50), "within budget, not supported")


class HeadWindows(unittest.TestCase):
    def test_short_records_are_ineligible_not_truncated(self):
        exp = np.array(["a"] * 5 + ["b"] * 3)
        t = np.array([60, 160, 260, 360, 460, 60, 110, 160], float)
        keep = CL.head_windows(exp, t, 300)
        self.assertEqual(keep.tolist(), [True, True, True, True, False, False, False, False])


class RuleBehaviour(unittest.TestCase):
    """The verdict rules must discriminate; every real pool here stops at rule 2."""

    @classmethod
    def setUpClass(cls):
        cls.dm = CL.decision_map(n_max=60)
        cls.oc = CL.operating_characteristics()

    def test_small_pools_cannot_support_but_can_exceed(self):
        small = self.dm[self.dm.n < CL.n_min(0.10)]
        self.assertTrue(small.k_supported_max.isna().all())
        self.assertTrue((small.original_verdict_at_k0 == "insufficient evidence").all())
        self.assertTrue(small[small.n >= 4].k_exceeded_min.notna().all())
        self.assertEqual(self.dm[self.dm.n == 21].iloc[0].k_exceeded_min, 6)

    def test_all_verdicts_reachable_from_n_min(self):
        big = self.dm[self.dm.n >= CL.n_min(0.10)]
        for col in ("k_supported_max", "k_inconclusive_min", "k_exceeded_min"):
            self.assertTrue(big[col].notna().all())
        at29 = self.dm[self.dm.n == 29].iloc[0]
        self.assertEqual(at29.k_supported_max, 0)          # only zero alarms support the claim
        self.assertEqual(at29.original_k_observed_fail_min, 3)      # 3/29 > 0.10 under the original rule
        self.assertEqual(at29.k_exceeded_min, 7)           # lower bound above 0.10 needs 7/29

    def test_supported_is_never_likely_when_the_true_rate_equals_alpha(self):
        at_alpha = self.oc[self.oc.true_p == 0.10]
        self.assertTrue((at_alpha.p_supported <= 0.05 + 1e-12).all())
        self.assertTrue((self.oc[self.oc.n < 29].p_supported == 0).all())

    def test_power_rises_with_pool_size_when_the_true_rate_is_low(self):
        low = self.oc[self.oc.true_p == 0.02].sort_values("n")
        self.assertAlmostEqual(low[low.n == 29].p_supported.iloc[0], 0.557, places=2)
        self.assertGreater(low[low.n == 200].p_supported.iloc[0], 0.99)
        self.assertGreater(self.oc[(self.oc.true_p == 0.20) & (self.oc.n == 29)].original_p_observed_fail.iloc[0], 0.9)
        self.assertGreater(self.oc[(self.oc.true_p == 0.30) & (self.oc.n == 29)].p_exceeded.iloc[0], 0.8)

    def test_clustering_inflates_support_at_the_boundary(self):
        cs = CL.cluster_sensitivity()
        at_alpha = cs[(cs.true_p == 0.10) & (cs.n == 60) & (cs.cluster_size == 10)].sort_values("icc")
        from scipy.stats import binom
        exact = sum(binom.pmf(k, 60, 0.10) for k in range(61) if CL.verdict(k, 60) == "supported")
        self.assertAlmostEqual(at_alpha[at_alpha.icc == 0].p_supported.iloc[0], exact, places=10)
        self.assertTrue(at_alpha.p_supported.is_monotonic_increasing)
        self.assertGreater(at_alpha[at_alpha.icc == 0.50].p_supported.iloc[0], 0.05)


if __name__ == "__main__":
    unittest.main()
