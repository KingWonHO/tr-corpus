"""Fixed physical-threshold alarms under the paper's evaluation (external review of 2026-10-03, experiment E2).

No fitting and no calibration: an alarm is raised at the first 10 s window
endpoint at which the causally reconstructed maximum surface temperature
reaches a fixed threshold.  60 degC is a typical maximum operating temperature
and the internal-temperature criterion of Wang et al. (2026); 80 degC is near
the onset of SEI decomposition.  Both are illustrative fixed rules, not tuned
detectors.  They are scored exactly like the learned models: the 77 paired
internal positives against every reference event (pre-trigger alarms fail),
the 25 calibration and 27 check negatives, and the 43 D6 positives and 13 D6
negatives, with full-record and first-300 s false alarms.

With ``--calibrated`` the same temperature level is used as a risk score and
its threshold is set like every learned model's: smallest threshold with at
most 10 % empirical FAR on the 25 calibration negatives.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalv2 as V          # noqa: E402
import run_e3 as R          # noqa: E402
import run_validation as RV  # noqa: E402
import survival as SV        # noqa: E402

THRESHOLDS = (60.0, 80.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="W60_native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrated", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    reg = pd.read_csv(os.path.join(R.OUT, "registry", "experiments.csv"))
    d, meta = R.load_windows(a.windows, splits=("train", "val", "test", "test_zeroshot"))
    j = list(d["features"]).index("T_surface_max")
    # float64: a float32 array would round the nextafter threshold back onto a tied peak
    value = np.where(d["mask"][:, -1, j] > 0, d["X"][:, -1, j], np.nan).astype(np.float64)
    plan = V.Plan(d, meta, reg, ["M1", "M2"])
    target = np.char.startswith(plan.ds, V.TARGET + "/")
    arc = np.flatnonzero(target & plan.evaluable)
    arc_neg = arc[d["y_tr"][arc] == 0]

    internal, positives, negatives = [], [], []
    if a.calibrated:
        level = np.nan_to_num(value, nan=-np.inf)
        tau = SV.calibrate_threshold(level[plan.cal_neg], d["experiment"][plan.cal_neg],
                                     np.ones(len(plan.cal_neg), bool), far=V.BUDGET)
        settings = [(float(tau), "T_surface_max >= %.1f C (calibrated, 10 %% on calibration negatives)" % tau)]
        print("calibrated level threshold: %.3f C" % tau)
    else:
        settings = [(t, "T_surface_max >= %g C" % t) for t in THRESHOLDS]
    for theta, name in settings:
        risk = np.nan_to_num(value >= theta).astype(float)
        # internal design: one alarm per held cell, re-scored against every reference event
        for held in plan.folds:
            te = np.flatnonzero((plan.ds == held) & plan.observed)
            order = np.argsort(d["t_end"][te], kind="mergesort")
            hit = np.flatnonzero(risk[te][order] >= 0.5)
            alarm = float(d["t_end"][te][order][hit[0]]) if len(hit) else None
            tg = plan.trig.get(held, np.nan)
            pre = alarm is not None and np.isfinite(tg) and alarm < tg
            for lab, onset in plan.onsets.items():
                on = onset.get(held, np.nan)
                if not np.isfinite(on):
                    continue
                det = bool(alarm is not None and alarm < on and not pre)
                internal.append(dict(model=name, held=held, label=lab, alarm_time=alarm, t_trigger=tg,
                                     pre_trigger=pre, onset=float(on), detected=det,
                                     lead=float(on - alarm) if det else np.nan))
        for pool, idx in [("D6", arc)]:
            pr = RV.positive_records(d, idx, risk[idx], 0.5, plan).assign(model=name, pool=pool)
            positives.append(pr)
        for pool, idx in [("calibration", plan.cal_neg), ("source check", plan.test_neg), ("D6", arc_neg)]:
            for seg, (h, anc) in {"full": (None, "tail"), "first 300 s": (300, "head"), "last 300 s": (300, "tail")}.items():
                nr = RV.negative_records(d, idx, risk[idx], 0.5, h, anc)
                negatives.append(nr.assign(model=name, pool=pool, segment=seg))
    pd.DataFrame(internal).to_csv(os.path.join(a.out, "internal_rows.csv"), index=False)
    pd.concat(positives).to_csv(os.path.join(a.out, "d6_positive_records.csv"), index=False)
    pd.concat(negatives).to_csv(os.path.join(a.out, "negative_records.csv"), index=False)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
