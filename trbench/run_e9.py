"""E9: hold the model fixed, swap only the negatives.

Every false-alarm rate in this benchmark is a statement about whichever
non-runaway experiments happened to be available, and sections 13-E, 14-E and
16-C all ran into the same wall from different directions: the same threshold
costs a few percent on #9's mechanical negatives and a third to nine tenths on
#12's ARC negatives.  E9 isolates that one variable.

The model is fitted once, the positive risk traces are computed once, and the
ONLY thing that changes between arms is which non-runaway experiments price the
alarm.  Anything that moves is attributable to the negative pool and to nothing
else.

Two confounds have to be removed for the comparison to mean anything.

OBSERVATION TIME.  A false-alarm rate defined as "fraction of negative
experiments raising at least one alarm" grows with how long you watch.  #9's
negatives are cut short -- median 1,429 s of record -- while #12's run to
10,800 s, so the raw comparison is 7.6x more opportunity on one side.  Every
rate here is therefore also reported over MATCHED windows of 5, 15, 30 and 60
minutes taken from the end of each negative record, so the pools are watched
for equally long.

FIRST-ALARM TIME.  A pool that alarms late is not the same as one that alarms
immediately, and a rate alone cannot tell them apart.  The median time from the
start of the matched window to the first alarm is reported alongside.

    uv run python trbench/run_e9.py --models rule,xgboost,lightgbm,timesfm
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import run_e3 as R          # noqa: E402
import evalv2 as V          # noqa: E402

WINDOWS_S = [300, 900, 1800, 3600]      # 5, 15, 30, 60 minutes
PANEL = "M1"


def _pool_stats(d, idx):
    """Record length per negative experiment, in seconds."""
    e = np.asarray(d["experiment"])[idx]
    t = d["t_end"][idx]
    return pd.Series(t).groupby(e).agg(lambda x: x.max() - x.min())


def _rate(d, idx, risk, tau, window_s=None):
    """Fraction of negative experiments alarming, and when they first do.

    `window_s` restricts each experiment to its own last `window_s` seconds, so
    two pools recorded for very different durations are watched for the same
    length of time.
    """
    e = np.asarray(d["experiment"])[idx]
    t = d["t_end"][idx]
    fired, firsts = [], []
    for k in np.unique(e):
        sel = e == k
        tt, rr = t[sel], risk[sel]
        o = np.argsort(tt, kind="mergesort")
        tt, rr = tt[o], rr[o]
        if window_s is not None:
            keep = tt >= tt[-1] - window_s
            tt, rr = tt[keep], rr[keep]
        if not len(tt):
            continue
        a = SV.first_alarm(tt, np.maximum.accumulate(rr), tau)
        fired.append(a is not None)
        if a is not None:
            firsts.append(a - tt[0])
    return (float(np.mean(fired)) if fired else np.nan,
            len(fired),
            float(np.median(firsts)) if firsts else np.nan)


def run(d, meta, reg, models, budget=V.BUDGET, seed=0):
    plan = V.Plan(d, meta, reg, [PANEL])
    ds = plan.ds
    is_t = np.char.startswith(ds, V.TARGET + "/")
    cols = np.flatnonzero(MW.panel_mask(d["features"], PANEL))
    arc_neg = np.flatnonzero(is_t & (d["y_tr"] == 0))

    src_len = _pool_stats(d, plan.test_neg)
    arc_len = _pool_stats(d, arc_neg)
    print("  source-check negatives %d exp, record median %.0f s"
          % (len(src_len), src_len.median()))
    print("  ARC negatives          %d exp, record median %.0f s"
          % (len(arc_len), arc_len.median()))

    rows = []
    for m in models:
        r_cal, r_src, r_arc = R.run_fold(
            d, cols, m, plan.pool, [plan.cal_neg, plan.test_neg, arc_neg], seed)
        # one threshold, from the calibration negatives, never touched again
        tau = SV.calibrate_threshold(r_cal, d["experiment"][plan.cal_neg],
                                     np.ones(len(plan.cal_neg), bool), far=budget)
        for pool, idx, risk in [("source_mech", plan.test_neg, r_src),
                                ("arc", arc_neg, r_arc)]:
            for w in [None] + WINDOWS_S:
                far, n, first = _rate(d, idx, risk, tau, w)
                rows.append(dict(model=m, tau=tau, pool=pool,
                                 window_s=("full" if w is None else w),
                                 n_exp=n, far=far, first_alarm_s=first))
        f_full = [r for r in rows if r["model"] == m and r["window_s"] == "full"]
        print("    %-9s tau %.4g | full-record FAR  %s"
              % (m, tau, "  ".join("%s %.3f" % (r["pool"], r["far"])
                                   for r in f_full)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rule,xgboost,lightgbm")
    ap.add_argument("--windows", default="W60_raw_causal")
    ap.add_argument("--budget", type=float, default=V.BUDGET)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    d, meta = R.load_windows(a.windows,
                             splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(os.path.join(R.OUT, "registry", "experiments.csv"))
    print("pool %d windows / %d experiments | %s"
          % (len(d["X"]), len(meta), a.windows))
    res = run(d, meta, reg, a.models.split(","), a.budget, a.seed)
    path = os.path.join(R.OUT, "results", "e9%s.csv" % R.out_tag(a.windows))
    R.save_merged(res, path)
    print("\n%.0fs -> %s (%d rows)" % (time.time() - t0, path, len(res)))


if __name__ == "__main__":
    main()
