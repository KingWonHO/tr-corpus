"""Alarm-cadence sensitivity (reports/36 section 4; external review of 2026-10-03).

Summarises ``trbench/run_review_20261006.py cadence``: every internal LOEO fold
refitted as stored on 10 s windows, the held cell scored on 10 s windows and on
windows ending every 1 s, the latter with the stored 10 s threshold (tau10:
only the alarm grid changes) and with a threshold recalibrated on the same
negatives' 1 s windows (tau1: what a 1 s monitor would have set).

Writes to tr-corpus/results/review_20261006/cadence/:
  C0_reproduction.csv   recomputed 10 s alarms against the stored run
  C1_summary.csv        per model: detection, reversals, lead profile, check FAR at 10 s / 1 s
  C2_rate_flips_by_dataset.csv   rate-threshold reversals, D1-D3 vs D5
  C3_alarm_shift.csv    how far the 1 s alarm moves relative to the 10 s alarm
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "tr-corpus", "results")
CAD = os.path.join(RES, "review_20261006", "cadence")
LEARNED = ["xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer"]
MODELS = ["rule"] + LEARNED
LEADS = [0, 30, 60, 120]
GRIDS = {"10 s": ("alarm10", "detected10"), "1 s, tau10": ("alarm1_tau10", "detected1_tau10"),
         "1 s, tau1": ("alarm1_tau1", "detected1_tau1")}
KEY = ["model", "panel", "seed", "held"]


def rng(s, fmt="%d"):
    s = pd.Series(s).dropna()
    if not len(s):
        return "—"
    lo, hi = s.min(), s.max()
    return (fmt % lo) if lo == hi else ("%s–%s" % (fmt % lo, fmt % hi))


def load():
    c = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(CAD, "e3a_*_cadence_*.csv")))],
                  ignore_index=True)
    s = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "validation_20260925_all7_rescored", "e3a_*s0to4.csv"))],
                  ignore_index=True)
    return c, s


def starts(c):
    """Earliest admissible alarm time per held cell on each grid (claims.lead_support)."""
    out = {}
    for grid, wdir in [("10 s", "W60_native"), ("1 s", "W60_native_s1")]:
        t0 = []
        for sp in ("train", "val", "test"):
            z = np.load(os.path.join(ROOT, "tr-corpus", "windows", wdir, sp + ".npz"), allow_pickle=True)
            t0.append(pd.Series(z["t_end"]).groupby(np.asarray(z["experiment"]).astype(str)).min())
        t0 = pd.concat(t0).groupby(level=0).min()
        tg = c.drop_duplicates("held").set_index("held").t_trigger
        out[grid] = pd.Series({k: max(t0[k], g) if np.isfinite(g) else t0[k] for k, g in tg.items()})
    return out


def reproduction(c, s):
    m = c[c.label == "L2@1.0"].merge(s[s.label == "L2@1.0"][KEY + ["tau", "alarm_time", "far_check"]], on=KEY)
    same_alarm = (m.alarm10 == m.alarm_time) | (m.alarm10.isna() & m.alarm_time.isna())
    same_tau = np.isclose(m.tau10, m.tau, rtol=1e-6, atol=1e-9)
    g = m.assign(a=same_alarm, t=same_tau, f=np.isclose(m.far_check10, m.far_check)).groupby("model")
    return g.agg(fits=("held", "size"), alarm_identical=("a", "mean"), tau_identical=("t", "mean"),
                 check_far_identical=("f", "mean")).reindex(MODELS).reset_index()


def flips(c, det, lab, cells=None):
    a = c[c.label == "L2@1.0"].set_index(KEY)[det]
    b = c[c.label == lab].set_index(KEY)[det]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").reset_index()
    if cells is not None:
        j = j[j.held.isin(cells)]
    return j.assign(f=j.a != j.b).groupby(["model", "panel", "seed"]).agg(n=("held", "size"), f=("f", "sum"))


def summary(c, start):
    rows = []
    l2all = c[c.label == "L2@1.0"]
    for m in MODELS:
        s = c[c.model == m]
        l2 = l2all[l2all.model == m]
        for grid, (al, det) in GRIDS.items():
            row = {"model": m, "grid": grid}
            k = l2.groupby(["panel", "seed"])[det].sum()
            row["L2 detected / 77"] = rng(k)
            for lab in ["L2@0.5", "L2@2.0", "L3 voltage", "ISC 25mV"]:
                f = flips(s, det, lab)
                row["flips " + lab] = "%s / %d" % (rng(f.f), f.n.max())
            st = start["10 s" if grid == "10 s" else "1 s"]
            span = l2.onset - l2.held.map(st)
            lead = l2.onset - l2[al]
            for ell in LEADS:
                el = (span > 0) if ell == 0 else (span >= ell)
                ok = l2[det].astype(bool) & ((ell == 0) | (lead >= ell)) & el
                met = ok.groupby([l2.panel, l2.seed]).sum()
                row["ℓ=%d met / observable" % ell] = "%s / %d" % (rng(met), l2[el].held.nunique())
            far = "far_check10" if grid == "10 s" else ("far_check1_tau1" if grid == "1 s, tau1" else None)
            if far:
                med = l2.groupby(["panel", "seed"])[far].median()
                row["check FAR (fold median)"] = rng(med, "%.3f")
            else:
                row["check FAR (fold median)"] = "not computed"
            rows.append(row)
    return pd.DataFrame(rows)


def rate_by_dataset(c):
    d13 = c[c.held.str.match(r"ds0[123]_")].held.unique()
    d5 = c[c.held.str.startswith("ds09_mech")].held.unique()   # corpus D5 (Lin et al.)
    rows = []
    for m in MODELS:
        s = c[c.model == m]
        for grid, (_, det) in GRIDS.items():
            row = {"model": m, "grid": grid}
            for name, cells in [("D1–D3", d13), ("D5", d5)]:
                for lab in ["L2@0.5", "L2@2.0", "L3 voltage"]:
                    f = flips(s, det, lab, cells)
                    row["%s %s" % (name, lab)] = "%s / %d" % (rng(f.f), f.n.max())
            rows.append(row)
    return pd.DataFrame(rows)


def alarm_shift(c):
    l2 = c[c.label == "L2@1.0"]
    rows = []
    for m in MODELS:
        s = l2[l2.model == m]
        both = s.dropna(subset=["alarm10", "alarm1_tau10"])
        d = both.alarm10 - both.alarm1_tau10          # > 0: the 1 s alarm is earlier
        bt = s.dropna(subset=["alarm10", "alarm1_tau1"])
        d1 = bt.alarm10 - bt.alarm1_tau1
        rows.append(dict(model=m, fits=len(s),
                         both_alarm_tau10=len(both),
                         earlier_0_9s=float(((d >= 0) & (d <= 9)).mean()),
                         earlier_gt9s=float((d > 9).mean()), later=float((d < 0).mean()),
                         median_shift_tau10=float(d.median()), p95_shift_tau10=float(d.quantile(0.95)),
                         alarm_only_1s_tau10=int((s.alarm10.isna() & s.alarm1_tau10.notna()).sum()),
                         alarm_only_10s=int((s.alarm10.notna() & s.alarm1_tau10.isna()).sum()),
                         median_shift_tau1=float(d1.median()),
                         tau1_above_tau10=float((s.tau1 > s.tau10).mean()),
                         alarm_lost_tau1=int((s.alarm10.notna() & s.alarm1_tau1.isna()).sum())))
    return pd.DataFrame(rows)


def main():
    c, s = load()
    start = starts(c)
    out = {"C0_reproduction": reproduction(c, s), "C1_summary": summary(c, start),
           "C2_rate_flips_by_dataset": rate_by_dataset(c), "C3_alarm_shift": alarm_shift(c)}
    for k, v in out.items():
        v.to_csv(os.path.join(CAD, k + ".csv"), index=False)
        print("== " + k)
        print(v.to_string(index=False))


if __name__ == "__main__":
    main()
