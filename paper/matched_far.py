"""M2 - M1 detection at matched realised check-negative FAR (review item 2.6).

The internal design calibrates each panel's threshold separately on the 25
calibration negatives, so the two panels reach the 27 check negatives at
different realised false-alarm rates (Table 5).  From the per-fold traces
stored by ``run_v2.py --curves`` this script re-thresholds both panels at the
same check-negative FAR level and counts detections, holding fold, seed and
held cell fixed.  Thresholding on the check set is hindsight -- the deployment
could not have done it -- so the table is a paired comparison at equal cost,
not a performance estimate; the calibration-budget row reproduces the stored
alarms and is the check that the traces are the ones the paper scored.

    paper/tables/S4j_matched_far.{csv,md}
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "paper")]
from make_figures import save_table, rng, MODEL_NAME    # noqa: E402

TRACES = os.path.join(ROOT, "tr-corpus", "results", "validation_20260925_e3a_trees", "traces")
LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)


def tau_at(peaks, far):
    """Smallest threshold whose experiment-level FAR on `peaks` is <= far."""
    p = np.sort(np.asarray(peaks, float))[::-1]
    k = int(np.floor(far * len(p)))
    return np.nextafter(p[k], np.inf) if k < len(p) else -np.inf


def detect(t, r, tau, onset, trig):
    i = np.flatnonzero(r >= tau)
    if not len(i):
        return False
    ta = float(t[i[0]])
    if np.isfinite(trig) and ta < trig:
        return False
    return ta < onset


def main():
    reg = pd.read_csv(os.path.join(ROOT, "tr-corpus", "registry", "experiments.csv"))
    onset = dict(zip(reg.dataset_id + "/" + reg.experiment_id, reg["t_onset_L2_1.0"]))
    rows = []
    for f in sorted(glob.glob(os.path.join(TRACES, "*.npz"))):
        name = os.path.basename(f)[:-4]
        model, panel, seed, held = name.split("_", 3)
        held = held.replace("__", "/", 1)
        z = np.load(f)
        on = onset.get(held, np.nan)
        if not np.isfinite(on):
            continue
        t, r, trig = z["t"], z["risk_running_max"], float(z["t_trigger"])
        row = dict(model=model, panel=panel, seed=int(seed[1:]), held=held,
                   stored=detect(t, r, float(z["tau"]), on, trig),
                   cal_budget=detect(t, r, tau_at(z["cal_peaks"], 0.10), on, trig))
        for lv in LEVELS:
            row["far%.2f" % lv] = detect(t, r, tau_at(z["check_peaks"], lv), on, trig)
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(ROOT, "paper", "tables", "S4j_matched_far_cells.csv"), index=False)
    if not len(d):
        print("no traces yet")
        return
    assert (d.stored == d.cal_budget).all(), "calibration-budget re-thresholding does not reproduce the stored alarms"
    cols = ["cal_budget"] + ["far%.2f" % lv for lv in LEVELS]
    label = {"cal_budget": "calibration budget (as reported)", **{"far%.2f" % lv: "check FAR <= %.2f" % lv for lv in LEVELS}}
    out = []
    for m in sorted(d.model.unique(), key=lambda x: ["xgboost", "lightgbm"].index(x) if x in ("xgboost", "lightgbm") else 9):
        for c in cols:
            g = d[d.model == m].groupby(["seed", "panel"])[c].sum().unstack("panel")
            g = g.dropna()
            if "M2" not in g.columns or "M1" not in g.columns or not len(g):
                continue
            n = d[(d.model == m) & (d.panel == "M1")].groupby("seed").size().iloc[0]
            out.append({"model": MODEL_NAME.get(m, m), "threshold rule": label[c], "seeds": len(g), "cells": int(n),
                        "M1 detected": rng(g["M1"], "%d"), "M2 detected": rng(g["M2"], "%d"),
                        "M2 - M1": rng(g["M2"] - g["M1"], "%+d"),
                        "seeds with M2 > M1": "%d of %d" % (int((g["M2"] > g["M1"]).sum()), len(g))})
    save_table(pd.DataFrame(out), "S4j_matched_far",
               "Source: %s (per-fold traces of the internal design, trees, 60 s horizon, L2, 77 cells). Each row re-thresholds both panels "
               "of the same fold and seed at the same experiment-level FAR on the 27 check negatives (a hindsight threshold, not the deployment's) "
               "and counts detections; the calibration-budget row uses the stored rule and reproduces the stored alarms exactly."
               % os.path.relpath(TRACES, ROOT).replace(os.sep, "/"))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
