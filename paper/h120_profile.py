"""Required-lead profile of the 120 s-horizon probes against the 60 s-horizon
probes (added at the 2026-09-24 review, item 2.1).

Both sets of alarms come from the internal leave-one-experiment-out design
(Section 3.2): one alarm per (model, panel, held cell, seed), scored against
L2.  A lead of ell is *observable* in a cell when L2 falls at least ell after
the earliest admissible alarm time (the later of the first prediction endpoint
and the trigger); success at ell is detection with lead >= ell.  The
denominators are the observable cells, as in Fig. 7b and Supplementary Table S8b.

    paper/tables/S8h_horizon120_profile.{csv,md}
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "trbench"), os.path.join(ROOT, "paper")]
import claims as C                                        # noqa: E402
from make_figures import save_table, rng, MODEL_NAME      # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results")
H60 = [str(p) for p in C.E3A]
H120 = sorted(glob.glob(os.path.join(RES, "validation_20260925_e3a_h120*", "*.csv")))
LEADS = (0, 30, 60, 120)


def first_endpoint(d):
    df = pd.DataFrame(dict(e=d["experiment"], t=d["t_end"]))
    return df.groupby("e").t.min().to_dict()


def profile(files, t_first, horizon):
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = rows[rows.label == "L2@1.0"].copy()
    rows["t_s"] = np.maximum(rows.held.map(t_first), rows.t_trigger.fillna(-np.inf))
    out = []
    for (m, p, s), g in rows.groupby(["model", "panel", "seed"]):
        r = dict(horizon_s=horizon, model=m, panel=p, seed=s, cells=len(g))
        for ell in LEADS:
            obs = (g.onset - g.t_s) >= ell
            ok = g.detected & ((ell == 0) | (g.lead >= ell)) & obs
            r["observable_%d" % ell] = int(obs.sum())
            r["met_%d" % ell] = int(ok.sum())
        r["median_lead_detected_s"] = float(g.loc[g.detected, "lead"].median()) if g.detected.any() else np.nan
        r["far_check_median"] = float(g.far_check.median())
        out.append(r)
    return pd.DataFrame(out)


def main():
    d, meta, reg, plan = C.load_meta()
    t_first = first_endpoint(d)
    p60 = profile(H60, t_first, 60)
    p120 = profile(H120, t_first, 120)
    both = pd.concat([p60, p120], ignore_index=True)
    both.to_csv(os.path.join(ROOT, "paper", "tables", "S8h_horizon120_profile_cells.csv"), index=False)
    rows = []
    for (h, m), g in both.groupby(["horizon_s", "model"], sort=False):
        if m == "rule":
            continue
        row = {"training horizon (s)": h, "model": MODEL_NAME.get(m, m), "seeds": g.seed.nunique(),
               "panels": ", ".join(sorted(g.panel.unique()))}
        for ell in LEADS:
            row["met / observable, ell = %d s" % ell] = "%s / %d" % (rng(g["met_%d" % ell], "%d"), int(g["observable_%d" % ell].iloc[0]))
        row["median lead of detected cells (s)"] = rng(g.median_lead_detected_s, "%g")
        row["check FAR (fold median)"] = rng(g.far_check_median, "%.2f")
        rows.append(row)
    save_table(pd.DataFrame(rows), "S8h_horizon120_profile",
               "Sources: %s (60 s horizon) and tr-corpus/results/validation_20260925_e3a_h120* (120 s horizon); internal design, panels M1 and M2, "
               "L2, same folds, calibration negatives and 10 %% budget. Ranges over the available seeds and both panels. A 120 s-horizon probe is trained "
               "to flag an onset within the next 120 s, so a window 120 s before onset is a training positive for it and a training negative for the 60 s probe."
               % "validation_20260925_all7_rescored")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
