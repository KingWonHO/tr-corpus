"""Mechanism tables behind Results 3.1, 3.3, 3.4 (Supplementary Tables S2f, S4g,
S4h, S5b), regenerated from the stored v1.1 outputs; no model is fitted.

    S2_flip_anatomy.csv          every verdict flip of Section 3.1 has the alarm between the two events
    S4g_negative_pools.csv       what the two negative pools look like (exposure, temperatures)
    S4g_first_alarm_anatomy.csv  when and at what temperature the frozen models first alarm
    S4h_representation_anatomy.csv  max-mean thermocouple gap, pre-onset window T, observation age
    S5b_d6_risk_band.csv         the D6 risk band relative to the frozen threshold

Pools follow the adjudication (protocol 29): D6 negatives are the 13 records
adjudicated non-runaway, D6 positives the 43 L2-labelled records adjudicated
runaway, source check negatives the 27 adjudicated non-runaway test records.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "trbench"), os.path.join(ROOT, "paper")]
import claims as C                                  # noqa: E402
from make_figures import save_table                # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results")
WIN = os.path.join(ROOT, "tr-corpus", "windows", "W60_native")
TREES = os.path.join(RES, "validation_20260925_native")
SEQ = os.path.join(RES, "validation_20260925_seq")
DS = {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3", "ds04_osf": "D4", "ds09_mech": "D5", "ds12_arc": "D6"}
DISP = {"L2@0.5": "L2$_{0.5}$", "L2@2.0": "L2$_{2}$", "L1 venting": "L1", "L3 voltage": "L3", "ISC 25mV": "25 mV"}
LEARNED = list(C.LEARNED)


def load_windows():
    parts = []
    for sp in ("train", "val", "test", "test_zeroshot"):
        z = np.load(os.path.join(WIN, sp + ".npz"), allow_pickle=True)
        feats = list(z["features"])
        idx = {n: feats.index(n) for n in ("T_surface_max", "T_surface_mean", "age__T_surface_max")}
        X, M = z["X"], z["mask"]
        parts.append(pd.DataFrame(dict(
            experiment=z["experiment"].astype(str), t_end=z["t_end"].astype(float), split=sp,
            Tmax=np.where(M[:, -1, idx["T_surface_max"]] > 0, X[:, -1, idx["T_surface_max"]], np.nan),
            Tmean=np.where(M[:, -1, idx["T_surface_mean"]] > 0, X[:, -1, idx["T_surface_mean"]], np.nan),
            age=np.where(M[:, -1, idx["age__T_surface_max"]] > 0, X[:, -1, idx["age__T_surface_max"]], np.nan))))
    return pd.concat(parts, ignore_index=True)


def registry():
    reg = pd.read_csv(os.path.join(ROOT, "tr-corpus", "registry", "experiments.csv"))
    reg = reg[reg.dataset_id.isin(DS)].copy()
    reg["key"] = reg.dataset_id + "/" + reg.experiment_id
    adj = pd.read_csv(os.path.join(ROOT, "tr-corpus", "registry", "outcome_adjudication.csv"))
    reg = reg.merge(adj[["experiment_id", "final_assignment"]], on="experiment_id", how="left")
    return reg


# ---------------------------------------------------------------- S2f
def flip_anatomy(reg):
    rows = pd.concat([pd.read_csv(str(p)) for p in C.E3A], ignore_index=True)
    rows = rows[rows.model.isin(LEARNED)]
    base = rows[rows.label == "L2@1.0"].set_index(["model", "panel", "seed", "held"])
    out = []
    for lab in ("L2@0.5", "L2@2.0", "L1 venting", "L3 voltage", "ISC 25mV"):
        alt = rows[rows.label == lab].set_index(["model", "panel", "seed", "held"])
        j = base[["detected", "alarm_time", "onset"]].join(alt[["detected", "onset"]], rsuffix="_alt", how="inner").reset_index()
        j["dataset"] = j.held.str.split("/").str[0].map(DS)
        j["flip"] = j.detected != j.detected_alt
        j["between"] = j.flip & (j.alarm_time >= np.minimum(j.onset, j.onset_alt)) & (j.alarm_time <= np.maximum(j.onset, j.onset_alt))
        for ds, g in list(j.groupby("dataset")) + [("all", j)]:
            per_fit = g.groupby(["model", "panel", "seed"]).flip.sum()
            f = g[g.flip]
            out.append({"event": DISP[lab], "dataset": ds, "cells": g.held.nunique(),
                        "event minus L2, median (s)": round(float((g.onset_alt - g.onset).groupby(g.held).first().median()), 1),
                        "flips per fit": "%d–%d" % (per_fit.min(), per_fit.max()),
                        "flip cases": int(g.flip.sum()), "alarm between events": int(g.between.sum()),
                        "lost": int((f.detected & ~f.detected_alt).sum()), "gained": int((~f.detected & f.detected_alt).sum())})
    df = pd.DataFrame(out)
    save_table(df, "S2_flip_anatomy",
               "Source: validation_20260925_all7_rescored (six learned models x 2 panels x 5 seeds, label v1.1, adjudicated cohort). "
               "A flip is a cell whose detection status differs between L2 and the alternative event for the same alarm; 'alarm between events' counts "
               "the flipped fit-cell cases whose alarm time lies between the two onsets.")
    return df


# ---------------------------------------------------------------- S4g
def first_alarm_anatomy(reg, win):
    key = reg.set_index("key")
    src_neg = key[(key.split == "test") & key.t_onset_L2.isna() & (key.final_assignment == "negative") & (key.dataset_id != "ds12_arc")].index
    d6_neg = key[(key.dataset_id == "ds12_arc") & key.t_onset_L2.isna() & (key.final_assignment == "negative")].index
    pools = {"source": src_neg, "D6": d6_neg}
    prow = []
    for pool, keys in pools.items():
        w = win[win.experiment.isin(keys)]
        g = w.groupby("experiment")
        span = g.t_end.max() - g.t_end.min()
        T0 = g.apply(lambda x: x.sort_values("t_end").Tmax.iloc[0])
        T300 = g.apply(lambda x: x[x.t_end <= x.t_end.min() + 300].Tmax.max())
        Tend = g.apply(lambda x: x.sort_values("t_end").Tmax.iloc[-1])
        prow.append(dict(pool=pool, records=len(keys), exposure_median_s=round(span.median()), exposure_min_s=round(span.min()), exposure_max_s=round(span.max()),
                         T_start_median=round(T0.median()), T_first300_median=round(T300.median()), T_end_median=round(Tend.median()), T_max_median=round(g.Tmax.max().median())))
    save_table(pd.DataFrame(prow), "S4g_negative_pools", "Source: W60_native windows, adjudicated negatives (label v1.1). Exposure is the span of prediction endpoints.")
    rows = []
    t0 = win.groupby("experiment").t_end.min()
    for run in (TREES, SEQ):
        nc = pd.read_csv(os.path.join(run, "negative_cells.csv"))
        nc = nc[(nc.age_mode == "no_age") & (nc.window_s.astype(str) == "full") & nc.eligible & nc.fired.eq(True)]
        for (m, pool), g in nc.groupby(["model", "pool"]):
            pname = "D6" if pool == "arc" else "source"
            temps = []
            for r in g.itertuples():
                te = t0[r.held] + r.first_alarm_s
                x = win[(win.experiment == r.held) & (np.abs(win.t_end - te) < 0.5)]
                temps.append(float(x.Tmax.iloc[0]) if len(x) and np.isfinite(x.Tmax.iloc[0]) else np.nan)
            temps = np.array(temps, float)
            rows.append(dict(pool=pname, model=m, alarms=len(g), first_alarm_median_s=round(g.first_alarm_s.median()),
                             first_alarm_min_s=round(g.first_alarm_s.min()), first_alarm_max_s=round(g.first_alarm_s.max()),
                             temp_at_alarm_median=round(np.nanmedian(temps)), temp_at_alarm_p10=round(np.nanpercentile(temps, 10)),
                             temp_at_alarm_p90=round(np.nanpercentile(temps, 90))))
    df = pd.DataFrame(rows).sort_values(["pool", "model"])
    save_table(df, "S4g_first_alarm_anatomy",
               "Source: validation_20260925_native and validation_20260925_seq negative_cells.csv (full record, values representation, seeds 0-4 pooled), "
               "temperatures from the W60_native windows at the alarming window's end. Times are on the exposure clock (from the first prediction endpoint).")
    return df


# ---------------------------------------------------------------- S4h
def representation_anatomy(reg, win):
    key = reg.set_index("key")
    win = win.merge(reg[["key", "dataset_id", "t_onset_L2", "final_assignment"]], left_on="experiment", right_on="key", how="inner")
    win["ds"] = win.dataset_id.map(DS)
    rows = []
    for ds in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        w = win[win.ds == ds]
        pos = w[w.t_onset_L2.notna() & (w.final_assignment == "positive") & (w.t_end < w.t_onset_L2) & (w.t_end >= w.t_onset_L2 - 60)]
        neg = w[w.t_onset_L2.isna() & (w.final_assignment == "negative")]
        for name, g in (("all", pd.concat([pos, neg])), ("runaway", pos), ("non-runaway", neg)):
            if not len(g):
                continue
            gap = (g.Tmax - g.Tmean).dropna()
            T = g.Tmax.dropna() if name != "non-runaway" else pd.Series(dtype=float)
            age = g.age.dropna()
            rows.append(dict(dataset=ds, records=name, windows=len(g),
                             gap_max_minus_mean_median=round(float(gap.median()), 1) if len(gap) else np.nan, gap_p90=round(float(gap.quantile(.9)), 1) if len(gap) else np.nan,
                             gap_max=round(float(gap.max()), 1) if len(gap) else np.nan,
                             T_preonset_median=round(float(T.median()), 1) if len(T) else np.nan, T_preonset_p10=round(float(T.quantile(.1)), 1) if len(T) else np.nan,
                             T_preonset_p90=round(float(T.quantile(.9)), 1) if len(T) else np.nan,
                             age_median=round(float(age.median()), 1), age_p90=round(float(age.quantile(.9)), 1), age_share_zero=round(float((age == 0).mean()), 2)))
    df = pd.DataFrame(rows)
    save_table(df, "S4h_representation_anatomy",
               "Source: W60_native windows, label v1.1, adjudicated pools. Gap: hottest thermocouple minus mean at the window end over windows ending within 60 s "
               "before L2 (runaway) or all windows (non-runaway). T_preonset: maximum surface temperature at the end of the same pre-onset windows. Age: observation age of the surface channel at the window end.")
    return df


# ---------------------------------------------------------------- S5b
def d6_risk_band(reg):
    d, meta, _, plan = C.load_meta()
    st = C.settings()
    st = st[st.run.isin(C.MAIN_RUNS) & (st.representation == "no_age") & st.model.isin(LEARNED)]
    onset = plan.onsets["L2@1.0"]
    rows = []
    for s in st.itertuples():
        z = np.load(s.file)
        idx, risk = z["arc_idx"], np.asarray(z["arc_risk"], float)
        e, t = d["experiment"][idx], d["t_end"][idx]
        pos, neg = [], []
        for k in np.unique(e):
            sel = e == k
            on = onset.get(k, np.nan)
            if np.isfinite(on):
                pre = sel & (t < on)
                if pre.any():
                    pos.append(risk[pre].max())
            else:
                neg.append(risk[sel].max())
        pos, neg = np.array(pos), np.array(neg)
        tau = s.tau_stored
        rows.append(dict(model=s.model, seed=s.seed, tau=round(tau, 4), d6_detected_before_L2=int((pos >= tau).sum()), d6_neg_alarm=int((neg >= tau).sum()),
                         pre_onset_max_p10=round(np.percentile(pos, 10), 4), pre_onset_max_med=round(np.median(pos), 4), pre_onset_max_p90=round(np.percentile(pos, 90), 4),
                         neg_max_med=round(np.median(neg), 4), pre_within_0p02=int((np.abs(pos - tau) <= 0.02).sum()), n_pos=len(pos), n_neg=len(neg)))
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(LEARNED)}
    df = df.sort_values(["model", "seed"], key=lambda s: s.map(order) if s.name == "model" else s)
    save_table(df, "S5b_d6_risk_band",
               "Source: validation_20260925_native and validation_20260925_seq predictions (values representation), adjudicated D6 pools (43 runaway, 13 non-runaway). "
               "Pre-onset maximum risk over the windows before L2 of each runaway record; non-runaway maximum over the full record.")
    return df


# ---------------------------------------------------------------- S4i
def shortcut_controls():
    run = os.path.join(RES, "validation_20260925_shortcut_controls")
    ps = pd.read_csv(os.path.join(run, "positive_summary.csv")); nr = pd.read_csv(os.path.join(run, "negative_rates.csv")); cal = pd.read_csv(os.path.join(run, "calibration.csv"))
    def rng(v, f="%d"):
        v = list(v); lo, hi = min(v), max(v)
        return (f % lo) if lo == hi else (f % lo) + "\u2013" + (f % hi)
    rows = []
    for arm, label in (("mask_only", "channel availability only (M1 masks, no values)"), ("age_only", "observation ages only (no sensor values)")):
        for m, name in (("xgboost", "XGBoost"), ("lightgbm", "LightGBM")):
            c = cal[(cal.model == m) & (cal.age_mode == arm)]; p = ps[(ps.model == m) & (ps.age_mode == arm)]; n = nr[(nr.model == m) & (nr.age_mode == arm)]
            def far(pool, ws, anc):
                q = n[(n.pool == pool) & (n.window_s.astype(str) == ws) & (n.anchor == anc)]
                return rng(q.n_alarm) + "/" + str(int(q.n_eligible.iloc[0]))
            src, arc = p[p.pool == "source_test"], p[p.pool == "arc"]
            lead = lambda g: "\u2014" if g.lead_median.isna().all() else rng(g.lead_median.dropna(), "%g")
            rows.append({"input": label, "model": name, "tau": rng(c.tau, "%.3f"), "calibration alarms": rng(c.n_cal_alarm) + "/" + str(int(c.n_cal.iloc[0])),
                         "source test detected": rng(src.n_detected) + "/" + str(int(src.n_events.iloc[0])), "source test lead median (s)": lead(src),
                         "D6 detected": rng(arc.n_detected) + "/" + str(int(arc.n_events.iloc[0])), "D6 lead median (s)": lead(arc),
                         "source check FAR full": far("source_mech", "full", "full"), "D6 FAR first 300 s": far("arc", "300", "head"), "D6 FAR full": far("arc", "full", "full")})
    df = pd.DataFrame(rows)
    save_table(df, "S4i_shortcut_controls",
               "Source: validation_20260925_shortcut_controls (trees, seeds 0-4; held-out design, adjudicated pools, label v1.1). mask_only: the nine window statistics are replaced by "
               "the per-channel availability fraction of the M1 channels; age_only: the model sees the per-channel observation ages and obs_age and no sensor value. "
               "A D6 lead equal to the full pre-onset span means the alarm stood from the first window.")
    return df


if __name__ == "__main__":
    reg = registry(); win = load_windows()
    print(flip_anatomy(reg).to_string(index=False))
    print(first_alarm_anatomy(reg, win).to_string(index=False))
    print(representation_anatomy(reg, win).to_string(index=False))
    print(d6_risk_band(reg).to_string(index=False))
    print(shortcut_controls().to_string(index=False))
