"""Supplementary tables and the worked claim assessment from the 2026-10-06 sensitivity analyses.

Reads the result files under tr-corpus/results (stored runs, claims_20260925 and
review_20261006) and writes paper/tables/<name>.csv for retype_supp_tables.py:

  S2j_flips_by_dataset        reference-event reversals by dataset group, rate-variant onset offsets
  S2k_confirmed_internal      internal design re-scored on source-confirmed records only
  S2k_confirmed_heldout       held-out design on source-confirmed records only
  S3b_training_length         sequence probes refitted for 6, 12 and 24 epochs (held-out design)
  S4l_alarm_cadence           internal design re-scored on windows ending every 1 s
  S8i_temperature_rule        surface-temperature level rules beside the learned probes
  S8j_disputed_negatives      two source-non-runaway / L2 records scored as check negatives
  S11c_d8_layout              D8 tensor held fixed, source aggregate placed in its slots
  T7_worked_example           main-text Table 7: one claim assessed end to end

    uv run python paper/review_tables.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "tr-corpus", "results")
REV = os.path.join(RES, "review_20261006")
TAB = os.path.join(HERE, "tables")
LEARNED = ["xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer"]
NAME = {"rule": "Rate rule", "xgboost": "XGBoost", "lightgbm": "LightGBM", "gru": "GRU", "mamba": "Mamba",
        "itransformer": "iTransformer", "convtransformer": "ConvTransformer"}
DSN = {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3", "ds09_mech": "D5"}


def rng(s, fmt="%d"):
    s = pd.Series(s).dropna()
    if not len(s):
        return "—"
    lo, hi = s.min(), s.max()
    return (fmt % lo) if lo == hi else ("%s–%s" % (fmt % lo, fmt % hi))


def span_rng(texts):
    """Combine 'a–b / n' strings into one range over the given rows."""
    lo, hi, n = [], [], set()
    for t in texts:
        k, d = [x.strip() for x in t.split("/")]
        a, b = (k.split("–") + [k])[:2] if "–" in k else (k, k)
        lo.append(float(a)); hi.append(float(b)); n.add(d)
    fmt = "%d" if all(float(x).is_integer() for x in lo + hi) else "%.3f"
    r = (fmt % min(lo)) if min(lo) == max(hi) else ("%s–%s" % (fmt % min(lo), fmt % max(hi)))
    return "%s / %s" % (r, "/".join(sorted(n)))


def num_rng(texts, fmt="%+.3f"):
    vals = []
    for t in texts:
        t = str(t).replace("−", "-")
        parts = [p for p in t.replace("–+", "|+").replace("–-", "|-").split("|")]
        vals += [float(p) for p in parts]
    lo, hi = min(vals), max(vals)
    return (fmt % lo) if lo == hi else ("%s to %s" % (fmt % lo, fmt % hi))


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(beta.ppf(1 - a, k + 1, n - k))


def cp_lower(k, n, a=0.05):
    return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))


def verdict(k, n, alpha=0.10):
    if cp_upper(k, n) <= alpha:
        return "supported"
    if cp_lower(k, n) > alpha:
        return "budget exceeded"
    return "inconclusive"


def write(name, df):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    print("== %s (%d rows)" % (name, len(df)))
    print(df.to_string(index=False))


# ---------------------------------------------------------------- S2j
def s2j():
    r2 = pd.read_csv(os.path.join(REV, "stored", "R2_flips_by_dataset.csv"))
    r3 = pd.read_csv(os.path.join(REV, "stored", "R3_rate_variant_offsets.csv"))
    rows = []
    for _, r in r2.iterrows():
        g = r["cells"]
        sub = r3[r3.dataset.isin(["D1", "D2", "D3"] if g == "D1–D3" else ([g] if g != "all" else r3.dataset.unique()))]
        same = {v: int(sub[sub.variant == v]["same onset"].sum()) for v in ("L2@0.5", "L2@2.0")}
        rows.append({"cells": g, "same onset L2@0.5 / L2@2.0": "%d / %d" % (same["L2@0.5"], same["L2@2.0"]),
                     "L2@0.5": r["L2@0.5"], "L2@2.0": r["L2@2.0"], "L3 voltage": r["L3 voltage"],
                     "L3 voltage (trees)": r["L3 voltage (trees)"], "ISC 25mV": r["ISC 25mV"]})
    write("S2j_flips_by_dataset", pd.DataFrame(rows))


# ---------------------------------------------------------------- S2k
def s2k():
    r1a = pd.read_csv(os.path.join(REV, "stored", "R1a_internal_source_confirmed.csv"))
    r1b = pd.read_csv(os.path.join(REV, "stored", "R1b_internal_trees_recalibrated.csv"))
    rows = []
    for sub, lab in [("all paired positives", "all"), ("source-confirmed", "source-confirmed"),
                     ("trace-adjudicated only", "trace-adjudicated only")]:
        s = r1a[r1a.subset == sub]
        seq, tree = s[~s.model.isin(["xgboost", "lightgbm"])], s[s.model.isin(["xgboost", "lightgbm"])]
        b = r1b[(r1b.threshold == ("stored" if sub == "all paired positives" else "stored, confirmed check"))
                & (r1b.subset == ("all paired positives" if sub != "source-confirmed" else "source-confirmed"))
                & r1b.model.isin(["xgboost", "lightgbm"])]
        rows.append({"subset": lab, "cells": int(s.cells.max()),
                     "L2 detected": span_rng([x + " / 0" for x in s["L2 detected"]]).split(" /")[0],
                     "flips L2@0.5": span_rng(s["flips L2@0.5"]), "flips L2@2.0": span_rng(s["flips L2@2.0"]),
                     "flips L3 voltage": span_rng(s["flips L3 voltage"]), "flips ISC 25mV": span_rng(s["flips ISC 25mV"]),
                     "M2−M1 sequence": num_rng(seq["M2−M1 detection rate"]), "M2−M1 trees": num_rng(tree["M2−M1 detection rate"]),
                     "ℓ=120 met / observable": span_rng(s["ℓ=120 met / observable"])})
    write("S2k_confirmed_internal", pd.DataFrame(rows))
    # tree check FAR, all 27 vs the 17 source-confirmed check negatives (stored thresholds)
    far = r1b[(r1b.subset == "all paired positives") & r1b.model.isin(["xgboost", "lightgbm"])
              & r1b.threshold.isin(["stored", "stored, confirmed check", "recalibrated on confirmed"])]
    print(far[["threshold", "model", "L2 detected", "check FAR (fold median)", "check records"]].to_string(index=False))

    r1c = pd.read_csv(os.path.join(REV, "stored", "R1c_heldout_source_confirmed.csv"))
    st = r1c[r1c.threshold == "stored"]
    rows = []
    for m in ["rule"] + LEARNED:
        a, c = st[(st.model == m) & (st.subset == "all")].iloc[0], st[(st.model == m) & (st.subset == "source-confirmed")].iloc[0]
        rows.append({"model": NAME[m],
                     "source test detected": "%s → %s" % (a["source test detected"], c["source test detected"]),
                     "D6 detected": "%s → %s" % (a["D6 detected"], c["D6 detected"]),
                     "source check alarms, first 300 s": "%s → %s" % (a["source check alarms, first 300 s"], c["source check alarms, first 300 s"]),
                     "D6 negatives alarming, full": a["D6 negatives alarming, full"]})
    write("S2k_confirmed_heldout", pd.DataFrame(rows))


# ---------------------------------------------------------------- S3b
def s3b():
    cur = pd.read_csv(os.path.join(REV, "convergence", "loss_curves.csv"))
    rows = []
    for m in ["gru", "mamba", "itransformer", "convtransformer"]:
        for E in (6, 12, 24):
            c = cur[(cur.model == m) & (cur.epochs == E)]
            last = c[c.epoch == E]
            d = os.path.join(REV, "convergence", "epochs_%d" % E)
            ps = pd.read_csv(os.path.join(d, "positive_summary.csv"))
            nr = pd.read_csv(os.path.join(d, "negative_rates.csv"))
            ps, nr = ps[(ps.model == m) & (ps.age_mode == "no_age")], nr[(nr.model == m) & (nr.age_mode == "no_age")]
            rows.append({"model": NAME[m], "epochs": E,
                         "training loss": rng(last.train_loss, "%.2f"), "held-out loss": rng(last.monitor_loss, "%.2f"),
                         "epoch of minimum held-out loss": rng(c.loc[c.groupby("seed").monitor_loss.idxmin()].epoch),
                         "source test detected": rng(ps[ps.pool == "source_test"].n_detected) + " / 13",
                         "source check alarming, full": rng(nr[(nr.pool == "source_mech") & (nr.window_s == "full")].n_alarm) + " / 27",
                         "D6 detected": rng(ps[ps.pool == "arc"].n_detected) + " / 43",
                         "D6 negatives alarming, full": rng(nr[(nr.pool == "arc") & (nr.window_s == "full")].n_alarm) + " / 13"})
    write("S3b_training_length", pd.DataFrame(rows))


# ---------------------------------------------------------------- S4l
def s4l():
    c1 = pd.read_csv(os.path.join(REV, "cadence", "C1_summary.csv"))
    keep = ["model", "grid", "L2 detected / 77", "flips L2@0.5", "flips L2@2.0", "flips L3 voltage", "flips ISC 25mV",
            "ℓ=60 met / observable", "ℓ=120 met / observable", "check FAR (fold median)"]
    t = c1[keep].copy()
    t["model"] = t.model.map(NAME)
    t["grid"] = t.grid.map({"10 s": "10 s", "1 s, tau10": "1 s, 10 s τ", "1 s, tau1": "1 s, recalibrated τ"})
    t["check FAR (fold median)"] = t["check FAR (fold median)"].replace({"not computed": "—"})
    write("S4l_alarm_cadence", t)


# ---------------------------------------------------------------- S8i
def s8i():
    e2 = pd.read_csv(os.path.join(REV, "fixed_rules", "E2_summary.csv"))
    rows = []
    for _, r in e2.iterrows():
        lab = "≥ 60 °C, fixed" if ">= 60" in r["rule"] else ("≥ 80 °C, fixed" if ">= 80" in r["rule"] else "≥ 110.4 °C, calibrated (10 %)")
        rows.append({"rule": lab, "L2 detected /77 (D1–D3, D5)": "%d (%d, %d)" % (r["L2 detected /77"], r["D1–D3 /16"], r["D5 /61"]),
                     "pre-trigger alarms": int(r["pre-trigger alarms"]), "ℓ=60 met / observable": r["ℓ=60 met/obs"].replace("/", " / "),
                     "ℓ=120 met / observable": r["ℓ=120 met/obs"].replace("/", " / "),
                     "flips L3 voltage": r["flips L3 voltage"].replace("/", " / "),
                     "calibration alarms, first 300 s": r["calibration first 300 s"].replace("/", " / "),
                     "source check alarms, first 300 s": r["source check first 300 s"].replace("/", " / "),
                     "D6 detected": "%d / 43" % r["D6 detected /43"],
                     "D6 negatives alarming, first 300 s": r["D6 first 300 s"].replace("/", " / "),
                     "D6 negatives alarming, full": r["D6 full"].replace("/", " / ")})
    write("S8i_temperature_rule", pd.DataFrame(rows))


# ---------------------------------------------------------------- S8j
def s8j():
    cf = pd.read_csv(os.path.join(RES, "claims_20260925", "claims_far.csv"))
    cf = cf[cf.run.isin(["native_final", "seq_matched"]) & (cf.representation == "no_age") & (cf.tau_kind == "full") & (cf.H_s == 300)
            & (cf.pool == "source check") & (cf.segment == "head")]
    dn = pd.read_csv(os.path.join(REV, "disputed_heldout", "negative_rates.csv"))
    dn = dn[(dn.age_mode == "no_age") & (dn.pool == "source_mech") & (dn.window_s == "300") & (dn.anchor == "head")]
    dp = pd.read_csv(os.path.join(REV, "disputed_heldout", "positive_summary.csv"))
    dp = dp[(dp.age_mode == "no_age") & (dp.pool == "source_test")]
    rows, flips = [], 0
    for m in ["rule"] + LEARNED:
        a, b = cf[cf.model == m].set_index("seed"), dn[dn.model == m].set_index("seed")
        seeds = sorted(set(a.index) & set(b.index))
        v27 = [verdict(int(a.loc[s, "k"]), int(a.loc[s, "n"])) for s in seeds]
        v29 = [verdict(int(b.loc[s, "n_alarm"]), int(b.loc[s, "n_eligible"])) for s in seeds]
        ch = sum(x != y for x, y in zip(v27, v29))
        flips += ch
        sup = [s for s, v in zip(seeds, v29) if v == "supported"]
        det = dp[(dp.model == m) & dp.seed.isin(sup)].n_detected if sup else pd.Series(dtype=float)
        rows.append({"model": NAME[m], "settings": len(seeds),
                     "alarms, 27 check negatives": rng(a.loc[seeds, "k"]) + " / 27",
                     "alarms, 29 check negatives": rng(b.loc[seeds, "n_alarm"]) + " / %d" % int(b.n_eligible.max()),
                     "inconclusive → supported": "%d / %d" % (sum(v == "supported" for v in v29), len(seeds)),
                     "source test detected (supported settings)": (rng(det) + " / 13") if len(det) else "—"})
    write("S8j_disputed_negatives", pd.DataFrame(rows))
    print("verdict changes:", flips)


# ---------------------------------------------------------------- S11c
def s11c():
    s = pd.read_csv(os.path.join(REV, "d8_layout", "summary.csv"))
    order = ["no_age", "d8layout_agg", "d8layout_max", "d8layout_mean", "surface_max_only", "surface_mean_only"]
    lab = {"no_age": "M1 (main)", "d8layout_agg": "D8 layout, M1 aggregates", "d8layout_max": "D8 layout, source maximum in both slots",
           "d8layout_mean": "D8 layout, source mean in both slots", "surface_max_only": "one channel, source maximum",
           "surface_mean_only": "one channel, source mean"}
    tgt = {"no_age": "D8 M1 tensor", "d8layout_agg": "D8 M1 tensor", "d8layout_max": "D8 M1 tensor", "d8layout_mean": "D8 M1 tensor",
           "surface_max_only": "D8 maximum probe", "surface_mean_only": "D8 probe mean"}
    rows = []
    for arm in order:
        x, l = s[(s.arm == arm) & (s.model == "xgboost")].iloc[0], s[(s.arm == arm) & (s.model == "lightgbm")].iloc[0]
        rows.append({"source training": lab[arm], "D8 input": tgt[arm],
                     "XGBoost vehicles alarming /292": str(x.vehicles_alarm).replace("-", "–"),
                     "LightGBM vehicles alarming /292": str(l.vehicles_alarm).replace("-", "–"),
                     "XGBoost episodes per 1,000 vehicle-h": str(x.episodes_per_1000_vehicle_h).replace("-", "–"),
                     "LightGBM episodes per 1,000 vehicle-h": str(l.episodes_per_1000_vehicle_h).replace("-", "–")})
    write("S11c_d8_layout", pd.DataFrame(rows))


# ---------------------------------------------------------------- Table 7
def worked():
    """XGBoost, panel M1: 'warns >= 60 s before L2 at <= 10 % alarm probability per 300 s'."""
    import sys
    sys.path.insert(0, HERE)
    import review_sensitivity as RS
    raw = pd.read_csv(os.path.join(RES, "validation_20260925_all7_rescored", "e3a_trees_s0to4.csv"))
    x = raw[(raw.model == "xgboost") & (raw.panel == "M1")]
    ls = pd.read_csv(os.path.join(RES, "claims_20260925", "lead_support_cells.csv"))
    start = ls[ls.event_set == "L2@1.0"].set_index("held").start_s
    out = {}
    l2 = x[x.label == "L2@1.0"].copy()
    l2["span"] = l2.onset - l2.held.map(start)
    out["conditional median lead (s)"] = rng(l2[l2.detected].groupby("seed").lead.median(), "%.0f")
    obs60 = l2[l2.span >= 60]
    out["observable at 60 s"] = "%d of 77" % obs60.held.nunique()
    out["met 60 s"] = rng(obs60.assign(ok=obs60.detected & (obs60.lead >= 60)).groupby("seed").ok.sum())
    d5 = ls[(ls.event_set == "L2@1.0") & (ls.dataset == "ds09_mech")]
    out["D5 median time from earliest admissible alarm to L2 (s)"] = "%.0f" % d5.span_s.median()
    out["D1-D3 minimum (s)"] = "%.0f" % ls[(ls.event_set == "L2@1.0") & ls.dataset.isin(["ds01_bak", "ds02_overcharge", "ds03_warwick"])].span_s.min()
    ep = pd.read_csv(os.path.join(RES, "claims_20260925", "event_profile_eligible.csv"))
    e = ep[(ep.model == "xgboost") & (ep.panel == "M1")]
    for es in ["L2@1.0", "E_TR", "E_elec"]:
        for ell in (60, 120):
            g = e[(e.event_set == es) & (e.lead_required_s == ell)]
            out["%s at %d s" % (es, ell)] = "%s of %d" % (rng(g.pass_all_eligible), g.eligible.max())
    # per voltage event, observable on that event's own span
    for lab in ["L3 voltage", "ISC 25mV"]:
        v = x[x.label == lab].copy()
        v["span"] = v.onset - v.held.map(start)
        o = v[v.span >= 60]
        out["%s at 60 s" % lab] = "%s of %d" % (rng(o.assign(ok=o.detected & (o.lead >= 60)).groupby("seed").ok.sum()), o.held.nunique())
    conf = RS.verdicts()
    c = l2[l2.held.map(conf).fillna(False) & (l2.span >= 60)]
    out["source-confirmed met 60 s"] = "%s of %d" % (rng(c.assign(ok=c.detected & (c.lead >= 60)).groupby("seed").ok.sum()), c.held.nunique())
    cf = pd.read_csv(os.path.join(RES, "claims_20260925", "claims_far.csv"))
    cf = cf[(cf.run == "native_final") & (cf.model == "xgboost") & (cf.representation == "no_age") & (cf.tau_kind == "full") & (cf.H_s == 300)]
    for pool, seg in [("source check", "head"), ("D6", "head"), ("D6", "full")]:
        g = cf[(cf.pool == pool) & (cf.segment == seg)]
        out["FAR %s %s" % (pool, seg)] = "%s of %d; upper %s; %s" % (rng(g.k), g.n.max(), rng(g.cp_upper95, "%.2f"),
                                                                    "/".join(sorted(set(v.split(" (")[0] for v in g.verdict))))
    r1c = pd.read_csv(os.path.join(REV, "stored", "R1c_heldout_source_confirmed.csv"))
    out["confirmed check, first 300 s"] = r1c[(r1c.threshold == "stored") & (r1c.subset == "source-confirmed") & (r1c.model == "xgboost")]["source check alarms, first 300 s"].iloc[0]
    dn = pd.read_csv(os.path.join(REV, "disputed_heldout", "negative_rates.csv"))
    dn = dn[(dn.model == "xgboost") & (dn.age_mode == "no_age") & (dn.pool == "source_mech") & (dn.window_s == "300") & (dn.anchor == "head")]
    out["29 check negatives, first 300 s"] = "%s of %d; %s" % (rng(dn.n_alarm), dn.n_eligible.max(),
                                                              "/".join(sorted(set(verdict(int(k), int(n)) for k, n in zip(dn.n_alarm, dn.n_eligible)))))
    df = pd.DataFrame({"quantity": list(out), "value": list(out.values())})
    write("T7_worked_example", df)


def main():
    s2j(); s2k(); s3b(); s4l(); s8i(); s8j(); s11c(); worked()


if __name__ == "__main__":
    main()
