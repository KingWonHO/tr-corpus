"""Sensitivity tables from stored results for the review of 2026-10-03 (reports/36).

No model is refitted.  Writes to tr-corpus/results/review_20261006/stored/.

R1  source-confirmed subsets (records whose physical outcome the source states)
    a. internal design, six learned models, stored alarms and thresholds,
       scored on the 37 source-confirmed paired positives;
    b. internal design, trees and rule, threshold recalibrated on the 12
       source-confirmed calibration negatives from the stored per-fold traces,
       FAR on the 17 source-confirmed check negatives;
    c. held-out design, all models, stored risks; threshold as stored and
       recalibrated on the 12 confirmed calibration negatives.
R2  reference-event reversals by dataset (D1, D2, D3, D5, D1-D3).
R3  onset offsets of the rate-threshold variants against the 10 s alarm grid.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "trbench"))
import evalv2 as V            # noqa: E402
import run_e3 as R            # noqa: E402
import run_validation as RV   # noqa: E402
import survival as SV         # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results")
OUT = os.path.join(RES, "review_20261006", "stored")
LEARNED = ["xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer"]
DSN = {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3", "ds04_osf": "D4", "ds09_mech": "D5", "ds12_arc": "D6"}
LABELS = ["L2@0.5", "L2@2.0", "L3 voltage", "ISC 25mV", "L1 venting"]
LEADS = (0, 30, 60, 120)


def rng(s, fmt="%d"):
    s = pd.Series(s).dropna()
    if not len(s):
        return "—"
    lo, hi = s.min(), s.max()
    return (fmt % lo) if lo == hi else ("%s–%s" % (fmt % lo, fmt % hi))


def verdicts():
    c = pd.read_csv(os.path.join(HERE, "tables", "S2g_outcome_reconciliation_cells.csv"))
    ds_of = {v: k for k, v in DSN.items()}
    c["key"] = c.dataset.map(ds_of).fillna(c.dataset) + "/" + c.experiment_id
    return dict(zip(c.key, c.source_verdict.eq("TR") | c.source_verdict.eq("non-TR")))


def internal_raw():
    raw = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "validation_20260925_all7_rescored", "e3a_*s0to4.csv"))])
    raw["dataset"] = raw.held.str.split("/").str[0].map(DSN)
    return raw


def spans():
    ls = pd.read_csv(os.path.join(RES, "claims_20260925", "lead_support_cells.csv"))
    ls = ls[ls.event_set == "L2@1.0"]
    return dict(zip(ls.held, ls.span_s))


def flips(sub, lab):
    a = sub[sub.label == "L2@1.0"][["model", "panel", "seed", "held", "detected"]]
    b = sub[sub.label == lab][["model", "panel", "seed", "held", "detected"]]
    m = a.merge(b, on=["model", "panel", "seed", "held"], suffixes=("", "_x"))
    g = m.assign(f=m.detected != m.detected_x).groupby(["model", "panel", "seed"]).agg(n=("held", "size"), f=("f", "sum"))
    return g


def summarise_internal(sub, span):
    """One row per model: detection, reversals, M2-M1, lead profile over fits."""
    rows = []
    for m in LEARNED:
        s = sub[sub.model == m]
        if not len(s):
            continue
        l2 = s[s.label == "L2@1.0"]
        det = l2.groupby(["panel", "seed"]).agg(k=("detected", "sum"), n=("held", "nunique"))
        row = {"model": m, "cells": int(det.n.max()), "L2 detected": rng(det.k)}
        for lab in ["L2@0.5", "L2@2.0", "L3 voltage", "ISC 25mV"]:
            f = flips(s, lab)
            row["flips " + lab] = "%s / %d" % (rng(f.f), f.n.max()) if len(f) else "—"
        p = l2.pivot_table(index=["seed", "held"], columns="panel", values="detected", aggfunc="first").dropna()
        dd = p.groupby(level=0).apply(lambda t: (t.M2.astype(int).sum() - t.M1.astype(int).sum()) / len(t))
        row["M2−M1 detection rate"] = rng(dd, "%+.3f")
        l2 = l2.assign(span=l2.held.map(span))
        for ell in LEADS:
            obs = l2[l2.span >= ell] if ell else l2[l2.span > 0]
            met = obs.assign(ok=obs.detected & (obs.lead >= ell)).groupby(["panel", "seed"]).ok.sum()
            row["ℓ=%d met / observable" % ell] = "%s / %d" % (rng(met), obs.held.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def r1a(raw, conf, span):
    raw = raw[raw.model.isin(LEARNED)]
    out = []
    for name, keep in [("all paired positives", None), ("source-confirmed", True), ("trace-adjudicated only", False)]:
        sub = raw if keep is None else raw[raw.held.map(conf).fillna(False) == keep]
        out.append(summarise_internal(sub, span).assign(subset=name))
    return pd.concat(out, ignore_index=True)


def r1b(raw, conf):
    """Trees and rule: recalibrate on confirmed calibration negatives from the stored traces."""
    rows = []
    onsets = raw.drop_duplicates(["held", "label"]).set_index(["held", "label"]).onset
    for f in sorted(glob.glob(os.path.join(RES, "validation_20260925_e3a_trees", "traces", "*.npz"))):
        base = os.path.basename(f)[:-4]
        m, panel, s, held = base.split("_", 3)
        held = held.replace("__", "/")
        z = np.load(f)
        ck, kk = z["cal_keys"].astype(str), z["check_keys"].astype(str)
        cmask = np.array([conf.get(k, False) for k in ck]); kmask = np.array([conf.get(k, False) for k in kk])
        tau_c = SV.calibrate_threshold(z["cal_peaks"][cmask], ck[cmask], np.ones(cmask.sum(), bool))
        t, rr, tg = z["t"], z["risk_running_max"], float(z["t_trigger"])
        for kind, tau, far_peaks in [("stored", float(z["tau"]), z["check_peaks"]),
                                     ("stored, confirmed check", float(z["tau"]), z["check_peaks"][kmask]),
                                     ("recalibrated on confirmed", tau_c, z["check_peaks"][kmask])]:
            hit = np.flatnonzero(rr >= tau)
            a = float(t[hit[0]]) if len(hit) else None
            pre = a is not None and np.isfinite(tg) and a < tg
            for lab in ["L2@1.0"] + LABELS:
                on = onsets.get((held, lab), np.nan)
                if not np.isfinite(on):
                    continue
                det = bool(a is not None and a < on and not pre)
                rows.append(dict(model=m, panel=panel, seed=int(s[1:]), held=held, label=lab, kind=kind,
                                 tau=tau, alarm_time=a, detected=det, lead=on - a if det else np.nan,
                                 far_check=float((far_peaks >= tau).mean()), n_check=len(far_peaks),
                                 n_cal=int(cmask.sum()) if kind.startswith("recal") else len(ck),
                                 confirmed=conf.get(held, False)))
    return pd.DataFrame(rows)


def r1b_summary(t):
    out = []
    for kind, sub in t.groupby("kind"):
        for subset, s2 in [("all paired positives", sub), ("source-confirmed", sub[sub.confirmed])]:
            for m, s in s2.groupby("model"):
                l2 = s[s.label == "L2@1.0"]
                det = l2.groupby(["panel", "seed"]).agg(k=("detected", "sum"), n=("held", "nunique"))
                far = l2.groupby(["panel", "seed"]).far_check.median()
                row = dict(threshold=kind, subset=subset, model=m, cells=int(det.n.max()), **{"L2 detected": rng(det.k)},
                           **{"check FAR (fold median)": rng(far, "%.2f"), "check records": int(l2.n_check.iloc[0])})
                for lab in ["L2@0.5", "L2@2.0", "L3 voltage", "ISC 25mV"]:
                    f = flips(s.assign(detected=s.detected), lab)
                    row["flips " + lab] = "%s / %d" % (rng(f.f), f.n.max()) if len(f) else "—"
                out.append(row)
    return pd.DataFrame(out)


def r1c(conf):
    """Held-out design from stored risks, stored and confirmed-recalibrated thresholds."""
    d, meta = R.load_windows("W60_native", splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(os.path.join(ROOT, "tr-corpus", "registry", "experiments.csv"))
    plan = V.Plan(d, meta, reg, ["M1"])
    rows = []
    for run in ["validation_20260925_native", "validation_20260925_seq"]:
        for f in sorted(glob.glob(os.path.join(RES, run, "predictions", "*_no_age_s*.npz"))):
            model, seed = os.path.basename(f).split("_no_age_s")
            seed = int(seed[:-4])
            z = np.load(f)
            cal_e = d["experiment"][z["cal_idx"]].astype(str)
            cm = np.array([conf.get(k, False) for k in cal_e])
            taus = {"stored": float(z["tau"]),
                    "recalibrated on confirmed": SV.calibrate_threshold(z["cal_risk"][cm], cal_e[cm], np.ones(cm.sum(), bool))}
            arc_idx, arc_r = z["arc_idx"], z["arc_risk"]
            neg = d["y_tr"][arc_idx] == 0
            for kind, tau in taus.items():
                for subset in ["all", "source-confirmed"]:
                    rec = {}
                    for pool, idx, r in [("source_test", z["source_positive_idx"], z["source_positive_risk"]), ("D6", arc_idx, arc_r)]:
                        pr = RV.positive_records(d, idx, r, tau, plan)
                        if subset != "all":
                            pr = pr[pr.held.map(conf).fillna(False)]
                        rec[pool] = (int(pr.detected.sum()), len(pr), pr.lead.median())
                    far = {}
                    for pool, idx, r in [("source", z["source_idx"], z["source_risk"]), ("D6", arc_idx[neg], arc_r[neg])]:
                        for seg, (h, anc) in {"full": (None, "tail"), "first 300 s": (300, "head")}.items():
                            nr = RV.negative_records(d, idx, r, tau, h, anc)
                            nr = nr[nr.eligible]
                            if subset != "all":
                                nr = nr[nr.held.map(conf).fillna(False)]
                            far[(pool, seg)] = (int(nr.fired.astype(bool).sum()), len(nr))
                    rows.append(dict(model=model, seed=seed, threshold=kind, subset=subset, tau=tau,
                                     src_det=rec["source_test"][0], src_n=rec["source_test"][1],
                                     d6_det=rec["D6"][0], d6_n=rec["D6"][1], d6_lead=rec["D6"][2],
                                     src_far_full=far[("source", "full")][0], src_neg_n=far[("source", "full")][1],
                                     src_far_head=far[("source", "first 300 s")][0],
                                     d6_far_full=far[("D6", "full")][0], d6_neg_n=far[("D6", "full")][1],
                                     d6_far_head=far[("D6", "first 300 s")][0], n_cal=len(set(cal_e[cm])) if kind != "stored" else len(set(cal_e))))
    return pd.DataFrame(rows)


def r1c_summary(t):
    out = []
    for (kind, subset, m), g in t.groupby(["threshold", "subset", "model"], sort=False):
        out.append({"threshold": kind, "subset": subset, "model": m, "calibration records": int(g.n_cal.iloc[0]),
                    "source test detected": "%s / %d" % (rng(g.src_det), g.src_n.iloc[0]),
                    "D6 detected": "%s / %d" % (rng(g.d6_det), g.d6_n.iloc[0]),
                    "source check alarms, full": "%s / %d" % (rng(g.src_far_full), g.src_neg_n.iloc[0]),
                    "source check alarms, first 300 s": "%s / %d" % (rng(g.src_far_head), g.src_neg_n.iloc[0]),
                    "D6 negatives alarming, full": "%s / %d" % (rng(g.d6_far_full), g.d6_neg_n.iloc[0]),
                    "D6 negatives alarming, first 300 s": "%s / %d" % (rng(g.d6_far_head), g.d6_neg_n.iloc[0])})
    return pd.DataFrame(out)


def r2(raw):
    raw = raw[raw.model.isin(LEARNED)]
    rows = []
    groups = [("D1", ["D1"]), ("D2", ["D2"]), ("D3", ["D3"]), ("D1–D3", ["D1", "D2", "D3"]), ("D5", ["D5"]), ("all", ["D1", "D2", "D3", "D5"])]
    for name, dss in groups:
        sub = raw[raw.dataset.isin(dss)]
        row = {"cells": name}
        for lab in ["L2@0.5", "L2@2.0", "L3 voltage", "ISC 25mV"]:
            f = flips(sub, lab)
            row[lab] = "%s / %d" % (rng(f.f), f.n.max()) if len(f) else "—"
            tr = flips(sub[sub.model.isin(["xgboost", "lightgbm"])], lab)
            row[lab + " (trees)"] = "%s / %d" % (rng(tr.f), tr.n.max()) if len(tr) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def r3(raw):
    on = raw.drop_duplicates(["held", "label"]).pivot(index="held", columns="label", values="onset")
    on["dataset"] = [DSN[h.split("/")[0]] for h in on.index]
    grid = raw[raw.label == "L2@1.0"].alarm_time.dropna()
    rows = []
    for lab in ["L2@0.5", "L2@2.0"]:
        g = (on[lab] - on["L2@1.0"]).abs()
        for ds, s in on.assign(g=g).groupby("dataset").g:
            s = s.dropna()
            rows.append({"variant": lab, "dataset": ds, "cells": len(s), "same onset": int((s == 0).sum()),
                         "offset 1–9 s": int(((s > 0) & (s < 10)).sum()), "offset ≥ 10 s": int((s >= 10).sum()),
                         "median |offset| s": float(s.median()), "max |offset| s": float(s.max())})
    t = pd.DataFrame(rows)
    t.attrs["grid"] = sorted(set((grid % 10).round(3)))
    return t


def save(df, name, note=""):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    with open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as fh:
        if note:
            fh.write(note + "\n\n")
        fh.write(df.to_markdown(index=False) + "\n")
    print("  %s (%d rows)" % (name, len(df)))


def main():
    conf = verdicts()
    raw = internal_raw()
    span = spans()
    save(r1a(raw, conf, span), "R1a_internal_source_confirmed",
         "Internal design, six learned models x M1/M2 x seeds 0–4, stored alarms and thresholds; ranges over fits. "
         "Source-confirmed: the source states the physical outcome (S2g).")
    t = r1b(raw[raw.model.isin(["rule", "xgboost", "lightgbm"])], conf)
    t.to_csv(os.path.join(OUT, "R1b_internal_trees_recalibrated_cells.csv"), index=False)
    save(r1b_summary(t), "R1b_internal_trees_recalibrated",
         "Internal design, rule and trees, from the stored per-fold traces. 'recalibrated on confirmed': tau re-set at 10 % FAR on the "
         "12 source-confirmed calibration negatives; FAR on the 17 source-confirmed check negatives.")
    t = r1c(conf)
    t.to_csv(os.path.join(OUT, "R1c_heldout_source_confirmed_cells.csv"), index=False)
    save(r1c_summary(t), "R1c_heldout_source_confirmed",
         "Held-out design, sensor values, stored per-window risks; ranges over seeds 0–4 (rule seed 0).")
    save(r2(raw), "R2_flips_by_dataset",
         "Status flips of the same alarm against L2@1.0, six learned models x M1/M2 x seeds 0–4 (trees: XGBoost, LightGBM).")
    t = r3(raw)
    save(t, "R3_rate_variant_offsets", "Absolute onset offset from L2@1.0 per paired cell. Alarm endpoints fall on a 10 s grid "
         "(alarm time mod 10 = %s s)." % t.attrs["grid"])


if __name__ == "__main__":
    main()
