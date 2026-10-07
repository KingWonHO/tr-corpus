"""E1: how much does the onset definition move the answer?

Three parts, in increasing cost and in the order the question actually splits:

  E1-0  label audit.  No model at all -- just how the L2 rate threshold and the
        four definitions relate to each other across the corpus.
  E1-A  same alarm, different yardstick.  Train once, then score the SAME risk
        trace against each definition of onset.  Isolates the effect of the
        yardstick from the effect of the model.
  E1-B  retrain per label.  The definition now enters training as well, so the
        model's alarm time and the sensor increment can both move.

E1-A and E1-B answer different questions and the gap between them is the point:
if a lead time changes under E1-A it is a reporting artefact, and if it changes
further under E1-B the definition has changed what the model learns.

    uv run python trbench/run_e1.py --part 0
    uv run python trbench/run_e1.py --part A --models xgboost
    uv run python trbench/run_e1.py --part B --models xgboost,lightgbm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import schema as S          # noqa: E402
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import run_e3 as R          # noqa: E402

OUT = R.OUT

# The four definitions plus the L2 threshold sweep.  #1 is the only dataset
# that carries all of them, which is why E1-A and E1-B run there.
LABELS = ["t_onset_L2_0.5", "t_onset_L2_1.0", "t_onset_L2_2.0",
          "t_onset_L2_unguarded", "t_onset_L1", "t_onset_L3", "t_isc"]

# The L2 sweep varies one PARAMETER of one rule; the rest are different physical
# EVENTS.  Mixing them in a single table conflates two kinds of sensitivity, so
# they are reported separately.
L2_SWEEP = ["t_onset_L2_0.5", "t_onset_L2_1.0", "t_onset_L2_2.0"]
CROSS_DEF = ["t_onset_L2_1.0", "t_onset_L1", "t_onset_L3", "t_isc"]

# Which channels a rule needs at all.  "Detected" and "computable" are not the
# same thing: a rule can have its sensor and still find no event, and reporting
# the detection count as coverage understates how many experiments were even
# eligible.
REQUIRES = {
    "t_onset_L2_0.5": S.GROUP_T_SURF + S.GROUP_TINT + ["T_vent", "T_vent_2"],
    "t_onset_L2_1.0": S.GROUP_T_SURF + S.GROUP_TINT + ["T_vent", "T_vent_2"],
    "t_onset_L2_2.0": S.GROUP_T_SURF + S.GROUP_TINT + ["T_vent", "T_vent_2"],
    "t_onset_L2_unguarded": S.GROUP_T_SURF + S.GROUP_TINT + ["T_vent", "T_vent_2"],
    "t_onset_L1": S.PRESSURE,
    "t_onset_L3": S.GROUP_V,
    "t_isc": S.GROUP_V,
}
PRETTY = {"t_onset_L2_0.5": "L2@0.5", "t_onset_L2_1.0": "L2@1.0",
          "t_onset_L2_2.0": "L2@2.0", "t_onset_L2_unguarded": "L2 unguarded",
          "t_onset_L1": "L1 venting", "t_onset_L3": "L3 voltage",
          "t_isc": "ISC 25mV"}


def registry():
    return pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))


def _key(r):
    return r["dataset_id"] + "/" + r["experiment_id"]


# ------------------------------------------------------------------ E1-0
def part0():
    """Label audit: no model, no training."""
    e = registry()
    m = e[e["role"] != "pretrain_only"].copy()
    m["key"] = m.apply(_key, axis=1)
    chans = m["channels_present"].fillna("").apply(lambda c: set(c.split(",")))
    rows = []
    for lab in LABELS:
        v = m[lab]
        elig = chans.apply(lambda c: any(x in c for x in REQUIRES[lab]))
        rows.append(dict(
            label=PRETTY[lab], column=lab,
            n_sensor_available=int(elig.sum()),
            n_detected=int(v.notna().sum()),
            detect_rate_pct=round(100 * v.notna().sum() / max(int(elig.sum()), 1), 1),
            n_datasets_detected=int(m.loc[v.notna(), "dataset_id"].nunique())))
    cover = pd.DataFrame(rows)

    # Pairwise offsets, per trigger, among experiments where BOTH labels fired.
    # Every row carries n and the full range: a median alone invites reading a
    # central value as a bound, and these spreads are wide (ISC - L1 under
    # heating is -384 s at the median but runs -630 to -138 s over 11 cells).
    off = []
    for a in LABELS:
        for b in LABELS:
            if a >= b:
                continue
            sub = m.dropna(subset=[a, b])
            for g, gg in sub.groupby("trigger"):
                if len(gg) < 2:
                    continue
                d = gg[a] - gg[b]
                off.append(dict(label_a=PRETTY[a], label_b=PRETTY[b], trigger=g,
                                n=len(gg), median_s=round(float(d.median()), 1),
                                min_s=round(float(d.min()), 1),
                                max_s=round(float(d.max()), 1),
                                iqr_s=round(float(d.quantile(.75) - d.quantile(.25)), 1),
                                same_rule=bool(a in L2_SWEEP and b in L2_SWEEP)))
    offsets = pd.DataFrame(off)

    # how far the L2 threshold alone moves the event, and whether it changes
    # whether there IS an event
    sweep = []
    for g, gg in m.groupby("trigger"):
        row = dict(trigger=g, n_experiments=len(gg))
        for lab in ["t_onset_L2_0.5", "t_onset_L2_1.0", "t_onset_L2_2.0"]:
            row["n_" + PRETTY[lab]] = int(gg[lab].notna().sum())
        both = gg.dropna(subset=["t_onset_L2_0.5", "t_onset_L2_2.0"])
        if len(both):
            d = both["t_onset_L2_2.0"] - both["t_onset_L2_0.5"]
            row["shift_0.5to2.0_median_s"] = round(float(d.median()), 1)
            row["shift_max_s"] = round(float(d.max()), 1)
        sweep.append(row)
    sweep = pd.DataFrame(sweep)
    return cover, offsets, sweep


# ------------------------------------------------------------------ E1-A/B
def _relabel(d, reg, label):
    """Rebuild (y_time, y_event) for an alternative onset definition."""
    onset = {_key(r): r[label] for _, r in reg.iterrows()}
    end = {_key(r): (r["crop_start_s"] if pd.notna(r.get("crop_start_s")) else 0.0)
           + r["duration_s"] for _, r in reg.iterrows()}
    exp = d["experiment"].astype(str)
    on = np.array([onset.get(k, np.nan) for k in exp], float)
    ed = np.array([end.get(k, np.nan) for k in exp], float)
    ev = np.isfinite(on)
    y_time = np.where(ev, on - d["t_end"], ed - d["t_end"])
    return y_time.astype(np.float32), ev.astype(np.uint8), onset


def _folds(d, meta, panels):
    """Experiments expressing every panel and reaching runaway under L2."""
    ds = d["experiment"].astype(str)
    elig = None
    for p in panels:
        e = set(MW.panel_experiments(p)["file"].map(
            lambda f: "%s/%s" % (f.split("/")[1], f.split("/")[2][:-8])))
        elig = e if elig is None else (elig & e)
    return ds, sorted(elig & set(meta[meta.t_onset.notna()].key)), elig


def partA(d, meta, reg, models, panels, seed=0):
    """One trained alarm, scored against every definition of onset.

    Any change here is purely a change of yardstick: the model, the fold and
    the risk trace are identical across rows.
    """
    ds, folds, elig = _folds(d, meta, panels)
    trig = dict(zip(meta.key, meta.t_trigger))
    rows = []
    for panel in panels:
        cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
        for held in folds:
            te = np.flatnonzero(ds == held)
            tr = np.flatnonzero((ds != held) & np.isin(ds, list(elig)))
            neg = np.flatnonzero((d["y_tr"] == 0) & (ds != held))
            for m in models:
                rte, rneg = R.run_fold(d, cols, m, tr, [te, neg], seed)
                for lab in LABELS:
                    _, _, onset = _relabel(d, reg, lab)
                    if not np.isfinite(onset.get(held, np.nan)):
                        continue
                    c = SV.operating_curve(
                        np.concatenate([d["t_end"][te], d["t_end"][neg]]),
                        np.concatenate([rte, rneg]),
                        np.concatenate([d["experiment"][te], d["experiment"][neg]]),
                        onset, trig,
                        np.concatenate([np.zeros(len(te), bool),
                                        d["y_tr"][neg] == 0]))
                    rows.append(dict(part="E1-A", panel=panel, model=m,
                                     held=held, label=PRETTY[lab],
                                     **SV.curve_summary(c)))
    return pd.DataFrame(rows)


def partB(d, meta, reg, models, panels, seed=0, labels=None):
    """Retrain with each definition as the target.

    The definition now shapes the hazard the model fits, so both the alarm time
    and the size of the sensor increment can move -- which is what separates a
    reporting artefact from a modelling one.
    """
    ds, folds, elig = _folds(d, meta, panels)
    trig = dict(zip(meta.key, meta.t_trigger))
    rows = []
    for lab in (labels or LABELS):
        y_time, y_event, onset = _relabel(d, reg, lab)
        dd = dict(d)
        dd["y_time"], dd["y_event"] = y_time, y_event
        dd["y_tr"] = y_event
        for panel in panels:
            cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
            for held in folds:
                if not np.isfinite(onset.get(held, np.nan)):
                    continue
                te = np.flatnonzero(ds == held)
                tr = np.flatnonzero((ds != held) & np.isin(ds, list(elig)))
                neg = np.flatnonzero((d["y_tr"] == 0) & (ds != held))
                for m in models:
                    rte, rneg = R.run_fold(dd, cols, m, tr, [te, neg], seed)
                    c = SV.operating_curve(
                        np.concatenate([dd["t_end"][te], dd["t_end"][neg]]),
                        np.concatenate([rte, rneg]),
                        np.concatenate([dd["experiment"][te], dd["experiment"][neg]]),
                        onset, trig,
                        np.concatenate([np.zeros(len(te), bool),
                                        d["y_tr"][neg] == 0]))
                    rows.append(dict(part="E1-B", panel=panel, model=m,
                                     held=held, label=PRETTY[lab],
                                     **SV.curve_summary(c)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0", choices=["0", "A", "B"])
    ap.add_argument("--models", default="xgboost")
    ap.add_argument("--panels", default="M1,M2,M3,M4,M5")
    ap.add_argument("--windows", default="W60")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="all",
                    choices=["all", "l2sweep", "crossdef"],
                    help="l2sweep varies one rule's threshold; crossdef changes "
                         "the physical event. They are different questions.")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    t0 = time.time()

    if a.part == "0":
        cover, offsets, sweep = part0()
        for name, df in [("e1_0_coverage", cover), ("e1_0_offsets", offsets),
                         ("e1_0_l2sweep", sweep)]:
            df.to_csv(os.path.join(OUT, "results",
                                   name + R.out_tag(a.windows) + ".csv"),
                      index=False)
        print(cover.to_string(index=False))
        print()
        print(sweep.to_string(index=False))
        return

    d, meta = R.load_windows(a.windows)
    reg = registry()
    models = a.models.split(",")
    panels = a.panels.split(",")
    labs = {"all": None, "l2sweep": L2_SWEEP, "crossdef": CROSS_DEF}[a.labels]
    if a.part == "A":
        res = partA(d, meta, reg, models, panels, a.seed)
    else:
        res = partB(d, meta, reg, models, panels, a.seed, labs)
    suffix = "" if a.labels == "all" else "_" + a.labels
    path = os.path.join(OUT, "results",
                        "e1_%s%s%s.csv" % (a.part, suffix, R.out_tag(a.windows)))
    res = R.save_merged(res, path)
    print("%d rows, %.0fs -> %s" % (len(res), time.time() - t0, path))


if __name__ == "__main__":
    main()
