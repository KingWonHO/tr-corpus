"""E6: which normalisation lets an operating point survive the move?

E11 measured the damage.  Baseline-only z-scoring is leak-free -- the mean and
standard deviation come from pre-trigger windows only -- but it does not put the
datasets on a common scale, because it divides by whatever the baseline noise
happened to be.  #1's TS0330A sits at 25.17 degC with a baseline standard
deviation of 0.0081 degC, so a 400 degC excursion is reported as z = 50,000; #9,
whose baseline is genuinely noisy, reports the same physics near z = 1.  A model
fitted on one of those scales has no reason to apply on another.

So the question is not "is z-scoring leaky" (it is not) but "is it
commensurable" (it is not).  Six candidate schemes, all causal -- every one uses
only pre-trigger statistics, so any of them could run online:

  zscore        (x - mu) / sigma, per experiment                 [current]
  zscore_floor  sigma floored at the instrument resolution
  zscore_floor5 the same floor, five times larger                [sensitivity]
  delta         x - mu, left in physical units
  raw           x, physical units, baseline offset retained
  global        (x - mu) / sigma from corpus-pooled baselines

Two things have to be controlled for, and both were found the hard way.

FIRST, the missing-value sentinel is not scheme-neutral.  `window_features`
fills an absent reading with 0 after summarising, and 0 means different things
in different schemes: under `raw` it is 0 degC, comfortably outside the 12-600
degC the corpus actually spans, so it reads as "absent"; under `global` it is
the pooled mean, 21.6 degC, which sits inside the real data.  Measured, this
breaks the affine relation between the two schemes' feature columns exactly on
the rows where a channel is missing (correlation falls to 0.64 on T_ambient),
and the corpus is roughly half missing.  So every scheme is run twice, once with
the 0 sentinel and once letting the trees take NaN natively, and a scheme
difference is only believed when it survives both.

SECOND, this measurement is seed-noisy where the scheme is bad.  Zero-shot AUC
under `zscore` ranges 0.455-0.634 across five seeds for XGBoost; the single-seed
0.455 first recorded for E11 was the bottom of that range.  Everything here is
therefore reported as a mean over seeds with its range, and single-seed numbers
are not quoted.

Each scheme is scored on three things, because a scheme that fixes transfer by
destroying the signal is not a fix:

  transfer     #12 zero-shot pre-onset AUC, and what the operating point costs
               when priced on source negatives versus #12's own
  integrity    #12 in-domain AUC.  This is NOT expected to be invariant: the
               per-experiment schemes give each cell its own transform, so the
               trees see a different feature space, not a monotone image of one
  case study   #1's M5-vs-M1 lead time, the one panel result that held up

    uv run python trbench/run_e6.py --models xgboost,lightgbm --seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import run_e3 as R          # noqa: E402
import run_e11 as E11       # noqa: E402

OUT = R.OUT
TARGET = E11.TARGET

# Instrument resolution, used as a floor on the baseline standard deviation.
# These are the smallest differences the sensors can be trusted to report, so a
# baseline quieter than this is measuring the digitiser, not the cell.  The
# voltage figure is deliberately a fifth of the 25 mV ISC criterion -- the floor
# must not be able to swallow the smallest event the corpus labels.  Every value
# is re-run at 5x under `zscore_floor5`; if the two agree the exact choice is
# not carrying the result.
RESOLUTION = {"T_": 0.5, "P_": 0.005, "V_": 0.005, "I": 0.05,
              "gas_": 0.01, "F_": 0.01}


def _floor_for(ch):
    for pre, v in RESOLUTION.items():
        if ch.startswith(pre) or ch == pre:
            return v
    return 0.0


# ------------------------------------------------------------------ features
def window_features(X, M, cols, missing="zero"):
    """R.window_features, with the missing-value sentinel made explicit.

    The upstream version always writes 0, which is what every earlier experiment
    used and is kept as the default so those results stay reproducible.  Here it
    is a parameter, because 0 is a different point in the data under different
    normalisations and would otherwise be read as a scheme effect.
    """
    X, M = X[:, :, cols], M[:, :, cols].astype(bool)
    Xm = np.where(M, X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last = np.where(M[:, -1, :], X[:, -1, :], np.nan)
        mean = np.nanmean(Xm, axis=1)
        std = np.nanstd(Xm, axis=1)
        mx = np.nanmax(Xm, axis=1)
        mn = np.nanmin(Xm, axis=1)
        slope = last - np.where(M[:, 0, :], X[:, 0, :], np.nan)
        dd = np.diff(Xm, axis=1)
        dmax = np.nanmax(dd, axis=1)
        dlast = dd[:, -1, :]
    avail = M.mean(axis=1)
    F = np.concatenate([last, mean, std, mx, mn, slope, dmax, dlast, avail],
                       axis=1).astype(np.float32)
    if missing == "nan":
        # XGBoost and LightGBM both learn a default direction for NaN, so the
        # absent reading needs no stand-in value at all.
        return np.where(np.isfinite(F), F, np.nan).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ------------------------------------------------------------------- schemes
def to_physical(d, stats):
    """Undo the stored z-score, exactly.

    Windows ship normalised, and regenerating them once per scheme would re-read
    the whole corpus six times for a transform that is a multiply and an add.
    The inverse follows make_windows exactly, including its two edge cases: a
    channel with no baseline entry was never scaled, and a masked slot holds a
    structural 0 that must stay 0 rather than becoming mu.  Verified to round
    trip within float32 precision (max 3e-2 on values whose median is 475).
    """
    X = np.array(d["X"], dtype=np.float32)
    M = d["mask"].astype(bool)
    ds = np.asarray(d["experiment"]).astype(str)
    feats = list(d["features"])
    for key in np.unique(ds):
        st = stats.get(key, {})
        if not st:
            continue
        sel = ds == key
        for j, ch in enumerate(feats):
            s = st.get(ch)
            if not s:
                continue                       # was never scaled
            X[sel, :, j] = X[sel, :, j] * (s["std"] or 1.0) + s["mean"]
    return np.where(M, X, 0.0).astype(np.float32)


def apply_scheme(Xp, d, stats, pooled, scheme):
    """Physical units -> the scheme's units.  Masked slots stay 0 throughout."""
    if scheme == "raw":
        return Xp
    M = d["mask"].astype(bool)
    X = np.array(Xp, dtype=np.float32)
    ds = np.asarray(d["experiment"]).astype(str)
    feats = list(d["features"])
    mult = 5.0 if scheme == "zscore_floor5" else 1.0
    for key in np.unique(ds):
        st = pooled if scheme == "global" else stats.get(key, {})
        if not st:
            continue
        sel = ds == key
        for j, ch in enumerate(feats):
            s = st.get(ch)
            if not s:
                continue
            sd = s["std"] or 1.0
            if scheme == "delta":
                sd = 1.0
            elif scheme in ("zscore_floor", "zscore_floor5"):
                sd = max(sd, _floor_for(ch) * mult)
            X[sel, :, j] = (X[sel, :, j] - s["mean"]) / sd
    return np.where(M, X, 0.0).astype(np.float32)


def pooled_stats(stats, feats):
    """Corpus-pooled baseline statistics, as make_windows computes them."""
    out = {}
    for ch in feats:
        v = [(s[ch]["mean"], s[ch]["std"], s[ch]["n"])
             for s in stats.values() if ch in s]
        if not v:
            continue
        w = np.array([x[2] for x in v], float)
        out[ch] = dict(mean=float(np.average([x[0] for x in v], weights=w)),
                       std=float(np.sqrt(np.average([x[1] ** 2 for x in v],
                                                    weights=w)) or 1.0))
    return out


def scale_spread(d, X, cols):
    """How far apart the datasets sit, in the scheme's own units."""
    V = np.where(d["mask"][:, :, cols].astype(bool), X[:, :, cols], np.nan)
    ds = np.array([s.split("/")[0]
                   for s in np.asarray(d["experiment"]).astype(str)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = {k: float(np.nanpercentile(np.abs(V[ds == k]), 50))
               for k in sorted(set(ds))}
    lo, hi = min(med.values()), max(med.values())
    return med, (hi / lo if lo > 0 else np.inf)


# ----------------------------------------------------------------- measuring
def _agg(vals):
    a = np.array([v for v in vals if np.isfinite(v)], float)
    if not len(a):
        return np.nan, np.nan, np.nan
    return float(a.mean()), float(a.min()), float(a.max())


def transfer(d, meta, F, y, w, models, seeds):
    """#12 zero-shot, over seeds: discrimination, inversion, and the two prices."""
    ds = d["experiment"].astype(str)
    is_t = np.char.startswith(ds, TARGET + "/")
    onset = dict(zip(meta.key, meta.t_onset))
    trig = dict(zip(meta.key, meta.t_trigger))
    tgt = np.flatnonzero(is_t)
    src = np.flatnonzero(~is_t)
    src_neg = np.flatnonzero(~is_t & (d["y_tr"] == 0))

    rows = []
    for m in models:
        auc, near, far, fs, ft, inv = [], [], [], [], [], []
        for s in seeds:
            r_t, r_s = E11._fit_predict(F, y, w, m, src, [tgt, src_neg], s)
            z = E11._diagnose(d, onset, tgt, r_t, "zeroshot", m)
            _, sp = E11._point(d, onset, trig, tgt, r_t, src_neg, r_s)
            _, tp = E11._point(d, onset, trig, tgt, r_t)
            auc.append(z["auc_preonset"])
            near.append(z["ttl_0_60"]); far.append(z["ttl_1800_up"])
            inv.append(float(z["ttl_1800_up"] > z["ttl_0_60"]))
            fs.append(sp["far_at_recall"]); ft.append(tp["far_at_recall"])
        a, alo, ahi = _agg(auc)
        rows.append(dict(model=m, n_seeds=len(seeds),
                         zs_auc_preonset=a, zs_auc_lo=alo, zs_auc_hi=ahi,
                         zs_risk_near=_agg(near)[0], zs_risk_far=_agg(far)[0],
                         inverted_seeds=int(sum(inv)),
                         far_src=_agg(fs)[0], far_tgt=_agg(ft)[0]))
    return rows


def integrity(d, meta, F, y, w, models, seed=0):
    """#12 in-domain LOEO -- expensive, so one seed, and only the AUC matters."""
    ds = d["experiment"].astype(str)
    is_t = np.char.startswith(ds, TARGET + "/")
    onset = dict(zip(meta.key, meta.t_onset))
    trig = dict(zip(meta.key, meta.t_trigger))
    out = {}
    for m in models:
        idx, r = E11._oof_indomain(d, F, y, w, m, is_t, seed)
        i = E11._diagnose(d, onset, idx, r, "indomain", m)
        _, p = E11._point(d, onset, trig, idx, r)
        out[m] = (i["auc_preonset"], p["lead_at_recall"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="xgboost,lightgbm")
    ap.add_argument("--schemes", default="zscore,zscore_floor,zscore_floor5,"
                                         "delta,raw,global")
    ap.add_argument("--missing", default="zero,nan")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--windows", default="W60")
    ap.add_argument("--skip-case", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    models = a.models.split(",")
    seeds = list(range(a.seeds))

    d, meta = R.load_windows(a.windows,
                             splits=("train", "val", "test", "test_zeroshot"))
    stats = json.load(open(os.path.join(OUT, "scalers", "baseline_stats.json")))
    feats = list(d["features"])
    pooled = pooled_stats(stats, feats)
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], R.HORIZON_S)
    m1 = np.flatnonzero(MW.panel_mask(feats, "M1"))
    print("pool %d windows / %d exp | %d channels | %d seeds"
          % (len(d["X"]), len(meta), len(feats), len(seeds)))

    Xp = to_physical(d, stats)
    print("physical units recovered (%.0fs)" % (time.time() - t0))

    rows, scales, cases = [], [], []
    for scheme in a.schemes.split(","):
        ts = time.time()
        X = apply_scheme(Xp, d, stats, pooled, scheme)
        med, spread = scale_spread(d, X, m1)
        scales.append(dict(scheme=scheme, spread=spread, **med))
        d["X"] = X
        for miss in a.missing.split(","):
            F = window_features(X, d["mask"], m1, miss)
            tr = transfer(d, meta, F, y, w, models, seeds)
            ig = integrity(d, meta, F, y, w, models, 0)
            for r in tr:
                r["id_auc_preonset"], r["id_lead_at_recall"] = ig[r["model"]]
                rows.append(dict(scheme=scheme, missing=miss, **r))
                print("  %-14s %-4s %-9s zs-AUC %.3f [%.3f-%.3f] inv %d/%d"
                      " | FAR %.2f->%.2f | in-dom %.3f"
                      % (scheme, miss, r["model"], r["zs_auc_preonset"], r["zs_auc_lo"],
                         r["zs_auc_hi"], r["inverted_seeds"], len(seeds),
                         r["far_src"], r["far_tgt"], r["id_auc_preonset"]))
        if not a.skip_case:
            # does the one surviving panel result survive the scheme?
            cs, _ = R.arm_e3b(d, meta, models, ["M1", "M5"], 0)
            piv = cs.pivot_table(index=["model", "held"], columns="panel",
                                 values="lead_at_budget")
            for m in models:
                p = piv.loc[m].dropna() if m in piv.index.get_level_values(0) \
                    else pd.DataFrame()
                if not len(p):
                    continue
                dl = p["M5"] - p["M1"]
                cases.append(dict(scheme=scheme, model=m, n=len(dl),
                                  delta_m5_m1=float(dl.median()),
                                  improved=int((dl > 0).sum())))
        print("  %-14s scale spread %.0fx (%.0fs)" % (scheme, spread,
                                                      time.time() - ts))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results", "e6.csv"), index=False)
    pd.DataFrame(scales).to_csv(os.path.join(OUT, "results", "e6_scales.csv"),
                                index=False)
    if cases:
        pd.DataFrame(cases).to_csv(os.path.join(OUT, "results", "e6_case.csv"),
                                   index=False)
    print("\n" + res.round(3).to_string(index=False))
    if cases:
        print("\n" + pd.DataFrame(cases).to_string(index=False))
    print("\n%.0fs -> results/e6.csv, e6_scales.csv, e6_case.csv"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
