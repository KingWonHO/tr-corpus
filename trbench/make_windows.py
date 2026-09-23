"""Turn the standardized corpus into model-ready windows.

One pass, one array per split, all channels kept.  The sensor panels M1..M6 are
NOT baked into separate files: each window carries its availability mask, so an
E3 run selects a panel by zeroing mask columns at training time.  That is the
only way the panels stay comparable -- rebuilding the windows per panel would
also change the windows themselves.

Windowing rules
    length          WINDOW_S seconds at 1 Hz
    stride          dense (DENSE_STRIDE) inside [t_onset - DENSE_PRE_S,
                    t_onset + ALPHA_S], sparse (SPARSE_STRIDE) elsewhere.
                    The pre-onset approach is what the paper measures, so it is
                    where the samples should be spent.
    truncation      nothing after t_onset + ALPHA_S (plan 1-6)
    normalization   chosen by --norm (plan E6, and see reports section 11):
                    `per_experiment` z-scores against that cell's own
                    pre-trigger baseline, `global` against the corpus-pooled
                    baseline, `raw` leaves the physical units alone.
                    All three are leak-free -- the statistics come from
                    pre-trigger windows only -- but only the last two are
                    COMMENSURABLE across datasets.  Per-experiment scaling
                    divides by whatever the baseline noise happened to be, and
                    a quiet baseline (#1 sits at 0.0081 degC) turns a 400 degC
                    excursion into z = 50,000 while a noisy one reports the
                    same physics near z = 1.  Measured, that spreads the
                    datasets 1431x and inverts zero-shot risk on #12.

Targets
    y_time          survival time: seconds from window end to onset when the
                    experiment reaches runaway, otherwise to the end of record
    y_event         1 = onset observed, 0 = RIGHT-CENSORED
    y_ttl           seconds from window end to t_onset, NaN when there is none
    y_pre           1 if the window ends before onset and within LEAD_HORIZON_S
    y_tr            1 if this experiment ever reaches thermal runaway

`y_time` / `y_event` are the pair a discrete-time survival head needs.  The 190
measured experiments that never reach runaway are not negatives -- they are
right-censored observations carrying the information that no onset happened up
to the end of the record -- and without a censoring time their windows have no
usable label at all.  Train on windows with `y_time > 0`; the ones at or past
onset are post-event.

Windows never straddle two experiments and every window records its source
experiment, so results can be aggregated at experiment level.

Window length is a swept parameter, not a per-trigger setting: giving the
mechanical experiments a shorter window than the heated ones would confound any
trigger comparison with a window-length difference.  Each length is written to
its own directory (`windows/W60/`, `windows/W30/`, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import schema as S     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tr-corpus")

WINDOW_S = 60
DENSE_STRIDE = 1
SPARSE_STRIDE = 10
DENSE_PRE_S = 600.0
LEAD_HORIZON_S = 600.0

# Trigger-side actuation is dropped outright: it tells the model when the abuse
# started, which is exactly what an early-warning model is supposed to infer.
FEATURES = [c for c in S.CHANNELS if c not in S.TRIGGER_SIDE]

# Causal build only.  Seconds since the last REAL observation on this row.
AGE_CH = "obs_age"


def _starts(n, t, t_onset, window_s=WINDOW_S, uniform=False):
    """Window start indices under the dense/sparse stride rule.

    `uniform` drops the onset-anchored dense region entirely.  The dense stride
    exists to spend samples where the paper measures, but it makes an event
    experiment sampled ten times more finely than a non-event one AROUND A TIME
    ONLY THE LABEL KNOWS -- so a false-alarm rate computed across the two is
    comparing different sampling densities.  The causal build uses one stride
    everywhere and pays for it in sample count.
    """
    last = n - window_s
    if last < 0:
        return np.empty(0, int)
    if uniform or t_onset is None or not np.isfinite(t_onset):
        return np.arange(0, last + 1, SPARSE_STRIDE)
    end_t = t[np.arange(window_s - 1, n)]          # window end time by start idx
    dense = (end_t >= t_onset - DENSE_PRE_S) & (end_t <= t_onset + S.ALPHA_S)
    idx = np.arange(0, last + 1)
    keep = np.where(dense[:last + 1],
                    idx % DENSE_STRIDE == 0,
                    idx % SPARSE_STRIDE == 0)
    return idx[keep]


def causalize(df):
    """Replace non-causal interpolation with a per-channel forward fill.

    `resample_channel` fills gaps with `np.interp`, which reads the NEXT real
    sample -- on #12 that means 90-96% of what a model sees was computed from
    the future.  Here every channel is carried forward from ITS OWN last real
    observation.

    Per channel matters, and the first attempt at this got it wrong.  Driving
    the fill from the row-level `is_interpolated` -- an OR across channels --
    froze #1's temperature and voltage, which are measured at a full 1 Hz with
    zero interpolation, because one slow channel sampled every 3.6 s set the row
    flag on 72% of rows.  That collapsed a real M4 increment to zero and was
    nearly reported as a finding (see the log, section 15-C).  `standardize` now
    keeps the per-channel flag that `resample_channel` always returned; where an
    older parquet lacks it the row flag is used, which is exact for #12 (one
    instrument, one clock) and a no-op for the datasets that interpolate nothing.

    `obs_age` is the age of the STALEST present channel, so it is panel-agnostic
    and conservative: a temperature-only panel on #1 is told the gas channel's
    age too.  For #12, where every present channel shares a clock, it is exact.
    """
    df = df.copy().reset_index(drop=True)
    t = df["time_s"].to_numpy(float)
    row_flag = (df["is_interpolated"].to_numpy() != 0
                if "is_interpolated" in df.columns else np.zeros(len(df), bool))

    age = np.full(len(df), -1.0)
    seen_any = np.zeros(len(df), bool)
    for ch in FEATURES:
        if ch not in df.columns:
            continue
        mcol = S.mask_col(ch)
        icol = S.interp_col(ch) if hasattr(S, "interp_col") else None
        itp = (df[icol].to_numpy() != 0) if (icol and icol in df.columns) else row_flag
        have = df[mcol].to_numpy(np.uint8).astype(bool) if mcol in df.columns else             np.isfinite(df[ch].to_numpy())
        real = have & ~itp
        idx = np.maximum.accumulate(np.where(real, np.arange(len(df)), -1))
        seen = idx >= 0
        src = np.where(seen, idx, 0)
        df[ch] = np.where(seen, df[ch].to_numpy(np.float32)[src], np.nan).astype("float32")
        if mcol in df.columns:
            df[mcol] = (df[mcol].to_numpy(np.uint8)[src] * seen).astype("uint8")
        a = np.where(seen, t - t[src], np.nan)
        age = np.where(seen & (np.nan_to_num(a, nan=-1.0) > age),
                       np.nan_to_num(a, nan=-1.0), age)
        seen_any |= seen

    df[AGE_CH] = np.where(seen_any, np.maximum(age, 0.0), np.nan).astype("float32")
    df[S.mask_col(AGE_CH)] = seen_any.astype("uint8")
    return df.loc[seen_any].reset_index(drop=True)


def _scaler(exp_rows, mode):
    """Baseline-only statistics, per experiment or pooled (plan E6).

    `raw` returns nothing: physical units are already common to every dataset,
    which is the whole reason for the option.
    """
    if mode == "raw":
        return {}
    stats = json.load(open(os.path.join(OUT, "scalers", "baseline_stats.json")))
    if mode == "per_experiment":
        return stats
    pooled = {}
    for ch in FEATURES:
        vals = [(s[ch]["mean"], s[ch]["std"], s[ch]["n"])
                for s in stats.values() if ch in s]
        if not vals:
            continue
        w = np.array([v[2] for v in vals], float)
        m = np.average([v[0] for v in vals], weights=w)
        sd = np.sqrt(np.average([v[1] ** 2 for v in vals], weights=w)) or 1.0
        pooled[ch] = dict(mean=float(m), std=float(sd), n=int(w.sum()))
    return {"__global__": pooled}


def build(split, norm="per_experiment", out_dir=None, window_s=WINDOW_S,
          causal=False):
    exp = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))
    rows = exp[exp["split"] == split]
    if not len(rows):
        print("no experiments in split", split)
        return None

    stats = _scaler(rows, norm)
    Xs, Ms, meta_rows = [], [], []
    y_ttl, y_pre, y_tr, src = [], [], [], []
    y_time, y_event, t_end_all, trig_all = [], [], [], []

    for _, r in rows.iterrows():
        df = pd.read_parquet(os.path.join(OUT, r["file"]))
        if causal:
            df = causalize(df)
        # plan 1-6: nothing beyond t_onset + ALPHA_S enters the analysis
        df = df[(df["qc_flag"].to_numpy() & S.QC_POST_ONSET) == 0]
        if len(df) < WINDOW_S:
            continue

        t = df["time_s"].to_numpy(float)
        t_onset = r["t_onset_L2"] if pd.notna(r["t_onset_L2"]) else None

        key = "%s/%s" % (r["dataset_id"], r["experiment_id"])
        st = ({} if norm == "raw" else
              stats.get(key, {}) if norm == "per_experiment" else
              stats["__global__"])

        feats = FEATURES + ([AGE_CH] if causal else [])
        F = np.zeros((len(df), len(feats)), np.float32)
        M = np.zeros((len(df), len(feats)), np.uint8)
        for j, ch in enumerate(feats):
            v = df[ch].to_numpy(np.float32)
            m = df[S.mask_col(ch)].to_numpy(np.uint8)
            s = st.get(ch)
            if s:
                v = (v - s["mean"]) / (s["std"] or 1.0)
            # Missing is not zero (plan 1-5).  The value slot is filled with 0
            # only so the tensor is dense; the mask is what says whether that 0
            # means anything, and a model must consume both.
            F[:, j] = np.where(m.astype(bool) & np.isfinite(v), v, 0.0)
            M[:, j] = m
        starts = _starts(len(df), t, t_onset, window_s, uniform=causal)
        if not len(starts):
            continue

        for i in starts:
            Xs.append(F[i:i + window_s])
            Ms.append(M[i:i + window_s])
        end_t = t[starts + window_s - 1]
        if t_onset is None:
            # right-censored: no onset was observed up to the end of record
            y_ttl.extend([np.nan] * len(starts))
            y_pre.extend([0] * len(starts))
            y_time.extend((float(t[-1]) - end_t).tolist())
            y_event.extend([0] * len(starts))
        else:
            ttl = t_onset - end_t
            y_ttl.extend(ttl.tolist())
            y_pre.extend(((ttl > 0) & (ttl <= LEAD_HORIZON_S)).astype(np.uint8).tolist())
            y_time.extend(ttl.tolist())
            y_event.extend([1] * len(starts))
        y_tr.extend([int(t_onset is not None)] * len(starts))
        # window end time on the experiment's own axis: the alarm rule needs it
        # directly, and reconstructing it from y_ttl is impossible for the
        # censored experiments, which have no onset to subtract from
        t_end_all.extend(end_t.tolist())
        trig_all.extend([r["trigger"]] * len(starts))
        src.extend([key] * len(starts))
        meta_rows.append(dict(key=key, dataset_id=r["dataset_id"],
                              experiment_id=r["experiment_id"],
                              n_windows=len(starts), t_onset=t_onset,
                              t_trigger=r["t_trigger"],
                              record_end_s=float(t[-1]), trigger=r["trigger"],
                              split=split))

    if not Xs:
        print("no windows in split", split)
        return None

    X = np.stack(Xs).astype(np.float32)
    M = np.stack(Ms).astype(np.uint8)
    # The scheme is part of the directory name for everything but the original
    # per-experiment build, whose path is kept so earlier results stay
    # reproducible against the files they were computed from.
    suffix = "" if norm == "per_experiment" else "_" + norm
    suffix += "_causal" if causal else ""
    out_dir = out_dir or os.path.join(OUT, "windows",
                                      "W%d%s" % (window_s, suffix))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s.npz" % split)
    yt = np.asarray(y_time, np.float32)
    ye = np.asarray(y_event, np.uint8)
    np.savez_compressed(
        path, X=X, mask=M,
        y_time=yt, y_event=ye,
        t_end=np.asarray(t_end_all, np.float64),
        trigger=np.asarray(trig_all),
        y_ttl=np.asarray(y_ttl, np.float32),
        y_pre=np.asarray(y_pre, np.uint8),
        y_tr=np.asarray(y_tr, np.uint8),
        experiment=np.asarray(src),
        features=np.asarray(FEATURES + ([AGE_CH] if causal else [])),
        window_s=window_s, hz=S.BASE_HZ, norm=norm)
    pd.DataFrame(meta_rows).to_csv(
        os.path.join(out_dir, "%s_experiments.csv" % split), index=False)
    usable = int((yt > 0).sum())
    print("%-14s %6d windows / %3d exp  X%s  %.1f MB  |  survival-usable %6d "
          "(event %5d, censored %5d)"
          % (split, len(X), len(meta_rows), X.shape, os.path.getsize(path) / 1e6,
             usable, int(((yt > 0) & (ye == 1)).sum()),
             int(((yt > 0) & (ye == 0)).sum())))
    return X.shape


def panel_mask(features, panel):
    """Column selector for one sensor panel (E3 is run by toggling this).

    `obs_age` rides along with every panel when the build provides it.  It is
    not a sensor -- it is how stale the sensors are, which in a causal build is
    metadata about whatever the panel selected, the same way the availability
    mask is.  Leaving it out would hand the model a forward-filled value with no
    way to know it is an hour old.
    """
    sensors = set(S.SENSOR_PANELS[panel]) - set(S.TRIGGER_SIDE)
    # Native builds carry an age for each sensor. A hidden sensor's clock must
    # not enter a smaller panel through a global maximum-age feature.
    allowed = sensors | {"age__" + ch for ch in sensors} | {AGE_CH}
    return np.array([f in allowed for f in features], bool)


def panel_experiments(panel, split=None):
    """Experiments that can express `panel` -- the eligible set for an E3 run.

    A panel comparison is only valid on experiments carrying BOTH panels, so a
    paired M_k vs M_j run must intersect these two sets, not pool them.
    """
    exp = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))
    if split is not None:
        exp = exp[exp["split"] == split]
    ok = exp["channels_present"].apply(
        lambda s: panel in S.panel_expressible(s.split(",")))
    return exp[ok]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="per_experiment",
                    choices=["per_experiment", "global", "raw"])
    ap.add_argument("--causal", action="store_true",
                    help="forward-fill from real observations only, add obs_age, "
                         "and use one window stride for every experiment")
    ap.add_argument("--splits", default="train,val,test,test_zeroshot")
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", default="60,30",
                    help="window lengths in seconds; each gets its own directory")
    a = ap.parse_args()
    for w in [int(x) for x in a.windows.split(",")]:
        print("== window %d s" % w)
        for sp in a.splits.split(","):
            build(sp, a.norm, a.out, window_s=w, causal=a.causal)
