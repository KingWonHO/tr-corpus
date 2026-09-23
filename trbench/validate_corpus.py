"""Post-build checks.  Fails loudly rather than shipping a quietly broken corpus.

Checks that have actually caught bugs during development are marked (*).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import schema as S           # noqa: E402
import make_windows as W     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tr-corpus")

RANGE = {          # plausible physical ranges after unit conversion
    "T_": (-50.0, 2500.0),
    "P_internal": (-5000.0, 20000.0),
    "P_chamber": (-5000.0, 20000.0),
    "V_cell": (-5.0, 60.0),
    "V_module": (-5.0, 1000.0),
    "I": (-1000.0, 1000.0),
    "gas_": (-10.0, 1.1e6),
    "F_": (-1e5, 1e5),
    "disp_": (-100.0, 500.0),
}


def _limits(ch):
    for k, v in RANGE.items():
        if ch.startswith(k) or ch == k:
            return v
    return None


def check():
    fails, warns = [], []
    exp = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))

    # (*) an empty registry means an adapter silently produced nothing
    if not len(exp):
        fails.append("registry is empty")
        return fails, warns, exp

    if exp[["dataset_id", "experiment_id"]].duplicated().any():
        fails.append("duplicate dataset_id/experiment_id in registry")

    for _, r in exp.iterrows():
        p = os.path.join(OUT, r["file"])
        if not os.path.exists(p):
            fails.append("missing file " + r["file"])

    for ds in exp["dataset_id"].unique():
        sub = exp[exp["dataset_id"] == ds]

        # (*) whole channels vanishing across a dataset means a mapping bug,
        # like #1's per-cell thermocouple numbering
        chans = [set(s.split(",")) for s in sub["channels_present"]]
        union = set().union(*chans)
        for ch in union:
            n = sum(ch in c for c in chans)
            if 0 < n < len(sub) * 0.5 and len(sub) > 4:
                warns.append("%s: %s present in only %d/%d experiments"
                             % (ds, ch, n, len(sub)))

        # (*) a time axis that collapses or runs backwards, like #4's `ms`
        # column and #12's misparsed .EXO header
        if sub["duration_s"].min() < 10:
            fails.append("%s: an experiment is shorter than 10 s" % ds)

    # onset must follow trigger, and both must lie inside the record
    for _, r in exp.iterrows():
        for lab in ("t_onset_L1", "t_onset_L2", "t_onset_L3"):
            v = r[lab]
            if pd.isna(v):
                continue
            if v < r["t_trigger"] - 1e-6:
                warns.append("%s/%s: %s (%.1f) precedes t_trigger (%.1f)"
                             % (r["dataset_id"], r["experiment_id"], lab,
                                v, r["t_trigger"]))
            hi = r["duration_s"] + (r.get("crop_start_s") or 0.0)
            if v > hi + 1.0:
                fails.append("%s/%s: %s beyond the record"
                             % (r["dataset_id"], r["experiment_id"], lab))

    # per-file content checks on a sample
    rng = np.random.default_rng(0)
    sample = exp.groupby("dataset_id", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 6), random_state=0))
    for _, r in sample.iterrows():
        df = pd.read_parquet(os.path.join(OUT, r["file"]))
        missing = [c for c in S.ALL_COLUMNS if c not in df.columns]
        if missing:
            fails.append("%s: missing columns %s" % (r["file"], missing[:4]))
            continue
        t = df["time_s"].to_numpy()
        if len(t) > 1 and not np.all(np.diff(t) > 0):
            fails.append("%s: time_s is not strictly increasing" % r["file"])
        for ch in S.CHANNELS:
            v = df[ch].to_numpy(float)
            m = df[S.mask_col(ch)].to_numpy(bool)
            if m.any() and not np.isfinite(v[m]).all():
                fails.append("%s: %s has NaN where mask says measured"
                             % (r["file"], ch))
            lim = _limits(ch)
            if lim is not None and m.any():
                lo, hi = np.nanmin(v[m]), np.nanmax(v[m])
                if lo < lim[0] or hi > lim[1]:
                    warns.append("%s: %s out of plausible range [%.4g, %.4g]"
                                 % (r["file"], ch, lo, hi))

    # (*) simulated data must never reach an evaluation split
    if "split" in exp.columns:
        bad = exp[(exp["role"] == "pretrain_only") & (exp["split"] != "pretrain")]
        if len(bad):
            fails.append("%d simulated experiments are in an evaluation split"
                         % len(bad))
        bad = exp[(exp["dataset_id"] == "ds12_arc") &
                  (exp["split"] != "test_zeroshot")]
        if len(bad):
            fails.append("%d ds12_arc experiments are not zero-shot held out"
                         % len(bad))

        # no experiment may appear in two splits
        for sp in exp["split"].unique():
            ids = set(exp[exp["split"] == sp]["file"])
            for other in exp["split"].unique():
                if other <= sp:
                    continue
                overlap = ids & set(exp[exp["split"] == other]["file"])
                if overlap:
                    fails.append("%s and %s share %d experiments"
                                 % (sp, other, len(overlap)))

    # windows: no experiment may straddle two splits, no trigger-side feature
    wdir = os.path.join(OUT, "windows")
    if os.path.isdir(wdir):
        seen = {}
        for f in sorted(os.listdir(wdir)):
            if not f.endswith(".npz"):
                continue
            z = np.load(os.path.join(wdir, f), allow_pickle=True)
            feats = list(z["features"])
            leak = [c for c in feats if c in S.TRIGGER_SIDE]
            if leak:
                fails.append("%s: trigger-side channels in features %s" % (f, leak))
            if z["X"].shape[:2] != z["mask"].shape[:2]:
                fails.append("%s: X and mask shapes disagree" % f)
            if not np.isfinite(z["X"]).all():
                fails.append("%s: non-finite values in X" % f)
            split = f[:-4]
            for k in set(z["experiment"].tolist()):
                if k in seen and seen[k] != split:
                    fails.append("experiment %s appears in both %s and %s"
                                 % (k, seen[k], split))
                seen[k] = split
    return fails, warns, exp


if __name__ == "__main__":
    fails, warns, exp = check()
    print("experiments: %d across %d datasets"
          % (len(exp), exp["dataset_id"].nunique()))
    print("\nWARN (%d)" % len(warns))
    for w in warns[:40]:
        print("  -", w)
    if len(warns) > 40:
        print("  ... %d more" % (len(warns) - 40))
    print("\nFAIL (%d)" % len(fails))
    for f in fails[:40]:
        print("  -", f)
    sys.exit(1 if fails else 0)
