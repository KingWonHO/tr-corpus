"""Experiment-level train / validation / test splits and LOEO folds.

Plan section 1-8 is explicit: the split unit is the EXPERIMENT, never the
window.  Two windows from the same cell are near-duplicates -- splitting at the
window level puts the same event on both sides of the split and inflates every
score.  The effective sample size is the number of experiments, not the number
of windows, and that is what this file records.

Split policy
    ds12_arc      -> test_zeroshot   (held out entirely, plan E11)
    ds15_sim      -> pretrain        (never evaluated, plan E8)
    everything else, stratified within (dataset x trigger x onset-present):
                     ~70% train / ~15% val / ~15% test
LOEO folds are emitted for the core datasets separately, since the plan runs
leave-one-experiment-out as the primary protocol and the fixed split only as the
fast development loop.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import schema as S     # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tr-corpus")
SEED = 20260907
FRACS = dict(train=0.70, val=0.15, test=0.15)


def make(seed=SEED):
    exp = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))
    exp["split"] = ""

    zero = exp["dataset_id"] == "ds12_arc"
    pre = exp["role"] == "pretrain_only"
    exp.loc[zero, "split"] = "test_zeroshot"
    exp.loc[pre, "split"] = "pretrain"

    rng = np.random.default_rng(seed)
    pool = exp[~(zero | pre)]

    # Stratify so each dataset, trigger type and TR/no-TR outcome is represented
    # in all three splits: #9 alone would otherwise dominate and the small core
    # datasets could land entirely in one split.
    key = (pool["dataset_id"].astype(str) + "|" +
           pool["trigger"].astype(str) + "|" +
           pool["t_onset_L2"].notna().map({True: "tr", False: "notr"}))

    for _, idx in pool.groupby(key).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            exp.loc[idx, "split"] = "train"
            continue
        if n == 2:
            exp.loc[idx[:1], "split"] = "train"
            exp.loc[idx[1:], "split"] = "val"
            continue
        n_tr = max(1, int(round(n * FRACS["train"])))
        n_va = max(1, int(round(n * FRACS["val"])))
        n_tr = min(n_tr, n - 2)
        n_va = min(n_va, n - n_tr - 1)
        exp.loc[idx[:n_tr], "split"] = "train"
        exp.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        exp.loc[idx[n_tr + n_va:], "split"] = "test"

    # A handful of #9 tests reach runaway before the penetrator force clears its
    # contact threshold: the punch approaches slowly and the force only passes
    # 10% of stroke after the short has already started.  t_trigger is late for
    # these, not t_onset early.  Flagged rather than patched, so any analysis
    # anchored on t_trigger can exclude them explicitly.
    exp["anchor_inconsistent"] = (
        exp["t_onset_L2"].notna() & (exp["t_onset_L2"] < exp["t_trigger"]))

    exp.to_csv(os.path.join(OUT, "registry", "experiments.csv"), index=False)
    cols = ["dataset_id", "experiment_id", "file", "split", "trigger", "soh",
            "chemistry", "duration_s", "t_trigger", "t_onset_L2",
            "anchor_inconsistent", "n_channels", "channels_present"]
    exp[cols].to_csv(os.path.join(OUT, "splits", "split_assignment.csv"),
                     index=False)

    # LOEO folds over the core measured datasets
    core = exp[(exp["role"] == "core")]
    folds = [dict(fold=i,
                  test=[r["file"]],
                  train=[f for f in core["file"] if f != r["file"]])
             for i, (_, r) in enumerate(core.iterrows())]
    with open(os.path.join(OUT, "splits", "loeo_folds.json"), "w") as f:
        json.dump(dict(n_folds=len(folds), unit="experiment", folds=folds), f)

    summary = (exp.groupby(["dataset_id", "split"]).size()
               .unstack(fill_value=0))
    summary.to_csv(os.path.join(OUT, "splits", "split_summary.csv"))
    return exp, summary


if __name__ == "__main__":
    exp, summary = make()
    print(summary.to_string())
    print("\nLOEO folds over core datasets:",
          int((exp["role"] == "core").sum()))
    print("\nTR present (L2 onset found) by split:")
    print(exp.assign(tr=exp["t_onset_L2"].notna())
          .groupby(["split", "tr"]).size().unstack(fill_value=0).to_string())
