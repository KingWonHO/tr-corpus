"""Split-sensitivity run: move source negatives from training into evaluation.

Protocol: reports/16_split_sensitivity_protocol.md, committed before this ran.
The frozen split leaves 25 calibration and 26 check negatives, below the 29 that
zero alarms would need to bound a 10 % false-alarm probability.  This asks
whether that shortfall is a property of the corpus or of the split: source
negatives are re-assigned by whole cell design, outcome-blind, so that the
evaluation pool is larger, and the two tree probes are refitted.  Positives keep
their roles and D6 is untouched.

    uv run python trbench/run_split_sensitivity.py --out tr-corpus/results/validation_20260918_split_sensitivity

Nothing here selects on results: the assignment is fixed by the hash of the cell
design name, and every seed is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cluster_bootstrap as CB   # noqa: E402  (cell-design rule, unchanged)
import run_validation as RV      # noqa: E402
import run_e3 as R               # noqa: E402  (window loader)

CHECK_TARGET = 60      # protocol 16 section 2.1: fill until at least this many
CAL_TARGET = 40


def assignment(reg):
    """Whole cell designs to evaluation, then calibration, then training."""
    reg = reg.copy()
    reg["key"] = reg.dataset_id + "/" + reg.experiment_id.astype(str)
    src_neg = reg[(reg.dataset_id != "ds12_arc") & (reg.role != "pretrain_only")
                  & reg["t_onset_L2_1.0"].isna()].copy()
    src_neg["cluster"] = src_neg.apply(CB.cell_model, axis=1)
    order = sorted(src_neg.cluster.unique(),
                   key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    sizes = src_neg.cluster.value_counts()
    roles, n_check, n_cal = {}, 0, 0
    for c in order:
        if n_check < CHECK_TARGET:
            roles[c], n_check = "test", n_check + sizes[c]
        elif n_cal < CAL_TARGET:
            roles[c], n_cal = "val", n_cal + sizes[c]
        else:
            roles[c] = "train"
    src_neg["new_split"] = src_neg.cluster.map(roles)
    return src_neg[["key", "cluster", "split", "new_split"]], order, roles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="W60_native")
    ap.add_argument("--models", default="xgboost,lightgbm")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    d, meta = R.load_windows(a.windows, splits=("train", "val", "test", "test_zeroshot"))
    reg_path = Path(R.OUT) / "registry" / "experiments.csv"
    reg = pd.read_csv(reg_path)
    moves, order, roles = assignment(reg)
    reg2 = reg.copy()
    reg2["key"] = reg2.dataset_id + "/" + reg2.experiment_id.astype(str)
    reg2 = reg2.set_index("key")
    reg2.loc[moves.key, "split"] = moves.new_split.to_numpy()
    reg2 = reg2.reset_index(drop=False).drop(columns="key")
    # positives and D6 must be untouched
    same = (reg.split == reg2.split) | reg["t_onset_L2_1.0"].isna()
    assert same.all(), "a positive changed role"
    assert (reg[reg.dataset_id == "ds12_arc"].split.to_numpy()
            == reg2[reg2.dataset_id == "ds12_arc"].split.to_numpy()).all(), "D6 changed"
    moves.to_csv(out / "split_assignment_sensitivity.csv", index=False)

    counts = moves.groupby("new_split").size().to_dict()
    print("source negatives:", counts, "clusters:", {r: [c for c in order if roles[c] == r] for r in set(roles.values())})

    RV.run(d, meta, reg2, a.models.split(","), [int(s) for s in a.seeds.split(",")],
           ["no_age"], out, n_boot=a.bootstrap)

    manifest = dict(protocol="reports/16_split_sensitivity_protocol.md",
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True).strip(),
                    arguments=vars(a), negatives_per_role=counts,
                    cluster_order=order, cluster_roles=roles,
                    registry_sha256=hashlib.sha256(reg_path.read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
