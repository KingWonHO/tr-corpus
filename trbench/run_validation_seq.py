"""Deep-tier companion to run_validation.py: same fixed source split, same
calibration, same head/tail matched-exposure accounting, for the sequence
models that consume the window tensor instead of summary features.

run_validation.py deliberately refuses anything but the rule and the two tree
baselines so that its bounded supplement stays auditable.  This script reuses
its record-level functions unchanged and only swaps the fitting step for
run_e3.run_fold, which dispatches sequence models on the (window, time,
channel) tensor.  Nothing about thresholds, eligibility or windows differs.

`--ages` takes the same predeclared M1 representations as run_validation.py
(no_age, with_age, surface_max_only, surface_mean_only), so the sequence tier
can answer the representation and observation-clock question on exactly the
arms the trees were run on rather than on no_age alone.

    uv run python trbench/run_validation_seq.py --windows W60_native_v2 \
        --models gru,mamba,itransformer,convtransformer --seeds 0,1,2 \
        --out tr-corpus/results/validation_20260914_native_deep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deep as DEEP                    # noqa: E402
import evalv2 as V                     # noqa: E402
import run_e3 as R                     # noqa: E402
import run_validation as RV            # noqa: E402
import survival as SV                  # noqa: E402


def run(d, meta, reg, models, seeds, age_modes, out, n_boot=1000, budget=V.BUDGET):
    plan = V.Plan(d, meta, reg, ["M1"])
    ds = plan.ds
    is_t = np.char.startswith(ds, V.TARGET + "/")
    tgt = np.flatnonzero(is_t & plan.observed)
    arc_pos = tgt[d["y_tr"][tgt] == 1]
    arc_neg = tgt[d["y_tr"][tgt] == 0]
    src_pos = np.flatnonzero(~is_t & plan.observed & (d["y_tr"] == 1)
                             & (np.asarray(d["split"]) == "test"))
    rng = np.random.default_rng(20260914)

    pos_rows, neg_rows, cal_rows = [], [], []
    for age_mode in age_modes:
        # the same predeclared M1 representations run_validation.py uses, so the
        # sequence tier answers Section 4.4 on exactly the tree-side arms
        cols = np.flatnonzero(RV.representation_mask(d["features"], age_mode))
        if not len(cols):
            raise ValueError("no channel survives representation %r" % age_mode)
        for m in models:
            for seed in seeds:
                t0 = time.time()
                r_cal, r_src_neg, r_src_pos, r_arc_pos, r_arc_neg = R.run_fold(
                    d, cols, m, plan.frozen_train,
                    [plan.cal_neg, plan.test_neg, src_pos, arc_pos, arc_neg], seed)
                tau = SV.calibrate_threshold(r_cal, d["experiment"][plan.cal_neg],
                                             np.ones(len(plan.cal_neg), bool), far=budget)
                cal = RV.negative_records(d, plan.cal_neg, r_cal, tau)
                tag = dict(model=m, seed=seed, age_mode=age_mode, tau=tau)
                cal_rows.append(dict(**tag, n_cal=len(cal),
                                     n_cal_alarm=int(cal.fired.sum())))
                for pool, idx, risk in [("source_test", src_pos, r_src_pos),
                                        ("arc", arc_pos, r_arc_pos)]:
                    p = RV.positive_records(d, idx, risk, tau, plan)
                    lo, hi = RV.median_ci(p.lead, rng, n_boot)
                    k = int(p.detected.sum())
                    rlo, rhi = RV.wilson(k, len(p))
                    pos_rows.append(dict(**tag,
                                         pool=pool, n_events=len(p), n_detected=k,
                                         recall=k / len(p), recall_ci_lo=rlo, recall_ci_hi=rhi,
                                         lead_median=p.lead.median(), lead_ci_lo=lo, lead_ci_hi=hi,
                                         n_no_preonset_context=int((p.n_preonset_windows == 0).sum())))
                for pool, idx, risk in [("source_mech", plan.test_neg, r_src_neg),
                                        ("arc", arc_neg, r_arc_neg)]:
                    for w, anchor in [(None, "full")] + [(h, a) for h in RV.HORIZONS
                                                         for a in ("head", "tail")]:
                        rec = RV.negative_records(d, idx, risk, tau, w, anchor if w else "tail")
                        neg_rows.append(dict(**tag,
                                             pool=pool, window_s=("full" if w is None else w),
                                             anchor=anchor, **RV.rate_summary(rec)))
                f = [r for r in neg_rows if r["model"] == m and r["seed"] == seed
                     and r["age_mode"] == age_mode and r["window_s"] == 300]
                print("  %-15s %-17s seed %d tau %.4f | %ds | ARC head %d/%d tail %d/%d"
                      " | src head %d/%d tail %d/%d"
                      % (m, age_mode, seed, tau, time.time() - t0,
                         *[x for r in f if r["pool"] == "arc" for x in (r["n_alarm"], r["n_eligible"])],
                         *[x for r in f if r["pool"] == "source_mech" for x in (r["n_alarm"], r["n_eligible"])]),
                      flush=True)

    os.makedirs(out, exist_ok=True)
    pd.DataFrame(pos_rows).to_csv(os.path.join(out, "positive_summary.csv"), index=False)
    pd.DataFrame(neg_rows).to_csv(os.path.join(out, "negative_rates.csv"), index=False)
    pd.DataFrame(cal_rows).to_csv(os.path.join(out, "calibration.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="W60_native_v2")
    ap.add_argument("--models", default=",".join(DEEP.SEQ_MODELS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ages", default="no_age",
                    help="predeclared M1 representations, as in run_validation.py")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = a.models.split(",")
    if any(m not in DEEP.SEQ_MODELS for m in models):
        raise ValueError("sequence models only; trees go through run_validation.py")
    age_modes = a.ages.split(",")
    unknown = set(age_modes) - RV.REPRESENTATIONS
    if unknown:
        raise ValueError("unknown representation(s): %s" % ", ".join(sorted(unknown)))
    if os.path.exists(a.out):
        raise FileExistsError("use a fresh output directory: " + a.out)

    # The age channels stay in the pool and each representation masks its own
    # columns, so "no_age" here selects exactly what the earlier global strip did.
    d, meta = R.load_windows(a.windows, splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(os.path.join(R.OUT, "registry", "experiments.csv"))
    print("pool %d windows / %d experiments | %s | representations %s"
          % (len(d["X"]), len(meta), a.windows, ",".join(age_modes)))

    base = HERE.parent
    manifest = dict(arguments=vars(a), status="exploratory supplement, sequence tier",
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                     cwd=base, text=True).strip(),
                    code_sha256={str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(HERE.rglob("*.py"))},
                    window_artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in sorted((Path(R.OUT) / "windows" / a.windows).glob("*.npz"))})
    os.makedirs(a.out, exist_ok=True)
    Path(a.out, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run(d, meta, reg, models, [int(s) for s in a.seeds.split(",")], age_modes,
        a.out, a.bootstrap)
    print("->", a.out)


if __name__ == "__main__":
    main()
