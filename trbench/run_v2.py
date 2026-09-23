"""Run the V2 evaluator.  One pass answers both the panel and the label question.

A single `crossfit_loeo` pass over a panel set produces, for every held cell,
one threshold and one alarm time, then measures that alarm against every onset
definition.  So the panel comparison (E3) and the label comparison (E1) come
out of the SAME alarms -- which is the point: under V1 they could not, because
each label re-picked its own operating point.

    uv run python trbench/run_v2.py --arm e3b --models rule,xgboost,lightgbm,timesfm
    uv run python trbench/run_v2.py --arm e3a --models rule,xgboost,lightgbm,timesfm
    uv run python trbench/run_v2.py --arm e11 --models rule,xgboost,lightgbm,timesfm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import run_e3 as R          # noqa: E402
import evalv2 as V          # noqa: E402
import make_windows as MW  # noqa: E402

ARMS = {
    "e3a": ["M1", "M2"],                          # the powered comparison
    "e3b": ["M1", "M2", "M3", "M4", "M5"],        # #1's full external chain
    "e3c": ["M1", "M2", "M4"],                    # #1 + #2, gas increment
    "e3d": ["M1", "M2", "M3", "M6i"],             # #3, internal sensor bound
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="e3b", choices=list(ARMS) + ["e11"])
    ap.add_argument("--models", default="rule,xgboost,lightgbm,timesfm")
    ap.add_argument("--windows", default="W60_raw")
    ap.add_argument("--budget", type=float, default=V.BUDGET)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated repetitions; retained as a seed column")
    ap.add_argument("--without-age", action="store_true",
                    help="sensor values only, for observation-clock ablation")
    ap.add_argument("--gas-linear-control", action="store_true",
                    help="NONCAUSAL control: change only BAK gas interpolation")
    ap.add_argument("--out", default=None,
                    help="fresh CSV path for a versioned supplement")
    a = ap.parse_args()
    if a.without_age and not a.out:
        raise ValueError("--without-age requires a separate --out to preserve other representations")
    t0 = time.time()
    models = a.models.split(",")

    splits = ("train", "val", "test", "test_zeroshot") if a.arm == "e11" else ("train", "val", "test")
    d, meta = R.load_windows(a.windows, splits=splits)
    reg = pd.read_csv(os.path.join(R.OUT, "registry", "experiments.csv"))
    if a.without_age:
        keep = np.array([f != MW.AGE_CH and not f.startswith("age__")
                         for f in d["features"]])
        d["X"], d["mask"] = d["X"][:, :, keep], d["mask"][:, :, keep]
        d["features"] = list(np.array(d["features"])[keep])
    if a.gas_linear_control:
        if not a.out or a.arm != "e3b":
            raise ValueError("gas control requires --arm e3b and a separate --out")
        import gas_control
        d = gas_control.apply(d)
    os.makedirs(os.path.join(R.OUT, "results"), exist_ok=True)
    tag = R.out_tag(a.windows)
    print("pool %d windows / %d experiments | arm %s | models %s"
          % (len(d["X"]), len(meta), a.arm, ",".join(models)))

    path = a.out or os.path.join(R.OUT, "results", "v2_%s%s.csv" % (a.arm, tag))
    if a.out and os.path.exists(path):
        raise FileExistsError("use a fresh supplement path: " + path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if a.out:
        base = Path(HERE)
        window_base = Path(R.OUT) / "windows" / a.windows
        manifest = dict(arguments=vars(a), status="exploratory supplement",
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=base.parent, text=True).strip(),
            code_sha256={str(p.relative_to(base.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(base.rglob("*.py"))},
            registry_sha256=hashlib.sha256((Path(R.OUT) / "registry" / "experiments.csv").read_bytes()).hexdigest(),
            environment_lock_sha256=hashlib.sha256((base.parent / "uv.lock").read_bytes()).hexdigest(),
            window_artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(window_base.glob("*.npz"))})
        Path(path + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    runs = []
    for seed in seeds:
        if a.arm == "e11":
            part = V.frozen_external(d, meta, reg, models, "M1", a.budget, seed)
        else:
            part = V.crossfit_loeo(d, meta, reg, models, ARMS[a.arm], a.budget, seed)
        part["seed"] = seed
        runs.append(part)
        res = pd.concat(runs, ignore_index=True)
        if a.out:
            res.to_csv(path, index=False)
        else:
            R.save_merged(res, path)
    print("\n%.0fs -> %s  (%d rows)" % (time.time() - t0, path, len(res)))


if __name__ == "__main__":
    main()
