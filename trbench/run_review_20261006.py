"""Sensitivity runs requested by the external review of 2026-10-03 (external review of 2026-10-03).

Both modes reproduce the internal design of ``run_v2.py --arm e3a`` (panels M1
and M2, LOEO over the 77 adjudicated source positives, sensor values only,
threshold at 10 % experiment-level FAR on the 25 calibration negatives) and
change exactly one thing.

cadence
    Fit every fold exactly as the stored run, on the 10 s windows, then score
    the held cell on windows that end every 1 s (``W60_native_s1``).  Two
    thresholds are reported: ``tau10`` is the stored calibration on the 10 s
    calibration windows (the alarm grid alone changes), ``tau1`` recalibrates on
    the same negatives' 1 s windows (what a 1 s monitor would have set).  The
    10 s alarm is recomputed in the same fit, so its agreement with the stored
    run is checked rather than assumed.

disputed
    (internal design) The two test-split D5 records whose source states no runaway but whose
    trace satisfies L2 (outcome adjudication D1 ``disagree_source_nonTR_L2``)
    are moved out of the training pool and into the check negatives.  Nothing
    else changes; the check pool grows from 27 to 29 records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_e3 as R          # noqa: E402
import evalv2 as V          # noqa: E402
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import deep as DEEP         # noqa: E402

PANELS = ["M1", "M2"]


def drop_age(d):
    keep = np.array([f != MW.AGE_CH and not f.startswith("age__") for f in d["features"]])
    d["X"], d["mask"] = d["X"][:, :, keep], d["mask"][:, :, keep]
    d["features"] = list(np.array(d["features"])[keep])
    return d


def fitter(d, cols, model, train_idx, seed, horizon):
    """Fit as ``run_e3.run_fold`` does; return score(dataset, index) -> risk."""
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], horizon)
    if model == "rule":
        cols = np.asarray([j for j in cols if d["features"][j] != MW.AGE_CH
                           and not d["features"][j].startswith("age__")])
    if model in DEEP.SEQ_MODELS:
        X, M = d["X"][:, :, cols], d["mask"][:, :, cols]
        predict = DEEP.fit_seq(model, X[train_idx], M[train_idx], y[train_idx], w[train_idx], seed)
        return lambda dd, idx: (predict(dd["X"][idx][:, :, cols], dd["mask"][idx][:, :, cols])
                                if len(idx) else np.zeros(0))
    Ftr = R.window_features(d["X"][train_idx], d["mask"][train_idx], cols)
    predict = R.fit_model(model, Ftr, y[train_idx], w[train_idx], seed)
    return lambda dd, idx: (predict(R.window_features(dd["X"][idx], dd["mask"][idx], cols))
                            if len(idx) else np.zeros(0))


def alarm_of(d, idx, risk, tau):
    order = np.argsort(d["t_end"][idx], kind="mergesort")
    return SV.first_alarm(d["t_end"][idx][order], np.maximum.accumulate(np.asarray(risk)[order]), tau)


def cadence(d10, meta, reg, d1, models, seed, horizon, budget):
    plan = V.Plan(d10, meta, reg, PANELS)
    print("  " + plan.describe(), flush=True)
    ds1 = np.asarray(d1["experiment"]).astype(str)
    obs1 = np.asarray(d1["label_observed"]).astype(bool)
    cal_keys, chk_keys = set(plan.ds[plan.cal_neg]), set(plan.ds[plan.test_neg])
    cal1 = np.flatnonzero(np.isin(ds1, list(cal_keys)) & obs1)
    chk1 = np.flatnonzero(np.isin(ds1, list(chk_keys)) & obs1)
    print("  1 s: cal-neg %d win / check-neg %d win" % (len(cal1), len(chk1)), flush=True)
    rows = []
    for panel in PANELS:
        cols = np.flatnonzero(MW.panel_mask(d10["features"], panel))
        assert list(np.array(d10["features"])[cols]) == list(np.array(d1["features"])[cols])
        for held in plan.folds:
            te10 = np.flatnonzero((plan.ds == held) & plan.observed)
            te1 = np.flatnonzero((ds1 == held) & obs1)
            tr = plan.train_idx(held)
            if not len(te10) or not len(tr):
                continue
            for m in models:
                score = fitter(d10, cols, m, tr, seed, horizon)
                r_cal10, r_chk10, r_te10 = score(d10, plan.cal_neg), score(d10, plan.test_neg), score(d10, te10)
                r_cal1, r_chk1, r_te1 = score(d1, cal1), score(d1, chk1), score(d1, te1)
                ones10, ones1 = np.ones(len(plan.cal_neg), bool), np.ones(len(cal1), bool)
                tau10 = SV.calibrate_threshold(r_cal10, d10["experiment"][plan.cal_neg], ones10, far=budget)
                tau1 = SV.calibrate_threshold(r_cal1, d1["experiment"][cal1], ones1, far=budget)
                a10 = alarm_of(d10, te10, r_te10, tau10)
                a1_t10 = alarm_of(d1, te1, r_te1, tau10)
                a1_t1 = alarm_of(d1, te1, r_te1, tau1)
                far10 = SV.false_alarm_rate(r_chk10, d10["experiment"][plan.test_neg],
                                            np.ones(len(plan.test_neg), bool), tau10)
                far1 = SV.false_alarm_rate(r_chk1, d1["experiment"][chk1], np.ones(len(chk1), bool), tau1)
                tg = plan.trig.get(held, np.nan)

                def det(a, on):
                    pre = a is not None and np.isfinite(tg) and a < tg
                    return bool(a is not None and a < on and not pre)

                for name, onset in plan.onsets.items():
                    on = onset.get(held, np.nan)
                    if not np.isfinite(on):
                        continue
                    rows.append(dict(model=m, panel=panel, held=held, label=name, onset=float(on),
                                     t_trigger=tg, tau10=tau10, tau1=tau1,
                                     alarm10=a10, alarm1_tau10=a1_t10, alarm1_tau1=a1_t1,
                                     detected10=det(a10, on), detected1_tau10=det(a1_t10, on),
                                     detected1_tau1=det(a1_t1, on),
                                     far_check10=far10, far_check1_tau1=far1,
                                     n_win10=len(te10), n_win1=len(te1)))
        print("    %s done (%d rows)" % (panel, len(rows)), flush=True)
    return pd.DataFrame(rows)


def disputed_keys(reg):
    adj = pd.read_csv(V.ADJUDICATION)
    x = adj[adj.outcome_evidence.str.contains("disagree_source_nonTR_L2", na=False)]
    split = dict(zip(reg.experiment_id, reg.split))
    keep = [e for e in x.experiment_id if split.get(e) == "test"]
    return {"ds09_mech/" + e for e in keep}


def disputed_plan(base_plan, keys):
    print("disputed check negatives:", sorted(keys), flush=True)

    class DisputedPlan(base_plan):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            moved = np.isin(self.ds, list(keys))
            self.pool = self.pool[~moved[self.pool]]
            self.frozen_train = self.frozen_train[~moved[self.frozen_train]]
            self.test_neg = np.concatenate([self.test_neg, np.flatnonzero(moved & self.observed)])
            self._check()

    return DisputedPlan


def manifest(path, args, window_dirs):
    base = Path(HERE)
    m = dict(arguments=vars(args), status="exploratory review sensitivity (external review 2026-10-03)",
             git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.parent, text=True).strip(),
             code_sha256={str(p.relative_to(base.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(base.rglob("*.py"))},
             registry_sha256=hashlib.sha256((Path(R.OUT) / "registry" / "experiments.csv").read_bytes()).hexdigest(),
             window_artifact_sha256={"%s/%s" % (w, p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for w in window_dirs for p in sorted((Path(R.OUT) / "windows" / w).glob("*.npz"))})
    Path(path + ".manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["cadence", "disputed", "disputed-heldout"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--windows", default="W60_native")
    ap.add_argument("--fine-windows", default="W60_native_s1")
    ap.add_argument("--budget", type=float, default=V.BUDGET)
    ap.add_argument("--horizon", type=float, default=R.HORIZON_S)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise FileExistsError(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.mode == "disputed-heldout":
        manifest(a.out.rstrip("/"), a, [a.windows])   # sibling file; run_validation needs an empty dir
    t0 = time.time()
    models = a.models.split(",")
    reg = pd.read_csv(os.path.join(R.OUT, "registry", "experiments.csv"))
    if a.mode == "disputed-heldout":
        import run_validation as RV
        V.Plan = disputed_plan(V.Plan, disputed_keys(reg))
        sys.argv = ["run_validation.py", "--windows", a.windows, "--models", a.models,
                    "--seeds", a.seeds, "--ages", "no_age", "--out", a.out]
        RV.main()
        return
    d10, meta = R.load_windows(a.windows, splits=("train", "val", "test"))
    d10 = drop_age(d10)
    dirs = [a.windows]
    if a.mode == "cadence":
        d1, _ = R.load_windows(a.fine_windows, splits=("train", "val", "test"))
        d1 = drop_age(d1)
        assert list(d1["features"]) == list(d10["features"]), "feature layouts differ"
        dirs.append(a.fine_windows)
    else:
        V.Plan = disputed_plan(V.Plan, disputed_keys(reg))
    manifest(a.out, a, dirs)
    runs = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        print("seed %d" % seed, flush=True)
        if a.mode == "cadence":
            part = cadence(d10, meta, reg, d1, models, seed, a.horizon, a.budget)
        else:
            part = V.crossfit_loeo(d10, meta, reg, models, PANELS, a.budget, seed, horizon=a.horizon)
        part["seed"] = seed
        runs.append(part)
        pd.concat(runs, ignore_index=True).to_csv(a.out, index=False)
    print("\n%.0fs -> %s" % (time.time() - t0, a.out), flush=True)


if __name__ == "__main__":
    main()
