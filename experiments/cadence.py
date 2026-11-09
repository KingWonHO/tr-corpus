# stored primary task states re-scored on causal 60 s windows ending every second
import time

import numpy as np
import pandas as pd

from core import fitting, native, tasks
from core import survival as SV
from core.perturb import series_from_windows
from core.windows import window_features
from experiments.base import Experiment

MODELS = {"D6_h600": ["xgboost", "lightgbm", "native_rate_rule"], "D5_h120": ["lstm", "gru", "temperature_rule"]}


# every 60 s window of the rebuilt 1 s series, one per second
def one_second(series):
    keys = []
    ends = []
    Xs = []
    Ms = []
    for k, s in series.items():
        n = len(s["val"])
        e = np.arange(59, n)
        idx = e[:, None] + np.arange(-59, 1)[None, :]
        v = np.nan_to_num(s["val"][idx], nan=0.0)
        m = s["mask"][idx]
        Xs.append(v[:, :, None].astype(np.float32))
        Ms.append(m[:, :, None])
        ends.append(s["t0"] + e.astype(float))
        keys.append(np.full(len(e), k))
    return np.concatenate(keys), np.concatenate(ends), np.concatenate(Xs), np.concatenate(Ms)


# record decisions from the trigger at the stored 10 s threshold and at one recalibrated at the 1 s cadence
def evaluate(t, r, keys, roles, c, lead, tau_fixed):
    order = np.lexsort((t, keys))
    ks = keys[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    per = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        j = order[a:b]
        k = ks[a]
        trig = float(c.at[k, "t_trigger"])
        tt = t[j]
        rr = r[j]
        trig = trig if np.isfinite(trig) else float(tt[0])
        valid = tt >= trig
        if valid.any():
            per[k] = (tt, rr, roles[k], trig, float(rr[valid].max()))
    cal = np.array([v[4] for v in per.values() if v[2] == "calibration"])
    tau_recal = SV.calibrate_threshold(cal, np.arange(len(cal)).astype(str), np.ones(len(cal), bool), 0.1)
    rows = []
    for k, (tt, rr, role, trig, _) in per.items():
        if role != "test":
            continue
        pos = bool(c.at[k, "scored_positive"])
        neg = bool(c.at[k, "scored_negative"])
        onset = float(c.at[k, "t_onset_L2_1.0"])
        first = float(tt[tt >= trig].min())
        rec = dict(key=k, positive=pos, negative=neg, opportunity=bool(pos and onset - first >= lead))
        for name, tau in [("fixed", tau_fixed), ("recal", tau_recal)]:
            hit = np.flatnonzero((tt >= trig) & (rr >= tau))
            alarm = float(tt[hit[0]]) if len(hit) else np.nan
            ok = bool(pos and np.isfinite(alarm) and alarm < onset and onset - alarm >= lead)
            rec[name + "_success"] = ok
            rec[name + "_first_decision"] = bool(ok and alarm == first)
            rec[name + "_neg_alarm"] = bool(neg and np.isfinite(alarm))
        rows.append(rec)
    return pd.DataFrame(rows), tau_recal


# 1 s decision cadence; the stored 10 s risks must be reproduced at the stored endpoints first
class Cadence(Experiment):
    name = 'cadence'
    help = 'stored states re-scored at a 1 s decision cadence'

    def run(self):
        t0 = time.perf_counter()
        tasks_dir = self.paths.need('preprocess/tasks')
        comparison = self.paths.need('task_comparison')
        d = tasks.load_inputs(tasks_dir)
        c = tasks.cohort(tasks_dir)
        prim = tasks.primary_records(comparison)
        rate_series = native.native_series(self.paths.arc_raw)
        out = []
        checks = []
        for task, lead in tasks.TASKS.items():
            for fold in range(5):
                ref = tasks.stored_risk(comparison, task, fold, "xgboost", 0)
                sel = np.isin(ref["role"], ["calibration", "test"])
                ix = ref["indices"][sel]
                role_of = dict(zip(ref["experiment"][sel].astype(str), ref["role"][sel]))
                series = series_from_windows(np.unique(ix), d)
                keys, t, X, M = one_second(series)
                F = None
                for model in MODELS[task]:
                    seeds = [0] if model.endswith("_rule") else range(5)
                    for seed in seeds:
                        st = prim[(prim.task == task) & (prim.arm == "own") & (prim.model == model)
                                  & (prim.seed == seed) & (prim.fold == fold) & (prim.role == "test")]
                        tau = float(st.tau.iloc[0])
                        folder = tasks.run_dir(comparison, task, fold, model, seed)
                        if model == "temperature_rule":
                            r = X[:, -1, 0].astype(float)
                        elif model == "native_rate_rule":
                            r = np.zeros(len(t))
                            for k in np.unique(keys):
                                j = np.flatnonzero(keys == k)
                                r[j] = native.window_risk(*rate_series[k], t[j])
                        elif model in ("xgboost", "lightgbm"):
                            if F is None:
                                F = window_features(X, M, np.array([0]))
                            r = tasks.predict(folder, dict(F=F, X=X, mask=M), np.arange(len(t))).astype(float)
                        else:
                            r = fitting.predict(folder, X, M, None, np.arange(len(t))).astype(float)
                        sr = tasks.stored_risk(comparison, task, fold, model, seed)
                        ss = np.isin(sr["role"], ["calibration", "test"])
                        lut = pd.Series(r, index=pd.MultiIndex.from_arrays([keys, np.round(t).astype(int)]))
                        at = list(zip(sr["experiment"][ss].astype(str), np.round(sr["t"][ss]).astype(int)))
                        got = lut.loc[at].to_numpy()
                        err = float(np.max(np.abs(got - sr["risk"][ss])))
                        checks.append(dict(task=task, fold=fold, model=model, seed=seed, max_abs_error=err))
                        assert err < 1e-5, (task, fold, model, seed, err)
                        res, tau1 = evaluate(t, r, keys, role_of, c, lead, tau)
                        out.append(res.assign(task=task, fold=fold, model=model, seed=seed, tau_10s=tau, tau_1s=tau1))
                print(task, fold, "done", round(time.perf_counter() - t0), "s", flush=True)
        rec = pd.concat(out, ignore_index=True)
        rec.to_parquet(self.out / "r3_cadence_1s_records.parquet", index=False)
        pd.DataFrame(checks).to_csv(self.out / "r3_cadence_1s_check.csv", index=False)
        rows = []
        for (task, model, seed), g in rec.groupby(["task", "model", "seed"]):
            p = g[g.positive]
            n = g[g.negative]
            counts = {}
            for th in ["fixed", "recal"]:
                for q in ["success", "first_decision", "neg_alarm"]:
                    col = f"{th}_{q}"
                    counts[col] = n[col].mean() if q == "neg_alarm" else p[col].sum()
            rows.append(dict(task=task, model=model, seed=seed, n_pos=len(p), opportunity=int(p.opportunity.sum()),
                             n_neg=len(n), **counts))
        s = pd.DataFrame(rows)
        s.to_csv(self.out / "r3_cadence_1s_seed.csv", index=False)
        s.groupby(["task", "model"]).mean(numeric_only=True).drop(columns="seed").to_csv(self.out / "r3_cadence_1s.csv")
        tasks.dump(self.out / "r3_cadence_1s_verification.json", dict(
            status="PASS", checks=len(checks), max_abs_error=float(pd.DataFrame(checks).max_abs_error.max()),
            seconds=time.perf_counter() - t0))
        print(s.groupby(["task", "model"]).mean(numeric_only=True).round(3).to_string())
        print("CADENCE COMPLETE")
