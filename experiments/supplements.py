# supplementary analyses on stored decisions: chemistry/format strata, untimed alarms, opportunity windows,
# requirement bounds and single-decision compute time
import platform
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from core import models, tasks
from core.windows import window_features
from experiments.base import Experiment

MODELS = tasks.LEARNED + ["temperature_rule", "rate_rule", "native_rate_rule"]


# stored tree state as a one-thread scoring function and its number of trees
def tree_scorer(folder, model):
    if model == "xgboost":
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(bytearray((folder / "model.ubj").read_bytes()))
        m.set_param({"nthread": 1})
        return m.inplace_predict, len(m.get_dump())
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(folder / "model.txt"))

    def score(F):
        return m.predict(F, num_threads=1)
    return score, m.num_trees()


# requirement bounds, chemistry strata, opportunity windows and compute
class Supplements(Experiment):
    name = 'supplements'
    help = 'requirement bounds, chemistry strata, opportunity windows and compute'

    def run(self):
        self.tasks_dir = self.paths.need('preprocess/tasks')
        self.comparison = self.paths.need('task_comparison')
        prim = tasks.primary_records(self.comparison)
        c = tasks.cohort(self.tasks_dir)
        d = tasks.load_inputs(self.tasks_dir)
        print(self.breakdown(prim, c).round(2).to_string())
        print(self.untimed().round(2).to_string())
        print(self.windows(c, d).to_string())
        print(self.bounds().round(3).to_string())
        print(self.compute().round(3).to_string())

    # chemistry and format strata of the own-source tasks
    def breakdown(self, prim, c):
        rows = []
        for task, lead in tasks.TASKS.items():
            g = prim[prim.task.eq(task) & prim.arm.eq("own") & prim.role.eq("test") & prim.model.isin(MODELS)].copy()
            g["chemistry"] = c.loc[g.key, "chemistry"].to_numpy()
            g["format"] = c.loc[g.key, "form_factor"].to_numpy()
            for by in ["chemistry", "format"]:
                for (model, seed, grp), h in g.groupby(["model", "seed", by]):
                    pos = h[h.positive]
                    neg = h[h.negative]
                    rows.append(dict(task=task, by=by, group=grp, model=model, seed=seed, n_positive=len(pos),
                                     success=float(pos[f"success_{lead}"].sum()), n_negative=len(neg),
                                     false_alarms=float(neg.negative_alarm.sum())))
        x = pd.DataFrame(rows)
        s = x.groupby(["task", "by", "group", "model"]).agg(
            n_positive=("n_positive", "first"), success=("success", "mean"),
            n_negative=("n_negative", "first"), false_alarms=("false_alarms", "mean")).reset_index()
        s.to_csv(self.out / "strata_chemistry_format.csv", index=False)
        return s

    # alarms on physical positives without an L2 time
    def untimed(self):
        u = pd.read_csv(self.comparison / "untimed_outcome_alarms.csv")
        u = u[u.arm.eq("own") & u.task.isin(list(tasks.TASKS))]
        s = u.groupby(["task", "model", "outcome"]).agg(records=("records", "first"), alarmed=("alarmed", "mean"),
                                                         alarmed_sd=("alarmed", "std")).reset_index()
        s.to_csv(self.out / "untimed_alarms.csv", index=False)
        return s

    # executable opportunity for D5 timed positives under shorter windows and a 1 s decision cadence
    def windows(self, c, d):
        pos = c[(c.dataset == "D5") & c.scored_positive.astype(bool)]
        keys = d["experiment"]
        order = np.lexsort((d["t_end"], keys))
        ks = keys[order]
        b = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
        first_end = {ks[a]: float(d["t_end"][order[a]]) for a in b[:-1]}
        rows = []
        for k, r in pos.iterrows():
            trig = float(r.t_trigger)
            onset = float(r["t_onset_L2_1.0"])
            fe = first_end[k]
            # first second covered by the stored windows
            start = fe - 59.0
            for window in [60, 30, 10, 1]:
                for cadence in [10, 1]:
                    earliest = max(trig, start + window - 1)
                    if cadence == 10:
                        # stored grid fe + 10 n, extended backwards
                        n = np.ceil((earliest - fe) / 10.0)
                        g0 = fe + 10.0 * n
                    else:
                        g0 = np.ceil(earliest)
                    for lead in [30, 60, 120]:
                        rows.append(dict(key=k, window_s=window, cadence_s=cadence, lead=lead,
                                         executable=bool(g0 <= onset - lead)))
            # trigger-limited bound: no history or grid limit
            for lead in [30, 60, 120]:
                rows.append(dict(key=k, window_s=0, cadence_s=0, lead=lead, executable=bool(trig <= onset - lead)))
        x = pd.DataFrame(rows)
        s = x.groupby(["window_s", "cadence_s", "lead"]).executable.sum().unstack("lead").reset_index()
        stored = s[(s.window_s == 60) & (s.cadence_s == 10)][[30, 60, 120]].to_numpy().ravel().tolist()
        assert stored == [41, 37, 9], stored
        s.to_csv(self.out / "opportunity_windows.csv", index=False)
        return s

    # one-sided 95 % Clopper-Pearson bounds for timely detection and false alarms of the primary detectors
    def bounds(self):
        a = pd.read_csv(self.comparison / "primary_seed.csv")
        a = a[a.arm.eq("own") & a.task.isin(list(tasks.TASKS)) & a.model.isin(MODELS)]
        rows = []
        for r in a.itertuples():
            s = round(r.success)
            n = int(r.n_positive)
            f = round(r.false_alarms)
            m = int(r.n_negative)
            lb = 0.0 if s == 0 else float(beta.ppf(0.05, s, n - s + 1))
            lb_opp = 0.0 if s == 0 else float(beta.ppf(0.05, s, r.opportunity - s + 1))
            ub = 1.0 if f == m else float(beta.ppf(0.95, f + 1, m - f))
            rows.append(dict(task=r.task, model=r.model, seed=r.seed, success=s, n_positive=n,
                             opportunity=int(r.opportunity), timely_lb95=lb, timely_opportunity_lb95=lb_opp,
                             false_alarms=f, n_negative=m, far_ub95=ub))
        x = pd.DataFrame(rows)
        x.to_csv(self.out / "requirement_bounds_seed.csv", index=False)
        s = x.groupby(["task", "model"]).agg(
            timely_lb_min=("timely_lb95", "min"), timely_lb_max=("timely_lb95", "max"),
            opp_lb_min=("timely_opportunity_lb95", "min"), opp_lb_max=("timely_opportunity_lb95", "max"),
            far_ub_min=("far_ub95", "min"), far_ub_max=("far_ub95", "max")).reset_index()
        s.to_csv(self.out / "requirement_bounds.csv", index=False)
        return s

    # median wall time of one decision (feature extraction + model) on one CPU thread; machine dependent
    def compute(self):
        torch.set_num_threads(1)
        with np.load(self.tasks_dir / "inputs.npz", allow_pickle=True) as z:
            X = z["X"][:2000]
            M = z["mask"][:2000].astype(bool)
        rows = []
        for model in tasks.LEARNED:
            folder = tasks.run_dir(self.comparison, "D6_h600", 0, model, 0)
            times = []
            if model in ["xgboost", "lightgbm"]:
                score, size = tree_scorer(folder, model)
                for i in range(1000):
                    t0 = time.perf_counter()
                    F = window_features(X[i:i + 1], M[i:i + 1], np.array([0]))
                    score(F)
                    times.append(time.perf_counter() - t0)
                unit = "trees"
            else:
                cp = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
                net = models.build(cp["model"], cp["c_in"], cp["window"])
                net.load_state_dict(cp["state_dict"])
                net.eval()
                size = sum(p.numel() for p in net.parameters())
                with torch.no_grad():
                    for i in range(1000):
                        t0 = time.perf_counter()
                        x = torch.as_tensor(models.prep(X[i:i + 1], M[i:i + 1], cp["mu"], cp["sd"]))
                        net(x)
                        times.append(time.perf_counter() - t0)
                unit = "parameters"
            t = np.array(times[50:]) * 1e3
            rows.append(dict(model=model, size=size, unit=unit, median_ms=float(np.median(t)),
                             p95_ms=float(np.percentile(t, 95))))
        x = pd.DataFrame(rows)
        x.to_csv(self.out / "compute_per_decision.csv", index=False)
        tasks.dump(self.out / "compute_platform.json", dict(processor=platform.processor(), machine=platform.machine(),
                                                            python=platform.python_version(), threads=1))
        return x
