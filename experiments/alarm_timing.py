# alarm timing: first-decision alarms, after-crossing successes and first false alarms on negatives
import numpy as np
import pandas as pd

from core import tasks
from experiments.base import Experiment

LEAD = {"D1_within": 60, "D2_within": 60, "D3_within": 60, "D4_within": 60, "D5_h120": 30, "D5_h300": 30,
        "D6_h600": 120, "thermal_to_D1": 60, "thermal_to_D2": 60, "thermal_to_D3": 60, "thermal_to_D4": 60}
CHECK = {"D5_h120": [60, 120, 300, 600], "D6_h600": [300, 3600, 10800, 36000]}
MODELS = tasks.LEARNED + ["temperature_rule", "rate_rule", "native_rate_rule"]


# Kaplan-Meier cumulative incidence 1 - S(t) at the grid points
def km_incidence(times, events, grid):
    order = np.argsort(times)
    t = np.asarray(times)[order]
    e = np.asarray(events)[order]
    s = 1.0
    at_risk = len(t)
    surv = []
    i = 0
    for g in grid:
        while i < len(t) and t[i] <= g:
            j = i
            d = 0
            while j < len(t) and t[j] == t[i]:
                d += e[j]
                j += 1
            if at_risk > 0:
                s *= 1 - d / at_risk
            at_risk -= (j - i)
            i = j
        surv.append(s)
    return 1 - np.array(surv)


# record key -> (window end times, last input value, last input mask), in time order
def record_series(d):
    keys = d["experiment"]
    last = d["X"][:, -1, 0]
    order = np.lexsort((d["t_end"], keys))
    ks = keys[order]
    b = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    out = {}
    for a, z in zip(b[:-1], b[1:]):
        j = order[a:z]
        out[ks[a]] = (d["t_end"][j], last[j], d["mask"][j, -1, 0])
    return out


# alarm timing relative to monitoring start for every task and fitting arm
class AlarmTiming(Experiment):
    name = 'alarm_timing'
    help = 'alarm timing: first-decision alarms and after-crossing successes'

    def run(self):
        tasks_dir = self.paths.need('preprocess/tasks')
        prim = tasks.primary_records(self.paths.need('task_comparison'))
        c = tasks.cohort(tasks_dir)
        series = record_series(tasks.load_inputs(tasks_dir))
        self.first_decision(prim, series)
        self.false_alarm_timing(prim, c, series)

    # timely successes issued at the first admissible decision versus after a below-threshold decision
    def first_decision(self, prim, series):
        p = prim[prim.role.eq("test")].copy()

        def temp_at(k, t):
            tt, xx, _ = series[k]
            return float(xx[min(np.searchsorted(tt, t), len(tt) - 1)])

        p["first_decision_alarm"] = np.isfinite(p.alarm) & ((p.alarm - p.first_admissible) <= 0.5)
        p["temp_first_decision"] = [temp_at(k, t) for k, t in zip(p.key, p.first_admissible)]
        rows = []
        for (task, arm, model, seed), g in p.groupby(["task", "arm", "model", "seed"]):
            lead = LEAD[task]
            pos = g[g.positive]
            neg = g[g.negative]
            succ = pos[pos[f"success_{lead}"]]
            fa = neg[np.isfinite(neg.alarm)]
            rows.append(dict(task=task, arm=arm, model=model, seed=seed, lead=lead,
                             n_positive=len(pos), n_negative=len(neg),
                             success=len(succ), success_first=int(succ.first_decision_alarm.sum()),
                             success_after_crossing=int((~succ.first_decision_alarm).sum()),
                             false_alarms=len(fa), false_alarms_first=int(fa.first_decision_alarm.sum())))
        x = pd.DataFrame(rows)
        x.to_csv(self.out / "alarm_timing_seed.csv", index=False)
        s = x.groupby(["task", "arm", "model"]).agg(
            lead=("lead", "first"), n_positive=("n_positive", "first"), n_negative=("n_negative", "first"),
            success=("success", "mean"), success_first=("success_first", "mean"),
            success_after_crossing=("success_after_crossing", "mean"), false_alarms=("false_alarms", "mean"),
            false_alarms_first=("false_alarms_first", "mean")).reset_index()
        s["first_share_success"] = s.success_first / s.success.replace(0, np.nan)
        s["first_share_false_alarm"] = s.false_alarms_first / s.false_alarms.replace(0, np.nan)
        s.to_csv(self.out / "alarm_timing_summary.csv", index=False)

        d5 = p[p.task.eq("D5_h120") & p.arm.eq("own") & p.model.isin(["lstm", "gru"])]
        temps = d5.groupby([d5.first_decision_alarm, d5.positive]).temp_first_decision.agg(
            ["count", "mean", "min", "max"])
        temps.to_csv(self.out / "d5_first_decision_temperature.csv")
        keys8 = sorted(d5[d5.positive & d5.success_30].key.unique())
        other = s[~s.task.str.startswith("D5")]
        tasks.dump(self.out / "alarm_timing_verification.json", dict(
            d5_recurrent_success_keys=keys8,
            no_first_decision_success_outside_D5=bool((other.success_first.fillna(0) == 0).all()),
            d5_recurrent_first_temperature=temps.reset_index().to_dict("records")))
        print(s[s.arm.eq("own")].round(2).to_string())
        print(temps.round(1).to_string())

    # time to the first false alarm on negative test records, censored at the end of monitoring,
    # and the ARC surface temperature at that alarm
    def false_alarm_timing(self, prim, c, series):
        rows = []
        curves = []
        state = []
        for task in tasks.TASKS:
            neg = prim[prim.task.eq(task) & prim.arm.eq("own") & prim.role.eq("test") & prim.negative.astype(bool)
                       & prim.model.isin(MODELS)]
            for (model, seed), g in neg.groupby(["model", "seed"]):
                tta = (g.alarm - g.first_admissible).to_numpy(float)
                event = np.isfinite(tta)
                time_ = np.where(event, tta, g.monitoring_s.to_numpy(float))
                grid = CHECK[task] + [float(g.monitoring_s.max())]
                inc = km_incidence(time_, event.astype(int), grid)
                for gp, v in zip(grid, inc):
                    curves.append(dict(task=task, model=model, seed=seed, t=gp, incidence=v,
                                       label="full" if gp == grid[-1] else str(int(gp))))
                rows.append(dict(task=task, model=model, seed=seed, negatives=len(g), alarmed=int(event.sum()),
                                 median_time_to_alarm_s=float(np.median(tta[event])) if event.any() else np.nan,
                                 median_monitoring_s=float(g.monitoring_s.median())))
                if task != "D6_h600":
                    continue
                for r in g.itertuples():
                    t, temp, m = series[r.key]
                    use = (t >= r.first_admissible) & m
                    at_alarm = float(temp[np.argmin(np.abs(t - r.alarm))]) if np.isfinite(r.alarm) else np.nan
                    state.append(dict(model=model, seed=seed, key=r.key, chemistry=c.at[r.key, "chemistry"],
                                      soc=c.at[r.key, "soc_pct"], monitoring_h=r.monitoring_s / 3600,
                                      alarm_h=(r.alarm - r.first_admissible) / 3600 if np.isfinite(r.alarm) else np.nan,
                                      temp_start=float(temp[use][0]), temp_max=float(temp[use].max()),
                                      temp_at_alarm=at_alarm))
        pd.DataFrame(rows).to_csv(self.out / "false_alarm_timing_seed.csv", index=False)
        cv = pd.DataFrame(curves)
        cv.to_csv(self.out / "false_alarm_incidence_seed.csv", index=False)
        incidence = cv.groupby(["task", "model", "label"]).incidence.agg(["mean", "std"]).reset_index()
        incidence.to_csv(self.out / "false_alarm_incidence.csv", index=False)
        st = pd.DataFrame(state)
        st.to_csv(self.out / "arc_negative_state.csv", index=False)
        per_rec = st.drop_duplicates("key")[["key", "chemistry", "soc", "monitoring_h", "temp_start", "temp_max"]]
        summ = st.dropna(subset=["temp_at_alarm"]).groupby("model").agg(
            alarms=("temp_at_alarm", "size"), temp_at_alarm_median=("temp_at_alarm", "median"),
            temp_at_alarm_min=("temp_at_alarm", "min"), temp_at_alarm_max=("temp_at_alarm", "max"),
            alarm_h_median=("alarm_h", "median"), alarm_h_min=("alarm_h", "min")).reset_index()
        summ.to_csv(self.out / "arc_false_alarm_state.csv", index=False)
        tasks.dump(self.out / "timing_summary.json", dict(
            arc_negatives=per_rec.to_dict("records"),
            arc_temp_start_range=[float(per_rec.temp_start.min()), float(per_rec.temp_start.max())],
            arc_temp_max_range=[float(per_rec.temp_max.min()), float(per_rec.temp_max.max())],
            arc_monitoring_h_range=[float(per_rec.monitoring_h.min()), float(per_rec.monitoring_h.max())]))
        print(summ.to_string())
        print(cv.groupby(["task", "model", "label"]).incidence.mean().unstack().round(3).to_string())
