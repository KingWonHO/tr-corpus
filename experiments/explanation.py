# feature attribution: exact tree SHAP contributions of the stored XGBoost and LightGBM task states
import numpy as np
import pandas as pd

from core import tasks
from core.windows import window_features
from experiments.base import Experiment


# contributions of the nine window summaries, without the bias column
def contributions(folder, F):
    if (folder / "model.txt").exists():
        import lightgbm as lgb
        return lgb.Booster(model_file=str(folder / "model.txt")).predict(F, pred_contrib=True, num_threads=4)[:, :-1]
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(bytearray((folder / "model.ubj").read_bytes()))
    return b.predict(xgb.DMatrix(F), pred_contribs=True)[:, :-1]


# mean absolute contributions over test windows from the trigger on, and contributions at first alarms
class Explanation(Experiment):
    name = 'explanation'
    help = 'feature attribution of the stored task models'

    def run(self):
        tasks_dir = self.paths.need('preprocess/tasks')
        comparison = self.paths.need('task_comparison')
        d = tasks.load_inputs(tasks_dir)
        F = window_features(d["X"], d["mask"], np.array([0]))
        c = tasks.cohort(tasks_dir)
        prim = tasks.primary_records(comparison)
        imp = []
        at_alarm = []
        for task in tasks.TASKS:
            for fold in range(5):
                for model in ["xgboost", "lightgbm"]:
                    for seed in range(5):
                        r = tasks.stored_risk(comparison, task, fold, model, seed)
                        test = r["role"] == "test"
                        ix = r["indices"][test]
                        keys = r["experiment"][test].astype(str)
                        t = r["t"][test]
                        trig = c.loc[keys, "t_trigger"].to_numpy(float)
                        use = t >= np.where(np.isfinite(trig), trig, -np.inf)
                        phi = contributions(tasks.run_dir(comparison, task, fold, model, seed), F[ix[use]])
                        a = np.abs(phi).mean(axis=0)
                        imp.append(dict(task=task, model=model, fold=fold, seed=seed,
                                        **dict(zip(tasks.FEATURES, a / a.sum()))))
                        st = prim[(prim.task == task) & (prim.arm == "own") & (prim.model == model)
                                  & (prim.seed == seed) & (prim.fold == fold) & (prim.role == "test")]
                        hits = st[np.isfinite(st.alarm) & ((st.positive & st.detected) | st.negative)]
                        kk = keys[use]
                        tt = t[use]
                        for h in hits.itertuples():
                            j = np.flatnonzero((kk == h.key) & (np.abs(tt - h.alarm) < 0.5))
                            if not len(j):
                                continue
                            kind = "detected positive" if h.positive else "false alarm"
                            at_alarm.append(dict(task=task, model=model, fold=fold, seed=seed, key=h.key, kind=kind,
                                                 **dict(zip(tasks.FEATURES, phi[j[0]]))))
                print(task, fold, "done", flush=True)
        imp = pd.DataFrame(imp)
        imp.to_csv(self.out / "explain_importance_runs.csv", index=False)
        summ = imp.groupby(["task", "model"])[tasks.FEATURES].mean().T
        summ.to_csv(self.out / "explain_importance.csv")
        al = pd.DataFrame(at_alarm)
        al.to_csv(self.out / "explain_at_alarm_runs.csv", index=False)
        al.groupby(["task", "model", "kind"])[tasks.FEATURES].mean().T.to_csv(self.out / "explain_at_alarm.csv")
        counts = {" | ".join(k): int(v) for k, v in al.groupby(["task", "model", "kind"]).size().items()}
        tasks.dump(self.out / "explain_summary.json", dict(runs=len(imp), alarm_windows=len(al), alarm_counts=counts))
        print(summ.round(3).to_string())
        print(al.groupby(["task", "model", "kind"])[tasks.FEATURES].mean().T.round(3).to_string())
