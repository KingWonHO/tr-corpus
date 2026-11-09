# synthetic sensor perturbations at the stored thresholds
import time
import zlib

import numpy as np
import pandas as pd

from core import native, tasks, windows
from core.perturb import PERTURBATIONS, RULES, evaluate, perturb_series, series_from_windows, windows_from_series
from experiments.base import Experiment, split_arg

NATIVE_TASK = 'D6_h600'
NATIVE_MODEL = 'native_rate_rule'
DELAYS = [0, 10, 30, 60]


# perturbed grid inputs re-scored by every stored state, and perturbed native ARC series for the native-rate rule
class Robustness(Experiment):
    name = 'robustness'
    help = 'synthetic sensor perturbations at the stored thresholds'

    @staticmethod
    def add_arguments(ap):
        ap.add_argument('--parts', default='grid,native',
                        help='grid: stored task states and grid rules; native: ARC native-rate rule')

    def run(self):
        self.tasks_dir = self.paths.need('preprocess/tasks')
        self.comparison = self.paths.need('task_comparison')
        self.c = tasks.cohort(self.tasks_dir)
        self.prim = tasks.primary_records(self.comparison)
        parts = split_arg(self.args.parts)
        if 'grid' in parts:
            self.grid()
        if 'native' in parts:
            self.native()

    def stored(self, task, model, seed, fold):
        p = self.prim
        return p[(p.task == task) & (p.arm == 'own') & (p.model == model) & (p.seed == seed)
                 & (p.fold == fold) & (p.role == 'test')]

    def grid(self):
        t_start = time.perf_counter()
        d = tasks.load_inputs(self.tasks_dir)
        c = self.c
        out = []
        checks = []
        for task, lead in tasks.TASKS.items():
            for fold in range(5):
                ref = tasks.stored_risk(self.comparison, task, fold, 'xgboost', 0)
                sel = np.isin(ref['role'], ['calibration', 'test'])
                ix = ref['indices'][sel]
                keys = ref['experiment'][sel].astype(str)
                roles = ref['role'][sel]
                t = ref['t'][sel]
                series = series_from_windows(np.unique(ix), d)
                order = np.argsort(ix)
                assert np.all(np.diff(ix[order]) > 0)
                for kind in PERTURBATIONS:
                    vals = {}
                    for k, s in series.items():
                        rng = np.random.default_rng(zlib.crc32((k + '|' + kind).encode()))
                        vals[k] = perturb_series(s, kind, rng)
                    X, M = windows_from_series(ix, series, vals, d)
                    F = windows.window_features(X, M, np.array([0]))
                    local = dict(X=X, mask=M, F=F)
                    n = np.arange(len(ix))
                    for model in tasks.LEARNED + RULES:
                        seeds = [0] if model in RULES else range(5)
                        for seed in seeds:
                            if model == 'temperature_rule':
                                r = X[:, -1, 0].astype(float)
                            elif model == 'rate_rule':
                                r = F[:, 6].astype(float)
                            else:
                                dest = tasks.run_dir(self.comparison, task, fold, model, seed)
                                r = tasks.predict(dest, local, n).astype(float)
                            st = self.stored(task, model, seed, fold)
                            tau = float(st.tau.iloc[0])
                            res, tau_recal = evaluate(t, r, keys, roles, c, lead, tau)
                            if kind == 'clean':
                                stored = st.set_index('key')
                                j = res.set_index('key')
                                same_s = (j.fixed_success == stored.loc[j.index, f'success_{lead}'].astype(bool)).all()
                                negs = j[j.negative].index
                                stored_neg = stored.loc[negs, 'negative_alarm'].astype(bool)
                                same_n = (j.loc[negs, 'fixed_neg_alarm'] == stored_neg).all()
                                same_tau = abs(tau_recal - tau) < 1e-9
                                checks.append(dict(task=task, fold=fold, model=model, seed=seed, success_match=bool(same_s),
                                                   negative_match=bool(same_n), tau_match=bool(same_tau)))
                                assert same_s and same_n and same_tau, (task, fold, model, seed)
                            res = res.assign(task=task, fold=fold, model=model, seed=seed, perturbation=kind,
                                             tau_fixed=tau, tau_recal=tau_recal)
                            out.append(res)
                    print(task, fold, kind, 'done', round(time.perf_counter() - t_start), 's', flush=True)
        rec = pd.concat(out, ignore_index=True)
        rec.to_parquet(self.out / 'robustness_records.parquet', index=False)
        pd.DataFrame(checks).to_csv(self.out / 'robustness_clean_replay.csv', index=False)

        rows = []
        for (task, model, seed, kind), g in rec.groupby(['task', 'model', 'seed', 'perturbation']):
            pos = g[g.positive]
            neg = g[g.negative]
            rows.append(dict(task=task, model=model, seed=seed, perturbation=kind,
                             n_positive=len(pos), n_negative=len(neg),
                             success_fixed=pos.fixed_success.sum(), far_fixed=neg.fixed_neg_alarm.mean(),
                             success_recal=pos.recal_success.sum(), far_recal=neg.recal_neg_alarm.mean()))
        seedwise = pd.DataFrame(rows)
        seedwise.to_csv(self.out / 'robustness_seed.csv', index=False)
        summ = seedwise.groupby(['task', 'model', 'perturbation']).agg(
            n_positive=('n_positive', 'first'), n_negative=('n_negative', 'first'),
            success_fixed=('success_fixed', 'mean'), success_fixed_sd=('success_fixed', 'std'),
            far_fixed=('far_fixed', 'mean'), success_recal=('success_recal', 'mean'),
            far_recal=('far_recal', 'mean')).reset_index()
        summ.to_csv(self.out / 'robustness_summary.csv', index=False)

        # a delivery delay of d seconds shifts every stored alarm by d
        lat = []
        p = self.prim
        test = p[p.role.eq('test') & p.arm.eq('own') & p.task.isin(list(tasks.TASKS))]
        for (task, model, seed), g in test.groupby(['task', 'model', 'seed']):
            lead = tasks.TASKS[task]
            pos = g[g.positive]
            for delay in DELAYS:
                ok = pos.detected & (pos.conditional_lead - delay >= lead)
                lat.append(dict(task=task, model=model, seed=seed, delay_s=delay, success=int(ok.sum()),
                                n_positive=len(pos)))
        lat = pd.DataFrame(lat)
        latency = lat.groupby(['task', 'model', 'delay_s']).agg(success=('success', 'mean'),
                                                                n_positive=('n_positive', 'first')).reset_index()
        latency.to_csv(self.out / 'latency_summary.csv', index=False)
        tasks.dump(self.out / 'robustness_verification.json',
                   dict(status='PASS', clean_replays=len(checks), records=len(rec), seconds=time.perf_counter() - t_start))
        print('ROBUSTNESS COMPLETE', len(checks), 'clean replays', flush=True)

    def native(self):
        t_start = time.perf_counter()
        lead = tasks.TASKS[NATIVE_TASK]
        series = native.native_series(self.paths.arc_raw)
        c = self.c
        out = []
        checks = []
        for fold in range(5):
            ref = tasks.stored_risk(self.comparison, NATIVE_TASK, fold, NATIVE_MODEL, 0)
            keys_all = ref['experiment'].astype(str)
            sel = np.isin(ref['role'], ['calibration', 'test'])
            keys = keys_all[sel]
            roles = ref['role'][sel]
            t = ref['t'][sel]
            st = self.stored(NATIVE_TASK, NATIVE_MODEL, 0, fold)
            tau = float(st.tau.iloc[0])
            for kind in PERTURBATIONS:
                r = np.zeros(len(t))
                for k in np.unique(keys):
                    j = np.flatnonzero(keys == k)
                    rng = np.random.default_rng(zlib.crc32((k + '|' + kind + '|native').encode()))
                    tt, xx = native.perturb(*series[k], kind, rng)
                    r[j] = native.window_risk(tt, xx, t[j])
                if kind == 'clean':
                    assert np.allclose(r, ref['risk'][sel], rtol=0, atol=1e-9), ('clean risk differs', fold)
                res, tau_recal = evaluate(t, r, keys, roles, c, lead, tau)
                if kind == 'clean':
                    stored = st.set_index('key')
                    g = res.set_index('key')
                    negs = g[g.negative].index
                    stored_neg = stored.loc[negs, 'negative_alarm'].astype(bool)
                    ok = dict(success=bool((g.fixed_success == stored.loc[g.index, f'success_{lead}'].astype(bool)).all()),
                              negative=bool((g.loc[negs, 'fixed_neg_alarm'] == stored_neg).all()),
                              tau=bool(abs(tau_recal - tau) < 1e-9))
                    checks.append(dict(fold=fold, risk_match=True, **{k + '_match': v for k, v in ok.items()}))
                    assert all(ok.values()), (fold, ok)
                out.append(res.assign(task=NATIVE_TASK, fold=fold, model=NATIVE_MODEL, seed=0, perturbation=kind,
                                      tau_fixed=tau, tau_recal=tau_recal))
            print(NATIVE_TASK, fold, 'native done', round(time.perf_counter() - t_start), 's', flush=True)
        rec = pd.concat(out, ignore_index=True)
        rec.to_parquet(self.out / 'robustness_native_records.parquet', index=False)
        pd.DataFrame(checks).to_csv(self.out / 'robustness_native_clean_replay.csv', index=False)
        rows = []
        for kind, g in rec.groupby('perturbation', sort=False):
            pos = g[g.positive]
            neg = g[g.negative]
            rows.append(dict(task=NATIVE_TASK, model=NATIVE_MODEL, perturbation=kind,
                             n_positive=len(pos), n_negative=len(neg), success_fixed=int(pos.fixed_success.sum()), far_fixed=neg.fixed_neg_alarm.mean(),
                             success_recal=int(pos.recal_success.sum()), far_recal=neg.recal_neg_alarm.mean(),
                             tau_fixed_mean=g.groupby('fold').tau_fixed.first().mean(),
                             tau_recal_mean=g.groupby('fold').tau_recal.first().mean()))
        summ = pd.DataFrame(rows)
        summ.to_csv(self.out / 'robustness_native_summary.csv', index=False)
        tasks.dump(self.out / 'robustness_native_verification.json',
                   dict(status='PASS', clean_replays=len(checks), records=len(rec), seconds=time.perf_counter() - t_start))
        print(summ.to_string(index=False))
        print('NATIVE ROBUSTNESS COMPLETE', flush=True)
