# revision analyses on stored states (events, severity, calibration, bounds)
import json
import zlib

import numpy as np
import pandas as pd
from scipy.stats import beta, mannwhitneyu

from core import events, native, tasks, windows
from core.perturb import evaluate, perturb_series, series_from_windows, windows_from_series
from experiments.base import Experiment, split_arg

JOBS = ['initial_temperature', 'severity_far', 'calibration_expectation', 'design_bounds', 'thresholds', 'model_size',
        'seed_spread', 'bootstrap', 'budget_area', 'pretrigger', 'availability', 'event_variants', 'combined']
COMBOS = {'bias+2_hold10': ['offset_+2', 'hold_10'], 'bias-2_hold10': ['offset_-2', 'hold_10']}
WITHIN = ['D1_within', 'D2_within', 'D3_within', 'D4_within']


def cp_lower(x, n):
    return 0.0 if x <= 0 else float(beta.ppf(0.05, x, n - x + 1))


def cp_upper(x, n):
    return 1.0 if x >= n else float(beta.ppf(0.95, x + 1, n - x))


# 1 degC/s held for 3 s, with T >= 60 degC or a >25 % voltage drop by then (GB 38031 C.5.2.7 logic)
def gb_onset(t, T, V, trig):
    t = np.asarray(t, float)
    T = np.asarray(T, float)
    if np.isfinite(T).sum() < 5:
        return None
    dt = float(np.median(np.diff(t)))
    Ts = events.smooth(T, max(3, int(round(1.0 / dt)) | 1))
    rate = np.gradient(Ts, t)
    n = max(1, int(round(3.0 / dt)))
    run = np.convolve((rate >= 1.0).astype(int), np.ones(n, int), mode='valid')
    vdrop = np.zeros(len(t), bool)
    if V is not None and np.isfinite(V).sum() > 5:
        base = V[(t <= trig) & np.isfinite(V)]
        v0 = float(np.median(base)) if len(base) else float(V[np.isfinite(V)][0])
        vdrop = np.maximum.accumulate(np.nan_to_num(V, nan=np.inf) < 0.75 * v0)
    for i in np.flatnonzero(run >= n):
        if Ts[i] >= 60.0 or vdrop[i]:
            return float(t[i])
    return None


# 95 % interval of a ratio of sums when whole designs are resampled
def design_ci(v, groups, reps, seed):
    v = np.asarray(v, float)
    groups = np.asarray(groups)
    names = np.unique(groups)
    sums = np.array([v[groups == g].sum() for g in names])
    counts = np.array([(groups == g).sum() for g in names])
    ix = np.random.default_rng(seed).integers(0, len(names), (reps, len(names)))
    vals = sums[ix].sum(1) / counts[ix].sum(1)
    return np.quantile(vals, [.025, .975])


# stored-state analyses: no model is refitted and no stored threshold is changed
class Review(Experiment):
    name = 'review'
    help = 'revision analyses on stored states (events, severity, calibration, bounds)'

    @staticmethod
    def add_arguments(ap):
        ap.add_argument('--jobs', default=','.join(JOBS), help='comma-separated subset of: ' + ', '.join(JOBS))

    def run(self):
        which = split_arg(self.args.jobs)
        unknown = sorted(set(which) - set(JOBS))
        if unknown:
            raise ValueError('unknown jobs: %s' % ', '.join(unknown))
        self.tasks_dir = self.paths.need('preprocess/tasks')
        self.comparison = self.paths.need('task_comparison')
        self.c = tasks.cohort(self.tasks_dir)
        self.prim = tasks.primary_records(self.comparison)
        self.d = tasks.load_inputs(self.tasks_dir)
        for name in JOBS:
            if name in which:
                getattr(self, name)()
                print('done', name, flush=True)
        print('REVIEW COMPLETE', flush=True)

    def harmonized(self, key):
        return pd.read_parquet(self.paths.data / self.c.at[key, 'file'])

    def own_test(self, task):
        p = self.prim
        return p[(p.task == task) & (p.arm == 'own') & (p.role == 'test')]

    def stored(self, task, model, seed, fold):
        p = self.prim
        return p[(p.task == task) & (p.arm == 'own') & (p.model == model) & (p.seed == seed)
                 & (p.fold == fold) & (p.role == 'test')]

    # D5 surface temperature at the trigger and at the first admissible decision by outcome
    def initial_temperature(self):
        prim = self.prim
        d = self.d
        C = self.c
        g = self.own_test('D5_h120')
        g = g[(g.model == 'lstm') & (g.seed == 0)]
        rows = []
        ix_by_key = pd.Series(np.arange(len(d['experiment']))).groupby(d['experiment']).apply(np.array).to_dict()
        for r in g.itertuples():
            if not (r.positive or r.negative):
                continue
            j = ix_by_key[r.key]
            jj = j[np.isclose(d['t_end'][j], r.first_admissible)]
            assert len(jj) == 1, r.key
            T_first = float(d['X'][jj[0], -1, 0]) if d['mask'][jj[0], -1, 0] else np.nan
            h = self.harmonized(r.key)
            pre = h[(h.time_s <= r.trigger) & h.T_surface_max.notna()]
            T_trig = float(pre.T_surface_max.iloc[-1]) if len(pre) else np.nan
            rows.append(dict(key=r.key, positive=r.positive, T_trigger=T_trig, T_first=T_first,
                             first_minus_trigger=r.first_admissible - r.trigger,
                             chemistry=C.at[r.key, 'chemistry']))
        x = pd.DataFrame(rows)
        rec = prim[(prim.task == 'D5_h120') & (prim.arm == 'own') & prim.model.isin(['lstm', 'gru']) & (prim.role == 'test')]
        fd = rec[(rec.alarm == rec.first_admissible)].key.unique()
        x['recurrent_first_decision_alarm'] = x.key.isin(fd)
        x.to_csv(self.out / 'r3_initial_temperature_records.csv', index=False)
        out = []
        for col in ['T_trigger', 'T_first']:
            p = x[x.positive][col].dropna()
            n = x[~x.positive][col].dropna()
            auc = mannwhitneyu(-p, -n).statistic / (len(p) * len(n))
            out.append(dict(quantity=col, n_pos=len(p), n_neg=len(n),
                            pos_median=p.median(), pos_q1=p.quantile(.25), pos_q3=p.quantile(.75),
                            neg_median=n.median(), neg_q1=n.quantile(.25), neg_q3=n.quantile(.75),
                            auc_lower_is_positive=auc))
        a = x[x.recurrent_first_decision_alarm]
        out.append(dict(quantity='T_trigger_first_decision_alarmed', n_pos=int(a.positive.sum()),
                        n_neg=int((~a.positive).sum()), pos_median=a[a.positive].T_trigger.median(), neg_median=a[~a.positive].T_trigger.median()))
        out.append(dict(quantity='first_minus_trigger_s', n_pos=int(x.positive.sum()), n_neg=int((~x.positive).sum()),
                        pos_median=x[x.positive].first_minus_trigger.median(),
                        neg_median=x[~x.positive].first_minus_trigger.median()))
        succ = rec[rec.positive & rec.success_30.astype(bool)].key.unique()
        fa = rec[rec.negative & rec.negative_alarm.astype(bool) & (rec.alarm == rec.first_admissible)].key.unique()
        x['group'] = np.select([x.key.isin(succ), x.key.isin(fa), x.positive],
                               ['recurrent success', 'first-decision false alarm', 'other positive'], 'other negative')
        for grp, h in x.groupby('group'):
            for col in ['T_trigger', 'T_first']:
                out.append(dict(quantity=f'{grp}|{col}', n_pos=len(h), pos_median=h[col].median(), pos_q1=h[col].min(),
                                pos_q3=h[col].max(), neg_median=h[col].mean()))
        pd.DataFrame(out).to_csv(self.out / 'r3_initial_temperature.csv', index=False)

    # D5 negative alarms by source severity, and alarms on the source-negative/L2-discordant records
    def severity_far(self):
        g = self.own_test('D5_h120').copy()
        sev = self.c.severity_score.reindex(g.key).to_numpy()
        g['group'] = np.select([g.negative & (sev <= 2), g.negative & (sev >= 3), g.negative & np.isnan(sev),
                                ~g.negative & ~g.positive & (sev >= 3)],
                               ['source severity 1-2', 'source severity 3-5', 'trace-inferred', 'discordant'], 'other')
        g['any_alarm'] = g.alarm.notna()
        rows = []
        for (model, grp), h in g[g.group != 'other'].groupby(['model', 'group']):
            per_seed = h.groupby('seed').any_alarm.sum()
            rows.append(dict(model=model, group=grp, n=h.key.nunique(), alarms_mean=per_seed.mean(),
                             rate=per_seed.mean() / h.key.nunique()))
        pd.DataFrame(rows).to_csv(self.out / 'r3_severity_far.csv', index=False)

    # expected test FAR implied by the calibration-negative count of each fold, against the observed FAR
    def calibration_expectation(self):
        rows = []
        for task in tasks.TASKS:
            g = self.own_test(task)
            folds = g[g.model.eq('xgboost') & g.seed.eq(0)].groupby('fold').agg(n_cal=('n_cal', 'first'),
                                                                                 n_neg=('negative', 'sum'))
            k = np.floor(0.1 * folds.n_cal)
            folds['expected_far'] = (k + 1) / (folds.n_cal + 1)
            exp = float((folds.expected_far * folds.n_neg).sum() / folds.n_neg.sum())
            for model, h in g.groupby('model'):
                obs = h[h.negative].groupby('seed').negative_alarm.mean().mean()
                rows.append(dict(task=task, model=model, n_cal_min=int(folds.n_cal.min()), n_cal_max=int(folds.n_cal.max()),
                                 expected_far=exp, observed_far=obs))
        pd.DataFrame(rows).to_csv(self.out / 'r3_calibration_expectation.csv', index=False)

    # Clopper-Pearson bounds with records and with designs as the unit
    def design_bounds(self):
        rows = []
        for task, lead in tasks.TASKS.items():
            g = self.own_test(task)
            for model, h in g.groupby('model'):
                p = h[h.positive]
                n = h[h.negative]
                dp = p.design.nunique()
                dn = n.design.nunique()
                s = p.groupby('seed')[f'success_{lead}'].sum().mean() / p.key.nunique()
                f = n.groupby('seed').negative_alarm.sum().mean() / n.key.nunique()
                rows.append(dict(task=task, model=model, n_pos=p.key.nunique(), pos_designs=dp, n_neg=n.key.nunique(),
                                 neg_designs=dn, timely=s, far=f,
                                 timely_lb_records=cp_lower(s * p.key.nunique(), p.key.nunique()),
                                 timely_lb_designs=cp_lower(s * dp, dp),
                                 far_ub_records=cp_upper(f * n.key.nunique(), n.key.nunique()),
                                 far_ub_designs=cp_upper(f * dn, dn), zero_alarm_ub_designs=1 - 0.05 ** (1 / dn)))
        pd.DataFrame(rows).to_csv(self.out / 'r3_design_bounds.csv', index=False)

    # stored rule thresholds in physical units
    def thresholds(self):
        p = self.prim
        g = p[p.task.isin(list(tasks.TASKS)) & p.arm.eq('own')
              & p.model.isin(['temperature_rule', 'rate_rule', 'native_rate_rule'])]
        out = g.groupby(['task', 'model']).tau.agg(['min', 'median', 'max']).reset_index()
        out.to_csv(self.out / 'r3_thresholds.csv', index=False)

    # trees, split nodes and parameters of the stored states
    def model_size(self):
        import lightgbm as lgb
        import torch
        import xgboost as xgb
        rows = []
        for task in tasks.TASKS:
            for fold in range(5):
                for seed in range(5):
                    for model in tasks.LEARNED:
                        p = tasks.run_dir(self.comparison, task, fold, model, seed)
                        if model == 'xgboost':
                            b = xgb.Booster()
                            b.load_model(str(p / 'model.ubj'))
                            df = b.trees_to_dataframe()
                            best = json.loads((p / 'fit.json').read_text())['best_epoch']
                            df = df[df.Tree < best]
                            rows.append(dict(task=task, model=model, trees=best, nodes=len(df),
                                             splits=int((df.Feature != 'Leaf').sum()), params=np.nan))
                        elif model == 'lightgbm':
                            b = lgb.Booster(model_file=str(p / 'model.txt'))
                            df = b.trees_to_dataframe()
                            bi = b.best_iteration if b.best_iteration > 0 else b.num_trees()
                            df = df[df.tree_index < bi]
                            rows.append(dict(task=task, model=model, trees=bi, nodes=len(df),
                                             splits=int(df.split_feature.notna().sum()), params=np.nan))
                        else:
                            cp = torch.load(p / 'model.pt', map_location='cpu', weights_only=False)
                            n = sum(int(v.numel()) for v in cp['state_dict'].values())
                            rows.append(dict(task=task, model=model, trees=np.nan, nodes=np.nan, splits=np.nan, params=n))
        x = pd.DataFrame(rows)
        x.to_csv(self.out / 'r3_model_size_states.csv', index=False)
        x.groupby(['task', 'model']).agg(['min', 'max']).to_csv(self.out / 'r3_model_size.csv')

    # bias + sample-and-hold combinations at the stored and recalibrated thresholds, with and without a 30 s delay
    def combined(self):
        d = self.d
        C = self.c
        rows = []
        for task, lead in tasks.TASKS.items():
            models = ['xgboost', 'lightgbm'] if task == 'D6_h600' else ['lstm', 'gru', 'temperature_rule']
            for fold in range(5):
                ref = tasks.stored_risk(self.comparison, task, fold, 'xgboost', 0)
                sel = np.isin(ref['role'], ['calibration', 'test'])
                ix = ref['indices'][sel]
                keys = ref['experiment'][sel].astype(str)
                roles = ref['role'][sel]
                t = ref['t'][sel]
                series = series_from_windows(np.unique(ix), d)
                for name, steps in COMBOS.items():
                    vals = {}
                    for k, s in series.items():
                        s2 = dict(s)
                        for kind in steps:
                            rng = np.random.default_rng(zlib.crc32((k + '|' + kind).encode()))
                            s2['val'] = perturb_series(s2, kind, rng)
                        vals[k] = s2['val']
                    X, M = windows_from_series(ix, series, vals, d)
                    F = windows.window_features(X, M, np.array([0]))
                    for model in models:
                        for seed in ([0] if model.endswith('_rule') else range(5)):
                            if model == 'temperature_rule':
                                r = X[:, -1, 0].astype(float)
                            else:
                                dest = tasks.run_dir(self.comparison, task, fold, model, seed)
                                r = tasks.predict(dest, dict(X=X, mask=M, F=F), np.arange(len(ix))).astype(float)
                            st = self.stored(task, model, seed, fold)
                            res, _ = evaluate(t, r, keys, roles, C, lead, float(st.tau.iloc[0]))
                            rows.append(res.assign(task=task, fold=fold, model=model, seed=seed, combo=name))
            if task == 'D6_h600':
                ser = native.native_series(self.paths.arc_raw)
                for fold in range(5):
                    ref = tasks.stored_risk(self.comparison, task, fold, 'native_rate_rule', 0)
                    sel = np.isin(ref['role'], ['calibration', 'test'])
                    keys = ref['experiment'][sel].astype(str)
                    roles = ref['role'][sel]
                    t = ref['t'][sel]
                    p = self.prim
                    st = p[(p.task == task) & (p.arm == 'own') & (p.model == 'native_rate_rule') & (p.fold == fold)
                           & (p.role == 'test')]
                    for name, steps in COMBOS.items():
                        r = np.zeros(len(t))
                        for k in np.unique(keys):
                            j = np.flatnonzero(keys == k)
                            tt, xx = ser[k]
                            for kind in steps:
                                tt, xx = native.perturb(tt, xx, kind, None)
                            r[j] = native.window_risk(tt, xx, t[j])
                        res, _ = evaluate(t, r, keys, roles, C, lead, float(st.tau.iloc[0]))
                        rows.append(res.assign(task=task, fold=fold, model='native_rate_rule', seed=0, combo=name))
        x = pd.concat(rows, ignore_index=True)
        out = []
        for (task, model, combo, seed), g in x.groupby(['task', 'model', 'combo', 'seed']):
            lead = tasks.TASKS[task]
            onset = C['t_onset_L2_1.0'].reindex(g.key).to_numpy()
            pos = g.positive.to_numpy()
            neg = g.negative.to_numpy()
            for delay in [0, 30]:
                for th in ['fixed', 'recal']:
                    a = g[th + '_alarm'].to_numpy() + delay
                    ok = pos & np.isfinite(a) & (a < onset) & (onset - a >= lead)
                    out.append(dict(task=task, model=model, combo=combo, seed=seed, delay=delay, threshold=th,
                                    success=int(ok.sum()), n_pos=int(pos.sum()),
                                    far=float(g[th + '_neg_alarm'][neg].mean())))
        summ = pd.DataFrame(out).groupby(['task', 'model', 'combo', 'delay', 'threshold']).agg(
            success=('success', 'mean'), n_pos=('n_pos', 'first'), far=('far', 'mean')).reset_index()
        summ.to_csv(self.out / 'r3_combined.csv', index=False)

    # GB 38031-style and LFP-specific thermal events, and re-scoring of the stored primary alarms against them
    def event_variants(self):
        C = self.c
        rows = []
        for k, r in C.iterrows():
            h = self.harmonized(k)
            t = h.time_s.to_numpy(float)
            if isinstance(r.T_probe, str) and r.T_probe in h:
                T = h[r.T_probe].to_numpy(float)
            else:
                T = h.T_surface_max.to_numpy(float)
            V = h.V_cell.to_numpy(float) if 'V_cell' in h and h.V_cell.notna().sum() > 5 else None
            censored = (isinstance(r.censored_channels, str) and 'T_surface' in r.censored_channels
                        and np.isfinite(r.T_censored_at) and r.T_censored_at >= 110)
            ceil = r.T_censored_at if censored else None
            trig = float(r.t_trigger) if np.isfinite(r.t_trigger) else 0.0
            l2 = events.onset_L2(t, T, 1.0, ceiling=ceil)
            lfp = events.onset_L2(t, T, 1.0 / 6.0, ceiling=ceil) if r.chemistry == 'LFP' else l2
            rows.append(dict(key=k, dataset=r.dataset, chemistry=r.chemistry, outcome=r.final_assignment,
                             positive=bool(r.scored_positive), negative=bool(r.scored_negative),
                             L2_stored=r['t_onset_L2_1.0'], L2_recomputed=l2, L2_GB=gb_onset(t, T, V, trig), L2_chem=lfp))
        ev = pd.DataFrame(rows)
        ev.to_csv(self.out / 'r3_event_variants_records.csv', index=False)
        same = (ev.L2_stored.isna() & ev.L2_recomputed.isna()) | np.isclose(ev.L2_stored, ev.L2_recomputed)
        summ = dict(records=len(ev), l2_reproduced=int(same.sum()))
        for v in ['L2_GB', 'L2_chem']:
            both = ev.L2_stored.notna() & ev[v].notna()
            dlt = (ev[v] - ev.L2_stored)[both]
            only_variant = {o: int(((ev[v].notna()) & ev.L2_stored.isna() & (ev.outcome == o)).sum())
                            for o in ['positive', 'negative', 'indeterminate']}
            summ[v] = dict(present=int(ev[v].notna().sum()), both=int(both.sum()), only_variant=only_variant,
                           only_l2=int((ev[v].isna() & ev.L2_stored.notna()).sum()),
                           delta_median=float(dlt.median()), delta_p10=float(dlt.quantile(.1)),
                           delta_p90=float(dlt.quantile(.9)), delta_zero=int((dlt == 0).sum()))
        rs = []
        lab = ev.set_index('key')
        for task, lead in tasks.TASKS.items():
            g = self.own_test(task)
            for v in ['L2_stored', 'L2_GB', 'L2_chem']:
                on = lab[v].reindex(g.key).to_numpy(float)
                pos = g.physical_outcome.eq('positive').to_numpy() & np.isfinite(on)
                a = g.alarm.to_numpy(float)
                ok = pos & np.isfinite(a) & (a < on) & (on - a >= lead)
                h = g.assign(timed=pos, ok=ok)
                for model, m in h.groupby('model'):
                    rs.append(dict(task=task, event=v, model=model, timed=int(m[m.seed == m.seed.min()].timed.sum()),
                                   success=m.groupby('seed').ok.sum().mean()))
        pd.DataFrame(rs).to_csv(self.out / 'r3_event_variants_rescore.csv', index=False)
        tasks.dump(self.out / 'r3_event_variants.json', summ)

    # convergence of the paired design-bootstrap intervals
    def bootstrap(self):
        rows = []
        for task, model, rule, metric in [('D5_h120', 'lstm', 'temperature_rule', 'success_30'),
                                          ('D5_h120', 'lstm', 'temperature_rule', 'negative_alarm'),
                                          ('D6_h600', 'xgboost', 'native_rate_rule', 'success_120')]:
            g = self.own_test(task)
            a = g[g.model == model].groupby(['key', 'design', 'positive', 'negative'])[metric].mean().rename('m')
            b = g[g.model == rule].groupby(['key', 'design', 'positive', 'negative'])[metric].mean().rename('r')
            z = pd.concat([a, b], axis=1).reset_index()
            z = z[z.negative] if metric == 'negative_alarm' else z[z.positive]
            dlt = (z.m - z.r).astype(float)
            for reps, seed in [(2000, 20261030)] + [(10000, s) for s in range(5)] + [(2000, s) for s in range(1, 6)]:
                lo, hi = design_ci(dlt, z.design, reps, seed)
                rows.append(dict(task=task, model=model, rule=rule, metric=metric, reps=reps, seed=seed,
                                 delta=dlt.mean(), low=lo, high=hi, designs=z.design.nunique()))
        pd.DataFrame(rows).to_csv(self.out / 'r3_bootstrap.csv', index=False)

    # alarms on pre-trigger windows of the heating and overcharge tests
    def pretrigger(self):
        C = self.c
        rows = []
        for task in WITHIN:
            for fold in range(3):
                for model in tasks.LEARNED:
                    for seed in range(5):
                        p = tasks.run_dir(self.comparison, task, fold, model, seed)
                        if not (p / 'risk.npz').exists():
                            continue
                        z = tasks.stored_risk(self.comparison, task, fold, model, seed)
                        st = self.stored(task, model, seed, fold)
                        if st.empty:
                            continue
                        tau = float(st.tau.iloc[0])
                        test = z['role'] == 'test'
                        for k in np.unique(z['experiment'][test]):
                            if C.at[str(k), 'dataset'] == 'D5':
                                continue
                            j = test & (z['experiment'] == k)
                            trig = float(C.at[str(k), 't_trigger'])
                            pre = j & (z['t'] < trig)
                            if not pre.any():
                                rows.append(dict(task=task, model=model, seed=seed, key=str(k), pre_windows=0, pre_s=0.0,
                                                 alarm=False))
                                continue
                            rows.append(dict(task=task, model=model, seed=seed, key=str(k), pre_windows=int(pre.sum()),
                                             pre_s=float(trig - z['t'][pre].min()),
                                             alarm=bool((z['risk'][pre] >= tau).any())))
        x = pd.DataFrame(rows)
        x.to_csv(self.out / 'r3_pretrigger_records.csv', index=False)
        summ = x.groupby(['task', 'model']).agg(records=('key', 'nunique'),
                                                with_pre=('pre_windows', lambda s: int((s > 0).sum() / 5)),
                                                pre_minutes=('pre_s', lambda s: s.sum() / 5 / 60),
                                                alarms=('alarm', lambda s: s.sum() / 5)).reset_index()
        summ.to_csv(self.out / 'r3_pretrigger.csv', index=False)

    # mean timely fraction and FAR over the 0-20 % calibration budgets
    def budget_area(self):
        x = pd.read_parquet(self.comparison / 'per_record.parquet')
        x = x[x.policy.eq('task_start') & x.role.eq('test')]
        rows = []
        for (task, arm, model), g in x.groupby(['task', 'arm', 'model']):
            lead = int(g.highlight_lead.iloc[0])
            pts = []
            for b, h in g.groupby('budget'):
                p = h[h.positive]
                n = h[h.negative]
                timely = p.groupby('seed')[f'success_{lead}'].mean().mean()
                far = n.groupby('seed').negative_alarm.mean().mean()
                pts.append((b, timely, far))
            pts = np.array(sorted(pts))
            if len(pts) < 4:
                continue
            span = pts[-1, 0] - pts[0, 0]
            rows.append(dict(task=task, arm=arm, model=model, lead=lead,
                             area_timely=np.trapezoid(pts[:, 1], pts[:, 0]) / span,
                             area_far=np.trapezoid(pts[:, 2], pts[:, 0]) / span,
                             **{f'timely_{int(b * 100)}': v for b, v, _ in pts},
                             **{f'far_{int(b * 100)}': f for b, _, f in pts}))
        pd.DataFrame(rows).to_csv(self.out / 'r3_budget_area.csv', index=False)

    # per-seed success counts (median, range) and false-alarm counts
    def seed_spread(self):
        rows = []
        for task, lead in tasks.TASKS.items():
            g = self.own_test(task)
            for model, h in g.groupby('model'):
                s = h[h.positive].groupby('seed')[f'success_{lead}'].sum()
                f = h[h.negative].groupby('seed').negative_alarm.sum()
                rows.append(dict(task=task, model=model, success_seeds=' '.join(map(str, s.astype(int).tolist())),
                                 success_median=s.median(), success_min=s.min(), success_max=s.max(),
                                 false_alarm_seeds=' '.join(map(str, f.astype(int).tolist()))))
        pd.DataFrame(rows).to_csv(self.out / 'r3_seed_spread.csv', index=False)

    # availability and fresh-sample fractions of the monitored ARC windows
    def availability(self):
        d = self.d
        p = self.prim
        g = p[(p.task == 'D6_h600') & (p.arm == 'own') & (p.model == 'xgboost') & (p.seed == 0)]
        keys = g.key.unique()
        j = np.flatnonzero(np.isin(d['experiment'], keys))
        j = j[d['t_end'][j] >= self.c['t_trigger'].reindex(d['experiment'][j]).to_numpy(float)]
        avail = d['mask'][j, :, 0].mean(1)
        fresh = (np.nan_to_num(d['age'][j], nan=np.inf) < 1.0).mean(1)
        std_zero = (d['X'][j, :, 0].std(1) == 0).mean()
        pd.DataFrame([dict(windows=len(j), avail_lt1=float((avail < 1).mean()), avail_median=float(np.median(avail)),
                           fresh_median=float(np.median(fresh)), fresh_zero=float((fresh == 0).mean()),
                           fresh_p90=float(np.quantile(fresh, .9)), std_zero=float(std_zero))]).to_csv(
            self.out / 'r3_availability.csv', index=False)
