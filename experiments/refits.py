# targeted refits (noise augmentation, window summaries, source balance)
import time
import zlib

import numpy as np
import pandas as pd

from core import fitting, tasks, windows
from core import survival as SV
from core.perturb import perturb_series, series_from_windows, windows_from_series
from experiments.base import Experiment, split_arg

D1_PLAN = 'thermal_to_D1__add_D5positive'
VARIANTS = {
    'arc_noiseaug': dict(plan='D6_h600__own', folds=range(5), models=['xgboost'], lead=120, horizon=600),
    'arc_fresh': dict(plan='D6_h600__own', folds=range(5), models=['xgboost'], lead=120, horizon=600),
    'd1_replay': dict(plan=D1_PLAN, folds=[0], models=['xgboost', 'lightgbm'], lead=60, horizon=300),
    'd1_balanced': dict(plan=D1_PLAN, folds=[0], models=['xgboost', 'lightgbm'], lead=60, horizon=300),
    'd5_relative': dict(plan='D5_h120__own', folds=range(5), models=['lstm', 'gru'], lead=30, horizon=120),
    'arc_nocap': dict(plan='D6_h600__own', folds=range(5), models=['xgboost'], lead=120, horizon=600),
}
EVAL = {'arc_noiseaug': ['clean', 'noise_1', 'noise_2'], 'arc_fresh': ['clean', 'noise_1', 'noise_2'],
        'd1_replay': ['clean'], 'd1_balanced': ['clean'], 'd5_relative': ['clean', 'offset_+2', 'offset_-2'],
        'arc_nocap': ['clean', 'noise_1']}
SEEDS = range(5)


# every array of results/preprocess/tasks/inputs.npz, with the window summaries F
def load_all(tasks_dir):
    with np.load(tasks_dir / 'inputs.npz', allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    d['experiment'] = d['experiment'].astype(str)
    d['mask'] = d['mask'].astype(bool)
    d['F'] = windows.window_features(d['X'], d['mask'], np.array([0]))
    return d


# window summaries that count only fresh native observations (age < 1 s) as data
def fresh_features(X, M, age):
    return windows.window_features(X, M & (np.nan_to_num(age, nan=np.inf) < 1.0)[:, :, None], np.array([0]))


# window-mean-removed temperature
def relative(X, M):
    v = np.where(M, X, np.nan)
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(v, axis=1, keepdims=True)
    return np.where(M, X - np.nan_to_num(mu), 0.0).astype(np.float32)


# windows with an independent 1 degC noise realization per native observation
def noisy_windows(ix, d, tag):
    series = series_from_windows(np.unique(ix), d)
    vals = {}
    for k, s in series.items():
        vals[k] = perturb_series(s, 'noise_1', np.random.default_rng(zlib.crc32((k + '|noise_1|' + tag).encode())))
    return windows_from_series(ix, series, vals, d)


# within each target class, D5 windows and the other sources carry equal total weight
def source_weights(ix, y, d, c):
    ds = c.loc[d['experiment'][ix], 'dataset'].to_numpy()
    w = np.zeros(len(ix))
    for cls in [0, 1]:
        for grp in [ds == 'D5', ds != 'D5']:
            j = (y[ix] == cls) & grp
            if j.any():
                w[j] = 1.0 / j.sum()
    return w * len(ix) / w.sum()


# task XGBoost configuration with optional fitting weights; returns the in-memory predictor
def xgb_fit(F, y, keys, tr, va, seed, dest, weight=None):
    import xgboost as xgb
    ratio = float((y[tr] == 0).sum()) / max(float(y[tr].sum()), 1.0)
    m = xgb.XGBClassifier(n_estimators=2000, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=2, scale_pos_weight=np.sqrt(ratio), tree_method='hist', device='cpu',
                          eval_metric='logloss', early_stopping_rounds=50, random_state=seed, n_jobs=4, verbosity=0)
    m.fit(F[tr], y[tr], sample_weight=weight, eval_set=[(F[va], y[va])],
          sample_weight_eval_set=[fitting.balanced_weights(y[va], keys[va])], verbose=False)
    m.save_model(dest / 'model.ubj')
    return lambda Fx: m.predict_proba(Fx)[:, 1], dict(best_iteration=int(m.best_iteration) + 1)


# task LightGBM configuration with optional fitting weights; saves the complete booster at the lowest validation loss
def lgb_fit(F, y, keys, tr, va, seed, dest, weight=None):
    import lightgbm as lgb
    pos = np.sqrt((y[tr] == 0).sum() / max(y[tr].sum(), 1))
    best = {'loss': np.inf}

    def snapshot(env):
        value = float(env.evaluation_result_list[0][2])
        if value < best['loss']:
            best.update(loss=value, text=env.model.model_to_string(num_iteration=-1))
    snapshot.order = 21
    m = lgb.LGBMClassifier(n_estimators=2000, max_depth=3, num_leaves=7, learning_rate=0.05, min_child_samples=30,
                           reg_lambda=1, scale_pos_weight=float(pos), subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    m.fit(F[tr], y[tr], sample_weight=weight, eval_set=[(F[va], y[va])],
          eval_sample_weight=[fitting.balanced_weights(y[va], keys[va])], eval_metric='binary_logloss',
          callbacks=[snapshot, lgb.early_stopping(50, verbose=False)])
    lgb.Booster(model_str=best['text']).save_model(str(dest / 'model.txt'), num_iteration=-1)
    b = lgb.Booster(model_file=str(dest / 'model.txt'))
    return lambda Fx: b.predict(Fx, num_threads=4), dict(best_iteration=int(m.best_iteration_))


# task-start evaluation at the 10 % budget with first-admissible timing; tau_fixed=None uses the recalibrated threshold
def evaluate(t, r, keys, roles, c, lead, tau_fixed=None):
    order = np.lexsort((t, keys))
    ks = keys[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    per = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        j = order[a:b]
        k = ks[a]
        trig = float(c.at[k, 't_trigger'])
        tt = t[j]
        rr = r[j]
        trig = trig if np.isfinite(trig) else float(tt[0])
        valid = tt >= trig
        if valid.any():
            per[k] = (tt, rr, roles[j[0]], trig, float(rr[valid].max()))
    cal = np.array([v[4] for v in per.values() if v[2] == 'calibration'])
    tau_recal = SV.calibrate_threshold(cal, np.arange(len(cal)).astype(str), np.ones(len(cal), bool), 0.1)
    rows = []
    for k, (tt, rr, role, trig, _) in per.items():
        if role != 'test':
            continue
        pos = bool(c.at[k, 'scored_positive'])
        neg = bool(c.at[k, 'scored_negative'])
        onset = float(c.at[k, 't_onset_L2_1.0'])
        first = float(tt[tt >= trig].min())
        rec = dict(key=k, positive=pos, negative=neg, first_admissible=first)
        for name, tau in [('fixed', tau_recal if tau_fixed is None else tau_fixed), ('recal', tau_recal)]:
            hit = np.flatnonzero((tt >= trig) & (rr >= tau))
            alarm = float(tt[hit[0]]) if len(hit) else np.nan
            ok = bool(pos and np.isfinite(alarm) and alarm < onset and onset - alarm >= lead)
            rec[name + '_success'] = ok
            rec[name + '_first_decision'] = bool(ok and alarm == first)
            rec[name + '_neg_alarm'] = bool(neg and np.isfinite(alarm))
        rows.append(rec)
    return pd.DataFrame(rows), tau_recal


# one family refitted on unchanged memberships, targets, caps and hyperparameters, changing only the stated element
class TargetedRefits(Experiment):
    name = 'refits'
    help = 'targeted refits (noise augmentation, window summaries, source balance)'

    @staticmethod
    def add_arguments(ap):
        ap.add_argument('--variants', default=','.join(VARIANTS), help='comma-separated subset of: ' + ', '.join(VARIANTS))

    def run(self):
        variants = split_arg(self.args.variants)
        unknown = sorted(set(variants) - set(VARIANTS))
        if unknown:
            raise ValueError('unknown variants: %s' % ', '.join(unknown))
        self.tasks_dir = self.paths.need('preprocess/tasks')
        self.comparison = self.paths.need('task_comparison')
        self.d = load_all(self.tasks_dir)
        self.c = tasks.cohort(self.tasks_dir)
        for variant in variants:
            rec = self.refit(variant)
            rec.to_parquet(self.out / f'retrain_r3_{variant}.parquet', index=False)
            if variant == 'd1_replay':
                self.replay_check(rec)
            rows = []
            for (model, seed, kind), g in rec.groupby(['model', 'seed', 'perturbation'], sort=False):
                pos = g[g.positive]
                neg = g[g.negative]
                rows.append(dict(variant=variant, model=model, seed=seed, perturbation=kind, n_positive=len(pos),
                                 n_negative=len(neg), success_fixed=int(pos.fixed_success.sum()),
                                 first_decision_fixed=int(pos.fixed_first_decision.sum()),
                                 far_fixed=neg.fixed_neg_alarm.mean(), success_recal=int(pos.recal_success.sum()),
                                 first_decision_recal=int(pos.recal_first_decision.sum()),
                                 far_recal=neg.recal_neg_alarm.mean()))
            seedwise = pd.DataFrame(rows)
            seedwise.to_csv(self.out / f'retrain_r3_{variant}_seed.csv', index=False)
            means = seedwise.groupby(['model', 'perturbation'], sort=False).mean(numeric_only=True)
            print(means.round(3).to_string(), flush=True)
        tasks.dump(self.out / 'retrain_r3_done.json', dict(variants=variants))
        print('REFITS COMPLETE', flush=True)

    # an unchanged refit must reproduce the stored record decisions and thresholds
    def replay_check(self, rec):
        prim = tasks.primary_records(self.comparison)
        checks = []
        for (model, seed), g in rec[rec.perturbation.eq('clean')].groupby(['model', 'seed']):
            st = prim[(prim.task == 'thermal_to_D1') & (prim.arm == 'add_D5positive') & (prim.model == model)
                      & (prim.seed == seed) & (prim.role == 'test')].set_index('key')
            g = g.set_index('key')
            negs = g[g.negative].index
            stored_neg = st.loc[negs, 'negative_alarm'].astype(bool)
            checks.append(dict(model=model, seed=seed,
                               success=bool((g.fixed_success == st.loc[g.index, 'success_60'].astype(bool)).all()),
                               negative=bool((g.loc[negs, 'fixed_neg_alarm'] == stored_neg).all()),
                               tau=float(abs(g.tau_fixed.iloc[0] - st.tau.iloc[0]))))
        checks = pd.DataFrame(checks)
        checks.to_csv(self.out / 'retrain_r3_d1_replay_check.csv', index=False)
        print(checks.to_string(), flush=True)
        assert checks.success.all() and checks.negative.all() and (checks.tau < 1e-9).all(), 'd1 replay differs'

    def refit(self, variant):
        d = self.d
        c = self.c
        v = VARIANTS[variant]
        y, w = SV.discrete_hazard_targets(d['y_time'], d['y_event'], v['horizon'])
        keys = d['experiment']
        obs = d['label_observed'].astype(bool)
        task, arm = v['plan'].split('__')
        out = []
        for fold in v['folds']:
            tag = f"{v['plan']}__f{fold}"
            z = pd.read_csv(self.tasks_dir / 'memberships' / f'{tag}.csv')
            idx = {r: np.flatnonzero(np.isin(keys, z.loc[z.role.eq(r), 'key']) & obs) for r in ['fit', 'validation']}
            tr = tasks.cap(idx['fit'][w[idx['fit']] > 0], keys, y)
            va = tasks.cap(idx['validation'][w[idx['validation']] > 0], keys, y)
            if variant == 'arc_nocap':
                tr = idx['fit'][w[idx['fit']] > 0]
                va = idx['validation'][w[idx['validation']] > 0]
            ref = tasks.stored_risk(self.comparison, task, fold, v['models'][0], 0, arm)
            sel = np.isin(ref['role'], ['calibration', 'test'])
            ix = ref['indices'][sel]
            ekeys = ref['experiment'][sel].astype(str)
            roles = ref['role'][sel]
            t = ref['t'][sel]
            series = series_from_windows(np.unique(ix), d)
            evalX = {}
            for kind in EVAL[variant]:
                vals = {}
                for k, s in series.items():
                    vals[k] = perturb_series(s, kind, np.random.default_rng(zlib.crc32((k + '|' + kind).encode())))
                evalX[kind] = windows_from_series(ix, series, vals, d)
            F = d['F']
            X = d['X']
            M = d['mask']
            if variant == 'arc_fresh':
                F = fresh_features(X, M, d['age'])
            if variant == 'd5_relative':
                X = relative(X, M)
            tr_f = tr
            va_f = va
            y_aug = y
            k_aug = keys
            if variant == 'arc_noiseaug':
                fv = np.concatenate([tr, va])
                Xn, Mn = noisy_windows(fv, d, 'augment')
                Fn = windows.window_features(Xn, Mn, np.array([0]))
                n0 = len(F)
                F = np.concatenate([F, Fn])
                y_aug = np.concatenate([y, y[fv]])
                k_aug = np.concatenate([keys, keys[fv]])
                tr_f = np.concatenate([tr, n0 + np.arange(len(tr))])
                va_f = np.concatenate([va, n0 + len(tr) + np.arange(len(va))])
            for model in v['models']:
                for seed in SEEDS:
                    dest = self.out / 'retrain_r3' / variant / tag / f'{model}_s{seed}'
                    dest.mkdir(parents=True, exist_ok=True)
                    t0 = time.perf_counter()
                    weight = source_weights(tr, y, d, c) if variant == 'd1_balanced' else None
                    if model == 'xgboost':
                        f, info = xgb_fit(F, y_aug, k_aug, tr_f, va_f, seed, dest, weight)
                    elif model == 'lightgbm':
                        f, info = lgb_fit(F, y_aug, k_aug, tr_f, va_f, seed, dest, weight)
                    else:
                        fitting.seq_fit(model, X, M, y, keys, tr, va, seed, tasks.SEQ_CONFIG, dest, max_epochs=100)
                        info = {}
                    tau_clean = None
                    for kind in EVAL[variant]:
                        Xe, Me = evalX[kind]
                        if variant == 'd5_relative':
                            r = fitting.predict(dest, relative(Xe, Me), Me, None, np.arange(len(ix)))
                        elif variant == 'arc_fresh':
                            r = f(fresh_features(Xe, Me, d['age'][ix]))
                        else:
                            r = f(windows.window_features(Xe, Me, np.array([0])))
                        res, tau_recal = evaluate(t, np.asarray(r, float), ekeys, roles, c, v['lead'], tau_clean)
                        if kind == 'clean':
                            tau_clean = tau_recal
                        out.append(res.assign(variant=variant, task=task, arm=arm, fold=fold, model=model, seed=seed,
                                              perturbation=kind, tau_fixed=tau_clean, tau_recal=tau_recal, **info))
                    print(variant, tag, model, seed, f'{time.perf_counter() - t0:.1f}s', flush=True)
        return pd.concat(out, ignore_index=True)
