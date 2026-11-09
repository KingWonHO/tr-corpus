# common-cohort audit: four-role fits on 77 positives and 173 negatives
import gc
import json
import platform
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch

from core import fitting
from core import survival as SV
from core import windows
from core.fitting import sha, write_json
from experiments.base import Experiment, split_arg
from preprocess import manifest

ARMS = {'design_h300': ('group', 300, 'all'), 'design_h60': ('group', 60, 'all'),
        'record_matched_h300': ('record', 300, 'all'), 'source_h300': ('group', 300, 'source'),
        'posttrigger_h300': ('group', 300, 'posttrigger')}
STAGES = ['fit', 'analyze', 'check']
ROLES = ['fit', 'validation', 'calibration', 'test']
LABELS = {'L2_0.5': 't_onset_L2_0.5', 'L2_2': 't_onset_L2_2.0', 'L3': 't_onset_L3', 'ISC25mV': 't_isc', 'L1': 't_onset_L1'}
KEYS = ['arm', 'model', 'panel', 'seed', 'checkpoint', 'calibration_horizon', 'calibration_evidence', 'budget']
LEADS = [0, 30, 60, 120]
PACKAGES = ['torch', 'numpy', 'pandas', 'scipy', 'xgboost', 'scikit-learn', 'mambapy']


def expected_jobs():
    jobs = set()
    for arm in ARMS:
        for fold in range(5):
            for panel in ['T', 'TV']:
                for model in ['xgboost', 'gru', 'mamba']:
                    if arm == 'posttrigger_h300' and (panel != 'TV' or model == 'gru'):
                        continue
                    for seed in range(5):
                        jobs.add((arm, fold, panel, model, seed))
    for arm in ['design_h300', 'record_matched_h300']:
        for fold in range(5):
            for model in ['temperature_rule', 'rate_rule']:
                jobs.add((arm, fold, 'T', model, 0))
    return jobs


# T: surface and ambient temperatures with masks; TV adds V_cell; no observation-age channels
def panel_columns(features, panel):
    cols = windows.panel_mask(features, 'M1')
    if panel == 'TV':
        cols = cols | np.array([f == 'V_cell' for f in features])
    return cols & np.array([not f.startswith('age__') and f != windows.AGE_CH for f in features])


# per-record alarms of one stored run under every calibration horizon, evidence subset and budget
def evaluate(path, c):
    info = json.loads(path.read_text())
    tag = {k: info[k] for k in ['arm', 'fold', 'panel', 'model', 'seed']}
    rows = []
    cal = []
    for ck, fn in [('selected', 'risk.npz'), ('epoch6', 'risk_epoch6.npz')]:
        if not (path.parent / fn).exists():
            continue
        z = np.load(path.parent / fn)
        keys = z['experiment'].astype(str)
        rr = z['risk'].astype(float)
        tt = z['t'].astype(float)
        per = {}
        for k in np.unique(keys):
            ix = np.flatnonzero(keys == k)
            ix = ix[np.argsort(tt[ix])]
            t, r = tt[ix], rr[ix]
            assert np.isfinite(r).all() and np.all(np.diff(t) > 0)
            long = t[-1] - t[0] >= 300
            per[k] = dict(t=t, r=r, role=str(z['role'][ix[0]]), full=r.max(),
                          head300=r[t <= t[0] + 300].max() if long else np.nan,
                          tail300=r[t >= t[-1] - 300].max() if long else np.nan)
        for calh in (['full', 'head300'] if ck == 'selected' else ['full']):
            for evidence in (['all', 'source'] if ck == 'selected' else ['all']):
                cp = {k: v[calh] for k, v in per.items()
                      if v['role'] == 'calibration' and np.isfinite(v[calh]) and (evidence == 'all' or c.loc[k, 'confirmed'])}
                for budget in ([0, .025, .05, .1, .15, .2, .3] if ck == 'selected' else [.1]):
                    ct = dict(tag, checkpoint=ck, calibration_horizon=calh, calibration_evidence=evidence, budget=budget)
                    if not cp:
                        cal.append(dict(ct, status='no_calibration_records', n_cal=0))
                        continue
                    kk = np.array(list(cp))
                    pr = np.array(list(cp.values()), float)
                    tau = SV.calibrate_threshold(pr, kk, np.ones(len(kk), bool), budget)
                    nf = int((pr >= tau).sum())
                    assert nf <= np.floor(budget * len(kk))
                    cal.append(dict(ct, status='ok', tau=tau, n_cal=len(kk), cal_fired=nf, cal_far=nf / len(kk)))
                    for k, v in per.items():
                        if v['role'] != 'test':
                            continue
                        rows.append(test_row(ct, k, v, c.loc[k], tau))
    return pd.DataFrame(rows), pd.DataFrame(cal), info


# alarm, lead and opportunity of one test record at threshold tau
def test_row(ct, k, v, cr, tau):
    t, r = v['t'], v['r']
    hit = np.flatnonzero(r >= tau)
    alarm = float(t[hit[0]]) if len(hit) else np.nan
    trigger = float(cr.t_trigger) if np.isfinite(cr.t_trigger) else float(t[0])
    allowed = t[t >= trigger]
    first = allowed[0] if len(allowed) else np.inf
    pre = bool(np.isfinite(alarm) and alarm < trigger)
    onset = float(cr['t_onset_L2_1.0'])
    det = bool(np.isfinite(onset) and np.isfinite(alarm) and not pre and alarm < onset)
    row = dict(ct, held=k, dataset=cr.dataset, design=cr.design, confirmed=bool(cr.confirmed),
               scored_positive=bool(cr.scored_positive), scored_negative=bool(cr.scored_negative),
               source_verdict=cr.source_verdict, outcome=cr.final_assignment, tau=tau, onset=onset, trigger=trigger,
               first_endpoint=float(t[0]), last_endpoint=float(t[-1]), first_admissible=float(first),
               alarm=alarm, pre_trigger=pre, detected=det, lead=onset - alarm if det else np.nan,
               no_alarm=not np.isfinite(alarm),
               late_alarm=bool(np.isfinite(alarm) and np.isfinite(onset) and alarm >= onset and not pre))
    for h in ['full', 'head300', 'tail300']:
        row[h + '_alarm'] = int(v[h] >= tau) if np.isfinite(v[h]) else np.nan
    for ell in LEADS:
        row['opportunity_' + str(ell)] = bool(np.isfinite(onset) and (onset - first >= ell if ell else onset > first))
        start = max(t[0], trigger)
        row['opportunity_continuous_' + str(ell)] = bool(np.isfinite(onset) and (onset - start >= ell if ell else onset > start))
        row['success_' + str(ell)] = bool(det and onset - alarm >= ell)
        assert not row['success_' + str(ell)] or row['opportunity_' + str(ell)]
    for lab, col in LABELS.items():
        oe = cr[col]
        row['detected_' + lab] = bool(np.isfinite(alarm) and not pre and alarm < oe) if np.isfinite(oe) else np.nan
    return row


def aggregate(x, out):
    rows = []
    refs = []
    for tags, s in x.groupby(KEYS, observed=True):
        for subset, keep in [('all', np.ones(len(s), bool)), ('source', s.confirmed), ('trace', ~s.confirmed)]:
            ss = s[keep]
            scopes = [('all', ss), ('D1-D3', ss[ss.dataset.isin(['D1', 'D2', 'D3'])]), ('D5', ss[ss.dataset.eq('D5')])]
            for scope, g in scopes:
                p = g[g.scored_positive]
                n = g[g.scored_negative]
                row = dict(zip(KEYS, tags))
                row.update(subset=subset, scope=scope, n_positive=len(p), n_negative=len(n),
                           positive_designs=p.design.nunique(), negative_designs=n.design.nunique(),
                           detected=int(p.detected.sum()), pre_trigger=int(p.pre_trigger.sum()), no_alarm=int(p.no_alarm.sum()),
                           late_alarm=int(p.late_alarm.sum()), median_lead=p.lead.median())
                for ell in LEADS:
                    op = int(p[f'opportunity_{ell}'].sum())
                    met = int(p[f'success_{ell}'].sum())
                    row.update({f'opp_{ell}': op, f'met_{ell}': met, f'rate_all_{ell}': met / len(p) if len(p) else np.nan,
                                f'rate_observable_{ell}': met / op if op else np.nan,
                                f'opp_continuous_{ell}': int(p[f'opportunity_continuous_{ell}'].sum())})
                for h in ['full', 'head300', 'tail300']:
                    nn = n[f'{h}_alarm'].dropna()
                    k = int(nn.sum())
                    num = len(nn)
                    row.update({f'neg_n_{h}': num, f'neg_k_{h}': k, f'far_{h}': k / num if num else np.nan})
                rows.append(row)
                if row['calibration_horizon'] == 'full' and np.isclose(row['budget'], .1) and row['checkpoint'] == 'selected':
                    for lab in LABELS:
                        b = p.dropna(subset=['detected_' + lab])
                        a = b.detected.astype(bool)
                        alt = b['detected_' + lab].astype(bool)
                        refs.append(dict(zip(KEYS, tags), subset=subset, scope=scope, label=lab, n=len(b),
                                         flips=int((a != alt).sum()), success_to_failure=int((a & ~alt).sum()),
                                         failure_to_success=int((~a & alt).sum())))
    pd.DataFrame(rows).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(refs).to_csv(out / 'reference_flips.csv', index=False)


# design-cluster bootstrap interval of a mean (2000 resamples, seed 20261030)
def conditional_interval(v, g):
    v = np.asarray(v, float)
    g = np.asarray(g, str)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if not len(v):
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, groups=0, degenerate=True)
    names = np.unique(g)
    s = np.array([v[g == k].sum() for k in names])
    n = np.array([(g == k).sum() for k in names])
    rng = np.random.default_rng(20261030)
    ix = rng.integers(0, len(names), (2000, len(names)))
    means = s[ix].sum(1) / n[ix].sum(1)
    ci = np.quantile(means, [.025, .975])
    return dict(mean=v.mean(), lo=ci[0], hi=ci[1], n=len(v), groups=len(names), degenerate=len(np.unique(v)) < 2)


def design_counts(s):
    pos = s.scored_positive
    neg = s.scored_negative
    return pd.Series(dict(n_positive=int(pos.sum()), met_60=int(s.loc[pos, 'success_60'].sum()),
                          met_120=int(s.loc[pos, 'success_120'].sum()), n_negative=int(neg.sum()),
                          full_alarms=int(s.loc[neg, 'full_alarm'].sum())))


def controls(x, c, native, out):
    p = x[x.checkpoint.eq('selected') & x.calibration_horizon.eq('full') & x.budget.eq(.1)]
    z = p[p.calibration_evidence.eq('all')]
    rows = []
    paired = []
    discord = []
    native_rows = []
    disputed = set(c.index[c.dataset.eq('D5') & c.source_verdict.eq('non-TR') & c['t_onset_L2_1.0'].notna()])
    assert len(disputed) == 7
    native = native.copy()
    native['held'] = native.dataset_id + '/' + native.experiment_id
    for tags, s in z.groupby(['arm', 'model', 'panel', 'seed']):
        tag = dict(zip(['arm', 'model', 'panel', 'seed'], tags))
        for subset, keep in [('all', np.ones(len(s), bool)), ('source', s.confirmed)]:
            a = s[keep]
            for metric, flag in [('success_60', 'scored_positive'), ('success_120', 'scored_positive'),
                                 ('full_alarm', 'scored_negative'), ('head300_alarm', 'scored_negative')]:
                b = a[a[flag]].dropna(subset=[metric])
                rows.append(dict(tag, subset=subset, metric=metric, **conditional_interval(b[metric], b.design)))
        for scenario, flag in [('eligible_only', s.scored_negative), ('add_source_nonTR_L2', s.scored_negative | s.held.isin(disputed))]:
            b = s[flag]
            added = b[b.held.isin(disputed)]
            discord.append(dict(tag, scenario=scenario, negative_n=len(b), alarms=int(b.full_alarm.sum()), far=b.full_alarm.mean(),
                                added_n=len(added), added_alarms=int(added.full_alarm.sum())))
        if tags[0] == 'design_h300':
            b = s[s.scored_positive].merge(native[['held', 'native_L2_s', 'native1s_L2_s', 'causal_L2_s']], on='held', validate='one_to_one')
            for col in ['native_L2_s', 'native1s_L2_s', 'causal_L2_s']:
                q = b[b[col].notna()]
                det = q.alarm.notna() & ~q.pre_trigger & (q.alarm < q[col])
                lead = q[col] - q.alarm
                native_rows.append(dict(tag, reference=col, n=len(q), missing=len(b) - len(q), detected=int(det.sum()),
                                        flips=int((det != q.detected).sum()), met_60=int((det & (lead >= 60)).sum()),
                                        met_120=int((det & (lead >= 120)).sum())))
    for tags, s in z.groupby(['arm', 'model', 'seed']):
        for metric, flag in [('success_60', 'scored_positive'), ('success_120', 'scored_positive'), ('full_alarm', 'scored_negative')]:
            q = s[s[flag]].pivot(index=['held', 'design'], columns='panel', values=metric)
            if not {'T', 'TV'}.issubset(q.columns):
                continue
            q = q.dropna()
            delta = q['TV'].astype(float) - q['T'].astype(float)
            paired.append(dict(zip(['arm', 'model', 'seed'], tags), metric=metric,
                               **conditional_interval(delta, q.index.get_level_values('design'))))
    pd.DataFrame(rows).to_csv(out / 'conditional_design_intervals.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'paired_sensor_intervals.csv', index=False)
    pd.DataFrame(discord).to_csv(out / 'discordant_negatives.csv', index=False)
    pd.DataFrame(native_rows).to_csv(out / 'native_reference.csv', index=False)
    # design concentration, shown without selecting or excluding any test design
    groups = ['arm', 'model', 'panel', 'seed', 'design', 'dataset', 'confirmed']
    q = z.groupby(groups, observed=True).apply(design_counts, include_groups=False).reset_index()
    q.to_csv(out / 'design_results.csv', index=False)


# common-cohort audit experiment (stages fit, analyze, check)
class CommonCohortAudit(Experiment):
    name = 'common_cohort'
    help = 'common-cohort audit: four-role fits on 77 positives and 173 negatives'

    @staticmethod
    def add_arguments(ap):
        ap.add_argument('--stage', default=','.join(STAGES), help='comma-separated subset of fit,analyze,check')
        ap.add_argument('--models', default='xgboost,gru,mamba,temperature_rule,rate_rule')
        ap.add_argument('--arms', default=','.join(ARMS))
        ap.add_argument('--folds', default='0,1,2,3,4')
        ap.add_argument('--seeds', default='0,1,2,3,4')
        ap.add_argument('--panels', default='T,TV')
        ap.add_argument('--partial', action='store_true', help='analyze/check the completed subset of the 670 jobs')
        ap.add_argument('--rebuild', action='store_true', help='re-evaluate every run instead of using analysis_cache')

    def run(self):
        stages = split_arg(self.args.stage)
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise SystemExit('unknown stage(s): ' + ','.join(sorted(unknown)))
        self.roles = self.paths.need('preprocess/roles')
        self.cohort = self.paths.need('preprocess/cohort') / 'cohort.csv'
        self.protocol = self.paths.configs / 'common_cohort_protocol.md'
        if 'fit' in stages:
            self.fit()
        if 'analyze' in stages:
            self.analyze()
        if 'check' in stages:
            self.check()

    def fit(self):
        a = self.args
        self.runtime_manifest()
        d, _ = windows.load_windows(self.paths.data)
        for arm in split_arg(a.arms):
            for fold in split_arg(a.folds, int):
                for panel in split_arg(a.panels):
                    for model in split_arg(a.models):
                        if arm == 'posttrigger_h300' and (panel != 'TV' or model not in ['xgboost', 'mamba']):
                            continue
                        if model.endswith('_rule') and (panel != 'T' or arm not in ['design_h300', 'record_matched_h300']):
                            continue
                        for seed in ([0] if model.endswith('_rule') else split_arg(a.seeds, int)):
                            try:
                                result = self.job(d, arm, fold, panel, model, seed)
                            except Exception:
                                print('FAILED', arm, fold, panel, model, seed, flush=True)
                                raise
                            print('DONE', arm, fold, panel, model, seed, result, flush=True)
        print('COMPLETE', flush=True)

    def runtime_manifest(self):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(self.out / 'runtime_manifest.json',
                   dict(captured_during_run=True, python=sys.version, platform=platform.platform(), gpu=gpu, cuda=torch.version.cuda,
                        packages={name: version(name) for name in PACKAGES}))

    def job(self, d, arm, fold, panel, model, seed):
        mode, horizon, evidence = ARMS[arm]
        dest = self.out / 'runs' / arm / f'f{fold}_{panel}_{model}_s{seed}'
        if (dest / 'complete.json').exists():
            return 'cached'
        dest.mkdir(parents=True, exist_ok=True)
        z = pd.read_csv(self.roles / f'roles_{mode}_f{fold}.csv')
        if evidence == 'source':
            z.loc[z.role.isin(['fit', 'validation']) & ~z.confirmed, 'role'] = 'excluded'
        keys = d['experiment'].astype(str)
        obs = d['label_observed'].astype(bool)
        y, w = SV.discrete_hazard_targets(d['y_time'], d['y_event'], horizon)
        idx = {r: np.flatnonzero(np.isin(keys, z.loc[z.role.eq(r), 'key']) & obs) for r in ROLES}
        tr = idx['fit'][w[idx['fit']] > 0]
        va = idx['validation'][w[idx['validation']] > 0]
        if evidence == 'posttrigger':
            tmap = z.set_index('key').t_trigger
            tg = np.array([tmap[k] for k in keys[tr]], float)
            tr = tr[d['t_end'][tr] >= tg]
        assert len(np.unique(y[tr])) == len(np.unique(y[va])) == 2
        cols = panel_columns(d['features'], panel)
        X = np.asarray(d['X'][:, :, cols], dtype=np.float32)
        M = d['mask'][:, :, cols].astype(bool)
        features = np.array(d['features'])[cols].tolist()
        F = windows.window_features(X, M, np.arange(X.shape[2])) if model == 'xgboost' or model.endswith('_rule') else None
        ev = np.concatenate([idx['calibration'], idx['test']])
        role = np.concatenate([np.full(len(idx['calibration']), 'calibration'), np.full(len(idx['test']), 'test')])
        z.to_csv(dest / 'membership.csv', index=False)
        np.savez_compressed(dest / 'indices.npz', fit=tr, validation=va, calibration=idx['calibration'], test=idx['test'])
        info = dict(arm=arm, mode=mode, horizon=horizon, window=60, labels=evidence, fold=fold, panel=panel, model=model, seed=seed,
                    features=features)
        if model.endswith('_rule'):
            if model == 'temperature_rule':
                risk = np.where(M[ev, -1, :], X[ev, -1, :], -np.inf).max(axis=1)
            else:
                risk = F[ev, 6 * len(features):7 * len(features)].max(axis=1)
            selected = None
            info.update(epochs_run=0, best_epoch=0, best_validation_loss=None, fit_s=0)
        else:
            selected = self.select(arm, fold, panel, model, seed, X, M, F, y, keys, tr, va)
            cfg = fitting.configuration(model, selected)
            if seed == 0:
                cp = self.out / 'selection' / arm / f'f{fold}_{panel}_{model}' / f'candidate{selected}'
                for f in cp.iterdir():
                    if f.is_file():
                        (dest / f.name).write_bytes(f.read_bytes())
                fit = json.loads((dest / 'fit.json').read_text())
            elif model == 'xgboost':
                fit = fitting.tree_fit(F, y, keys, tr, va, seed, cfg, dest)
            else:
                fit = fitting.seq_fit(model, X, M, y, keys, tr, va, seed, cfg, dest)
            info.update(fit)
            risk = fitting.predict(dest, X, M, F, ev)
            replay = fitting.predict(dest, X, M, F, ev)
            err = float(np.max(np.abs(replay - risk)))
            assert err <= 1e-7, err
            info['full_replay_max_abs_diff'] = err
            if model != 'xgboost':
                r6 = fitting.predict(dest, X, M, F, ev, 'epoch6.pt')
                np.savez_compressed(dest / 'risk_epoch6.npz', risk=r6, indices=ev, role=role, experiment=keys[ev], t=d['t_end'][ev])
        assert np.isfinite(risk).all()
        np.savez_compressed(dest / 'risk.npz', risk=risk, indices=ev, role=role, experiment=keys[ev], t=d['t_end'][ev])
        info['selected_candidate'] = selected
        info['protocol_sha256'] = sha(self.protocol)
        write_json(dest / 'complete.json', info)
        del X, M, F
        gc.collect()
        torch.cuda.empty_cache()
        return f"epoch={info['best_epoch']}/{info['epochs_run']}, loss={info['best_validation_loss']}"

    # seed 0 fits the three fixed candidates and keeps the lowest validation loss for every seed
    def select(self, arm, fold, panel, model, seed, X, M, F, y, keys, tr, va):
        selection = self.out / 'selection' / arm / f'f{fold}_{panel}_{model}'
        selection.mkdir(parents=True, exist_ok=True)
        selpath = selection / 'selected.json'
        if not selpath.exists():
            assert seed == 0, 'Seed 0 must select config first'
            scores = []
            for ci in range(3):
                cp = selection / f'candidate{ci}'
                cp.mkdir(exist_ok=True)
                cfg = fitting.configuration(model, ci)
                if (cp / 'fit.json').exists():
                    fit = json.loads((cp / 'fit.json').read_text())
                else:
                    if model == 'xgboost':
                        fit = fitting.tree_fit(F, y, keys, tr, va, 0, cfg, cp)
                    else:
                        fit = fitting.seq_fit(model, X, M, y, keys, tr, va, 0, cfg, cp)
                    print('CANDIDATE', arm, fold, panel, model, ci, 'ep', fit['best_epoch'], 'run', fit['epochs_run'],
                          'loss', round(fit['best_validation_loss'], 5), flush=True)
                scores.append(fit['best_validation_loss'])
            write_json(selpath, dict(candidate=int(np.argmin(scores)), validation_losses=scores, selected_by='seed0 validation only'))
        return json.loads(selpath.read_text())['candidate']

    def analyze(self):
        out = self.out
        paths = sorted((out / 'runs').glob('*/*/complete.json'))
        infos = [json.loads(p.read_text()) for p in paths]
        done = {(i['arm'], i['fold'], i['panel'], i['model'], i['seed']) for i in infos}
        expected = expected_jobs()
        assert done <= expected
        assert len(done) == len(infos)
        coverage = dict(expected=len(expected), complete=len(done), missing=sorted(expected - done),
                        status='complete' if done == expected else 'PARTIAL')
        (out / 'coverage.json').write_text(json.dumps(coverage, indent=2))
        if not self.args.partial:
            assert done == expected, coverage
        c = pd.read_csv(self.cohort).set_index('key')
        frames = []
        calframes = []
        cache = out / 'analysis_cache'
        cache.mkdir(exist_ok=True)
        for path in paths:
            label = path.parent.parent.name + '__' + path.parent.name
            rp = cache / (label + '.parquet')
            cp = cache / (label + '_cal.parquet')
            if self.args.rebuild or not rp.exists() or not cp.exists():
                x, cal, _ = evaluate(path, c)
                x.to_parquet(rp, index=False)
                cal.to_parquet(cp, index=False)
            frames.append(pd.read_parquet(rp))
            calframes.append(pd.read_parquet(cp))
        x = pd.concat(frames, ignore_index=True)
        x.to_parquet(out / 'per_record.parquet', index=False)
        pd.concat(calframes, ignore_index=True).to_csv(out / 'calibration.csv', index=False)
        diagnostics = [{k: v for k, v in i.items() if k not in ['features', 'config']} for i in infos]
        pd.DataFrame(diagnostics).to_csv(out / 'fit_diagnostics.csv', index=False)
        if not self.args.partial:
            p = x[x.checkpoint.eq('selected') & x.calibration_horizon.eq('full') & x.calibration_evidence.eq('all') & x.budget.eq(.1)]
            for _, g in p.groupby(['arm', 'model', 'panel', 'seed']):
                assert len(g) == 267 and g.held.nunique() == 267
                assert g.scored_positive.sum() == 77 and g.scored_negative.sum() == 173
                assert (g.scored_positive & g.confirmed).sum() == 37
                assert (g.scored_negative & g.confirmed).sum() == 90
        aggregate(x, out)
        controls(x, c, pd.read_csv(self.paths.configs / 'native_sampling_cells.csv'), out)
        print('Analysis', coverage['status'], len(done), 'of', len(expected), 'jobs;', len(x), 'record rows', flush=True)

    # independent checks of role membership, checkpoint restoration and saved risks
    def check(self):
        out = self.out
        checks = 0
        frozen = json.loads((self.roles / 'input_manifest.json').read_text())['sha256']
        for name, h in frozen.items():
            assert sha(manifest.resolve(self.paths, name)) == h, name
            checks += 1
        yy = np.array([0, 0, 0, 1, 1, 1, 1])
        kk = np.array(['a', 'a', 'b', 'c', 'c', 'c', 'd'])
        assert np.allclose(fitting.balanced_weights(yy, kk), [.125, .125, .25, 1 / 12, 1 / 12, 1 / 12, .25])
        checks += 1
        peaks = np.array([.1, .1, .9, .9])
        threshold = SV.calibrate_threshold(peaks, np.array(['a', 'b', 'c', 'd']), np.ones(4, bool), .25)
        assert threshold > .9 and np.sum(peaks >= threshold) == 0
        checks += 1
        for fold in range(5):
            group = pd.read_csv(self.roles / f'roles_group_f{fold}.csv')
            record = pd.read_csv(self.roles / f'roles_record_f{fold}.csv')
            for flag in ['scored_positive', 'scored_negative']:
                assert group.loc[group.role.eq('fit'), flag].sum() == record.loc[record.role.eq('fit'), flag].sum()
                checks += 1
            for frame in [group, record]:
                assert frame.key.is_unique
                for role in ['fit', 'validation']:
                    assert frame.loc[frame.role.eq(role), 'final_assignment'].ne('indeterminate').all()
                    checks += 1
                if frame is group:
                    assert frame[frame.role.ne('excluded')].groupby('design').role.nunique().max() == 1
                    checks += 1
        paths = sorted((out / 'runs').glob('*/*/complete.json'))
        expected = expected_jobs()
        done = set()
        details = []
        d, _ = windows.load_windows(self.paths.data)
        keys = d['experiment'].astype(str)
        cohort = pd.read_csv(self.cohort)
        paired = cohort[cohort.paired & cohort.dataset.isin(['D1', 'D2', 'D3', 'D5'])]
        assert np.allclose(paired.t_trigger, paired.window_t_trigger, equal_nan=True)
        checks += 1
        pos = paired[paired.scored_positive]
        assert np.allclose(pos['t_onset_L2_1.0'], pos.window_t_onset)
        checks += 1
        for _, row in pos.iterrows():
            q = (keys == row.key) & d['label_observed'] & d['y_event'].astype(bool)
            assert q.any() and np.allclose(d['t_end'][q] + d['y_time'][q], row['t_onset_L2_1.0'], atol=.001, rtol=0)
            checks += 1
        protocol = sha(self.protocol)
        arrays = {}
        for p in paths:
            info = json.loads(p.read_text())
            tag = (info['arm'], info['fold'], info['panel'], info['model'], info['seed'])
            done.add(tag)
            assert tag in expected
            assert info['protocol_sha256'] == protocol
            checks += 1
            z = pd.read_csv(p.parent / 'membership.csv')
            ix = np.load(p.parent / 'indices.npz')
            y, w = SV.discrete_hazard_targets(d['y_time'], d['y_event'], info['horizon'])
            for role in ROLES:
                assert np.isin(keys[ix[role]], z.loc[z.role.eq(role), 'key']).all()
                checks += 1
                assert d['label_observed'][ix[role]].all()
                checks += 1
            assert not set(ix['fit']) & set(ix['validation'])
            assert (w[ix['fit']] > 0).all() and (w[ix['validation']] > 0).all()
            checks += 2
            if info['arm'] == 'source_h300':
                assert z.loc[z.role.isin(['fit', 'validation']), 'confirmed'].all()
                checks += 1
            if info['arm'] == 'posttrigger_h300':
                tg = z.set_index('key').t_trigger
                assert np.all(d['t_end'][ix['fit']] >= np.array([tg[k] for k in keys[ix['fit']]]))
                checks += 1
            col = panel_columns(d['features'], info['panel'])
            assert np.array(d['features'])[col].tolist() == info['features']
            checks += 1
            risk = np.load(p.parent / 'risk.npz')
            assert np.array_equal(risk['indices'], np.r_[ix['calibration'], ix['test']])
            checks += 1
            if info['model'].endswith('_rule'):
                continue
            if info['panel'] not in arrays:
                X = np.asarray(d['X'][:, :, col], np.float32)
                M = d['mask'][:, :, col].astype(bool)
                arrays[info['panel']] = (X, M, windows.window_features(X, M, np.arange(X.shape[2])))
            X, M, F = arrays[info['panel']]
            loc = np.unique(np.linspace(0, len(risk['risk']) - 1, min(128, len(risk['risk'])), dtype=int))
            pred = fitting.predict(p.parent, X, M, F, risk['indices'][loc])
            err = float(np.max(np.abs(pred - risk['risk'][loc])))
            assert err < 2e-5, (tag, err)
            checks += 1
            vlerr = None
            if info['seed'] == 0:
                va = ix['validation']
                pp = np.clip(fitting.predict(p.parent, X, M, F, va).astype(float), 1e-7, 1 - 1e-7)
                loss = -(y[va] * np.log(pp) + (1 - y[va]) * np.log1p(-pp))
                v = float(np.sum(fitting.balanced_weights(y[va], keys[va]) * loss))
                vlerr = abs(v - info['best_validation_loss'])
                assert vlerr < 3e-4, (tag, v, info['best_validation_loss'])
                checks += 1
            hist = pd.read_csv(p.parent / 'history.csv')
            assert int(np.argmin(hist.validation_loss)) + 1 == info['best_epoch']
            checks += 1
            if info['model'] != 'xgboost':
                assert 10 <= info['epochs_run'] <= 100
                assert (p.parent / 'epoch6.pt').exists()
                checks += 2
                cp = torch.load(p.parent / 'model.pt', map_location='cpu', weights_only=False)
                vals = np.where(M[ix['fit']], X[ix['fit']], np.nan)
                mu = np.nan_to_num(np.nanmean(vals, axis=(0, 1)))
                sd = np.nan_to_num(np.nanstd(vals, axis=(0, 1)), nan=1.)
                sd = np.where(sd > 1e-6, sd, 1.)
                assert np.allclose(mu, cp['mu']) and np.allclose(sd, cp['sd'])
                checks += 1
            details.append(dict(arm=info['arm'], fold=info['fold'], model=info['model'], panel=info['panel'], seed=info['seed'],
                                sample_replay_error=err, restored_validation_loss_error=vlerr))
        if not self.args.partial:
            assert done == expected, (len(done), len(expected))
        pd.DataFrame(details).to_csv(out / 'independent_replay_checks.csv', index=False)
        result = dict(checks=checks, jobs=len(done), expected=len(expected), status='PASS' if done == expected else 'PARTIAL PASS',
                      notes='Validation objective checked on all seed-0 fits; sampled replay on all learned fits. '
                            'Initial full-risk repeat used restored checkpoints twice, not live-to-saved equality.')
        (out / 'verification.json').write_text(json.dumps(result, indent=2))
        print(result, flush=True)
