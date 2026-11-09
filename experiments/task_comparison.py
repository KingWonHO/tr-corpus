# task-specific comparisons with fitting additions
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import fitting, native, survival, tasks, windows
from core import task_analysis
from experiments.base import Experiment, split_arg
from preprocess import manifest

STAGES = ['fit', 'native_rate', 'analyze', 'audit', 'replay', 'interpret']
ROLES = ['fit', 'validation', 'calibration', 'test', 'transfer_case']
EVALUATED = ['validation', 'calibration', 'test', 'transfer_case']
CODE = Path(__file__)


# a cached accounting file is reused only when it is newer than the run's risk file
def cached(path, run):
    return path.exists() and path.stat().st_mtime >= (run / 'risk.npz').stat().st_mtime


# task fits (four learned families, five seeds, grid rules), ARC native-rate rule, alarm accounting, audit, tables
class TaskComparison(Experiment):
    name = 'task_comparison'
    help = 'task-specific comparisons with fitting additions'

    @staticmethod
    def add_arguments(ap):
        ap.add_argument('--stage', default=','.join(STAGES), help='comma list of ' + ', '.join(STAGES))
        ap.add_argument('--models', default='xgboost,lightgbm,gru,lstm,temperature_rule,rate_rule',
                        help='fit stage: learned models and grid rules')
        ap.add_argument('--tasks', default='', help='fit stage: only these tasks')
        ap.add_argument('--plans', default='', help='fit stage: only these plan tags')
        ap.add_argument('--seeds', default='0,1,2,3,4', help='fit stage: seeds of the learned models')
        ap.add_argument('--limit', type=int, default=0, help='fit stage: stop after this many settings')
        ap.add_argument('--partial', action='store_true', help='analyze and audit an incomplete set of runs')

    def run(self):
        self.tasks_dir = self.paths.need('preprocess/tasks')
        chosen = split_arg(self.args.stage)
        unknown = set(chosen) - set(STAGES)
        if unknown:
            raise ValueError('unknown stage %s' % sorted(unknown))
        for stage in STAGES:
            if stage in chosen:
                getattr(self, stage)()

    def inputs(self):
        with np.load(self.tasks_dir / 'inputs.npz', allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        d['experiment'] = d['experiment'].astype(str)
        d['mask'] = d['mask'].astype(bool)
        d['F'] = windows.window_features(d['X'], d['mask'], np.array([0]))
        return d

    def fit(self):
        models = split_arg(self.args.models)
        unknown = set(models) - set(tasks.LEARNED + task_analysis.GRID_RULES)
        if unknown:
            raise ValueError('unknown model %s' % sorted(unknown))
        d = self.inputs()
        plans = pd.read_csv(self.tasks_dir / 'plans.csv')
        if self.args.tasks:
            plans = plans[plans.task.isin(split_arg(self.args.tasks))]
        if self.args.plans:
            plans = plans[plans.tag.isin(split_arg(self.args.plans))]
        done = 0
        for p in plans.itertuples(index=False):
            for model in models:
                if model.endswith('_rule') and p.arm != 'own':
                    continue
                seeds = [0] if model.endswith('_rule') else split_arg(self.args.seeds, int)
                for seed in seeds:
                    try:
                        result = self.fit_one(d, p, model, seed)
                    except Exception:
                        print('FAILED', p.tag, model, seed, flush=True)
                        raise
                    print('DONE', p.tag, model, seed, result, flush=True)
                    done += 1
                    if self.args.limit and done >= self.args.limit:
                        return
        print('WORKER_COMPLETE', done, flush=True)

    # one model and seed on one plan: capped fit/validation windows, full evaluation trajectories
    def fit_one(self, d, p, model, seed):
        dest = self.out / 'runs' / p.tag / f'{model}_s{seed}'
        if (dest / 'complete.json').exists():
            return 'cached'
        dest.mkdir(parents=True, exist_ok=True)
        membership = self.tasks_dir / 'memberships' / p.membership
        z = pd.read_csv(membership)
        keys = d['experiment']
        obs = d['label_observed'].astype(bool)
        y, w = survival.discrete_hazard_targets(d['y_time'], d['y_event'], p.horizon)
        idx = {r: np.flatnonzero(np.isin(keys, z.loc[z.role.eq(r), 'key']) & obs) for r in ROLES}
        tr = tasks.cap(idx['fit'][w[idx['fit']] > 0], keys, y)
        va = tasks.cap(idx['validation'][w[idx['validation']] > 0], keys, y)
        info = dict(task=p.task, arm=p.arm, fold=int(p.fold), horizon=int(p.horizon),
                    highlight_lead=int(p.highlight_lead), model=model, seed=seed, membership=p.membership,
                    membership_sha256=fitting.sha(membership), code_sha256=fitting.sha(CODE))
        rule = model.endswith('_rule')
        if not rule and (len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2):
            reason = 'fit/validation lacks both forecast target classes; split unchanged'
            tasks.dump(dest / 'infeasible.json', dict(info, reason=reason))
            return 'infeasible'
        held = {k: v for k, v in idx.items() if k not in ['fit', 'validation']}
        np.savez_compressed(dest / 'indices.npz', fit=tr, validation=va, **held)
        ev = np.concatenate([idx[r] for r in EVALUATED])
        roles = np.concatenate([np.full(len(idx[r]), r) for r in EVALUATED])
        if rule:
            risk = d['X'][ev, -1, 0] if model == 'temperature_rule' else d['F'][ev, 6]
            fit = dict(epochs_run=0, best_epoch=0, fit_s=0)
        else:
            if model == 'lightgbm':
                fit = tasks.lgb_fit(d['F'], y, keys, tr, va, seed, dest)
            elif model == 'xgboost':
                fit = fitting.tree_fit(d['F'], y, keys, tr, va, seed, dict(tasks.TREE_CONFIG), dest)
            else:
                fit = fitting.seq_fit(model, d['X'], d['mask'], y, keys, tr, va, seed, dict(tasks.SEQ_CONFIG), dest,
                                      max_epochs=100)
            risk = tasks.predict(dest, d, ev)
            # independent reload, same full trajectory batch ordering
            again = tasks.predict(dest, d, ev)
            err = float(np.max(np.abs(again - risk)))
            assert err < 1e-7, err
            info['replay_max_abs_error'] = err
        assert np.isfinite(risk).all()
        np.savez_compressed(dest / 'risk.npz', risk=risk, indices=ev, role=roles, experiment=keys[ev], t=d['t_end'][ev])
        info.update(fit)
        tasks.dump(dest / 'complete.json', info)
        gc.collect()
        torch.cuda.empty_cache()
        return f"epoch {info['best_epoch']}/{info['epochs_run']}; fit {info['fit_s']:.2f}s"

    # ARC rate from the original acquisition intervals, one setting per ARC own-source split
    def native_rate(self):
        own = pd.read_csv(self.tasks_dir / 'plans.csv').query("arm=='own' and task=='D6_h600'")
        tasks.dump(self.out / 'native_rate_manifest.json', dict(code_sha256=fitting.sha(CODE),
                                                                input_sha256=fitting.sha(self.tasks_dir / 'inputs.npz'),
                                                                settings=len(own)))
        with np.load(self.tasks_dir / 'inputs.npz', allow_pickle=True) as z:
            d = {k: z[k] for k in ['experiment', 'label_observed', 't_end']}
        observed = d['label_observed'].astype(bool)
        risk = np.full(len(d['t_end']), np.nan)
        audit = []
        for key, (t, x) in native.native_series(self.paths.arc_raw).items():
            ii = np.flatnonzero(d['experiment'] == key)
            if not len(ii):
                continue
            risk[ii] = native.window_risk(t, x, d['t_end'][ii])
            audit.append(dict(key=key, native_points=len(t), min_native_interval=float(np.diff(t).min()),
                              median_native_interval=float(np.median(np.diff(t))), windows=len(ii)))
        assert len(audit) == 66, len(audit)
        pd.DataFrame(audit).to_csv(self.out / 'arc_native_rate_inputs.csv', index=False)
        for row in own.itertuples(index=False):
            dest = self.out / 'runs' / row.tag / 'native_rate_rule_s0'
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'complete.json').exists():
                continue
            membership = self.tasks_dir / 'memberships' / row.membership
            q = pd.read_csv(membership)
            idx = {}
            for r in EVALUATED:
                idx[r] = np.flatnonzero(np.isin(d['experiment'], q.loc[q.role.eq(r), 'key']) & observed)
            ev = np.concatenate([idx[r] for r in EVALUATED])
            role = np.concatenate([np.full(len(idx[r]), r) for r in EVALUATED])
            assert np.isfinite(risk[ev]).all()
            np.savez_compressed(dest / 'risk.npz', risk=risk[ev], indices=ev, role=role, experiment=d['experiment'][ev],
                                t=d['t_end'][ev])
            np.savez_compressed(dest / 'indices.npz', fit=np.array([], int), **idx)
            info = {k: getattr(row, k) for k in ['task', 'arm', 'fold', 'horizon', 'highlight_lead', 'membership']}
            info.update(model='native_rate_rule', seed=0, epochs_run=0, best_epoch=0, fit_s=0,
                        code_sha256=fitting.sha(CODE), membership_sha256=fitting.sha(membership))
            tasks.dump(dest / 'complete.json', info)
        print('NATIVE_RATE_COMPLETE', len(own), flush=True)

    # record-level accounting of every run under both policies and four budgets, seed summaries, paired additions
    def analyze(self):
        cache = self.out / 'analysis_cache'
        cache.mkdir(exist_ok=True)
        c = pd.read_csv(self.tasks_dir / 'cohort.csv').set_index('key')
        with np.load(self.tasks_dir / 'inputs.npz', allow_pickle=True) as z:
            age = z['age']
        runs = self.out / 'runs'
        paths = sorted(runs.glob('*/*/complete.json'))
        allrows = []
        allcal = []
        for p in paths:
            tag = p.parent.parent.name + '__' + p.parent.name
            pf = cache / (tag + '.parquet')
            pc = cache / (tag + '_cal.parquet')
            if cached(pf, p.parent) and cached(pc, p.parent):
                r = pd.read_parquet(pf)
                cal = pd.read_parquet(pc)
            else:
                r, cal = task_analysis.evaluate(p, c, age)
                r.to_parquet(pf, index=False)
                cal.to_parquet(pc, index=False)
            allrows.append(r)
            allcal.append(cal)
        x = pd.concat(allrows, ignore_index=True)
        ca = pd.concat(allcal, ignore_index=True)
        x.to_parquet(self.out / 'per_record.parquet', index=False)
        ca.to_csv(self.out / 'calibration.csv', index=False)
        s = task_analysis.seed_summary(x)
        s.to_csv(self.out / 'summary_seed.csv', index=False)
        task_analysis.seed_mean(s).to_csv(self.out / 'summary.csv', index=False)
        if not self.args.partial:
            task_analysis.paired_additions(x).to_csv(self.out / 'paired_additions.csv', index=False)
        expected = len(task_analysis.expected_settings(pd.read_csv(self.tasks_dir / 'plans.csv')))
        inf = list(runs.glob('*/*/infeasible.json'))
        status = 'complete' if len(paths) + len(inf) == expected else 'partial'
        coverage = dict(status=status, completed=len(paths), infeasible=len(inf), expected=expected,
                        per_record_rows=len(x))
        tasks.dump(self.out / 'coverage.json', coverage)
        if not self.args.partial:
            assert status == 'complete', (len(paths), len(inf), expected)
        print('ANALYZED', status, len(paths), len(x), flush=True)

    # independent integrity checks of splits, saved states, trajectories, decisions and aggregates
    def audit(self):
        check = task_analysis.Checks()
        frozen = json.loads((self.tasks_dir / 'manifest.json').read_text())
        for name, digest in frozen['sha256'].items():
            check(fitting.sha(manifest.resolve(self.paths, name)) == digest, 'frozen input ' + name)
        plans = pd.read_csv(self.tasks_dir / 'plans.csv')
        expected = task_analysis.expected_settings(plans)
        memberships = self.tasks_dir / 'memberships'
        cohort = pd.read_csv(self.tasks_dir / 'cohort.csv').set_index('key')
        with np.load(self.tasks_dir / 'inputs.npz', allow_pickle=True) as data:
            keys = data['experiment'].astype(str)
            times = data['t_end']
            ytime = data['y_time']
            yevent = data['y_event'].astype(bool)
            observed = data['label_observed'].astype(bool)
        found = set()
        info_rows = []
        val_rows = []
        paths = sorted((self.out / 'runs').glob('*/*/complete.json'))
        records = pd.read_parquet(self.out / 'per_record.parquet')
        cg = pd.read_csv(self.out / 'calibration.csv').groupby(['task', 'arm', 'fold', 'model', 'seed'])
        rg = records.groupby(['task', 'arm', 'fold', 'model', 'seed'])
        checked_accounting = 0
        for number, path in enumerate(paths):
            m = json.loads(path.read_text())
            tag = path.parent.parent.name
            identity = (tag, m['model'], m['seed'])
            check(identity in expected and identity not in found, str(identity))
            found.add(identity)
            check(m['membership_sha256'] == fitting.sha(memberships / m['membership']), 'membership ' + str(identity))
            q = pd.read_csv(memberships / m['membership']).set_index('key')
            development = q[q.role.isin(['fit', 'validation'])]
            check((development.scored_positive | development.scored_negative).all(), 'eligible development')
            check(not development.scale.eq('module').any(), 'module exclusion')
            if m['task'] == 'D6_h600':
                check(not cohort.loc[development.index, 'chemistry'].eq('SIB').any(), 'sodium exclusion')
            with np.load(path.parent / 'indices.npz') as z:
                ix = {k: z[k] for k in z.files}
            task_analysis.check_indices(check, identity, m, ix, q, keys, observed, ytime, yevent)
            learned = not m['model'].endswith('_rule')
            if learned:
                h = pd.read_csv(path.parent / 'history.csv')
                check(m['best_epoch'] == int(np.argmin(h.validation_loss)) + 1, 'best epoch ' + str(identity))
                check(np.isclose(m['best_validation_loss'], h.validation_loss.min(), rtol=1e-6), 'best loss')
                check(m['epochs_run'] == len(h), 'history length')
                check(m['replay_max_abs_error'] < 1e-7, 'full trajectory reload')
                if m['model'] in ['gru', 'lstm']:
                    check(10 <= len(h) <= 100, 'epoch limit')
                info_rows.append(m)
            with np.load(path.parent / 'risk.npz') as z:
                r = {k: z[k] for k in z.files}
            check(np.array_equal(r['experiment'].astype(str), keys[r['indices']]), 'risk keys')
            check(np.array_equal(r['t'], times[r['indices']]), 'risk times')
            check(np.isfinite(r['risk']).all(), 'finite risk')
            check(np.array_equal(q.loc[r['experiment'].astype(str), 'role'].to_numpy(), r['role']), 'risk roles')
            if learned:
                covered, vl = task_analysis.validation_loss(r, ix['validation'], keys, ytime, yevent, m['horizon'])
                check(covered, 'validation coverage')
                err = abs(vl - m['best_validation_loss'])
                check(err < 2e-4, ('independent validation loss', identity, vl, m['best_validation_loss']))
                val_rows.append(dict(task=m['task'], arm=m['arm'], fold=m['fold'], model=m['model'], seed=m['seed'],
                                     saved=m['best_validation_loss'], recomputed=vl, error=err))
            ident = tuple(m[k] for k in ['task', 'arm', 'fold', 'model', 'seed'])
            if ident not in cg.groups or ident not in rg.groups:
                check(self.args.partial, 'analysis missing ' + str(identity))
                continue
            ca = cg.get_group(ident)
            ca = ca[ca.policy.eq('task_start') & ca.budget.eq(.1)].iloc[0]
            ar = rg.get_group(ident)
            ar = ar[ar.policy.eq('task_start') & ar.budget.eq(.1)].set_index('key')
            checked_accounting += task_analysis.check_decisions(check, identity, r, cohort, ca, ar)
            if number % 100 == 0:
                print('AUDIT', number, len(paths), flush=True)
        missing = expected - found
        check(self.args.partial or not missing, 'missing settings ' + str(len(missing)))
        task_analysis.check_aggregates(check, pd.read_csv(self.out / 'summary_seed.csv'), records)
        pd.DataFrame(info_rows).to_csv(self.out / 'training_summary.csv', index=False)
        pd.DataFrame(val_rows).to_csv(self.out / 'validation_loss_replay.csv', index=False)
        verification = dict(status='PASS' if not missing else 'PARTIAL_PASS', checks=check.count, settings=len(found),
                            expected=len(expected), missing=len(missing), primary_record_decisions=checked_accounting,
                            independent_validation_losses=len(val_rows), complete=not missing)
        tasks.dump(self.out / 'verification.json', verification)
        print('VERIFY', check.count, len(found), 'missing', len(missing), flush=True)

    # reload the seed-0 states of the first own-source fold of every task and compare with the saved risk
    def replay(self):
        d = tasks.load_inputs(self.tasks_dir)
        d['F'] = windows.window_features(d['X'], d['mask'], np.array([0]))
        p = pd.read_csv(self.tasks_dir / 'plans.csv')
        p = p[p.arm.eq('own')].sort_values('fold').groupby('task', sort=True).head(1)
        rows = []
        for row in p.itertuples():
            for model in ['lightgbm', 'xgboost', 'gru', 'lstm']:
                folder = self.out / 'runs' / row.tag / (model + '_s0')
                with np.load(folder / 'risk.npz') as z:
                    r = {k: z[k] for k in z.files}
                # even coverage of all saved roles and record trajectories, endpoints included
                pos = np.unique(np.linspace(0, len(r['indices']) - 1, min(4096, len(r['indices']))).round().astype(int))
                pred = tasks.predict(folder, d, r['indices'][pos])
                err = float(np.max(abs(pred - r['risk'][pos])))
                assert err < 2e-6, (row.tag, model, err)
                rows.append(dict(task=row.task, model=model, seed=0, checked_windows=len(pos), max_abs_error=err))
                print('REPLAY', row.task, model, err, flush=True)
        pd.DataFrame(rows).to_csv(self.out / 'independent_state_replay.csv', index=False)
        replayed = dict(status='PASS', states=len(rows), windows=sum(r['checked_windows'] for r in rows),
                        max_abs_error=max(r['max_abs_error'] for r in rows))
        tasks.dump(self.out / 'state_replay_verification.json', replayed)

    # derived tables: role counts, primary summaries, evidence strata, model-rule pairs, opportunity causes, policy
    def interpret(self):
        plans = pd.read_csv(self.tasks_dir / 'plans.csv')
        lead = plans.groupby('task').highlight_lead.first().to_dict()
        roles = task_analysis.role_counts(plans, self.tasks_dir / 'memberships')
        roles.to_csv(self.out / 'role_counts.csv', index=False)
        x = pd.read_parquet(self.out / 'per_record.parquet')
        primary, test = task_analysis.primary_test(x, lead)
        seed = task_analysis.primary_seed(test, lead)
        seed.to_csv(self.out / 'primary_seed.csv', index=False)
        task_analysis.primary_summary(seed).to_csv(self.out / 'primary_summary.csv', index=False)
        task_analysis.evidence_strata(test, lead).to_csv(self.out / 'evidence_strata.csv', index=False)
        task_analysis.untimed_alarms(test).to_csv(self.out / 'untimed_outcome_alarms.csv', index=False)
        cases = primary[primary.role.eq('transfer_case')].copy()
        cases.to_csv(self.out / 'transfer_cases.csv', index=False)
        paired = task_analysis.paired_model_rule(test, lead)
        paired.to_csv(self.out / 'paired_model_rule.csv', index=False)
        characteristics = pd.read_csv(self.tasks_dir / 'data_characteristics.csv')
        causes, summary = task_analysis.opportunity_constraints(characteristics)
        causes.to_csv(self.out / 'opportunity_constraints.csv', index=False)
        summary.to_csv(self.out / 'opportunity_constraints_summary.csv', index=False)
        policy = task_analysis.policy_sensitivity(pd.read_csv(self.out / 'summary_seed.csv'), lead)
        policy.to_csv(self.out / 'policy_sensitivity.csv', index=False)
        print('INTERPRETED', len(policy), len(paired), len(cases), flush=True)
