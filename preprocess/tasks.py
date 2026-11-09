# tasks step: task cohort, compact maximum-surface-temperature inputs, task memberships and the data audit
import hashlib
import itertools
import json
import zipfile

import numpy as np
import pandas as pd

from core import windows
from core.fitting import sha
from core.tasks import dump
from preprocess import manifest

ROLES = ['fit', 'validation', 'calibration', 'test']
MEMBER_COLUMNS = ['key', 'dataset', 'design', 'scale', 'scored_positive', 'scored_negative', 'eligible', 'confirmed', 'role']
SHORTLIST = {'lightgbm', 'xgboost', 'lstm', 'gru'}
P17_FILES = ['paired_source_comparisons.csv', 'source_probe_summary.csv']


def stable(x):
    return hashlib.sha256(('task20261030/' + str(x)).encode()).hexdigest()


# selected channels of a 3-d array inside an npz archive, read in blocks of 2048 windows
def read_columns(path, member, cols):
    with zipfile.ZipFile(path) as archive, archive.open(member + '.npy') as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(f)
        assert not fortran
        out = np.empty((shape[0], shape[1], len(cols)), dtype=dtype)
        width = int(np.prod(shape[1:])) * dtype.itemsize
        for a in range(0, shape[0], 2048):
            n = min(2048, shape[0] - a)
            raw = f.read(n * width)
            assert len(raw) == n * width
            out[a:a + n] = np.frombuffer(raw, dtype=dtype).reshape((n, *shape[1:]))[:, :, cols]
    return out


# T_surface_max, its mask and observation age of every window, including the zero-shot target windows
def compact(data_dir, zeroshot, path):
    d, _ = windows.load_windows(data_dir)
    features = list(d['features'])
    ti = features.index('T_surface_max')
    ai = features.index('age__T_surface_max')
    data = {k: np.asarray(d[k]) for k in ['y_time', 'y_event', 'label_observed', 't_end', 'experiment', 'trigger']}
    data.update(X=d['X'][:, :, ti:ti + 1].copy(), mask=d['mask'][:, :, ti:ti + 1].copy(), age=d['X'][:, :, ai].copy())
    del d
    with np.load(zeroshot, allow_pickle=True) as z:
        names = list(z['features'])
        ti = names.index('T_surface_max')
        ai = names.index('age__T_surface_max')
        more = {k: z[k] for k in ['y_time', 'y_event', 't_end', 'experiment', 'trigger']}
        if 'label_observed' in z.files:
            more['label_observed'] = z['label_observed']
        else:
            more['label_observed'] = np.ones(len(more['t_end']), bool)
    vals = read_columns(zeroshot, 'X', [ti, ai])
    more['X'] = vals[:, :, :1]
    more['age'] = vals[:, :, 1]
    more['mask'] = read_columns(zeroshot, 'mask', [ti])
    for k in data:
        data[k] = np.concatenate([data[k], more[k]])
    data['experiment'] = data['experiment'].astype(str)
    assert np.isfinite(data['X'][data['mask'].astype(bool)]).all()
    np.savez_compressed(path, **data)
    print('COMPACT', data['X'].shape, flush=True)


# five design-held-out folds of one source, balanced over 5000 seeded permutations; validation designs by size only
def design_roles(c, domain):
    z = c[c.dataset.eq(domain)].copy()
    agg = z.groupby('design').agg(p=('scored_positive', 'sum'), n=('scored_negative', 'sum'), total=('key', 'size'))
    gs = agg.index.to_numpy()
    rng = np.random.default_rng(20261030)
    best = None
    for it in range(5000):
        order = rng.permutation(len(gs))
        assignment = {gs[j]: i % 5 for i, j in enumerate(order)}
        counts = np.array([[agg.loc[[g for g in gs if assignment[g] == f], col].sum() for col in ['p', 'n', 'total']]
                           for f in range(5)])
        if counts[:, 0].min() < 1 or counts[:, 1].min() < 2:
            continue
        # all three development folds need at least two positive-bearing designs
        if any(sum(agg.loc[g, 'p'] > 0 for g in gs if assignment[g] not in [f, (f + 1) % 5]) < 2 for f in range(5)):
            continue
        score = float(np.sum((counts / counts.mean(0) - 1) ** 2))
        if best is None or score < best[0]:
            best = (score, assignment)
    assert best is not None, domain
    z['outer_fold'] = z.design.map(best[1])
    out = []
    for f in range(5):
        q = z.copy()
        q['role'] = 'excluded'
        eligible = q.scored_positive | q.scored_negative
        q.loc[q.outer_fold.eq(f), 'role'] = 'test'
        q.loc[q.outer_fold.eq((f + 1) % 5) & q.scored_negative, 'role'] = 'calibration'
        pool = q[~q.outer_fold.isin([f, (f + 1) % 5]) & eligible]
        options = []
        for n in [1, 2, 3]:
            for combo in itertools.combinations(sorted(pool.design.unique()), n):
                va = pool[pool.design.isin(combo)]
                tr = pool[~pool.design.isin(combo)]
                if min(va.scored_positive.sum(), va.scored_negative.sum(), tr.scored_positive.sum(), tr.scored_negative.sum()) < 1:
                    continue
                score = abs(len(va) / len(pool) - .25) + .2 * abs(va.scored_positive.mean() - pool.scored_positive.mean())
                options.append((score, stable(combo), combo))
        assert options, (domain, f)
        v = min(options)[2]
        q.loc[pool.index, 'role'] = 'fit'
        q.loc[pool[pool.design.isin(v)].index, 'role'] = 'validation'
        out.append(q)
    return out


# the two sodium-ion ARC cells are out-of-family transfer cases, never Li-ion fit, validation, calibration or test
def out_of_family(q, sib):
    mask = q.key.isin(sib)
    q.loc[mask, 'role'] = q.loc[mask, 'role'].map(lambda r: 'transfer_case' if r == 'test' else 'excluded')
    return q


# preprocessing step 'tasks'
class TasksStep:
    def __init__(self, paths):
        self.paths = paths
        self.out = paths.out('preprocess/tasks')
        self.plans = []
        self.counts = []
        self.sib = set()

    def add(self, task, arm, f, h, base, donors=None, strict=True, highlight=60):
        q = base.copy()
        q['role'] = q.role.astype(str)
        if donors is not None and len(donors):
            assert set(q.key).isdisjoint(donors.key)
            new = donors.copy()
            new['role'] = 'fit'
            q = pd.concat([q, new], ignore_index=True)
        q = q[MEMBER_COLUMNS].sort_values(['role', 'key'])
        assert q.key.is_unique
        for a, b in itertools.combinations(ROLES, 2):
            assert set(q[q.role.eq(a)].key).isdisjoint(q[q.role.eq(b)].key)
            if strict:
                assert set(q[q.role.eq(a)].design).isdisjoint(q[q.role.eq(b)].design), (task, arm, f, a, b)
        assert q.loc[q.role.isin(['fit', 'validation']), 'eligible'].all()
        assert q.loc[q.role.eq('calibration'), 'scored_negative'].all()
        if task == 'D6_h600':
            q = out_of_family(q, self.sib)
        tag = f'{task}__{arm}__f{f}'
        path = self.out / 'memberships' / f'{tag}.csv'
        q.to_csv(path, index=False)
        self.plans.append(dict(task=task, arm=arm, fold=f, horizon=h, highlight_lead=highlight, membership=path.name,
                               design_disjoint=strict, tag=tag))
        for role, g in q.groupby('role'):
            self.counts.append(dict(task=task, arm=arm, fold=f, role=role, n=len(g), positive=int(g.scored_positive.sum()),
                                    negative=int(g.scored_negative.sum()), designs=g.design.nunique()))

    def shortlist(self):
        prior = pd.read_csv(self.paths.configs / 'p7_inner_policy_selection.csv')
        rows = prior[prior.run_id.eq('p7_nested_L60_h300') & prior.budget.eq(.1) & ~prior.model.str.startswith('rule_')]
        rank = rows.groupby('model').agg(timely=('tune_timely60', 'mean'), far=('tune_far', 'mean'), n=('model', 'size'))
        rank['screening_utility'] = rank.timely - rank.far
        rank = rank.sort_values(['screening_utility', 'timely', 'far'], ascending=[False, False, True])
        rank['selected'] = np.arange(len(rank)) < 4
        rank.to_csv(self.out / 'model_selection.csv')
        models = rank.index[:4].tolist()
        assert set(models) == SHORTLIST
        dump(self.out / 'model_selection.json',
             dict(models=models, criterion='existing inner-validation mean timely@60 minus FAR, fixed 10% calibration budget; '
                                           'retrospective shortlist', new_test_selection=False))
        return models

    def memberships(self, c):
        d5 = design_roles(c, 'D5')
        arc = design_roles(c, 'D6')
        thermal = c[c.dataset.isin(['D1', 'D2', 'D3', 'D4']) & c.scale.eq('cell') & c.scored_positive]
        assert len(thermal) == 23
        modules = c[c.dataset.eq('D4') & c.scale.eq('module')]
        assert len(modules) == 2
        for f, q in enumerate(d5):
            additions = {'own': None, **{f'add_{s}': thermal[thermal.dataset.eq(s)] for s in ['D1', 'D2', 'D3', 'D4']},
                         'add_heating': thermal[thermal.dataset.isin(['D1', 'D3', 'D4'])], 'add_thermal': thermal}
            for arm, donors in additions.items():
                self.add('D5_h120', arm, f, 120, q, donors, highlight=30)
            for arm in ['own', 'add_thermal']:
                self.add('D5_h300', arm, f, 300, q, additions[arm], highlight=30)
        for source in ['D1', 'D2', 'D3', 'D4']:
            s = thermal[thermal.dataset.eq(source)].copy()
            s = s.iloc[sorted(range(len(s)), key=lambda i: stable(s.iloc[i].key))].copy()
            s['f'] = np.arange(len(s)) % 3
            for f in range(3):
                q = s.copy()
                q['role'] = np.where(q.f.eq(f), 'test', 'fit')
                dev = q[q.role.eq('fit')]
                n = max(1, int(round(len(dev) * .25)))
                va = sorted(dev.index, key=lambda i: stable(('val', q.loc[i, 'key'])))[:n]
                q.loc[va, 'role'] = 'validation'
                neg = d5[f][d5[f].role.isin(['calibration', 'test']) & d5[f].scored_negative].copy()
                self.add(source + '_within', 'own', f, 300, pd.concat([q, neg], ignore_index=True), strict=False)
        for f, source in enumerate(['D1', 'D2', 'D3', 'D4']):
            q = thermal.copy()
            q['role'] = np.where(q.dataset.eq(source), 'test', 'fit')
            donors = q[q.role.eq('fit')]
            # validation is a whole donor source chosen by size, never by performance
            sizes = donors.groupby('dataset').size()
            vs = min(sizes.index, key=lambda k: (abs(sizes[k] / len(donors) - .25), stable(k)))
            q.loc[q.dataset.eq(vs), 'role'] = 'validation'
            ng = d5[f][d5[f].role.isin(['calibration', 'test']) & d5[f].scored_negative]
            mo = modules.copy()
            mo['role'] = 'transfer_case'
            base = pd.concat([q, ng, mo], ignore_index=True)
            pool = d5[f][d5[f].role.eq('fit')]
            additions = {'own': None, 'add_D5negative': pool[pool.scored_negative], 'add_D5positive': pool[pool.scored_positive],
                         'add_D5both': pool}
            for arm, dd in additions.items():
                self.add('thermal_to_' + source, arm, f, 300, base, dd, strict=True)
        for f, q in enumerate(arc):
            additions = {'own': None, 'add_thermal': thermal, 'add_D5': d5[f][d5[f].role.eq('fit')]}
            for arm, dd in additions.items():
                self.add('D6_h600', arm, f, 600, q, dd, highlight=120)

    def arc_record(self, c):
        arc = c[c.dataset.eq('D6') & ~c.key.isin(self.sib)]
        dump(self.out / 'arc_chemistry_amendment.json',
             dict(reason='Source audit identified two sodium-ion records in ARC. They are out-of-family transfer cases, '
                         'never Li-ion fit/validation/calibration/test.',
                  primary_Liion_records=len(arc), primary_timed_positive=int(arc.scored_positive.sum()),
                  primary_negative=int(arc.scored_negative.sum()),
                  physical_positive_without_L2=int((arc.final_assignment.eq('positive') & arc['t_onset_L2_1.0'].isna()).sum()),
                  indeterminate=int(arc.final_assignment.eq('indeterminate').sum()), out_of_family_cases=sorted(self.sib),
                  memberships=[p['membership'] for p in self.plans if p['task'] == 'D6_h600']))

    def run(self):
        p = self.paths
        (self.out / 'memberships').mkdir(exist_ok=True)
        c = pd.read_csv(p.need('preprocess/cohort') / 'cohort.csv')
        assert len(c) == 343 and c.key.is_unique
        c['eligible'] = c.scored_positive | c.scored_negative
        c.to_csv(self.out / 'cohort.csv', index=False)
        inventory = c.groupby(['dataset', 'scale', 'trigger', 'final_assignment'], dropna=False).agg(
            n=('key', 'size'), designs=('design', 'nunique'), timed_positive=('scored_positive', 'sum'),
            negative=('scored_negative', 'sum'))
        inventory.reset_index().to_csv(self.out / 'inventory.csv', index=False)
        models = self.shortlist()
        self.sib = set(c.loc[c.dataset.eq('D6') & c.chemistry.eq('SIB'), 'key'])
        assert len(self.sib) == 2
        self.memberships(c)
        pd.DataFrame(self.plans).to_csv(self.out / 'plans.csv', index=False)
        pd.DataFrame(self.counts).to_csv(self.out / 'role_counts.csv', index=False)
        if not (self.out / 'inputs.npz').exists():
            compact(p.data, p.windows / 'test_zeroshot.npz', self.out / 'inputs.npz')
        protocol = p.configs / 'task_protocol.md'
        files = [protocol, p.configs / 'p7_inner_policy_selection.csv'] + [self.out / n for n in
                 ['inputs.npz', 'cohort.csv', 'plans.csv', 'model_selection.csv']] + sorted((self.out / 'memberships').glob('*.csv'))
        dump(self.out / 'manifest.json', dict(sha256=manifest.hashes(p, files), models=models, seed_count=5, plans=len(self.plans),
                                              expected_model_fits=len(self.plans) * len(models) * 5, retrospective=True))
        (self.out / 'protocol.md').write_bytes(protocol.read_bytes())
        self.arc_record(c)
        print('FROZEN', len(self.plans), 'plans', len(self.plans) * 20, 'learned fits', flush=True)
        self.audit()

    # every record described, and the frozen task construction verified
    def audit(self):
        out = self.out
        c = pd.read_csv(out / 'cohort.csv').set_index('key')
        plans = pd.read_csv(out / 'plans.csv')
        with np.load(out / 'inputs.npz', allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        keys = d['experiment'].astype(str)
        rows = []
        checks = 0
        assert set(keys) == set(c.index)
        checks += 1
        for k, g in c.iterrows():
            ii = np.flatnonzero(keys == k)
            ii = ii[np.argsort(d['t_end'][ii])]
            ii = ii[d['label_observed'][ii].astype(bool)]
            tt = d['t_end'][ii]
            x = d['X'][ii, :, 0]
            m = d['mask'][ii, :, 0].astype(bool)
            age = d['age'][ii]
            assert len(ii) > 0 and np.all(np.diff(tt) > 0)
            checks += 1
            onset = g['t_onset_L2_1.0']
            trigger = float(g.t_trigger) if np.isfinite(g.t_trigger) else tt[0]
            if g.scored_positive:
                e = d['y_event'][ii].astype(bool)
                assert e.any() and np.allclose(d['y_time'][ii][e] + tt[e], onset, atol=.002, rtol=0), k
                checks += 1
            row = dict(key=k, dataset=g.dataset, scale=g.scale, trigger_type=g.trigger, design=g.design, cell_type=g.cell_type,
                       chemistry=g.chemistry, capacity_ah=g.capacity_ah, soc_pct=g.soc_pct, source_supported=bool(g.confirmed),
                       outcome=g.final_assignment, timed_positive=bool(g.scored_positive), negative=bool(g.scored_negative),
                       t_trigger=trigger, onset=onset, windows=len(ii), first_endpoint=tt[0], last_endpoint=tt[-1],
                       elapsed_prediction_s=tt[-1] - tt[0], temperature_min=float(x[m].min()), temperature_max=float(x[m].max()),
                       temperature_valid_fraction=float(m.mean()), fresh_last_fraction=float((age[:, -1] < 1).mean()),
                       median_observation_age_s=float(np.median(age[:, -1])),
                       median_fresh_points_per_window=float(np.median((age < 1).sum(1))), channels=g.channels_present,
                       censored_channels=g.censored_channels)
            for ell in [0, 10, 30, 60, 120, 300]:
                good = (tt >= trigger) & ((tt <= onset - ell) if ell else (tt < onset))
                row[f'opp_{ell}'] = bool(g.scored_positive and good.any())
                row[f'fresh2_{ell}'] = bool(g.scored_positive and np.any(good & ((age < 1).sum(1) >= 2)))
                jj = np.flatnonzero(good)
                row[f'fresh2_at_deadline_{ell}'] = bool(g.scored_positive and len(jj) and (age[jj[-1]] < 1).sum() >= 2)
                row[f'age_at_deadline_{ell}'] = float(age[jj[-1], -1]) if g.scored_positive and len(jj) else np.nan
            rows.append(row)
        a = pd.DataFrame(rows)
        a.to_csv(out / 'data_characteristics.csv', index=False)
        summary = a.groupby(['dataset', 'scale']).agg(
            records=('key', 'size'), nominal_design_groups=('design', 'nunique'), raw_cell_type_labels=('cell_type', 'nunique'),
            timed_positive=('timed_positive', 'sum'), negative=('negative', 'sum'),
            median_fresh_last_fraction=('fresh_last_fraction', 'median'), median_observation_age_s=('median_observation_age_s', 'median'),
            median_prediction_duration_s=('elapsed_prediction_s', 'median'), opportunity_30=('opp_30', 'sum'),
            opportunity_60=('opp_60', 'sum'), opportunity_120=('opp_120', 'sum'), fresh2_60=('fresh2_60', 'sum'),
            fresh2_at_deadline_60=('fresh2_at_deadline_60', 'sum'), median_age_at_deadline_60=('age_at_deadline_60', 'median'))
        summary.reset_index().to_csv(out / 'data_characteristics_summary.csv', index=False)
        for p in plans.itertuples():
            q = pd.read_csv(out / 'memberships' / p.membership)
            for r in ROLES:
                assert (q.role == r).any(), (p.tag, r)
                checks += 1
            for r, s in itertools.combinations(ROLES, 2):
                assert set(q[q.role.eq(r)].key).isdisjoint(q[q.role.eq(s)].key)
                checks += 1
                if p.design_disjoint:
                    assert set(q[q.role.eq(r)].design).isdisjoint(q[q.role.eq(s)].design)
                    checks += 1
            assert not q[q.role.isin(['fit', 'validation'])].scale.eq('module').any()
            checks += 1
            if p.arm != 'own':
                base = plans[(plans.task == p.task) & (plans.fold == p.fold) & (plans.arm == 'own')].iloc[0]
                b = pd.read_csv(out / 'memberships' / base.membership)
                for r in ['validation', 'calibration', 'test', 'transfer_case']:
                    assert set(q[q.role.eq(r)].key) == set(b[b.role.eq(r)].key)
                    checks += 1
                assert set(b[b.role.eq('fit')].key).issubset(q[q.role.eq('fit')].key)
                checks += 1
        # the carried P17 diagnostics are the files recorded when that analysis completed
        recorded = json.loads((self.paths.configs / 'p17_source_heterogeneity_sha256.json').read_text())
        for name in P17_FILES:
            assert sha(self.paths.configs / ('p17_' + name)) == recorded[name]
            checks += 1
        dump(out / 'data_audit.json', dict(status='PASS', checks=checks, records=len(a), plans=len(plans), cell_module_separated=True,
                                           matched_target_validation_calibration_test=True))
        print('DATA_AUDIT_PASS', checks, len(a), flush=True)
