# roles step: fit, validation, calibration and test records of every outer fold of the common cohort
import hashlib
import itertools
import json

import pandas as pd
import torch

from core.fitting import write_json
from preprocess import manifest

ROLES = ['fit', 'validation', 'calibration', 'test']


def stable(x):
    return hashlib.sha256(('selected20261030/' + str(x)).encode()).hexdigest()


# design-split roles with validation designs chosen from outcome counts only, and the record split with
# fit counts matched to the design split
def role_table(s, fold):
    pool = s[~s.group_fold.isin([fold, (fold + 1) % 5]) & s.final_assignment.ne('indeterminate')]
    rec = s[~s.record_fold.isin([fold, (fold + 1) % 5]) & s.final_assignment.ne('indeterminate')]
    cap = (int(rec.scored_positive.sum()) - 3, int(rec.scored_negative.sum()) - 8)
    options = []
    groups = sorted(pool.design.unique())
    for k in range(1, min(5, len(groups))):
        for combo in itertools.combinations(groups, k):
            va = pool[pool.design.isin(combo)]
            tr = pool[~pool.design.isin(combo)]
            p = int(tr.scored_positive.sum())
            n = int(tr.scored_negative.sum())
            if not (8 <= p <= cap[0] and 20 <= n <= cap[1]):
                continue
            if va.scored_positive.sum() < 3 or va.scored_negative.sum() < 8:
                continue
            if min((va.scored_positive & va.confirmed).sum(), (va.scored_negative & va.confirmed).sum()) < 1:
                continue
            if min((tr.scored_positive & tr.confirmed).sum(), (tr.scored_negative & tr.confirmed).sum()) < 2:
                continue
            score = abs(len(va) / len(pool) - .25) + .2 * abs(va.scored_positive.mean() - pool.scored_positive.mean())
            options.append((score, stable(combo), combo))
    if not options:
        raise ValueError(f'No independent validation design split for fold {fold}')
    val_designs = min(options)[2]
    g = s.copy()
    g['role'] = 'excluded'
    g.loc[g.group_fold.eq(fold), 'role'] = 'test'
    g.loc[g.group_fold.eq((fold + 1) % 5) & g.scored_negative, 'role'] = 'calibration'
    g.loc[pool.index, 'role'] = 'fit'
    g.loc[pool[pool.design.isin(val_designs)].index, 'role'] = 'validation'
    fit = g[g.role.eq('fit')]
    rr = s.copy()
    rr['role'] = 'excluded'
    rr.loc[rr.record_fold.eq(fold), 'role'] = 'test'
    rr.loc[rr.record_fold.eq((fold + 1) % 5) & rr.scored_negative, 'role'] = 'calibration'
    rr.loc[rec.index, 'role'] = 'validation'
    for col in ['scored_positive', 'scored_negative']:
        n = int(fit[col].sum())
        idx = sorted(rec[rec[col]].index, key=lambda i: stable(rec.loc[i, 'key']))
        assert len(idx) > n
        rr.loc[idx[:n], 'role'] = 'fit'
    return g, rr


# preprocessing step 'roles'; existing role files must stay unchanged
class RolesStep:
    def __init__(self, paths):
        self.paths = paths
        self.out = paths.out('preprocess/roles')

    def run(self):
        p = self.paths
        cohort_dir = p.need('preprocess/cohort')
        s = pd.read_csv(cohort_dir / 'split_assignment.csv')
        counts = []
        for f in range(5):
            for mode, z in zip(['group', 'record'], role_table(s, f)):
                path = self.out / f'roles_{mode}_f{f}.csv'
                if path.exists():
                    pd.testing.assert_frame_equal(pd.read_csv(path), z, check_dtype=False)
                else:
                    z.to_csv(path, index=False)
                for role, sub in z.groupby('role'):
                    counts.append(dict(mode=mode, fold=f, role=role, n=len(sub), designs=sub.design.nunique(),
                                       positive=int(sub.scored_positive.sum()), negative=int(sub.scored_negative.sum()),
                                       confirmed_positive=int((sub.scored_positive & sub.confirmed).sum()),
                                       confirmed_negative=int((sub.scored_negative & sub.confirmed).sum())))
                for a, b in itertools.combinations(ROLES, 2):
                    assert set(z[z.role.eq(a)].key).isdisjoint(z[z.role.eq(b)].key)
                    if mode == 'group':
                        assert set(z[z.role.eq(a)].design).isdisjoint(z[z.role.eq(b)].design)
        pd.DataFrame(counts).to_csv(self.out / 'role_counts.csv', index=False)
        protocol = p.configs / 'common_cohort_protocol.md'
        files = [protocol, cohort_dir / 'cohort.csv', cohort_dir / 'split_assignment.csv'] + sorted(p.windows.glob('*'))
        inputs = manifest.hashes(p, files)
        path = self.out / 'input_manifest.json'
        if path.exists():
            assert json.loads(path.read_text())['sha256'] == inputs, 'Frozen inputs changed'
        else:
            write_json(path, dict(sha256=inputs, protocol='retrospective; four disjoint roles; no test-based selection',
                                  torch=torch.__version__))
        (self.out / 'protocol.md').write_bytes(protocol.read_bytes())
        print(pd.read_csv(self.out / 'role_counts.csv').to_string(index=False))
