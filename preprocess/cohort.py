# cohort step: record table of the six physical sources and the design-wise and record-wise outer folds
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd

from core import cohort as C
from core import windows
from core.schema import DATASET_LABEL as DS
from preprocess import manifest


def stable(s):
    return hashlib.sha256(('JES-review-20261029/' + str(s)).encode()).hexdigest()


def describe(plan):
    return ('pool %d windows / %d exp | cal-neg %d win / %d exp | check-neg %d win / %d exp | %d LOEO folds'
            % (len(plan.pool), len(set(plan.ds[plan.pool])), len(plan.cal_neg), len(set(plan.ds[plan.cal_neg])),
               len(plan.test_neg), len(set(plan.ds[plan.test_neg])), len(plan.folds)))


def cohort_table(reg, meta, plan, adjudication, reconciliation):
    reg = reg[reg.dataset_id.isin(DS)].copy()
    reg['key'] = reg.dataset_id + '/' + reg.experiment_id
    reg['dataset'] = reg.dataset_id.map(DS)
    reg['design'] = reg.apply(C.cell_model, axis=1)
    adj = pd.read_csv(adjudication)
    reg = reg.merge(adj.drop(columns=['physical_outcome'], errors='ignore'), on=['dataset', 'experiment_id'], how='left')
    ev = pd.read_csv(reconciliation)
    reg = reg.merge(ev[['dataset', 'experiment_id', 'source_verdict']], on=['dataset', 'experiment_id'], how='left')
    reg['confirmed'] = reg.source_verdict.isin(['TR', 'non-TR'])
    reg['paired'] = reg.key.isin(plan.elig)
    reg['scored_positive'] = reg.final_assignment.eq('positive') & reg['t_onset_L2_1.0'].notna()
    reg['scored_negative'] = reg.final_assignment.eq('negative') & reg['t_onset_L2_1.0'].isna()
    mm = meta.drop_duplicates('key').set_index('key')
    for col in ['t_trigger', 't_onset', 't_first', 't_last', 'n_windows']:
        if col in mm:
            reg['window_' + col] = reg.key.map(mm[col])
    return reg


# five outer folds; negative-bearing designs are spread first, then by size; calibration is the next fold
def make_splits(c):
    c = c[c.paired].copy()
    g = c.groupby('design').agg(n=('key', 'size'), neg=('scored_negative', 'sum'), pos=('scored_positive', 'sum'))
    g['hash'] = [stable(k) for k in g.index]
    g = g.sort_values(['neg', 'n', 'hash'], ascending=[False, False, True])
    folds = {}
    total = np.zeros(5, int)
    negatives = np.zeros(5, int)
    for design, row in g.iterrows():
        if row.neg:
            j = min(range(5), key=lambda k: (int(negatives[k]), int(total[k]), k))
        else:
            j = min(range(5), key=lambda k: (int(total[k]), k))
        folds[design] = j
        total[j] += int(row.n)
        negatives[j] += int(row.neg)
    c['group_fold'] = c.design.map(folds)
    # record-wise comparison folds, stratified by dataset and outcome without predictions
    c['record_fold'] = -1
    for _, sub in c.groupby(['dataset', 'final_assignment'], dropna=False):
        ix = sorted(sub.index, key=lambda k: stable(c.loc[k, 'key']))
        for j, i in enumerate(ix):
            c.loc[i, 'record_fold'] = j % 5
    return c


def split_counts(s):
    counts = []
    for mode in ['group', 'record']:
        for f in range(5):
            fold = s[f'{mode}_fold']
            for role, sel in [('test', fold == f), ('calibration', fold == (f + 1) % 5), ('train', ~fold.isin([f, (f + 1) % 5]))]:
                ss = s[sel]
                counts.append(dict(mode=mode, fold=f, role=role, n=len(ss), designs=ss.design.nunique(),
                                   positive=int(ss.scored_positive.sum()), negative=int(ss.scored_negative.sum()),
                                   confirmed_positive=int((ss.scored_positive & ss.confirmed).sum()),
                                   confirmed_negative=int((ss.scored_negative & ss.confirmed).sum())))
    return pd.DataFrame(counts)


# preprocessing step 'cohort'
class CohortStep:
    def __init__(self, paths):
        self.paths = paths
        self.out = paths.out('preprocess/cohort')

    def run(self):
        p = self.paths
        experiments = p.registry / 'experiments.csv'
        adjudication = p.registry / 'outcome_adjudication.csv'
        reconciliation = p.configs / 'outcome_reconciliation_cells.csv'
        d, meta = windows.load_windows(p.data)
        reg = pd.read_csv(experiments)
        plan = C.Plan(d, meta, reg, ['M1', 'M2'], str(adjudication))
        c = cohort_table(reg, meta, plan, adjudication, reconciliation)
        c.to_csv(self.out / 'cohort.csv', index=False)
        s = make_splits(c)
        s.to_csv(self.out / 'split_assignment.csv', index=False)
        counts = split_counts(s)
        counts.to_csv(self.out / 'split_counts.csv', index=False)
        files = [experiments, adjudication, reconciliation] + sorted(p.windows.glob('*'))
        info = {'status': 'review-response retrospective analysis; no prospective validation',
                'python': sys.version, 'platform': platform.platform(),
                'input_sha256': manifest.hashes(p, files),
                'n_windows': len(d['X']), 'features': d['features'], 'plan': describe(plan)}
        (self.out / 'input_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        print(describe(plan))
        print(c.groupby(['dataset', 'final_assignment']).size().to_string())
        print(counts.to_string(index=False))
