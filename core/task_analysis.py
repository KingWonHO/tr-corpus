# alarm accounting, summaries, integrity checks and derived tables of the task-specific runs
import json

import numpy as np
import pandas as pd

from core import survival, tasks

LEADS = [0, 10, 30, 60, 120, 300]
BUDGETS = [0, .05, .1, .2]
POLICIES = ['task_start', 'archival_first']
GROUP = ['task', 'arm', 'model', 'policy', 'budget', 'seed']
GRID_RULES = ['temperature_rule', 'rate_rule']
SEEDS = [0, 1, 2, 3, 4]


# every (plan tag, model, seed) a complete run contains
def expected_settings(plans):
    expected = set()
    for p in plans.itertuples():
        for model in tasks.LEARNED:
            expected.update((p.tag, model, seed) for seed in SEEDS)
        if p.arm == 'own':
            expected.update((p.tag, m, 0) for m in GRID_RULES)
            if p.task == 'D6_h600':
                expected.add((p.tag, 'native_rate_rule', 0))
    return expected


def threshold(peaks, budget):
    a = np.asarray(peaks, float)
    return survival.calibrate_threshold(a, np.arange(len(a)).astype(str), np.ones(len(a), bool), budget)


def first(t, r, tau, start=None):
    q = np.ones(len(t), bool) if start is None else t >= start
    hit = np.flatnonzero(q & (r >= tau))
    return float(t[hit[0]]) if len(hit) else np.nan


# record-level decisions and calibration rows of one run (path = its complete.json)
def evaluate(path, c, age):
    meta = json.loads(path.read_text())
    tag = {k: meta[k] for k in ['task', 'arm', 'model', 'seed', 'fold', 'horizon', 'highlight_lead']}
    rows = []
    cal = []
    with np.load(path.parent / 'risk.npz') as z:
        keys = z['experiment'].astype(str)
        t = z['t']
        r = z['risk'].astype(float)
        ix = z['indices']
        roles = z['role']
    order = np.lexsort((t, keys))
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    per = {}
    for a, b in zip(starts[:-1], starts[1:]):
        j = order[a:b]
        k = keys[j[0]]
        tt = t[j]
        rr = r[j]
        role = str(roles[j[0]])
        assert np.all(roles[j] == role) and np.all(np.diff(tt) > 0)
        trig = float(c.loc[k, 't_trigger'])
        trig = trig if np.isfinite(trig) else float(tt[0])
        valid = tt >= trig
        if not valid.any():
            continue
        per[k] = dict(t=tt, r=rr, indices=ix[j], role=role, trigger=trig, first=float(tt[valid][0]),
                      peak=float(rr.max()), task_peak=float(rr[valid].max()))
    for policy in POLICIES:
        peakname = 'task_peak' if policy == 'task_start' else 'peak'
        cp = [v[peakname] for v in per.values() if v['role'] == 'calibration']
        assert cp
        for budget in BUDGETS:
            tau = threshold(cp, budget)
            fired = int((np.asarray(cp) >= tau).sum())
            assert fired <= np.floor(budget * len(cp))
            base = dict(tag, policy=policy, budget=budget, tau=tau, n_cal=len(cp))
            cal.append(dict(base, cal_fired=fired, cal_far=fired / len(cp)))
            for k, v in per.items():
                if v['role'] not in ['test', 'transfer_case']:
                    continue
                cr = c.loc[k]
                tt = v['t']
                rr = v['r']
                onset = float(cr['t_onset_L2_1.0'])
                pos = bool(cr.scored_positive)
                neg = bool(cr.scored_negative)
                alarm = first(tt, rr, tau, v['trigger'] if policy == 'task_start' else None)
                pre = bool(np.isfinite(alarm) and alarm < v['trigger'])
                detected = bool(pos and np.isfinite(alarm) and not pre and alarm < onset)
                start = v['first'] if policy == 'task_start' else float(tt[0])
                end = float(tt[-1])
                head = (tt >= start) & (tt <= start + 300)
                row = dict(base, key=k, role=v['role'], dataset=cr.dataset, design=cr.design, scale=cr.scale,
                           confirmed=bool(cr.confirmed), physical_outcome=cr.final_assignment,
                           positive=pos, negative=neg, onset=onset, trigger=v['trigger'], first_admissible=v['first'],
                           monitoring_s=end - start, alarm=alarm, detected=detected, pretrigger=pre,
                           no_alarm=not np.isfinite(alarm),
                           late_alarm=bool(pos and np.isfinite(alarm) and not pre and alarm >= onset),
                           conditional_lead=onset - alarm if detected else np.nan,
                           negative_alarm=int(np.isfinite(alarm)) if neg else np.nan,
                           negative_head300=int(np.any(rr[head] >= tau)) if neg and end - start >= 300 else np.nan)
                aa = age[v['indices']]
                for ell in LEADS:
                    if ell:
                        allowed = (tt >= v['trigger']) & (tt <= onset - ell)
                    else:
                        allowed = (tt >= v['trigger']) & (tt < onset)
                    opportunity = bool(pos and allowed.any())
                    row['opportunity_' + str(ell)] = opportunity
                    row['success_' + str(ell)] = bool(detected and onset - alarm >= ell)
                    # acquisition support (two fresh points in an allowed window), separate from the opportunity
                    row['fresh2_' + str(ell)] = bool(pos and np.any(allowed & (np.sum(aa < 1, axis=1) >= 2)))
                    assert not row['success_' + str(ell)] or opportunity
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(cal)


# test-record counts and rates per task, arm, model, policy, budget and seed
def seed_summary(x):
    out = []
    for tags, g in x[x.role.eq('test')].groupby(GROUP, observed=True):
        assert g.key.is_unique, (tags, len(g), g.key.nunique())
        p = g[g.positive]
        n = g[g.negative]
        row = dict(zip(GROUP, tags))
        row.update(n_positive=len(p), n_negative=len(n), n_all=len(g), positive_designs=p.design.nunique(),
                   negative_designs=n.design.nunique(), detected=int(p.detected.sum()),
                   pretrigger=int(p.pretrigger.sum()), miss=int(p.no_alarm.sum()), late=int(p.late_alarm.sum()),
                   median_lead=float(p.conditional_lead.median()), far=float(n.negative_alarm.mean()),
                   false_alarms=float(n.negative_alarm.sum()), far_head300=float(n.negative_head300.mean()),
                   n_head300=int(n.negative_head300.notna().sum()), negative_hours=float(n.monitoring_s.sum() / 3600))
        for ell in LEADS:
            met = int(p[f'success_{ell}'].sum())
            op = int(p[f'opportunity_{ell}'].sum())
            row.update({f'met_{ell}': met, f'opp_{ell}': op, f'rate_{ell}': met / len(p) if len(p) else np.nan,
                        f'conditional_rate_{ell}': met / op if op else np.nan,
                        f'fresh2_{ell}': int(p[f'fresh2_{ell}'].sum())})
        out.append(row)
    return pd.DataFrame(out)


def seed_mean(s):
    keys = GROUP[:-1]
    metrics = [c for c in s if c not in GROUP]
    a = s.groupby(keys)[metrics].agg(['mean', 'std'])
    a.columns = ['_'.join(c) for c in a.columns]
    return a.reset_index()


# 95 % design-cluster bootstrap interval of a record mean
def ci(v, groups):
    v = np.asarray(v, float)
    groups = np.asarray(groups)
    names = np.unique(groups)
    if len(names) < 2:
        return np.nan, np.nan
    sums = np.array([v[groups == g].sum() for g in names])
    counts = np.array([(groups == g).sum() for g in names])
    rng = np.random.default_rng(20261030)
    ix = rng.integers(0, len(names), (2000, len(names)))
    vals = sums[ix].sum(1) / counts[ix].sum(1)
    return np.quantile(vals, [.025, .975])


# fitting additions against the own-source arm on unchanged target records (seeds averaged per record)
def paired_additions(x):
    s = x[x.role.eq('test') & x.policy.eq('task_start') & x.budget.eq(.1)]
    cols = ['negative_alarm'] + [f'success_{ell}' for ell in LEADS]
    a = s.groupby(['task', 'arm', 'model', 'key', 'design', 'positive', 'negative'])[cols].mean().reset_index()
    out = []
    for (task, model), g in a.groupby(['task', 'model']):
        base = g[g.arm.eq('own')]
        for arm, t in g[~g.arm.eq('own')].groupby('arm'):
            z = base.merge(t, on=['task', 'model', 'key', 'design', 'positive', 'negative'], suffixes=('_base', '_add'),
                           validate='one_to_one')
            assert len(z) == len(base) == len(t)
            for metric in cols:
                v = z[z.negative] if metric == 'negative_alarm' else z[z.positive]
                delta = v[metric + '_add'] - v[metric + '_base']
                lo, hi = ci(delta, v.design)
                out.append(dict(task=task, model=model, addition=arm, metric=metric, n=len(v),
                                designs=v.design.nunique(), base=v[metric + '_base'].mean(),
                                added=v[metric + '_add'].mean(), delta=delta.mean(), low=lo, high=hi))
    return pd.DataFrame(out)


# counting assertion used by the integrity audit
class Checks:
    def __init__(self):
        self.count = 0

    def __call__(self, condition, message):
        assert condition, message
        self.count += 1


# threshold by enumerating attainable operating points, tied maxima included
def independent_threshold(peaks, budget):
    values = np.unique(np.asarray(peaks, dtype=float))
    candidates = np.r_[values[0], np.nextafter(values, np.inf)]
    allowed = int(np.floor(len(peaks) * budget))
    return next(float(t) for t in candidates if np.count_nonzero(np.asarray(peaks) >= t) <= allowed)


# split, censoring and both-class checks of the saved indices of one run
def check_indices(check, identity, m, ix, q, keys, observed, ytime, yevent):
    for role, ids in ix.items():
        check(set(keys[ids]).issubset(set(q[q.role.eq(role)].index)), 'index role ' + str(identity))
        check(np.all(observed[ids]), 'unobserved windows')
        if role in ['fit', 'validation'] and m['model'] != 'native_rate_rule':
            target = (yevent[ids] & (ytime[ids] <= m['horizon'])).astype(int)
            known = np.where(yevent[ids], ytime[ids] > 0, ytime[ids] > m['horizon'])
            check(np.all(known), 'censor/postevent training ' + str(identity))
            check(len(np.unique(target)) == 2, 'both target classes ' + str(identity))


# validation loss recomputed from the saved probabilities of a learned run
def validation_loss(r, ids, keys, ytime, yevent, horizon):
    order = np.argsort(r['indices'])
    loc = np.searchsorted(r['indices'][order], ids)
    covered = np.array_equal(r['indices'][order][loc], ids)
    pred = r['risk'][order][loc].astype(float)
    y = (yevent[ids] & (ytime[ids] <= horizon)).astype(int)
    loss = -(y * np.log(np.clip(pred, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - pred, 1e-15, 1)))
    frame = pd.DataFrame(dict(key=keys[ids], y=y, loss=loss))
    return covered, float(frame.groupby(['y', 'key']).loss.mean().groupby(level=0).mean().mean())


# primary-policy threshold and first-alarm/lead accounting re-derived from the saved risk; returns decisions checked
def check_decisions(check, identity, r, cohort, ca, ar):
    rr = r['risk'].astype(float)
    rkeys = r['experiment'].astype(str)
    order = np.lexsort((r['t'], rkeys))
    kk = rkeys[order]
    cuts = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1], True])
    per = {}
    peaks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        j = order[a:b]
        key = kk[a]
        t = r['t'][j]
        score = rr[j]
        trigger = cohort.loc[key, 't_trigger']
        if not np.isfinite(trigger):
            trigger = t[0]
        valid = t >= trigger
        if not valid.any():
            continue
        if r['role'][j[0]] == 'calibration':
            peaks.append(score[valid].max())
        if r['role'][j[0]] in ['test', 'transfer_case']:
            per[key] = (t[valid], score[valid])
    tau = independent_threshold(peaks, .1)
    check(tau == ca.tau or np.isclose(tau, ca.tau, rtol=1e-14, atol=1e-14), 'threshold ' + str(identity))
    for key, (t, score) in per.items():
        v = ar.loc[key]
        hit = np.flatnonzero(score >= tau)
        ta = float(t[hit[0]]) if len(hit) else np.nan
        check((np.isnan(ta) and np.isnan(v.alarm)) or ta == v.alarm, 'first alarm ' + str(identity))
        for ell in LEADS:
            onset = cohort.loc[key, 't_onset_L2_1.0']
            pos = cohort.loc[key, 'scored_positive']
            good = bool(pos and np.isfinite(ta) and ta < onset and onset - ta >= ell)
            opportunity = bool(pos and np.any(t <= onset - ell if ell else t < onset))
            check(good == v[f'success_{ell}'] and opportunity == v[f'opportunity_{ell}'], 'lead accounting')
    return len(per)


# summary_seed.csv numerators, denominators and FAR recounted from the record table
def check_aggregates(check, s, records):
    ag = records[records.role.eq('test')].groupby(GROUP)
    for v in s.itertuples():
        g = ag.get_group(tuple(getattr(v, k) for k in GROUP))
        p = g[g.positive]
        n = g[g.negative]
        check(g.key.is_unique and len(p) == v.n_positive and len(n) == v.n_negative, 'denominators')
        for ell in LEADS:
            check(int(p[f'success_{ell}'].sum()) == getattr(v, f'met_{ell}'), 'numerator')
            check(int(p[f'opportunity_{ell}'].sum()) == getattr(v, f'opp_{ell}'), 'opportunity denominator')
        check(np.isclose(n.negative_alarm.mean(), v.far, equal_nan=True), 'test FAR')


# membership counts per plan and role
def role_counts(plans, memberships):
    roles = []
    for t in plans.itertuples():
        q = pd.read_csv(memberships / t.membership)
        for role, g in q.groupby('role'):
            roles.append(dict(task=t.task, arm=t.arm, fold=t.fold, role=role, n=len(g),
                              positive=int(g.scored_positive.sum()), negative=int(g.scored_negative.sum()),
                              designs=g.design.nunique()))
    return pd.DataFrame(roles)


# primary-policy test records with the success and opportunity at each task's highlighted lead
def primary_test(x, lead):
    primary = x[x.policy.eq('task_start') & x.budget.eq(.1)]
    test = primary[primary.role.eq('test')].copy()
    test['primary_lead'] = test.task.map(lead)
    test['primary_success'] = [getattr(t, 'success_' + str(lead[t.task])) for t in test.itertuples()]
    test['primary_opportunity'] = [getattr(t, 'opportunity_' + str(lead[t.task])) for t in test.itertuples()]
    return primary, test


def primary_seed(test, lead):
    seed = []
    for tags, g in test.groupby(['task', 'arm', 'model', 'seed']):
        pos = g[g.positive]
        neg = g[g.negative]
        opportunities = pos.primary_opportunity.sum()
        seed.append(dict(zip(['task', 'arm', 'model', 'seed'], tags)) | dict(
            lead_s=lead[tags[0]], n_positive=len(pos), n_negative=len(neg), opportunity=int(opportunities),
            success=float(pos.primary_success.sum()), timely=float(pos.primary_success.mean()),
            conditional=float(pos.primary_success.sum() / opportunities) if opportunities else np.nan,
            far=float(neg.negative_alarm.mean()), far300=float(neg.negative_head300.mean()),
            false_alarms=float(neg.negative_alarm.sum()), median_lead=pos.conditional_lead.median()))
    return pd.DataFrame(seed)


def primary_summary(a):
    metrics = [k for k in a if k not in ['task', 'arm', 'model', 'seed']]
    summary = a.groupby(['task', 'arm', 'model'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(v) for v in summary.columns]
    return summary.reset_index()


# outcome-evidence strata; denominators kept, seeds averaged only afterwards
def evidence_strata(test, lead):
    evidence = []
    for tags, g in test.groupby(['task', 'arm', 'model', 'seed', 'dataset', 'confirmed']):
        pos = g[g.positive]
        neg = g[g.negative]
        evidence.append(dict(zip(['task', 'arm', 'model', 'seed', 'dataset', 'source_supported'], tags)) | dict(
            n_positive=len(pos), n_negative=len(neg), primary_lead=lead[tags[0]],
            success=int(pos.primary_success.sum()), opportunity=int(pos.primary_opportunity.sum()),
            timely=pos.primary_success.mean(), far=neg.negative_alarm.mean()))
    return pd.DataFrame(evidence)


# alarms on physical outcomes without a timing endpoint (no lead time is assigned)
def untimed_alarms(test):
    untimed = test[(~test.positive) & (~test.negative)]
    unt = []
    for tags, g in untimed.groupby(['task', 'arm', 'model', 'seed', 'physical_outcome']):
        unt.append(dict(zip(['task', 'arm', 'model', 'seed', 'outcome'], tags)) | dict(
            records=g.key.nunique(), alarmed=int(g.alarm.notna().sum()), not_alarmed=int(g.alarm.isna().sum())))
    return pd.DataFrame(unt)


# each learned model against each physical rule on identical own-arm records
def paired_model_rule(test, lead):
    cols = ['primary_success', 'primary_opportunity', 'negative_alarm', 'negative_head300', 'detected', 'no_alarm',
            'late_alarm', 'pretrigger']
    own = test[test.arm.eq('own')]
    z = own.groupby(['task', 'model', 'key', 'design', 'positive', 'negative'])[cols].mean().reset_index()
    paired = []
    for task, g in z.groupby('task'):
        for rule, b in g[g.model.str.endswith('_rule')].groupby('model'):
            for model, t in g[~g.model.str.endswith('_rule')].groupby('model'):
                joined = b.merge(t, on=['task', 'key', 'design', 'positive', 'negative'], suffixes=('_rule', '_model'),
                                 validate='one_to_one')
                assert len(joined) == len(t) == len(b)
                for metric, pop in [('primary_success', 'positive'), ('negative_alarm', 'negative')]:
                    v = joined[joined[pop]]
                    delta = v[metric + '_model'] - v[metric + '_rule']
                    lo, hi = ci(delta, v.design)
                    paired.append(dict(task=task, model=model, rule=rule, metric=metric, lead_s=lead[task],
                                       records=len(v), designs=v.design.nunique(),
                                       rule_value=v[metric + '_rule'].mean(), model_value=v[metric + '_model'].mean(),
                                       delta=delta.mean(), low=lo, high=hi))
    return pd.DataFrame(paired)


# why a timed positive lacks a decision opportunity at a lead (the two observable constraints can overlap)
def opportunity_constraints(characteristics):
    c = characteristics[characteristics.timed_positive].copy()
    rows = []
    for ell in [30, 60, 120, 300]:
        for t in c.itertuples():
            deadline = t.onset - ell
            a = deadline < t.first_endpoint
            b = deadline < t.t_trigger
            if a and b:
                cause = 'both'
            elif a:
                cause = 'history_window_only'
            elif b:
                cause = 'trigger_only'
            elif not getattr(t, f'opp_{ell}'):
                cause = 'decision_grid'
            else:
                cause = 'opportunity_present'
            rows.append(dict(key=t.key, dataset=t.dataset, scale=t.scale, lead=ell, cause=cause, deadline=deadline,
                             first_window=t.first_endpoint, trigger=t.t_trigger))
    causes = pd.DataFrame(rows)
    summary = causes.groupby(['dataset', 'scale', 'lead', 'cause']).size().rename('records').reset_index()
    return causes, summary


# task-start against archival-first policy at the 10 % budget, separate from the pooling effects
def policy_sensitivity(sy, lead):
    sy = sy[sy.budget.eq(.1)]
    a = sy[sy.policy.eq('task_start')]
    b = sy[sy.policy.eq('archival_first')]
    z = a.merge(b, on=['task', 'arm', 'model', 'seed'], suffixes=('_task', '_archival'), validate='one_to_one')
    delta = []
    for t in z.itertuples():
        name = 'rate_' + str(lead[t.task])
        delta.append(getattr(t, name + '_task') - getattr(t, name + '_archival'))
    z['primary_delta'] = delta
    z['far_delta'] = z.far_task - z.far_archival
    return z[['task', 'arm', 'model', 'seed', 'primary_delta', 'far_delta', 'pretrigger_archival']]
