"""E11: does anything learned on the core datasets survive on #12?

#12 is the worst case the plan set aside for exactly this -- an accelerating-rate
calorimeter, temperature only, event-driven sampling that is >99% interpolated
on the 1 Hz grid, and two sodium-ion cells that are out of family for a
lithium-ion corpus.  It never enters training anywhere else in this benchmark.

Two arms, and the distance between them is the answer:

  zeroshot   fit on the core datasets, predict on #12
  indomain   leave-one-experiment-out inside #12 itself

If both fail, the task is hard on this data.  If only zero-shot fails, what
failed is transfer.

Both arms are scored the same way, and the operating point is not the primary
output.  A first run summarised the in-domain arm by averaging 45 single-
experiment curves and reported "0/45 detected", which hid the actual behaviour:
that model separates runaway from non-runaway cells at AUC 0.85 but fires only
inside the last minute.  So each arm now yields one out-of-fold risk per window
over all 66 experiments, and three things are read off it -- experiment-level
discrimination, when the risk actually rises relative to onset, and only then
the calibrated operating point.

#12 is also the first target domain in this benchmark that carries its own
negatives -- 21 of its 66 experiments never reach runaway -- so the alarm
threshold can be calibrated on source negatives, as a deployment would have to,
and separately on target negatives, as an oracle would.  The gap between those
two operating points is a direct reading of the domain shift, and it is the one
measurement no other arm here could make.

    uv run python trbench/run_e11.py --models rule,xgboost,lightgbm
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import run_e3 as R          # noqa: E402
import deep as DEEP         # noqa: E402
import foundation as FDN    # noqa: E402

OUT = R.OUT
TARGET = "ds12_arc"
PANEL = "M1"                # #12 carries surface temperature and nothing else


def _fit_predict(F, y, w, model, tr_idx, eval_idx, seed=0, d=None, cols=None):
    """As run_fold, but over features computed once for the whole pool.

    The window summary does not depend on the fold, and the in-domain arm refits
    66 times; recomputing it per fold was the entire cost of this experiment.
    The same holds for a frozen encoder's representation, so `F` carries that
    too.  Sequence models are the exception -- they read the window tensor, so
    they take `d`/`cols` and ignore `F`.
    """
    if model in DEEP.SEQ_MODELS:
        X, M = d["X"][:, :, cols], d["mask"][:, :, cols]
        predict = DEEP.fit_seq(model, X[tr_idx], M[tr_idx],
                               y[tr_idx], w[tr_idx], seed)
        return [predict(X[i], M[i]) for i in eval_idx]
    if model in FDN.FEAT_MODELS:
        predict = FDN.fit_head(F[tr_idx], y[tr_idx], w[tr_idx], seed)
        return [predict(F[i]) for i in eval_idx]
    predict = R.fit_model(model, F[tr_idx], y[tr_idx], w[tr_idx], seed)
    return [predict(F[i]) for i in eval_idx]


def model_features(d, cols, model, summary):
    """Which matrix a model family reads.  None means "read the windows"."""
    if model in DEEP.SEQ_MODELS:
        return None
    if model in FDN.FEAT_MODELS:
        return R.pool_features(d, cols, model)
    return summary


def _auc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def _diagnose(d, onset, idx, risk, tag, model):
    """Discrimination and timing, before any threshold is chosen.

    Two AUCs, and the distance between them is the whole point.  `auc_experiment`
    scores each cell by its peak risk over the entire record, so a detector that
    merely reacts to runaway once it is under way scores near 1 -- the dT/dt rule
    reaches 0.979 that way.  `auc_preonset` scores runaway cells by their peak
    over windows strictly BEFORE onset, which is the only evidence an alarm could
    have acted on, and non-runaway cells over their whole record.  A method that
    anticipates keeps its AUC; one that only reacts collapses toward chance.
    """
    e = np.asarray(d["experiment"])[idx]
    t = d["t_end"][idx]
    peak = pd.Series(risk).groupby(e).max()
    is_tr = np.array([np.isfinite(onset.get(k, np.nan)) for k in peak.index])
    pre = pd.Series(np.where([np.isfinite(onset.get(k, np.nan)) for k in e],
                             t < np.array([onset.get(k, np.inf) for k in e]),
                             True))
    peak_pre = pd.Series(np.where(pre, risk, -np.inf)).groupby(e).max()
    row = dict(model=model, arm=tag, n_tr=int(is_tr.sum()),
               n_nontr=int((~is_tr).sum()),
               auc_experiment=_auc(peak.values[is_tr], peak.values[~is_tr]),
               auc_preonset=_auc(peak_pre.values[is_tr], peak_pre.values[~is_tr]),
               peak_tr_median=float(np.median(peak.values[is_tr])),
               peak_nontr_median=float(np.median(peak.values[~is_tr])))
    ttl = d["y_time"][idx]
    for lo, hi, name in [(0, 60, "ttl_0_60"), (60, 300, "ttl_60_300"),
                         (300, 1800, "ttl_300_1800"), (1800, np.inf, "ttl_1800_up")]:
        s = np.isfinite(ttl) & (ttl >= lo) & (ttl < hi)
        row[name] = float(np.median(risk[s])) if s.any() else np.nan
    return row


def _at_recall(curve, target=0.5):
    """What the cheapest alarm reaching `target` recall actually costs.

    Every learned arm here returns NaN at the 10% budget, which says an alarm was
    unaffordable but not by how much.  This reads the curve the other way round:
    fix the recall, report the false-alarm rate it demands.
    """
    ok = curve[(curve["recall"] >= target) & np.isfinite(curve["lead_median"])]
    if not len(ok):
        return dict(far_at_recall=np.nan, lead_at_recall=np.nan, tau_at_recall=np.nan)
    cost = ok["far_external"].fillna(ok["far_pretrigger"])
    b = ok.loc[cost.idxmin()]
    return dict(far_at_recall=float(cost.min()),
                lead_at_recall=float(b["lead_median"]),
                tau_at_recall=float(b["tau"]))


def _point(d, onset, trig, idx, risk, neg_idx=None, neg_risk=None):
    """Curve over held-out experiments, priced against a negative pool.

    `neg_idx=None` means the pool is already inside `idx` -- which is the case
    whenever #12 is scored whole, since it brings its own 21 non-runaway cells.
    Appending them a second time would duplicate every window of the evaluation
    set, so the two cases are kept apart here rather than at each call site.
    """
    if neg_idx is None:
        t, r, e = d["t_end"][idx], risk, d["experiment"][idx]
        neg = d["y_tr"][idx] == 0
    else:
        t = np.concatenate([d["t_end"][idx], d["t_end"][neg_idx]])
        r = np.concatenate([risk, neg_risk])
        e = np.concatenate([d["experiment"][idx], d["experiment"][neg_idx]])
        neg = np.concatenate([np.zeros(len(idx), bool), d["y_tr"][neg_idx] == 0])
    c = SV.operating_curve(t, r, e, onset, trig, neg)
    return c, dict(SV.curve_summary(c), **_at_recall(c))


def _rank_align(F, src, tgt):
    """Map every feature to its within-domain quantile, separately per domain.

    Baseline-only z-scoring is leak-free but it does NOT put the datasets on a
    common scale: the baseline of an ARC hold is nearly flat, so its standard
    deviation is tiny and the resulting z-scores are enormous.  Measured across
    the corpus, the median normalised temperature runs from 1.2 (#9) to 3051
    (#1) to 1215 (#12) -- three orders of magnitude.  A model fitted on one of
    those scales cannot be expected to apply on another, which would explain the
    inverted zero-shot risk without any physics being involved.

    This is a DIAGNOSTIC, not an operating result: the quantiles are taken over
    each domain as a whole, including windows after onset, so it is not causal
    and could not run online.  Its only job is to say whether the inversion
    survives once the scale mismatch is removed.
    """
    G = np.array(F, copy=True)
    for grp in (src, tgt):
        G[grp] = np.column_stack([
            pd.Series(F[grp][:, j]).rank(pct=True).values
            for j in range(F.shape[1])])
    return G.astype(np.float32)


def _oof_indomain(d, F, y, w, model, is_t, seed=0, cols=None):
    """Leave-one-experiment-out over ALL 66 experiments, negatives included.

    Holding out only the runaway cells would leave the negatives permanently in
    training, so their scores would be in-sample and the discrimination would be
    flattered.  Every experiment is held out in turn, and the returned risk is
    out-of-fold everywhere.
    """
    ds = np.asarray(d["experiment"]).astype(str)
    idx = np.flatnonzero(is_t)
    risk = np.full(len(idx), np.nan)
    pos = {k: i for i, k in enumerate(idx)}
    for held in sorted(set(ds[is_t])):
        te = np.flatnonzero(ds == held)
        tr = np.flatnonzero(is_t & (ds != held))
        r, = _fit_predict(F, y, w, model, tr, [te], seed, d, cols)
        risk[[pos[i] for i in te]] = r
    return idx, risk


TAG = ""      # set from --windows in main(); part of the oof filename


def run(d, meta, F, y, w, models, seed=0, cols=None):
    ds = d["experiment"].astype(str)
    is_t = np.char.startswith(ds, TARGET + "/")
    onset = dict(zip(meta.key, meta.t_onset))
    trig = dict(zip(meta.key, meta.t_trigger))

    tgt = np.flatnonzero(is_t)
    tgt_neg = np.flatnonzero(is_t & (d["y_tr"] == 0))          # 21 ARC non-TR
    src = np.flatnonzero(~is_t)
    src_neg = np.flatnonzero(~is_t & (d["y_tr"] == 0))         # indentation
    tgt_ev = [k for k in sorted(set(ds[is_t])) if np.isfinite(onset.get(k, np.nan))]
    print("  target %d exp / %d windows (%d reach TR, %d do not)"
          % (len(set(ds[is_t])), len(tgt), len(tgt_ev), len(set(ds[tgt_neg]))))
    print("  source %d exp / %d windows (%d negative exp)"
          % (len(set(ds[src])), len(src), len(set(ds[src_neg]))))

    rows, curves, diag = [], [], []
    summary = F
    for m in models:
        t0 = time.time()
        F = model_features(d, cols, m, summary)
        # -- zero-shot: one fit on the core corpus, applied whole to #12 ------
        r_t, r_s = _fit_predict(F, y, w, m, src, [tgt, src_neg], seed, d, cols)
        diag.append(_diagnose(d, onset, tgt, r_t, "zeroshot", m))
        # the same scores, priced against a source pool and a target pool
        for tag, ni, nr in [("zeroshot/src-calib", src_neg, r_s),
                            ("zeroshot/tgt-calib", None, None)]:
            c, o = _point(d, onset, trig, tgt, r_t, ni, nr)
            c.insert(0, "model", m)
            c.insert(0, "arm", tag)
            curves.append(c)
            rows.append(dict(model=m, arm=tag, n_eval_exp=len(tgt_ev), **o))
        print("    %-9s zero-shot AUC %.3f / pre-onset %.3f (%.0fs)"
              % (m, diag[-1]["auc_experiment"], diag[-1]["auc_preonset"],
                 time.time() - t0))

        # -- zero-shot with the scale mismatch removed (diagnostic only) -----
        G = _rank_align(F, src, tgt) if F is not None else None
        if G is None:
            g_t = g_s = None          # sequence models read windows, not features
        else:
            g_t, g_s = _fit_predict(G, y, w, m, src, [tgt, src_neg], seed, d, cols)
        if g_t is not None:
            diag.append(_diagnose(d, onset, tgt, g_t, "zeroshot-rankalign", m))
            c, o = _point(d, onset, trig, tgt, g_t, src_neg, g_s)
            c.insert(0, "model", m)
            c.insert(0, "arm", "zeroshot-rankalign")
            curves.append(c)
            rows.append(dict(model=m, arm="zeroshot-rankalign",
                             n_eval_exp=len(tgt_ev), **o))
            print("    %-9s rank-aligned AUC %.3f / pre-onset %.3f"
                  % (m, diag[-1]["auc_experiment"], diag[-1]["auc_preonset"]))
        else:
            print("    %-9s rank-align skipped (sequence model reads windows)" % m)

        # -- in-domain: LOEO over #12, one curve on the out-of-fold scores ----
        idx, r_o = _oof_indomain(d, F, y, w, m, is_t, seed, cols)
        diag.append(_diagnose(d, onset, idx, r_o, "indomain", m))
        c, o = _point(d, onset, trig, idx, r_o)
        c.insert(0, "model", m)
        c.insert(0, "arm", "indomain")
        curves.append(c)
        rows.append(dict(model=m, arm="indomain", n_eval_exp=len(tgt_ev), **o))
        np.save(os.path.join(OUT, "results", "e11_oof_%s%s.npy" % (m, TAG)), r_o)
        print("    %-9s in-domain AUC %.3f / pre-onset %.3f, lead@r50 %ss (%.0fs)"
              % (m, diag[-1]["auc_experiment"], diag[-1]["auc_preonset"],
                 o["lead_at_recall"], time.time() - t0))
    return (pd.DataFrame(rows), pd.concat(curves, ignore_index=True),
            pd.DataFrame(diag))


def by_group(d, meta, reg, F, y, w, model, seed=0, cols=None):
    """Zero-shot broken out by cell format, chemistry and state of charge.

    #12 spans 18650/21700/4680 and includes two SODIUM-ION cells that are out of
    family for a lithium-ion training corpus.  Pooling them would hide whichever
    way that goes, so the groups are reported with their sizes attached -- the
    smallest are two experiments and are read as observations, not estimates.
    """
    ds = d["experiment"].astype(str)
    is_t = np.char.startswith(ds, TARGET + "/")
    tgt = np.flatnonzero(is_t)
    src = np.flatnonzero(~is_t)
    src_neg = np.flatnonzero(~is_t & (d["y_tr"] == 0))
    F = model_features(d, cols, model, F)
    r_t, r_s = _fit_predict(F, y, w, model, src, [tgt, src_neg], seed, d, cols)

    onset = dict(zip(meta.key, meta.t_onset))
    trig = dict(zip(meta.key, meta.t_trigger))
    e = reg[reg.dataset_id == TARGET].copy()
    e["key"] = e.dataset_id + "/" + e.experiment_id
    rows = []
    for gcol in ["form_factor", "chemistry", "soc_pct"]:
        if gcol not in e.columns:
            continue
        for g, gg in e.groupby(gcol):
            keys = set(gg.key)
            n_tr = sum(1 for k in keys if np.isfinite(onset.get(k, np.nan)))
            if not n_tr:
                continue
            sel = np.flatnonzero([k in keys for k in d["experiment"][tgt]])
            _, o = _point(d, onset, trig, tgt[sel], r_t[sel], src_neg, r_s)
            o["auc_experiment"] = _diagnose(
                d, onset, tgt[sel], r_t[sel], "zeroshot", model)["auc_experiment"]
            rows.append(dict(group=gcol, value=g, model=model,
                             n_exp=len(gg), n_tr=n_tr, **o))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rule,xgboost,lightgbm")
    ap.add_argument("--windows", default="W60")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    tag = R.out_tag(a.windows)

    d, meta = R.load_windows(a.windows,
                             splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))
    cols = np.flatnonzero(MW.panel_mask(d["features"], PANEL))
    print("pool: %d windows / %d experiments | panel %s = %d channels"
          % (len(d["X"]), len(meta), PANEL, len(cols)))

    F = R.window_features(d["X"], d["mask"], cols)
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], R.HORIZON_S)
    print("features %s computed in %.0fs" % (str(F.shape), time.time() - t0))

    globals()["TAG"] = tag
    models = a.models.split(",")
    res, curves, diag = run(d, meta, F, y, w, models, a.seed, cols)
    grp = by_group(d, meta, reg, F, y, w, models[-1], a.seed, cols)

    R.save_merged(res, os.path.join(OUT, "results", "e11%s.csv" % tag))
    R.save_merged(diag, os.path.join(OUT, "results", "e11_diagnostic%s.csv" % tag))
    R.save_merged(grp, os.path.join(OUT, "results", "e11_groups%s.csv" % tag))
    R.save_merged(curves, os.path.join(OUT, "results", "curve_e11%s.csv" % tag))
    print("\n" + res.to_string(index=False))
    print("\n%.0fs -> results/e11.csv, e11_diagnostic.csv, e11_groups.csv, curve_e11.csv"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
