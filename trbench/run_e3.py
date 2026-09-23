"""E3 runner: sensor panel -> hazard model -> calibrated alarm -> lead time.

A panel is selected by masking columns of the shared window tensor, never by
rebuilding the windows, so two panels differ in exactly one thing: which
channels the model may see.

Every model family reduces to the same interface -- one risk score per window --
so a rule, a gradient-boosted tree and a tabular foundation model are compared
at the same operating point on the same folds.

    uv run python trbench/run_e3.py --arm smoke
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
import schema as S          # noqa: E402
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import deep as DEEP         # noqa: E402
import foundation as FDN    # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tr-corpus")
SPLITS = ("train", "val", "test")

# Stage 4 case studies.  Each is the set of panels a paired comparison needs;
# the eligible experiments are those expressing all of them (8, 13 and 3).
CASES = {
    "E3-B": ["M1", "M2", "M3", "M4", "M5"],   # #1 -- full external chain
    "E3-C": ["M1", "M2", "M4"],               # #1 + #2 -- gas increment
    "E3-D": ["M1", "M2", "M3", "M6i"],        # #3 -- internal sensor bound
}
HORIZON_S = 60.0            # risk of onset within this many seconds


# ------------------------------------------------------------------- loading
def load_windows(window_dir="W60", splits=SPLITS):
    """Concatenate splits into one pool.

    LOEO needs every experiment of a dataset together, and the fixed split
    scatters #1's eight cells across train/val/test.  The split column is kept
    so a fixed-split arm can still filter on it.
    """
    base = os.path.join(OUT, "windows", window_dir)
    parts, meta = [], []
    for sp in splits:
        p = os.path.join(base, "%s.npz" % sp)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        parts.append({k: z[k] for k in
                      ("X", "mask", "y_time", "y_event", "y_tr",
                       "t_end", "experiment", "trigger")})
        # Native records may extend beyond the frozen registry's observation
        # endpoint. Keep their causal features, but never treat unknown tails
        # as labelled negative follow-up.
        parts[-1]["label_observed"] = (z["label_observed"].astype(bool)
            if "label_observed" in z.files else np.ones(len(parts[-1]["X"]), bool))
        parts[-1]["split"] = np.full(len(z["X"]), sp)
        meta.append(pd.read_csv(os.path.join(base, "%s_experiments.csv" % sp)))
        features = list(z["features"])
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["features"] = features
    d["window_dir"] = window_dir
    return d, pd.concat(meta, ignore_index=True)


# ------------------------------------------------------------------ features
def save_merged(df, path):
    """Write results without discarding models this run did not cover.

    The filename tag distinguishes window builds, not model sets, so running
    `--arm cases --models tabpfn` against an existing file used to replace the
    tree-model rows with 91 TabPFN rows -- which is what happened, and is the
    third time in this project that a partial re-run silently destroyed a
    complete one.  Rows for models the current run produced are replaced; every
    other model's rows are carried forward.
    """
    if os.path.exists(path) and "model" in df.columns:
        old = pd.read_csv(path)
        if "model" in old.columns and set(old.columns) == set(df.columns):
            keep = old[~old["model"].isin(df["model"].unique())]
            if len(keep):
                df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def out_tag(window_dir):
    """Result-file suffix naming the window build a run came from.

    E6 showed the normalisation changes the answer, so results from different
    builds must not land on the same filename -- an earlier run of this suite
    silently overwrote one model's results with another's, and the same mistake
    across normalisation schemes would be far harder to notice.  The original
    per-experiment build keeps the bare name so earlier files stay valid.
    """
    return "" if window_dir == "W60" else "_" + window_dir.replace("W60_", "")


def window_features(X, M, cols):
    """Summarise each window into a fixed vector for the tabular models.

    The availability mask is carried as a feature, not silently dropped: a
    channel that is absent and a channel that reads zero must remain
    distinguishable downstream (plan 1-5).
    """
    X, M = X[:, :, cols], M[:, :, cols].astype(bool)
    Xm = np.where(M, X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last = np.where(M[:, -1, :], X[:, -1, :], np.nan)
        mean = np.nanmean(Xm, axis=1)
        std = np.nanstd(Xm, axis=1)
        mx = np.nanmax(Xm, axis=1)
        mn = np.nanmin(Xm, axis=1)
        slope = last - np.where(M[:, 0, :], X[:, 0, :], np.nan)
        d = np.diff(Xm, axis=1)
        dmax = np.nanmax(d, axis=1)
        dlast = d[:, -1, :]
    avail = M.mean(axis=1)
    F = np.concatenate([last, mean, std, mx, mn, slope, dmax, dlast, avail], axis=1)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# -------------------------------------------------------------------- models
def fit_model(name, Ftr, ytr, wtr, seed=0):
    """-> callable mapping a feature matrix to a risk score.

    Fitting is separated from prediction because every fold scores two sets --
    the held-out experiment and the calibration pool -- and refitting for each
    doubled the cost for no reason.
    """
    if name == "rule":
        # dT/dt proxy: steepest one-step rise of the strongest channel.  No
        # fitting, which is the point -- it is the floor every model must clear.
        n = Ftr.shape[1] // 9
        return lambda F: F[:, 6 * n:7 * n].max(axis=1)

    keep = wtr > 0
    Ftr, ytr = Ftr[keep], ytr[keep]
    if ytr.sum() < 2 or (ytr == 0).sum() < 2:
        return lambda F: np.zeros(len(F))

    if name == "xgboost":
        import xgboost as xgb
        pos = float((ytr == 0).sum()) / max(float(ytr.sum()), 1.0)
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            scale_pos_weight=pos, tree_method="hist", device="cuda",
            eval_metric="logloss", random_state=seed, verbosity=0)
        m.fit(Ftr, ytr)
        return lambda F: m.predict_proba(F)[:, 1]

    if name == "lightgbm":
        # LightGBM's DLL needs the OpenMP runtime, and vcomp140.dll is not in
        # System32 on this machine -- the only copy ships inside scikit-learn.
        # Importing sklearn first loads it into the process; without this,
        # `import lightgbm` succeeds and the first fit dies on a missing
        # dependency, which is why an earlier availability probe passed (it had
        # imported aeon, and so sklearn, beforehand).
        import sklearn  # noqa: F401
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, is_unbalance=True,
            random_state=seed, verbose=-1)
        m.fit(Ftr, ytr)
        return lambda F: m.predict_proba(F)[:, 1]

    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        # TabPFN is an in-context learner with a bounded context; subsample the
        # training pool rather than truncating it, keeping every positive.
        rng = np.random.default_rng(seed)
        cap = 8000
        if len(Ftr) > cap:
            pos = np.flatnonzero(ytr == 1)
            neg = np.flatnonzero(ytr == 0)
            neg = rng.choice(neg, size=max(cap - len(pos), 1), replace=False)
            idx = np.concatenate([pos, neg])
            Ftr, ytr = Ftr[idx], ytr[idx]
        m = TabPFNClassifier(device="cuda", random_state=seed)
        m.fit(Ftr, ytr)
        return lambda F: np.concatenate(
            [m.predict_proba(F[i:i + 4096])[:, 1] for i in range(0, len(F), 4096)])

    if name == "rocket":
        # MultiRocket + logistic head.  The classifier wrappers in aeon sit on a
        # ridge classifier, whose predict_proba is a one-hot of the decision --
        # useless for a threshold sweep -- so the transform is used directly and
        # a calibrated linear head put on top.
        from aeon.transformations.collection.convolution_based import MultiRocket
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        head = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1))
        head.fit(Ftr, ytr)
        return lambda F: head.predict_proba(F)[:, 1]

    raise ValueError("unknown model " + name)


_ROCKET_CACHE = {}


def rocket_features(X, M, cols, seed=0, n_kernels=2500):
    """MultiRocket over the raw window, not the summary statistics.

    The convolution family is meant to see the waveform; feeding it the same
    eight summary numbers as the trees would make it a different-shaped copy of
    the tree entry rather than a separate family.
    """
    from aeon.transformations.collection.convolution_based import MultiRocket
    Z = np.transpose(np.where(M[:, :, cols].astype(bool), X[:, :, cols], 0.0),
                     (0, 2, 1)).astype(np.float32)     # (N, C, L) for aeon
    key = (Z.shape, seed, n_kernels)
    tr = _ROCKET_CACHE.get(key)
    if tr is None:
        tr = MultiRocket(n_kernels=n_kernels, n_jobs=8, random_state=seed)
        tr.fit(Z)
        _ROCKET_CACHE[key] = tr
    return np.nan_to_num(tr.transform(Z), nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------- frozen-feature pool
_POOL_CACHE = {}


def pool_features(d, cols, model, need=None):
    """Backbone representations, computed once and reused across folds.

    A frozen encoder gives a window the same representation whatever fold it
    lands in, so this is hoisted out of the LOEO loop entirely -- otherwise
    E3-A would encode the corpus 99 times over.  `need` lets an arm that only
    touches part of the pool pay for only that part.
    """
    key = (model, d["window_dir"], tuple(cols))
    F = FDN.features(model, d, cols, d["window_dir"], OUT, need)
    _POOL_CACHE[key] = F
    return F


# ------------------------------------------------------------------ one fold
def run_fold(d, cols, model, train_idx, eval_idx, seed=0, horizon=HORIZON_S):
    """Fit on `train_idx`, score every index set in `eval_idx`.

    Three families reach the data differently and this is the only place that
    knows it: sequence models take the window tensor, frozen encoders take a
    slice of the pool-wide representation, everything else takes the window
    summary.  The target, the weights and the returned risk score are identical
    across all three.
    """
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], horizon)
    if model == "rule":
        cols = np.asarray([j for j in cols if d["features"][j] != MW.AGE_CH
                           and not d["features"][j].startswith("age__")])

    if model in DEEP.SEQ_MODELS:
        X, M = d["X"][:, :, cols], d["mask"][:, :, cols]
        predict = DEEP.fit_seq(model, X[train_idx], M[train_idx],
                               y[train_idx], w[train_idx], seed)
        return [predict(X[i], M[i]) for i in eval_idx]

    if model in FDN.FEAT_MODELS:
        need = np.concatenate([np.asarray(train_idx)] +
                              [np.asarray(i) for i in eval_idx])
        F = pool_features(d, cols, model, need)
        predict = FDN.fit_head(F[train_idx], y[train_idx], w[train_idx], seed)
        return [predict(F[i]) for i in eval_idx]

    feat = rocket_features if model == "rocket" else window_features
    if model == "rocket":
        _ROCKET_CACHE.clear()          # kernels are refitted per fold
    Ftr = feat(d["X"][train_idx], d["mask"][train_idx], cols)
    predict = fit_model(model, Ftr, y[train_idx], w[train_idx], seed)
    return [predict(feat(d["X"][i], d["mask"][i], cols)) for i in eval_idx]


def curve_eval(d, meta, risk, test_idx, calib_idx=None, calib_risk=None):
    """Operating curve for one fold, plus its within-budget summary."""
    onset = dict(zip(meta.key, meta.t_onset))
    trig = dict(zip(meta.key, meta.t_trigger))
    if calib_idx is not None:
        times = np.concatenate([d["t_end"][test_idx], d["t_end"][calib_idx]])
        rr = np.concatenate([risk, calib_risk])
        exp = np.concatenate([d["experiment"][test_idx], d["experiment"][calib_idx]])
        neg = np.concatenate([np.zeros(len(test_idx), bool),
                              d["y_tr"][calib_idx] == 0])
    else:
        times, rr, exp = d["t_end"][test_idx], risk, d["experiment"][test_idx]
        neg = d["y_tr"][test_idx] == 0
    c = SV.operating_curve(times, rr, exp, onset, trig, neg)
    return c, SV.curve_summary(c)


def evaluate(d, meta, risk, test_idx, calib_idx, calib_risk, far=0.10):
    """Calibrate on `calib_idx`, report on `test_idx`.

    The two must be disjoint from each other AND from the training fold.  A
    threshold set on in-sample predictions is set on scores the model has
    already fitted, so the negatives look quieter than they are: the first run
    of this smoke test aimed at FAR 0.10 and delivered 0.35 on held-out data.
    """
    onset = dict(zip(meta.key, meta.t_onset))
    trig = dict(zip(meta.key, meta.t_trigger))
    tau = SV.calibrate_threshold(calib_risk, d["experiment"][calib_idx],
                                 d["y_tr"][calib_idx] == 0, far=far)
    return tau, SV.summarize(
        d["t_end"][test_idx], risk, d["experiment"][test_idx],
        onset, trig, d["y_tr"][test_idx] == 0, tau)


# ---------------------------------------------------------------------- arms
def arm_case(d, meta, models, panels, label, seed=0):
    """Case study: leave-one-experiment-out over the cells that express EVERY
    panel in `panels`.

    Requiring the whole set is what makes the comparison paired -- an experiment
    that carries M1 but not M5 cannot contribute to an M5-vs-M1 difference, and
    pooling it in would compare datasets instead of sensors.

    None of these datasets contains a non-runaway cell, so the alarm threshold
    comes from the corpus-wide negative pool, which is mechanical indentation.
    The resulting operating point is therefore an upper bound rather than a
    calibrated one; see the log for why that matters.
    """
    ds = d["experiment"].astype(str)
    elig = None
    for p in panels:
        e = set(MW.panel_experiments(p)["file"].map(
            lambda f: "%s/%s" % (f.split("/")[1], f.split("/")[2][:-8])))
        elig = e if elig is None else (elig & e)
    folds = sorted(elig & set(meta[meta.t_onset.notna()].key))
    print("  %s: %d panels x %d folds x %d models" % (label, len(panels),
                                                      len(folds), len(models)))
    rows, curves = [], []
    for panel in panels:
        cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
        for held in folds:
            te = np.flatnonzero(ds == held)
            tr = np.flatnonzero((ds != held) & np.isin(ds, list(elig)))
            neg = np.flatnonzero((d["y_tr"] == 0) & (ds != held))
            if not len(te) or not len(tr):
                continue
            for m in models:
                rte, rneg = run_fold(d, cols, m, tr, [te, neg], seed)
                c, cs = curve_eval(d, meta, rte, te, neg, rneg)
                c.insert(0, "held", held); c.insert(0, "model", m)
                c.insert(0, "panel", panel); c.insert(0, "arm", label)
                curves.append(c)
                rows.append(dict(arm=label, panel=panel, model=m,
                                 held=held, **cs))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def arm_e3b(d, meta, models, panels, seed=0):
    """E3-B: the M1->M5 chain inside #1, leave-one-experiment-out over 8 cells.

    #1 contains no non-runaway experiment, so its alarm threshold cannot be
    calibrated inside the dataset.  The corpus-wide negative pool is used
    instead, which means the operating point is 'false alarms on indentation
    that did not escalate' -- state it, do not hide it.
    """
    ds = d["experiment"].astype(str)
    in_ds01 = np.char.startswith(ds, "ds01_bak/")
    exps = sorted(set(ds[in_ds01]))
    # #1 has no non-runaway cell, so the threshold comes from the corpus-wide
    # negative pool -- restricted to `val` so it is never the training fold.
    neg_pool = np.flatnonzero((d["y_tr"] == 0) & (d["split"] == "val"))
    rows, curves = [], []

    for panel in panels:
        cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
        for held in exps:
            te = np.flatnonzero(in_ds01 & (ds == held))
            tr = np.flatnonzero(in_ds01 & (ds != held))
            for m in models:
                risk_te, risk_cal = run_fold(d, cols, m, tr, [te, neg_pool], seed)
                c, cs = curve_eval(d, meta, risk_te, te, neg_pool, risk_cal)
                _, s = evaluate(d, meta, risk_te, te, neg_pool, risk_cal)
                c.insert(0, "held", held.split("/")[-1])
                c.insert(0, "model", m); c.insert(0, "panel", panel)
                curves.append(c)
                rows.append(dict(arm="E3-B", panel=panel, model=m,
                                 held=held, **s, **cs))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def arm_e3a(d, meta, models, seed=0, panels=("M1", "M2")):
    """E3-A: M1 vs M2, leave-one-experiment-out over every experiment that can
    express both panels.

    The fixed split cannot carry this comparison -- its test half holds four
    heating events and one overcharge event, so a trigger-stratified claim rests
    on single digits.  LOEO uses all 267 eligible experiments as test folds in
    turn, at the cost of one fit per fold.
    """
    ds = d["experiment"].astype(str)
    elig = {}
    for panel in panels:
        cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
        elig[panel] = set(MW.panel_experiments(panel)["file"].map(
            lambda f: "%s/%s" % (f.split("/")[1], f.split("/")[2][:-8])))
    both = elig[panels[0]]
    for p in panels[1:]:
        both &= elig[p]
    # only experiments that reach runaway can contribute a lead time
    ev = meta[meta.t_onset.notna()].key
    folds = sorted(both & set(ev))

    rows, curves = [], []
    for panel in panels:
        cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
        for held in folds:
            te = np.flatnonzero(ds == held)
            tr = np.flatnonzero((ds != held) & np.isin(ds, list(both)))
            neg = np.flatnonzero((d["y_tr"] == 0) & (ds != held))
            if not len(te) or not len(tr):
                continue
            for m in models:
                risk_te, risk_neg = run_fold(d, cols, m, tr, [te, neg], seed)
                c, cs = curve_eval(d, meta, risk_te, te, neg, risk_neg)
                trig = d["trigger"][te][0]
                c.insert(0, "held", held); c.insert(0, "trigger", trig)
                c.insert(0, "model", m); c.insert(0, "panel", panel)
                curves.append(c)
                rows.append(dict(arm="E3-A", panel=panel, model=m,
                                 held=held, trigger=trig, **cs))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="smoke",
                    choices=["smoke", "e3a", "e3b", "cases"])
    ap.add_argument("--models", default="rule,xgboost")
    ap.add_argument("--panels", default="M1,M2,M3,M4,M5")
    ap.add_argument("--windows", default="W60")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    models = a.models.split(",")
    panels = a.panels.split(",")

    t0 = time.time()
    d, meta = load_windows(a.windows)
    print("pool: %d windows / %d experiments  (%s)"
          % (len(d["X"]), len(meta), a.windows))

    out, cur = [], []
    if a.arm in ("smoke", "e3a"):
        r, c = arm_e3a(d, meta, models, a.seed)
        out.append(r)
        cur.append(c)
    if a.arm in ("smoke", "e3b"):
        r, c = arm_e3b(d, meta, models, panels, a.seed)
        out.append(r)
        cur.append(c)
    if a.arm == "cases":
        for label, ps in CASES.items():
            r, c = arm_case(d, meta, models, ps, label, a.seed)
            out.append(r)
            cur.append(c)
    res = pd.concat(out, ignore_index=True)

    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    tag = out_tag(a.windows)
    path = os.path.join(OUT, "results", "e3_%s%s.csv" % (a.arm, tag))
    cpath = os.path.join(OUT, "results", "curve_%s%s.csv" % (a.arm, tag))
    res = save_merged(res, path)
    save_merged(pd.concat(cur, ignore_index=True), cpath)
    print("\n%.0fs -> %s\n        %s" % (time.time() - t0, path, cpath))
    return res


if __name__ == "__main__":
    main()
