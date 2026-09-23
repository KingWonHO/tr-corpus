"""The foundation tier: Chronos-2 and TimesFM-2.5 as frozen encoders.

The protocol flagged three problems with putting pretrained forecasters in a
detection benchmark (reports/02 section 5), and all three are handled here.

(1) THEY ARE FORECASTERS, NOT CLASSIFIERS.  Both are used as frozen encoders:
    the backbone produces a representation of the window, and a logistic head
    fitted on the same discrete-time hazard target turns it into the one risk
    score per window that every other model in the benchmark produces.  Nothing
    about the backbone is trained, so "zero-shot" means what it usually means
    for these models -- no gradient reaches the pretrained weights.

(2) MOST CANNOT TAKE MULTIVARIATE INPUT WITH MISSING MASKS.  Chronos-2 is
    multivariate-native and takes the panel as one multivariate series.  TimesFM
    is univariate, so each channel is encoded separately and the per-channel
    representations are concatenated; a channel absent for the whole window
    contributes zeros, and the availability fraction is appended so the head can
    tell an absent channel from a flat one.  This asymmetry is a property of the
    tools, not of the panels, and it is reported as such.

(3) EVERYTHING MUST SHARE THE SURVIVAL HEAD.  It does -- see `fit_head`.

The reason this is affordable at all: the backbone is frozen, so a window's
representation does not depend on the fold.  It is computed ONCE for the whole
pool, cached on disk, and the 99 LOEO folds then refit only a logistic
regression.  Recomputing per fold would have multiplied the cost by ~100.

TimesFM exposes no `embed()`, only `forecast()`.  Its inner module returns the
final hidden state from `forward`, so the representation is taken with a
forward hook on the last transformer block during a forecast call -- the same
tensor an `embed()` would have returned, obtained without reimplementing the
tokenizer.
"""
from __future__ import annotations

import os

import numpy as np

FEAT_MODELS = ("chronos2", "timesfm")

CHRONOS_REPO = "amazon/chronos-2"
TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
BATCH = 256


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fill(X, M):
    """Masked slots carry the channel's own window mean, or zero if fully absent.

    A pretrained forecaster has no missingness convention of ours; handing it a
    structural 0 in the middle of a 300 degC series would read as a real
    excursion.  The mean is the least eventful value that keeps the series
    on-scale, and the mask is carried separately into the head regardless.
    """
    Xm = np.where(M, X, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(Xm, axis=1, keepdims=True)
    mean = np.nan_to_num(mean)
    return np.where(M, X, np.broadcast_to(mean, X.shape)).astype(np.float32)


# ------------------------------------------------------------------ Chronos-2
def _chronos_embed(X, M):
    """(N, T, C) -> (N, D).  One multivariate series per window."""
    import torch
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained(CHRONOS_REPO, device_map=_device())
    filled = _fill(X, M)
    out = []
    for i in range(0, len(filled), BATCH):
        chunk = filled[i:i + BATCH]
        # Chronos-2 takes a multivariate window as (T, C).
        inputs = [torch.from_numpy(w) for w in chunk]
        with torch.no_grad():
            emb, _ = pipe.embed(inputs, batch_size=len(inputs))
        # one tensor per series: (tokens, D) or (C, tokens, D); pool everything
        # but the feature axis so the head sees a fixed width whatever the
        # patching decided.
        for e in emb:
            t = e.float()
            while t.dim() > 2:
                t = t.mean(dim=0)
            out.append(t.mean(dim=0).cpu().numpy())
        del emb
    return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------- TimesFM 2.5
PROJ_DIM = 768          # Chronos-2's native width; see _timesfm_embed


def _timesfm_embed(X, M):
    """(N, T, C) -> (N, PROJ_DIM + C).  Univariate encoder run per channel.

    `forecast()` calls the transformer stack twice per batch: once to encode the
    context and once to decode the horizon.  The first call is the window's
    representation, so the hook takes that one and ignores the rest; taking the
    last would hand the head a partially decoded future instead.

    Stacking six channels of 1280 units gives 7,680 features against ~42,000
    training windows, which is a head with more capacity than the one Chronos-2
    gets and would confound backbone quality with head width.  A fixed seeded
    Gaussian projection brings it to Chronos-2's 768.  The projection is drawn
    from a seed and never sees the data, so it cannot leak; it is applied to
    every split identically.
    """
    import torch
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
    model.compile(timesfm.ForecastConfig(
        max_context=X.shape[1], max_horizon=8, normalize_inputs=True,
        per_core_batch_size=BATCH))
    inner = getattr(model, "model", model)
    blocks = getattr(inner, "stacked_xf", None)
    if blocks is None:
        raise RuntimeError("cannot locate TimesFM transformer stack for hooking")

    caught = []

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        caught.append(h.detach())

    handle = blocks[-1].register_forward_hook(hook)
    try:
        filled = _fill(X, M)
        avail = M.astype(np.float32).mean(axis=1)           # (N, C)
        n, _, c = filled.shape
        per_channel = []
        for ch in range(c):
            reps = []
            for i in range(0, n, BATCH):
                hi = min(i + BATCH, n)
                series = [filled[j, :, ch].astype(np.float32) for j in range(i, hi)]
                caught.clear()
                with torch.no_grad():
                    model.forecast(horizon=8, inputs=series)
                if not caught:
                    raise RuntimeError("TimesFM hook captured nothing")
                h = caught[0].float()                       # context pass
                reps.append(h.mean(dim=1).cpu().numpy()[:hi - i])
            per_channel.append(np.concatenate(reps))
        R = np.concatenate(per_channel, axis=1).astype(np.float32)
    finally:
        handle.remove()

    rng = np.random.default_rng(0)
    P = rng.normal(0.0, 1.0 / np.sqrt(PROJ_DIM),
                   size=(R.shape[1], PROJ_DIM)).astype(np.float32)
    # A fully absent channel contributed only its own fill; the head needs to
    # know that, and `avail` is exactly the tree models' feature of the same name.
    return np.concatenate([R @ P, avail], axis=1).astype(np.float32)


# ---------------------------------------------------------------------- cache
def features(model, d, cols, window_dir, out_root, need=None):
    """Frozen representations for the rows an arm will actually touch.

    Grows a persistent cache rather than encoding the pool up front.  The case
    studies run on 24 experiments out of 343, and TimesFM encodes one channel at
    a time, so encoding the whole pool for the five-panel chain would have cost
    hours for rows no fold ever reads.  Rows already encoded are never redone,
    which also means a rerun after a crash resumes instead of restarting.
    """
    tag = "%s_%s_%dch" % (model, window_dir, len(cols))
    path = os.path.join(out_root, "features", tag + ".npz")
    n = len(d["X"])
    F, done = None, np.zeros(n, bool)
    if os.path.exists(path):
        z = np.load(path)
        if len(z["done"]) == n:
            F, done = z["F"], z["done"]
        else:
            print("  cache %s covers %d rows, pool has %d -- rebuilding"
                  % (tag, len(z["done"]), n))

    need = np.arange(n) if need is None else np.unique(np.asarray(need))
    todo = need[~done[need]]
    if len(todo):
        print("  %s: encoding %d of %d windows x %d channels ..."
              % (tag, len(todo), len(need), len(cols)))
        X = d["X"][todo][:, :, cols]
        M = d["mask"][todo][:, :, cols].astype(bool)
        E = _chronos_embed(X, M) if model == "chronos2" else _timesfm_embed(X, M)
        if F is None or F.shape[1] != E.shape[1]:
            F = np.zeros((n, E.shape[1]), np.float32)
        F[todo] = E
        done[todo] = True
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, F=F, done=done)
        print("  -> %s  %s  (%d/%d rows cached)"
              % (tag, tuple(F.shape), int(done.sum()), n))
    return F


def fit_head(Ftr, ytr, wtr, seed=0):
    """The shared survival head: logistic regression on frozen features.

    Deliberately the smallest head that can express the target.  Anything larger
    starts training a model on top of the representation, and the question here
    is what the pretrained representation already contains.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    keep = wtr > 0
    Ftr, ytr, wtr = Ftr[keep], ytr[keep], wtr[keep]
    if ytr.sum() < 2 or (ytr == 0).sum() < 2:
        return lambda F: np.zeros(len(F))
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=seed))
    pipe.fit(np.nan_to_num(Ftr), ytr, logisticregression__sample_weight=wtr)
    return lambda F: pipe.predict_proba(np.nan_to_num(F))[:, 1]
