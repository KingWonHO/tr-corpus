# causal 60 s windows from the Zenodo window build, and their tabular summaries
import os
import warnings

import numpy as np
import pandas as pd

from core import schema as S

SPLITS = ("train", "val", "test")
AGE_CH = "obs_age"
LEAD_HORIZON_S = 600.0


# all splits of one window build concatenated into one pool
def load_windows(data_dir, window_dir="W60_native", splits=SPLITS):
    base = os.path.join(str(data_dir), "windows", window_dir)
    parts, meta = [], []
    for sp in splits:
        p = os.path.join(base, "%s.npz" % sp)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        parts.append({k: z[k] for k in ("X", "mask", "y_time", "y_event", "y_tr", "t_end", "experiment", "trigger")})
        # tails beyond the registry endpoint keep their features but never count as follow-up
        parts[-1]["label_observed"] = (z["label_observed"].astype(bool) if "label_observed" in z.files
                                       else np.ones(len(parts[-1]["X"]), bool))
        parts[-1]["split"] = np.full(len(z["X"]), sp)
        meta.append(pd.read_csv(os.path.join(base, "%s_experiments.csv" % sp)))
        features = list(z["features"])
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    d["features"] = features
    d["window_dir"] = window_dir
    return d, pd.concat(meta, ignore_index=True)


# nine summaries per channel: last, mean, std, max, min, slope, max step, last step, availability
def window_features(X, M, cols):
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


# feature columns of one sensor panel, with the matching observation-age channels
def panel_mask(features, panel):
    sensors = set(S.SENSOR_PANELS[panel]) - set(S.TRIGGER_SIDE)
    allowed = sensors | {"age__" + ch for ch in sensors} | {AGE_CH}
    return np.array([f in allowed for f in features], bool)


# registry rows that can express one panel
def panel_experiments(registry, panel):
    ok = registry["channels_present"].apply(lambda s: panel in S.panel_expressible(s.split(",")))
    return registry[ok]
