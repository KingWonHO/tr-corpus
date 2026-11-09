# native-interval temperature-rate rule on the original ARC observations
import numpy as np

from core import arc


# record key -> (time from first observation, surface temperature)
def native_series(raw_dir):
    out = {}
    for raw in arc.load(raw_dir):
        t, x = arc.clean_series(*raw.series["T_surface_mid"])
        out[raw.dataset_id + "/" + raw.experiment_id] = (t - t[0], x)
    return out


# delivered (arrival time, value) pairs under one perturbation
def perturb(t, x, kind, rng):
    if kind == "clean":
        return t, x
    if kind.startswith("noise_"):
        return t, x + rng.normal(0.0, float(kind.split("_")[1]), len(x))
    if kind.startswith("offset_"):
        return t, x + float(kind.split("_")[1])
    if kind.startswith("hold_"):
        k = float(kind.split("_")[1])
        tick = np.ceil(t / k - 1e-9) * k
        last = np.r_[tick[1:] != tick[:-1], True]
        return tick[last], x[last]
    raise ValueError(kind)


# maximum native rate arriving inside each causal 60 s window, zero if none
def window_risk(t, x, end):
    rate = np.diff(x) / np.diff(t)
    u = t[1:]
    lo = np.searchsorted(u, end - 59, side="left")
    hi = np.searchsorted(u, end, side="right")
    rr = np.zeros(len(end))
    for j in np.flatnonzero(hi > lo):
        rr[j] = rate[lo[j]:hi[j]].max()
    return rr
