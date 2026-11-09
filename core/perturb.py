# synthetic sensor perturbations of the 1 s task inputs and the task-start evaluation at the 10 % budget
import numpy as np
import pandas as pd

from core import survival as SV

PERTURBATIONS = ["clean", "noise_1", "noise_2", "offset_+2", "offset_-2", "hold_5", "hold_10"]
RULES = ["temperature_rule", "rate_rule"]


# each record's 1 s series rebuilt from its overlapping causal windows
def series_from_windows(ix, d):
    keys = d["experiment"][ix]
    out = {}
    for k in np.unique(keys):
        w = ix[keys == k]
        w = w[np.argsort(d["t_end"][w])]
        t0 = int(round(d["t_end"][w[0]])) - 59
        t1 = int(round(d["t_end"][w[-1]]))
        n = t1 - t0 + 1
        val = np.full(n, np.nan, np.float32)
        msk = np.zeros(n, bool)
        age = np.full(n, np.nan, np.float32)
        for j in w:
            s = int(round(d["t_end"][j])) - 59 - t0
            seg = slice(s, s + 60)
            x, m, a = d["X"][j, :, 0], d["mask"][j, :, 0], d["age"][j]
            prev = val[seg]
            ok = ~np.isnan(prev) & m
            assert np.allclose(prev[ok], x[ok], atol=1e-4), ("inconsistent overlap", k)
            val[seg] = np.where(m, x, val[seg])
            msk[seg] |= m
            age[seg] = np.where(np.isnan(age[seg]), a, age[seg])
        out[k] = dict(t0=t0, val=val, mask=msk, age=age, windows=w)
    return out


# noise (carried with each native observation), constant bias or sample-and-hold
def perturb_series(s, kind, rng):
    v, m, a = s["val"].copy(), s["mask"], s["age"]
    if kind == "clean":
        return v
    if kind.startswith("noise_"):
        sd = float(kind.split("_")[1])
        e = rng.normal(0.0, sd, len(v)).astype(np.float32)
        fresh = np.nan_to_num(a, nan=0.0) < 1.0
        idx = np.where(fresh, np.arange(len(v)), 0)
        np.maximum.accumulate(idx, out=idx)
        return np.where(m, v + e[idx], v)
    if kind.startswith("offset_"):
        return np.where(m, v + float(kind.split("_")[1]), v)
    if kind.startswith("hold_"):
        k = int(kind.split("_")[1])
        pos = np.arange(len(v))
        src = pos - (pos % k)
        held = v[src]
        return np.where(m & m[src], held, v)
    raise ValueError(kind)


# windows rebuilt from (perturbed) series values
def windows_from_series(ix, series, values, d):
    X = np.empty((len(ix), 60, 1), np.float32)
    M = d["mask"][ix]
    pos = {j: n for n, j in enumerate(ix)}
    for k, s in series.items():
        v = values[k]
        for j in s["windows"]:
            st = int(round(d["t_end"][j])) - 59 - s["t0"]
            X[pos[j], :, 0] = np.where(M[pos[j], :, 0], v[st:st + 60], d["X"][j, :, 0])
    return X, M


# record decisions at a stored threshold and at one recalibrated on the calibration negatives
def evaluate(t, r, keys, roles, c, lead, tau_fixed):
    order = np.lexsort((t, keys))
    ks = keys[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    per = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        j = order[a:b]
        k = ks[a]
        trig = float(c.at[k, "t_trigger"])
        tt, rr = t[j], r[j]
        trig = trig if np.isfinite(trig) else float(tt[0])
        valid = tt >= trig
        if valid.any():
            per[k] = (tt, rr, roles[j[0]], trig, float(rr[valid].max()))
    cal = np.array([v[4] for v in per.values() if v[2] == "calibration"])
    tau_recal = SV.calibrate_threshold(cal, np.arange(len(cal)).astype(str), np.ones(len(cal), bool), 0.1)
    rows = []
    for k, (tt, rr, role, trig, _) in per.items():
        if role != "test":
            continue
        pos, neg = bool(c.at[k, "scored_positive"]), bool(c.at[k, "scored_negative"])
        onset = float(c.at[k, "t_onset_L2_1.0"])
        rec = dict(key=k, positive=pos, negative=neg)
        for name, tau in [("fixed", tau_fixed), ("recal", tau_recal)]:
            hit = np.flatnonzero((tt >= trig) & (rr >= tau))
            alarm = float(tt[hit[0]]) if len(hit) else np.nan
            det = bool(pos and np.isfinite(alarm) and alarm < onset)
            rec[name + "_alarm"] = alarm
            rec[name + "_success"] = bool(det and onset - alarm >= lead)
            rec[name + "_neg_alarm"] = bool(neg and np.isfinite(alarm))
        rows.append(rec)
    return pd.DataFrame(rows), tau_recal
