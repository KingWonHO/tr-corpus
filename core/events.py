# guarded thermal-rate onset (L2) on a temperature trace
import numpy as np
import pandas as pd
from scipy.signal import medfilt

from core import schema as S


def smooth(v, w):
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return v
    f = pd.Series(v).interpolate(limit_direction="both").to_numpy()
    k = max(3, int(w) | 1)
    return medfilt(f, kernel_size=min(k, (len(f) - 1) | 1))


# dT/dt >= thresh held for sustain_s, above t_min, confirmed by a further rise (or a sensor ceiling)
def onset_L2(t, T, thresh=S.L2_PRIMARY, sustain_s=S.L2_SUSTAIN_S, t_min=S.L2_T_FLOOR, rise_min=S.L2_RISE_MIN,
             horizon_s=S.L2_HORIZON_S, guarded=True, ceiling=None):
    t, T = np.asarray(t, float), np.asarray(T, float)
    if np.isfinite(T).sum() < 5:
        return None
    dt = float(np.median(np.diff(t)))
    Ts = smooth(T, max(3, int(round(1.0 / dt)) | 1))
    rate = np.gradient(Ts, t)
    n = max(1, int(round(sustain_s / dt)))
    hot = rate >= thresh
    run = np.convolve(hot.astype(int), np.ones(n, int), mode="valid")
    for i in np.flatnonzero(run >= n):
        if not guarded:
            return float(t[i])
        if Ts[i] < t_min:
            continue
        j = min(len(Ts) - 1, i + int(round(horizon_s / dt)))
        peak = np.nanmax(Ts[i:j + 1])
        if peak - Ts[i] < rise_min:
            if ceiling is None or peak < ceiling - 1.0:
                continue
        return float(t[i])
    return None
