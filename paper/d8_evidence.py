"""Evidence behind the D8 interpretation (review item 4.14): what the field
windows look like and which D5 availability pattern they match.

    paper/tables/S4k_d8_windows.{csv,md}
        Field pack-temperature statistics of the scored charging segments
        (window-end maximum temperature, rise over the segment, maximum), for
        all normal vehicles and separately for vehicles that alarm and stay
        silent under each frozen tree arm; and the share of D5 training windows
        whose M1 availability pattern equals the D8 pattern.

Reads the scored-snippet list and rebuilds the causal windows with the same
function the D8 case used (trbench/external_d8._d8_windows); no model runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "trbench"), os.path.join(ROOT, "trbench", "adapters"), os.path.join(ROOT, "paper")]
import external_d8 as D8                       # noqa: E402
import native_windows as NW                    # noqa: E402
from make_figures import save_table            # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results", "external_20260925_d8_field")
EXT = os.path.join(ROOT, "tr-corpus", "external", "d8_field")
WIN = os.path.join(ROOT, "tr-corpus", "windows", "W60_native")
M1 = ["T_surface_neg", "T_surface_pos", "T_surface_mid", "T_surface_max", "T_surface_mean", "T_ambient"]


def field_windows():
    sample = pd.read_csv(os.path.join(EXT, "scored_snippets.csv"))
    feats = list(NW.OUTPUT_FEATURES)
    rows = []
    for r in sample.itertuples():
        out = D8._d8_windows(r.brand, r.split, r.file)
        if out is None:
            continue
        tend, f, m = out
        it = feats.index("T_surface_max")
        T = np.where(m[:, -1, it] > 0, f[:, -1, it], np.nan)
        pattern = tuple(int(m[:, -1, feats.index(c)].max() > 0) for c in M1)
        rows.append(dict(brand=r.brand, car=r.car, snippet=r.file, T_median=np.nanmedian(T), T_first=T[np.isfinite(T)][0] if np.isfinite(T).any() else np.nan,
                         T_last=T[np.isfinite(T)][-1] if np.isfinite(T).any() else np.nan, T_max=np.nanmax(T), span_s=float(tend[-1] - tend[0]),
                         pattern="".join(map(str, pattern))))
    return pd.DataFrame(rows)


def d5_pattern_share(pattern):
    z = np.load(os.path.join(WIN, "train.npz"), allow_pickle=True)
    feats = list(z["features"]); e = z["experiment"].astype(str); m = z["mask"]
    d5 = np.char.startswith(e, "ds09_mech/")
    pat = np.stack([m[d5][:, -1, feats.index(c)] > 0 for c in M1], axis=1).astype(int)
    target = np.array([int(c) for c in pattern])
    return float((pat == target).all(axis=1).mean()), int(d5.sum())


def main():
    w = field_windows()
    w["rise"] = w.T_last - w.T_first
    pv = pd.read_csv(os.path.join(RES, "per_vehicle.csv"))
    veh = w.groupby(["brand", "car"]).agg(T_median=("T_median", "median"), rise=("rise", "median"), T_max=("T_max", "max"), segments=("snippet", "size")).reset_index()
    rows = [dict(group="all normal vehicles", vehicles=len(veh), segments=int(veh.segments.sum()),
                 T_median_median=round(veh.T_median.median(), 1), T_median_p10_p90="%.1f–%.1f" % (veh.T_median.quantile(.1), veh.T_median.quantile(.9)),
                 rise_median=round(w.rise.median(), 1), rise_p90=round(w.rise.quantile(.9), 1), T_max=round(veh.T_max.max(), 1))]
    for arm, model in (("no_age", "xgboost"), ("no_age", "lightgbm"), ("surface_max_only", "xgboost"), ("surface_max_only", "lightgbm"),
                       ("surface_mean_only", "xgboost"), ("surface_mean_only", "lightgbm")):
        g = pv[(pv.arm == arm) & (pv.model == model)].groupby(["brand", "car"]).snippets_alarm.sum().reset_index()
        j = veh.merge(g, on=["brand", "car"], how="left")
        for name, sel in (("alarming", j.snippets_alarm > 0), ("silent", j.snippets_alarm == 0)):
            k = j[sel]
            if not len(k):
                continue
            rows.append(dict(group="%s / %s / %s (any seed)" % (arm, model, name), vehicles=len(k), segments=int(k.segments.sum()),
                             T_median_median=round(k.T_median.median(), 1), T_median_p10_p90="%.1f–%.1f" % (k.T_median.quantile(.1), k.T_median.quantile(.9)),
                             rise_median=np.nan, rise_p90=np.nan, T_max=round(k.T_max.max(), 1)))
    pattern = w.pattern.mode().iloc[0]
    share, n = d5_pattern_share(pattern)
    df = pd.DataFrame(rows)
    save_table(df, "S4k_d8_windows",
               "Source: the %d scored D8 charging segments rebuilt with trbench/external_d8._d8_windows (causal 60 s windows, 10 s cadence) and "
               "external_20260925_d8_field/per_vehicle.csv. Temperatures are the maximum pack probe at the window end (degC); the rise is last minus first window of a segment. "
               "The D8 M1 availability pattern is %s over (%s); %.0f %% of the %d D5 training windows carry the same pattern." % (len(w), pattern, ", ".join(M1), 100 * share, n))
    w.to_csv(os.path.join(ROOT, "paper", "tables", "S4k_d8_windows_segments.csv"), index=False)
    print(df.to_string(index=False)); print("pattern", pattern, "D5 share", share)


if __name__ == "__main__":
    main()
