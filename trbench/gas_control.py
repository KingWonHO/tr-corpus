"""Explicit noncausal negative control, changing only BAK gas interpolation.

This is never a deployment result. It holds native-build timestamps, all other
channels, labels, splits, models and seeds constant, unlike the historical
comparison that changed stride and observation-age features at the same time.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "adapters"))
import ds01_bak


def apply(d):
    if not d["window_dir"].startswith("W60_native"):
        raise ValueError("gas control requires the native build")
    if any(f.startswith("age__") or f == "obs_age" for f in d["features"]):
        raise ValueError("gas control is specified for --without-age only")
    col = d["features"].index("gas_total_mol")
    length = d["X"].shape[1]
    n_changed = 0
    for key in np.unique(d["experiment"]):
        if not str(key).startswith("ds01_bak/"):
            continue
        exp = str(key).split("/")[1]
        p = HERE.parent / ds01_bak.RAW / exp / f"export_{exp}.xlsx"
        frame = pd.read_excel(p, skiprows=[1, 2])
        tt = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float) / 1000
        vv = pd.to_numeric(frame["n"], errors="coerce").to_numpy(float) / 1000
        good = np.isfinite(tt) & np.isfinite(vv)
        tt, vv = tt[good], vv[good]
        # The export time column is already elapsed time on the MAT origin;
        # the adapter no longer re-zeroes it (see ds01_bak.py), and neither
        # does this control.
        order = np.argsort(tt, kind="stable")
        tt, vv = tt[order], vv[order]
        unique = np.r_[True, np.diff(tt) > 0]
        tt, vv = tt[unique], vv[unique]
        idx = np.flatnonzero(d["experiment"] == key)
        times = d["t_end"][idx, None] - np.arange(length-1, -1, -1)[None, :]
        # Same outside-support hold as the native builder after the last
        # observation; only interior interpolation uses future observations.
        values = np.interp(times, tt, vv, left=np.nan, right=vv[-1])
        mask = np.isfinite(values)
        d["X"][idx, :, col] = np.where(mask, values, 0).astype(np.float32)
        d["mask"][idx, :, col] = mask.astype(np.uint8)
        n_changed += 1
    if n_changed != 8:
        raise ValueError(f"expected 8 BAK records, got {n_changed}")
    return d
