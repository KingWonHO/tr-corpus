"""Stage 1 of the outcome adjudication (protocol 29): per-experiment diagnostics for D5 (ds09_mech) and D6 (ds12_arc).

Ported from the 2026-09-24 audit script (reports/26) so that the adjudication is reproducible from the repository.

Replays trbench.common.onset_L2 with per-candidate bookkeeping so that, for a
record without L2, the failing guard can be named.  Writes a pickle of the
per-record diagnostics for the reconciliation script.
"""
import os, sys, glob, re, warnings, pickle
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "trbench"))
import schema as S
from common import _smooth
import importlib.util
spec = importlib.util.spec_from_file_location("ds12", os.path.join(ROOT, "trbench/adapters/ds12_arc.py"))
ds12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds12)
warnings.filterwarnings("ignore")

OUT = os.path.join(ROOT, "tr-corpus", "registry", "adjudication_cache")
os.makedirs(OUT, exist_ok=True)


def l2_diag(t, T, ceiling=None, thresh=S.L2_PRIMARY, sustain_s=S.L2_SUSTAIN_S,
            t_min=S.L2_T_FLOOR, rise_min=S.L2_RISE_MIN, horizon_s=S.L2_HORIZON_S):
    """Replay of common.onset_L2 returning (t_L2, t_unguarded, candidate summary)."""
    t, T = np.asarray(t, float), np.asarray(T, float)
    out = dict(t_L2=None, t_unguarded=None, n_cand=0, n_below_floor=0,
               n_conf_fail=0, best_rise=np.nan, best_rise_T=np.nan,
               best_rise_t=np.nan, max_rate=np.nan, max_rate_T=np.nan,
               first_cand_T=np.nan, first_cand_end_gap=np.nan)
    if np.isfinite(T).sum() < 5:
        out["note"] = "fewer than 5 finite samples"
        return out
    dt = float(np.median(np.diff(t)))
    Ts = _smooth(T, max(3, int(round(1.0 / dt)) | 1))
    rate = np.gradient(Ts, t)
    out["max_rate"] = float(np.nanmax(rate))
    out["max_rate_T"] = float(Ts[int(np.nanargmax(rate))])
    n = max(1, int(round(sustain_s / dt)))
    hot = rate >= thresh
    run = np.convolve(hot.astype(int), np.ones(n, int), mode="valid")
    cands = np.flatnonzero(run >= n)
    out["n_cand"] = int(len(cands))
    if len(cands):
        out["t_unguarded"] = float(t[cands[0]])
        out["first_cand_T"] = float(Ts[cands[0]])
        out["first_cand_end_gap"] = float(t[-1] - t[cands[0]])
    best = -np.inf
    for i in cands:
        if Ts[i] < t_min:
            out["n_below_floor"] += 1
            continue
        j = min(len(Ts) - 1, i + int(round(horizon_s / dt)))
        peak = np.nanmax(Ts[i:j + 1])
        rise = peak - Ts[i]
        if rise > best:
            best, out["best_rise_T"], out["best_rise_t"] = rise, float(Ts[i]), float(t[i])
        if rise < rise_min and (ceiling is None or peak < ceiling - 1.0):
            out["n_conf_fail"] += 1
            continue
        out["t_L2"] = float(t[i])
        break
    out["best_rise"] = float(best) if np.isfinite(best) else np.nan
    return out


def max_rise_60(t, T):
    """Largest rise over any 60 s forward window (smoothed as in the rule)."""
    T = np.asarray(T, float)
    if np.isfinite(T).sum() < 5:
        return np.nan
    dt = float(np.median(np.diff(t)))
    Ts = _smooth(T, max(3, int(round(1.0 / dt)) | 1))
    w = int(round(60.0 / dt))
    if len(Ts) <= w:
        return float(np.nanmax(Ts) - np.nanmin(Ts))
    s = pd.Series(Ts)
    fwd_max = s[::-1].rolling(w + 1, min_periods=1).max()[::-1].to_numpy()
    return float(np.nanmax(fwd_max - Ts))


reg = pd.read_csv(os.path.join(ROOT, "tr-corpus/registry/experiments.csv"))
reg = reg[reg.dataset_id.isin(["ds09_mech", "ds12_arc"])].copy()

rows = []
for _, r in reg.iterrows():
    df = pd.read_parquet(os.path.join(ROOT, "tr-corpus", r.file))
    probe = r.T_probe if isinstance(r.T_probe, str) else "T_surface_max"
    t = df.time_s.to_numpy(float)
    T = df[probe].to_numpy(float)
    ok = np.isfinite(T)
    d = dict(experiment_id=r.experiment_id, dataset_id=r.dataset_id, probe=probe,
             n_rows=len(df), t_first=t[0], t_last=t[-1],
             T_first=float(T[ok][0]) if ok.any() else np.nan,
             T_max=float(np.nanmax(T)) if ok.any() else np.nan,
             t_Tmax=float(t[int(np.nanargmax(T))]) if ok.any() else np.nan,
             T_last=float(T[ok][-1]) if ok.any() else np.nan,
             T_min=float(np.nanmin(T)) if ok.any() else np.nan,
             frac_T_finite=float(ok.mean()),
             probe_interp_frac=float(df.get(probe + "_interp", pd.Series(np.zeros(len(df)))).mean()),
             max_rise_60=max_rise_60(t, T),
             qc_sat_any=bool((df.qc_flag.to_numpy() & S.QC_SATURATED).any()),
             qc_detach_any=bool((df.qc_flag.to_numpy() & S.QC_TC_DETACH).any()),
             V_present=bool(df.mask_V_cell.any()),
             V_last=float(df.V_cell.dropna().iloc[-1]) if df.mask_V_cell.any() else np.nan,
             V_first=float(df.V_cell.dropna().iloc[0]) if df.mask_V_cell.any() else np.nan,
             V_min=float(df.V_cell.min()) if df.mask_V_cell.any() else np.nan)
    # surface channels individually (D5 has several)
    surf = [c for c in df.columns if c.startswith("T_surface_") and not c.endswith("_interp")
            and not c.startswith("T_surface_max") and not c.startswith("T_surface_mean")
            and not c.startswith("mask_") and df[c].notna().any()]
    d["n_surface_ch"] = len(surf)
    d["surface_ch"] = ",".join(surf)
    dg = l2_diag(t, T, ceiling=(r.T_censored_at if pd.notna(r.T_censored_at) and
                                isinstance(r.censored_channels, str) and probe in r.censored_channels.split(",")
                                else None))
    # replicate the ceiling inheritance of standardize(): aggregate inherits max T_ ceiling
    if pd.notna(r.T_censored_at) and isinstance(r.censored_channels, str):
        hits = [c for c in r.censored_channels.split(",") if c.startswith("T_")]
        if hits and probe == "T_surface_max":
            dg = l2_diag(t, T, ceiling=float(r.T_censored_at))
    d.update({("l2_" + k): v for k, v in dg.items()})
    d["match_registry_L2"] = (pd.isna(r.t_onset_L2) and dg["t_L2"] is None) or \
        (pd.notna(r.t_onset_L2) and dg["t_L2"] is not None and abs(r.t_onset_L2 - dg["t_L2"]) < 1e-6)
    d["match_registry_L2u"] = (pd.isna(r.t_onset_L2_unguarded) and dg["t_unguarded"] is None) or \
        (pd.notna(r.t_onset_L2_unguarded) and dg["t_unguarded"] is not None and abs(r.t_onset_L2_unguarded - dg["t_unguarded"]) < 1e-6)
    d["end_gap_after_L2"] = (t[-1] - r.t_onset_L2) if pd.notna(r.t_onset_L2) else np.nan
    rows.append(d)

diag = pd.DataFrame(rows)
diag.to_pickle(os.path.join(OUT, "diag.pkl"))

# ---------------------------------------------------------------- D6 raw native sampling
base = os.path.join(ROOT, "Dataset/12/TR_KIT ARC")
files = sorted(glob.glob(base + "/**/*.txt", recursive=True) + glob.glob(base + "/**/*.EXO", recursive=True))
raw_rows = []
regD6 = reg[reg.dataset_id == "ds12_arc"].set_index("experiment_id")
for f in files:
    form = os.path.basename(os.path.dirname(f)); stem = os.path.splitext(os.path.basename(f))[0]
    eid = form + "__" + stem
    if eid not in regD6.index:
        continue
    df = ds12._read(f)
    t = df.iloc[:, 0].to_numpy(float) * 60.0; T = df.iloc[:, 1].to_numpy(float)
    rate = df.iloc[:, 2].to_numpy(float) if df.shape[1] > 2 else np.full(len(t), np.nan)
    t0 = t[0]
    r = regD6.loc[eid]
    dt = np.diff(t)
    lo = r.crop_start_s if pd.notna(r.crop_start_s) else 0.0
    hi = lo + r.duration_s
    in_win = (t - t0 >= lo - 1e-6) & (t - t0 <= hi + 1e-6)
    d = dict(experiment_id=eid, n_native=len(t), raw_T0=T[0], raw_Tmax=float(np.nanmax(T)),
             raw_Tend=T[-1], raw_t_end_s=t[-1] - t0, raw_t0_min=t0 / 60,
             raw_rate_max_c_per_min=float(np.nanmax(rate)) if np.isfinite(rate).any() else np.nan,
             own_rate_max_c_per_min=float(np.nanmax(np.diff(T) / np.maximum(dt, 1e-9) * 60)),
             dt_median=float(np.median(dt)), dt_p90=float(np.percentile(dt, 90)), dt_min=float(dt.min()),
             n_native_in_window=int(in_win.sum()), frac_rows_native=in_win.sum() / r.n_samples,
             T_at_first_10cpm=np.nan, t_first_10cpm_rel=np.nan)
    # first time the ARC's own rate column exceeds 10 C/min (common ARC TR convention)
    if np.isfinite(rate).any():
        idx = np.flatnonzero(rate >= 10.0)
        if len(idx):
            d["T_at_first_10cpm"] = float(T[idx[0]]); d["t_first_10cpm_rel"] = float(t[idx[0]] - t0)
    if pd.notna(r.t_onset_L2):
        tl2 = r.t_onset_L2 + t0
        m = (t >= tl2 - 300) & (t <= tl2 + 300)
        dtw = np.diff(t[m]) if m.sum() > 1 else np.array([np.nan])
        d.update(n_native_pm300=int(m.sum()), dt_median_pm300=float(np.nanmedian(dtw)),
                 dt_p90_pm300=float(np.nanpercentile(dtw, 90)) if np.isfinite(dtw).any() else np.nan,
                 n_native_after_L2=int((t > tl2).sum()), raw_end_minus_L2_s=float(t[-1] - tl2))
    else:
        d.update(n_native_pm300=np.nan, dt_median_pm300=np.nan, dt_p90_pm300=np.nan,
                 n_native_after_L2=np.nan, raw_end_minus_L2_s=np.nan)
    raw_rows.append(d)
raw = pd.DataFrame(raw_rows)
raw.to_pickle(os.path.join(OUT, "d6raw.pkl"))

