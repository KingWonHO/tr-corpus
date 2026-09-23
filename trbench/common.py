"""Shared preprocessing core.  Adapters only produce raw channel series;
everything after that (anchoring, labels, resampling, QC, masking) lives here so
all seven datasets go through identical code."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.signal import medfilt

import schema as S


# ---------------------------------------------------------------- raw record
@dataclass
class RawExperiment:
    """What an adapter hands back: irregular, native-rate data in fixed units.

    series: channel -> (t_seconds, values).  Each channel keeps its own time
    base; they are only put on a common grid inside `standardize`.
    """
    dataset_id: str
    experiment_id: str
    series: dict
    meta: dict = field(default_factory=dict)
    t_trigger: float = None      # adapter may know it exactly
    notes: list = field(default_factory=list)


# ------------------------------------------------------------------- helpers
def _finite(t, v):
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    n = min(len(t), len(v))
    t, v = t[:n], v[:n]
    ok = np.isfinite(t) & np.isfinite(v)
    return t[ok], v[ok]


def _sorted_unique(t, v):
    o = np.argsort(t, kind="mergesort")
    t, v = t[o], v[o]
    keep = np.ones(len(t), bool)
    keep[1:] = np.diff(t) > 0
    return t[keep], v[keep]


def resample_channel(t, v, grid, native_dt=None):
    """Put one channel on `grid`.

    Downsampling uses a mean over each target bin, which is the anti-aliasing
    step the plan (1-4) requires for kHz pressure data; a plain pick-every-Nth
    would fold vent transients back into the band.  Upsampling is linear and the
    interpolated samples are reported so they can be masked.
    """
    t, v = _finite(t, v)
    if len(t) < 2:
        return np.full(len(grid), np.nan), np.zeros(len(grid), bool)
    t, v = _sorted_unique(t, v)

    step = grid[1] - grid[0] if len(grid) > 1 else 1.0
    if native_dt is None:
        native_dt = float(np.median(np.diff(t)))

    inside = (grid >= t[0] - step) & (grid <= t[-1] + step)

    if native_dt < step * 0.75:                      # downsample -> bin mean
        edges = np.concatenate([grid - step / 2, [grid[-1] + step / 2]])
        idx = np.digitize(t, edges) - 1
        ok = (idx >= 0) & (idx < len(grid))
        sums = np.bincount(idx[ok], weights=v[ok], minlength=len(grid))
        cnts = np.bincount(idx[ok], minlength=len(grid))
        out = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        interp = np.zeros(len(grid), bool)
        gap = np.isnan(out) & inside          # bins that caught no native sample
        if gap.any() and np.isfinite(out).sum() >= 2:
            good = np.isfinite(out)
            out[gap] = np.interp(grid[gap], grid[good], out[good])
            interp[gap] = True
    else:                                            # up / same rate -> linear
        out = np.interp(grid, t, v, left=np.nan, right=np.nan)
        interp = inside.copy()
        # samples landing within a quarter native step of a real sample are real
        j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        near = np.minimum(np.abs(grid - t[j - 1]), np.abs(grid - t[j]))
        interp &= near > native_dt * 0.25

    out[~inside] = np.nan
    interp &= np.isfinite(out)
    return out, interp


# ----------------------------------------------------------------- qc checks
def flag_tc_detachment(v, ambient=40.0, drop=150.0, hold=20):
    """Thermocouple fell off: a >`drop` degC collapse to near ambient that then
    stays there.  Everything after the collapse is untrustworthy (plan 1-6)."""
    bad = np.zeros(len(v), bool)
    ok = np.isfinite(v)
    if ok.sum() < hold + 2:
        return bad
    d = np.diff(v, prepend=v[0])
    cand = np.where((d < -drop) & (v < ambient + 30))[0]
    for i in cand:
        tail = v[i:i + hold]
        if np.isfinite(tail).sum() >= hold * 0.5 and np.nanmax(tail) < ambient + 30:
            bad[i:] = True
            break
    return bad


def flag_saturation(v, frac=0.999, min_run=5):
    """Channel pinned at its own max (or min) for a run of samples."""
    bad = np.zeros(len(v), bool)
    ok = np.isfinite(v)
    if ok.sum() < min_run * 2:
        return bad
    for lim in (np.nanmax(v), np.nanmin(v)):
        at = ok & (np.abs(v - lim) <= abs(lim) * (1 - frac) + 1e-9)
        idx = np.flatnonzero(at)
        if len(idx) < min_run:
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)
        for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]):
            if b - a + 1 >= min_run:
                bad[idx[a]:idx[b] + 1] = True
    return bad


# -------------------------------------------------------------- label timing
def _smooth(v, w):
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return v
    f = pd.Series(v).interpolate(limit_direction="both").to_numpy()
    k = max(3, int(w) | 1)
    return medfilt(f, kernel_size=min(k, (len(f) - 1) | 1))


def onset_L2(t, T, thresh=S.L2_PRIMARY, sustain_s=S.L2_SUSTAIN_S,
             t_min=S.L2_T_FLOOR, rise_min=S.L2_RISE_MIN, horizon_s=S.L2_HORIZON_S,
             guarded=True, ceiling=None):
    """dT/dt >= thresh degC/s held for `sustain_s`, plus two physical guards.

    A bare rate threshold is not enough on real data.  Two things trip it that
    are not thermal runaway:

      * the heater turn-on ramp -- #4 C2 climbs 23->37 degC in 12 s at t=98 s,
        a genuine 1.2 degC/s that is simply the abuse heater coming up;
      * thermocouple glitches -- #4 M1 shows a 34->45->37 degC spike at t=128 s
        in a test whose real runaway is at t=2615 s.

    So the crossing must also happen above `t_min` degC and be followed by a
    rise of at least `rise_min` degC within `horizon_s`, which no ramp artefact
    or glitch survives.  Pass guarded=False to get the bare rate crossing; the
    corpus stores both so E1 can quantify how much the guards move the label.

    `ceiling` handles CENSORED sensors.  #9 recorded much of its temperature
    with an infrared camera whose range was set to Tmax = 150 degC, and 25 of
    its 252 experiments sit pinned at exactly 150.24 degC.  On a censored trace
    the confirming rise can be arithmetically impossible -- three cells that the
    authors scored EUCAR 7 (rupture, smoke, fire) got no L2 label at all for
    that reason.  When the trace reaches its ceiling inside the horizon, that
    saturation IS the confirmation.
    """
    t, T = np.asarray(t, float), np.asarray(T, float)
    if np.isfinite(T).sum() < 5:
        return None
    dt = float(np.median(np.diff(t)))
    Ts = _smooth(T, max(3, int(round(1.0 / dt)) | 1))
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


def _first_guarded_voltage_crossing(t, v, low, sustain_s, baseline,
                                    recovery_s=3.0,
                                    dropout_min_s=10.0):
    """First sustained crossing after rejecting a clear logger dropout.

    The ordinary persistence rule remains authoritative.  A candidate is
    rejected only when all three artefact signatures occur together: the
    voltage stays low for at least ``dropout_min_s``, reaches near zero (within
    10% of baseline), and later returns to within 1% of baseline for
    ``recovery_s``.  Thus reversible electrochemical excursions and the source
    dataset's published first-25 mV criterion are preserved.  This narrow guard
    targets D3 test 1's 64 s near-zero outage followed by full recovery.
    """
    t, v, low = np.asarray(t, float), np.asarray(v, float), np.asarray(low, bool)
    if len(t) < 2:
        return None
    dt = float(np.median(np.diff(t)))
    n = max(1, int(round(sustain_s / dt)))
    nr = max(1, int(round(recovery_s / dt)))
    ndrop = max(n, int(round(dropout_min_s / dt)))
    # Work by complete low runs.  Iterating every overlapping persistence
    # window would reject the front of a dropout and then incorrectly accept a
    # shorter window near the end of that same artefact.
    starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
    for i in starts:
        j = i
        while j + 1 < len(low) and low[j + 1]:
            j += 1
        if j - i + 1 < n:
            continue
        long_dropout = (j - i + 1) >= ndrop
        near_zero = (np.isfinite(v[i:j + 1]).any() and
                     np.nanmin(np.abs(v[i:j + 1])) <= 0.10 * abs(baseline))
        recovery_level = baseline - max(0.01 * abs(baseline), 0.01)
        recovered = np.isfinite(v) & (v >= recovery_level)
        tail = recovered[j + 1:]
        full_recovery = False
        if len(tail) >= nr:
            recovery_run = np.convolve(tail.astype(int), np.ones(nr, int),
                                       mode="valid")
            full_recovery = bool(np.any(recovery_run >= nr))
        if long_dropout and near_zero and full_recovery:
            continue
        return float(t[i])
    return None


def onset_isc(t, V, t_trigger=0.0, drop_v=S.ISC_DROP_V, sustain_s=2.0,
              recovery_s=3.0):
    """Internal short circuit, by the #9 protocol's own criterion: open-circuit
    voltage down `drop_v` (25 mV) from the pre-abuse baseline.

    This is far more sensitive than L3's 20% collapse -- 25 mV on a 4 V cell is
    0.6% -- and it is the earliest electrical precursor in the corpus, which is
    exactly the quantity an early-warning study wants.  Computed wherever
    V_cell exists, not only for #9.
    """
    t, V = np.asarray(t, float), np.asarray(V, float)
    ok = np.isfinite(V)
    if ok.sum() < 10:
        return None
    t_trigger = float(t_trigger or 0.0)
    pre = ok & (t <= t_trigger)
    if pre.sum() < 5:
        pre = np.zeros(len(t), bool)
        pre[np.flatnonzero(ok)[:max(5, ok.sum() // 50)]] = True
    base = float(np.nanmedian(V[pre]))
    if not np.isfinite(base):
        return None
    low = ok & (V < base - drop_v) & (t > t_trigger)
    return _first_guarded_voltage_crossing(t, V, low, sustain_s, base,
                                           recovery_s)


def onset_L1(t, P, t_trigger=0.0, site="in_cell",
             gauge_floor_kpa=-110.0):
    """Venting time from pressure.  The sign depends on where the transducer is.

    in_cell (#3): gas builds up inside the sealed cell until the burst disc
        opens, so venting is the steepest DROP after the maximum.
    chamber (#1, #4): the transducer sits in the vessel around the cell and
        sees nothing until the vent gas arrives, so venting is the steepest
        RISE.  Looking for a drop here finds the vessel depressurising minutes
        later -- for #4 C1 that put "venting" 935 s AFTER thermal runaway.

    Merging the two channels, or using one rule for both, produces a label that
    is not just noisy but inverted.
    """
    t, P = np.asarray(t, float), np.asarray(P, float)
    ok = np.isfinite(P)
    if ok.sum() < 10:
        return None
    # Gauge pressure cannot be lower than a vacuum (about -101.3 kPa).  Leave a
    # small tolerance for calibration noise, but exclude more-negative values
    # before smoothing and differentiation.  Two D3 transducers rail at
    # -2.2/-2.7 MPa after venting; without this guard that later rail transition
    # is incorrectly selected as the vent itself.
    if site == "in_cell":
        P = np.where(P >= gauge_floor_kpa, P, np.nan)
        if np.isfinite(P).sum() < 10:
            return None
    Ps = _smooth(P, 5)
    Ps = np.where(t >= (t_trigger or 0.0), Ps, np.nan)
    if np.isfinite(Ps).sum() < 10:
        return None
    span = np.nanmax(Ps) - np.nanmin(Ps)
    if not np.isfinite(span) or span <= 0:
        return None
    rate = np.gradient(np.nan_to_num(Ps, nan=np.nanmin(Ps)), t)

    if site == "chamber":
        i = int(np.nanargmax(rate))
        return float(t[i]) if rate[i] >= 0.05 * span else None

    i_max = int(np.nanargmax(Ps))
    if i_max >= len(t) - 3:
        return None
    tail = rate[i_max:]
    i_rel = int(np.nanargmin(tail))
    if tail[i_rel] > -0.05 * span:      # no real collapse
        return None
    return float(t[i_max + i_rel])


def onset_L3(t, V, t_trigger=0.0, frac=0.2, sustain_s=3.0,
             recovery_s=3.0):
    """Cell voltage collapse: first SUSTAINED crossing below (1-frac) x the
    pre-abuse level.

    The baseline is taken strictly before t_trigger.  Under overcharge (#2) the
    voltage climbs to ~45 V within the first minute, so a baseline measured over
    "the first 2% of samples" would sit above the true resting voltage and mark
    an onset at t~3 s, before the abuse even starts.
    """
    t, V = np.asarray(t, float), np.asarray(V, float)
    ok = np.isfinite(V)
    if ok.sum() < 10:
        return None
    t_trigger = float(t_trigger or 0.0)
    pre = ok & (t <= t_trigger) if (t <= t_trigger).sum() >= 5 else None
    base = float(np.nanmedian(V[pre])) if pre is not None else \
        float(np.nanmedian(V[ok][:max(5, ok.sum() // 50)]))
    if not np.isfinite(base) or abs(base) < 1e-6:
        return None

    low = ok & (V < base * (1 - frac)) & (t > t_trigger)
    return _first_guarded_voltage_crossing(t, V, low, sustain_s, base,
                                           recovery_s)


# -------------------------------------------------------------- standardize
def _surface_channels(cols):
    named = ["T_surface_neg", "T_surface_pos", "T_surface_mid"]
    extra = sorted(k for k in cols if k.startswith("T_surface_x"))
    return [k for k in named if k in cols] + extra


def standardize(raw, hz=S.BASE_HZ):
    """Irregular native channels -> one regular grid + masks + labels."""
    ser = {k: _finite(*v) for k, v in raw.series.items()}
    ser = {k: v for k, v in ser.items() if len(v[0]) >= 2}
    if not ser:
        raise ValueError(raw.experiment_id + ": no usable channel")

    t0 = min(v[0][0] for v in ser.values())
    t1 = max(v[0][-1] for v in ser.values())
    step = 1.0 / hz
    grid = np.arange(0.0, (t1 - t0) + step / 2, step)
    ser = {k: (t - t0, v) for k, (t, v) in ser.items()}

    # Per-channel interpolation is kept, not folded into one OR.  #1's channels
    # have different native rates, so an OR flags 69-79% of its rows and a causal
    # rebuild driven by it freezes channels that were actually measured -- which
    # is exactly how a real M4 increment got reported as vanishing (see the log,
    # section 15-C).  The OR is still written for backward compatibility.
    cols, masks, interps = {}, {}, {}
    interp_any = np.zeros(len(grid), bool)
    for ch, (t, v) in ser.items():
        out, itp = resample_channel(t, v, grid)
        cols[ch] = out
        masks[ch] = np.isfinite(out)
        interps[ch] = itp
        interp_any |= itp

    # ---- qc, before the surface aggregates are derived from these channels
    qc = np.zeros(len(grid), np.uint8)
    for ch in list(cols):
        if not ch.startswith("T_"):
            continue
        det = flag_tc_detachment(cols[ch])
        if det.any():
            cols[ch] = np.where(det, np.nan, cols[ch])
            masks[ch] = masks[ch] & ~det
            qc |= (det.astype(np.uint8) * S.QC_TC_DETACH)
            raw.notes.append("%s: thermocouple detachment masked from t=%.1fs"
                             % (ch, grid[det.argmax()]))
    # Saturation is checked on temperature too, not just P and V: #9's infrared
    # camera was capped at 150 degC and 25 of its experiments sit pinned there,
    # so their peak temperature is censored, not measured.
    censored = {}
    for ch in list(cols):
        if not (ch.startswith(("T_", "P_", "V_")) or ch == "F_expansion"):
            continue
        sat = flag_saturation(cols[ch])
        if sat.sum() >= 5 and sat.mean() > 0.001:
            qc |= (sat.astype(np.uint8) * S.QC_SATURATED)
            hi = float(np.nanmax(cols[ch]))
            if np.isclose(np.nanmedian(cols[ch][sat]), hi, rtol=1e-3):
                censored[ch] = hi
                raw.notes.append("%s censored: pinned at %.2f for %d samples"
                                 % (ch, hi, int(sat.sum())))
    qc |= (interp_any.astype(np.uint8) * S.QC_INTERP)

    # derived aggregates over every surface TC the raw file had
    surf = _surface_channels(cols)
    if surf:
        M = np.vstack([cols[c] for c in surf])
        with warnings.catch_warnings():          # all-NaN columns are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            cols["T_surface_max"] = np.nanmax(M, axis=0)
            cols["T_surface_mean"] = np.nanmean(M, axis=0)
        got = np.vstack([masks[c] for c in surf]).any(axis=0)
        masks["T_surface_max"] = got
        masks["T_surface_mean"] = got
        # An aggregate is interpolated exactly where every member it could have
        # drawn from was.  Falling back to the row-level OR instead marked #1's
        # aggregates 72% interpolated when their members are measured at a full
        # 1 Hz, and that alone moved the M4 increment (log, section 16).
        agg = np.vstack([interps.get(c, np.zeros(len(grid), bool)) | ~masks[c]
                         for c in surf]).all(axis=0)
        interps["T_surface_max"] = agg
        interps["T_surface_mean"] = agg

    # ---- anchors and labels
    Tprobe, Tname = cols.get("T_surface_max"), "T_surface_max"
    if Tprobe is None or np.isfinite(Tprobe).sum() < 5:
        Tprobe = None
        for c in ("T_internal_core", "T_vent", "T_surface_mid", "T_ambient"):
            if c in cols and np.isfinite(cols[c]).sum() >= 5:
                Tprobe, Tname = cols[c], c
                break
    # the aggregate inherits its members' ceiling
    ceiling = censored.get(Tname)
    if ceiling is None and Tname == "T_surface_max" and censored:
        hits = [v for k, v in censored.items() if k.startswith("T_")]
        ceiling = max(hits) if hits else None
    if raw.t_trigger is not None:
        # t0 is the earliest sample across channels; several datasets index
        # their first sample as t=1, so rebasing a trigger of 0 would give -1.
        t_trig = max(0.0, float(raw.t_trigger) - t0)
    else:
        t_trig = 0.0
        raw.notes.append("t_trigger not instrumented; set to record start")

    labels = {}
    for th in S.L2_THRESHOLDS:
        labels["t_onset_L2_%s" % th] = (
            onset_L2(grid, Tprobe, th, ceiling=ceiling)
            if Tprobe is not None else None)
    labels["t_onset_L2"] = labels["t_onset_L2_%s" % S.L2_PRIMARY]
    labels["t_onset_L2_unguarded"] = (
        onset_L2(grid, Tprobe, S.L2_PRIMARY, guarded=False)
        if Tprobe is not None else None)
    labels["t_isc"] = (onset_isc(grid, cols["V_cell"], t_trig)
                       if "V_cell" in cols else None)
    labels["t_onset_L1"] = None
    for pch in S.PRESSURE:               # in-cell first, it is the direct measure
        if pch in cols:
            labels["t_onset_L1"] = onset_L1(grid, cols[pch], t_trig,
                                            S.PRESSURE_SITE[pch])
            labels["L1_source"] = pch
            if labels["t_onset_L1"] is not None:
                break
    labels["t_onset_L3"] = (onset_L3(grid, cols["V_cell"], t_trig)
                            if "V_cell" in cols else None)

    t_on = labels["t_onset_L2"]
    if t_on is None:
        t_on = labels["t_onset_L1"] if labels["t_onset_L1"] is not None else labels["t_onset_L3"]

    if t_on is not None:
        qc |= ((grid > t_on + S.ALPHA_S).astype(np.uint8) * S.QC_POST_ONSET)

    # ---- assemble
    df = pd.DataFrame(index=range(len(grid)))
    df["time_s"] = grid
    df["t_rel_trigger"] = grid - t_trig
    df["t_rel_onset"] = (grid - t_on) if t_on is not None else np.nan
    for ch in S.CHANNELS:
        df[ch] = cols.get(ch, np.full(len(grid), np.nan)).astype("float32")
        df[S.mask_col(ch)] = masks.get(ch, np.zeros(len(grid), bool)).astype("uint8")
    df["is_interpolated"] = interp_any.astype("uint8")
    # A channel resampled here has its own flag.  Derived aggregates (the
    # T_surface_* family, the pressure-derived gas of #1) are built further down
    # from several of these, so they fall back to the row-level OR -- which
    # over-flags rather than under-flags, the safe direction for a causal build.
    for ch in S.CHANNELS:
        itp = interps.get(ch)
        if itp is None:
            itp = interp_any
        df[S.interp_col(ch)] = np.asarray(itp)[:len(df)].astype("uint8")
    df["qc_flag"] = qc

    # ---- crop pathologically long, mostly-interpolated records
    # ARC heat-wait-seek runs last 6-50 hours and emit ~200 event-driven samples,
    # so on a 1 Hz grid >99% of the rows are interpolation, not measurement.
    # Keep the physically meaningful window and say so, rather than shipping
    # hundreds of megabytes of fabricated points.
    crop_start = None
    if len(grid) and grid[-1] > S.MAX_DURATION_S:
        end = (t_on + S.ALPHA_S) if t_on is not None else grid[-1]
        lo, hi = max(0.0, end - S.CROP_PRE_S), min(grid[-1], end)
        if hi - lo >= 600:
            keep = (grid >= lo) & (grid <= hi)
            df = df[keep].reset_index(drop=True)
            crop_start = float(lo)
            raw.notes.append(
                "record is %.0f s long and %.1f%% interpolated; cropped to "
                "[t_end-%.0f s, t_end] = [%.0f, %.0f] s of the original axis"
                % (grid[-1], 100 * interp_any.mean(), S.CROP_PRE_S, lo, hi))

    meta = dict(raw.meta)
    meta.update(
        dataset_id=raw.dataset_id, experiment_id=raw.experiment_id,
        n_samples=int(len(df)),
        duration_s=float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) else 0.0,
        record_duration_s=float(grid[-1]) if len(grid) else 0.0,
        crop_start_s=crop_start,
        hz=hz, t_trigger=float(t_trig),
        t_onset=None if t_on is None else float(t_on),
        T_probe=Tname if Tprobe is not None else None,
        T_censored_at=ceiling,
        censored_channels=",".join(sorted(censored)) or None,
        channels_present=sorted(c for c in S.CHANNELS if df[S.mask_col(c)].any()),
        notes=list(dict.fromkeys(raw.notes)),
    )
    for k, v in labels.items():
        meta[k] = v if (v is None or isinstance(v, str)) else float(v)
    return df, meta


def baseline_scaler_stats(df, meta):
    """Mean/std fitted only on the pre-trigger baseline (plan 1-7): using the
    whole sequence would let the TR excursion into the statistics and leak the
    future into the normalizer."""
    t_trig = meta.get("t_trigger") or 0.0
    win = df[df["time_s"] <= max(t_trig, float(df["time_s"].iloc[0]) + 30.0)]
    if len(win) < 10:
        win = df.head(max(10, len(df) // 20))
    out = {}
    for ch in S.CHANNELS:
        v = win[ch].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) >= 5:
            out[ch] = dict(mean=float(np.mean(v)),
                           std=float(np.std(v)) or 1.0, n=int(len(v)))
    return out
