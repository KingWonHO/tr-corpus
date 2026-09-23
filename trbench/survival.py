"""Discrete-time survival head, alarm rule, and lead-time evaluation.

The task is streaming: at every second a model emits a hazard, the survival
curve turns that into a risk at any horizon, an alarm fires the first time that
risk crosses a threshold, and the lead time is how far ahead of onset the alarm
came.  Everything a model family has to supply is one number per window, so the
same objective works for rules, trees, kernel methods and deep nets alike --
which is the precondition for comparing sensor panels across model families at
all.

Two rules the corpus forces, both implemented here rather than left to the
caller:

  * the alarm threshold is CALIBRATED on non-runaway experiments, never picked
    by hand, so every panel and model is compared at one operating point;
  * false alarms are counted per EXPERIMENT, not per hour.  The test split holds
    26 non-runaway experiments spanning 10.7 h, which is far too thin to
    estimate a 1-per-10-hour rate but perfectly adequate as a proportion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ targets
def discrete_hazard_targets(y_time, y_event, horizon_s=1.0):
    """Per-window binary target for a discrete-time hazard head.

    1 when the event falls inside the next `horizon_s`, 0 when the window is
    known to survive it.  Right-censored windows whose censoring time is
    shorter than the horizon carry no information and are masked out -- keeping
    them as zeros would teach the model that censoring means safety.

    Returns (target, sample_weight_mask).
    """
    y_time = np.asarray(y_time, float)
    y_event = np.asarray(y_event).astype(bool)

    target = ((y_event) & (y_time > 0) & (y_time <= horizon_s)).astype(np.float32)
    known = (y_time > horizon_s) | (y_event & (y_time > 0))
    return target, known.astype(np.float32)


def survival_from_hazard(h, steps):
    """S = (1-h)^steps -- risk of onset within `steps` seconds at hazard h."""
    h = np.clip(np.asarray(h, float), 1e-9, 1 - 1e-9)
    return (1.0 - h) ** float(steps)


# -------------------------------------------------------------------- alarm
def first_alarm(times, risk, tau):
    """Time of the first threshold crossing, or None if the run never alarms."""
    hit = np.flatnonzero(np.asarray(risk, float) >= tau)
    return float(np.asarray(times, float)[hit[0]]) if len(hit) else None


def _by_experiment(experiment, *arrays):
    order = np.argsort(experiment, kind="mergesort")
    keys = np.asarray(experiment)[order]
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yield keys[a], idx, [np.asarray(x)[idx] for x in arrays]


def calibrate_threshold(risk, experiment, is_negative, far=0.10):
    """Smallest float64 threshold with empirical calibration FAR <= `far`.

    Calibration experiments must be excluded from model fitting and evaluation.
    Tied peaks are never split: with an inclusive >= alarm, the threshold is
    placed immediately above the first peak that cannot be allowed to alarm.
    This bounds the empirical calibration rate, not an unseen population FAR.
    """
    risk = np.asarray(risk, float)
    experiment = np.asarray(experiment)
    neg = np.asarray(is_negative).astype(bool)
    if not 0 <= far <= 1:
        raise ValueError("far must be between zero and one")
    if risk.ndim != 1 or risk.shape != neg.shape or risk.shape != experiment.shape:
        raise ValueError("risk, experiment and is_negative must be aligned vectors")
    if not neg.any():
        raise ValueError("no non-runaway experiments to calibrate on")
    if not np.isfinite(risk[neg]).all():
        raise ValueError("non-finite calibration risk")

    # each negative experiment is summarised by its own peak risk: it alarms at
    # threshold tau exactly when that peak reaches tau
    peaks = np.array([r[0].max() for _, _, r in
                      _by_experiment(np.asarray(experiment)[neg], risk[neg])])
    peaks = np.sort(peaks)[::-1]
    k = int(np.floor(far * len(peaks)))          # experiments allowed to alarm
    if k >= len(peaks):
        return float(peaks[-1])
    return float(np.nextafter(peaks[k], np.inf))


# ------------------------------------------------------------------ metrics
def lead_times(times, risk, experiment, t_onset, tau):
    """Per-experiment lead time in seconds.

    Positive means the alarm preceded onset.  An experiment that reaches onset
    without ever alarming is a miss and returns NaN -- it must be reported as
    recall, not silently dropped from the lead-time average, which would make a
    model that only fires on easy events look like the best one.
    """
    out = {}
    for key, _, (tt, rr) in _by_experiment(experiment, times, risk):
        onset = t_onset.get(key)
        if onset is None or not np.isfinite(onset):
            continue
        ta = first_alarm(tt, rr, tau)
        out[key] = np.nan if ta is None else float(onset - ta)
    return out


def false_alarm_rate(risk, experiment, is_negative, tau):
    """Fraction of non-runaway experiments raising at least one alarm."""
    neg = np.asarray(is_negative).astype(bool)
    if not neg.any():
        return np.nan
    fired = [float(r[0].max()) >= tau for _, _, r in
             _by_experiment(np.asarray(experiment)[neg], np.asarray(risk)[neg])]
    return float(np.mean(fired))


def bootstrap_ci(values, n=2000, q=(2.5, 97.5), seed=0):
    """Bootstrap CI of the median over EXPERIMENTS, ignoring misses.

    Resampling is over experiments because that is the independent unit; over
    windows it would treat one cell as hundreds of samples.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) < 2:
        # one detection still has a median; it just has no interval
        return (float(v[0]), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(v, size=(n, len(v)), replace=True), axis=1)
    return float(np.median(v)), float(np.percentile(meds, q[0])), \
        float(np.percentile(meds, q[1]))


def summarize(times, risk, experiment, t_onset, t_trigger, is_negative, tau):
    """One operating point, reported the way section 4 of the protocol asks."""
    lead = lead_times(times, risk, experiment, t_onset, tau)
    detected = {k: v for k, v in lead.items() if np.isfinite(v) and v > 0}
    med, lo, hi = bootstrap_ci(list(detected.values()))

    # relative lead time: seconds alone let a trigger with a 13-hour
    # trigger-to-onset interval win by construction
    rel = []
    for k, v in detected.items():
        span = t_onset.get(k, np.nan) - t_trigger.get(k, np.nan)
        if np.isfinite(span) and span > 0:
            rel.append(min(v / span, 1.0))

    return dict(
        tau=float(tau),
        n_events=len(lead),
        event_recall=len(detected) / len(lead) if lead else np.nan,
        lead_median=med, lead_ci_lo=lo, lead_ci_hi=hi,
        lead_relative_median=float(np.median(rel)) if rel else np.nan,
        far=false_alarm_rate(risk, experiment, is_negative, tau),
    )


# ------------------------------------------------------- operating curve (A)
def _running_max(times, risk):
    """Sort by time and return (t, running max of risk).

    An alarm at threshold tau fires at the first time the running max reaches
    tau, so one sorted pass answers every threshold by binary search instead of
    rescanning the series per threshold.
    """
    o = np.argsort(np.asarray(times, float), kind="mergesort")
    t = np.asarray(times, float)[o]
    return t, np.maximum.accumulate(np.asarray(risk, float)[o])


def operating_curve(times, risk, experiment, t_onset, t_trigger,
                    is_negative=None, n_points=80):
    """Lead time, recall and false alarms across the whole threshold range.

    Reporting one operating point requires a negative pool that the model can
    actually discriminate.  #1 has no non-runaway cell, so its threshold has to
    be borrowed from another trigger's negatives -- and a model trained on
    heated cells scores mechanical-indentation negatives near zero, collapsing
    the threshold to ~0.001 against peak risks of 0.97.  At that point lowering
    the threshold only ever helps: lead time, recall and the apparent quality of
    a vaguer model all rise together, with nothing pushing back.

    So the curve is reported instead, and it carries a cost term that needs no
    negative experiments at all: `far_pretrigger`, the fraction of runaway
    experiments whose alarm fires BEFORE the abuse was applied.  Such an alarm
    cannot be a detection -- there was nothing to detect yet -- so it bounds how
    far the threshold can honestly be lowered even inside a dataset that
    contains only runaway cells.
    """
    experiment = np.asarray(experiment)
    tracks = {k: _running_max(t, r) for k, _, (t, r)
              in _by_experiment(experiment, times, risk)}

    ev = [k for k in tracks if np.isfinite(t_onset.get(k, np.nan))]
    neg = ([] if is_negative is None else
           sorted(set(experiment[np.asarray(is_negative).astype(bool)])))
    neg_peaks = np.array([tracks[k][1][-1] for k in neg if k in tracks])

    allr = np.concatenate([r for _, r in tracks.values()])
    taus = np.unique(np.quantile(allr, np.linspace(0, 1, n_points)))[::-1]

    rows = []
    for tau in taus:
        leads, rel, pre = [], [], 0
        for k in ev:
            t, R = tracks[k]
            i = int(np.searchsorted(R, tau, side="left"))
            if i >= len(t):
                continue                      # never alarms
            ta = float(t[i])
            on, tg = t_onset[k], t_trigger.get(k, np.nan)
            if np.isfinite(tg) and ta < tg:
                pre += 1                      # fired before the abuse began
                continue
            if ta >= on:
                continue                      # not an early warning
            leads.append(on - ta)
            if np.isfinite(tg) and on > tg:
                rel.append(min((on - ta) / (on - tg), 1.0))
        rows.append(dict(
            tau=float(tau),
            recall=len(leads) / len(ev) if ev else np.nan,
            lead_median=float(np.median(leads)) if leads else np.nan,
            lead_relative=float(np.median(rel)) if rel else np.nan,
            far_pretrigger=pre / len(ev) if ev else np.nan,
            far_external=(float(np.mean(neg_peaks >= tau))
                          if len(neg_peaks) else np.nan),
        ))
    return pd.DataFrame(rows)


def curve_summary(curve, budget=0.10):
    """One row: the earliest alarm affordable within the false-alarm budget.

    BOTH cost terms bind wherever both are defined.  The pre-trigger term alone
    is far too weak when the fold holds a single test experiment -- it is then
    0 or 1, so any threshold that avoids one early alarm passes, and the summary
    happily returns an operating point at which 79-100% of the negative
    experiments also alarm.  That is not an operating point, it is "always
    alarm", and it drove every panel difference in a first E3-A run to exactly
    zero because the metric had saturated.

    Where no negative pool exists (#1 has no non-runaway cell) only the
    pre-trigger term applies, and the summary should then be read as an upper
    bound rather than an operating point.
    """
    ok = curve[curve["far_pretrigger"] <= budget]
    if curve["far_external"].notna().any():
        ok = ok[ok["far_external"] <= budget]
    ok = ok[np.isfinite(ok["lead_median"])]
    if not len(ok):
        return dict(lead_at_budget=np.nan, recall_at_budget=np.nan,
                    lead_relative_at_budget=np.nan, tau_at_budget=np.nan,
                    far_external_at_budget=np.nan, lead_max=np.nan)
    best = ok.loc[ok["lead_median"].idxmax()]
    return dict(lead_at_budget=float(best["lead_median"]),
                recall_at_budget=float(best["recall"]),
                lead_relative_at_budget=float(best["lead_relative"]),
                tau_at_budget=float(best["tau"]),
                far_external_at_budget=float(best["far_external"]),
                lead_max=float(np.nanmax(curve["lead_median"])))
