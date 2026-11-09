# hazard targets, alarm threshold calibration and per-record alarm helpers
import numpy as np


# 1 when the event falls inside the horizon, with a mask for windows known to survive it
def discrete_hazard_targets(y_time, y_event, horizon_s=1.0):
    y_time = np.asarray(y_time, float)
    y_event = np.asarray(y_event).astype(bool)
    target = ((y_event) & (y_time > 0) & (y_time <= horizon_s)).astype(np.float32)
    known = (y_time > horizon_s) | (y_event & (y_time > 0))
    return target, known.astype(np.float32)


# arrays grouped by record
def by_experiment(experiment, *arrays):
    order = np.argsort(experiment, kind="mergesort")
    keys = np.asarray(experiment)[order]
    bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yield keys[a], idx, [np.asarray(x)[idx] for x in arrays]


# smallest threshold that lets at most a fraction `far` of calibration negatives alarm
def calibrate_threshold(risk, experiment, is_negative, far=0.10):
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
    peaks = np.array([r[0].max() for _, _, r in by_experiment(np.asarray(experiment)[neg], risk[neg])])
    peaks = np.sort(peaks)[::-1]
    k = int(np.floor(far * len(peaks)))
    if k >= len(peaks):
        return float(peaks[-1])
    return float(np.nextafter(peaks[k], np.inf))


# time of the first threshold crossing, or None
def first_alarm(times, risk, tau):
    hit = np.flatnonzero(np.asarray(risk, float) >= tau)
    return float(np.asarray(times, float)[hit[0]]) if len(hit) else None


# fraction of negative records whose peak risk reaches the threshold
def false_alarm_rate(risk, experiment, is_negative, tau):
    neg = np.asarray(is_negative).astype(bool)
    if not neg.any():
        return np.nan
    fired = [float(r[0].max()) >= tau for _, _, r in by_experiment(np.asarray(experiment)[neg], np.asarray(risk)[neg])]
    return float(np.mean(fired))
