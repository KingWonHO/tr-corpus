"""Evaluator V2: one alarm per fold, chosen without looking at the answer.

V1 had two defects that invalidate any number derived from `curve_summary`.

  THRESHOLD CHOSEN ON THE TEST FOLD.  `curve_summary` takes the operating point
  that MAXIMISES lead time on the very experiment being scored.  Measured over
  E1-A, 76 of the 120 (model, panel, cell) combinations that produce a finite
  threshold get a DIFFERENT one for each onset definition -- so "the same alarm,
  reported against a different yardstick" was never the same alarm.

  RELABELLING MOVES EXPERIMENTS INTO BOTH ARMS.  The negative pool is fixed by
  the L2 flag while the event times are relabelled, so an experiment that never
  reaches runaway under L2 can carry a finite ISC time and be counted as an
  event AND as a negative.  158 of #9's 169 L2-unobserved cells have an ISC time
  and 75 have an L3 time; the recall denominator inflates accordingly, which is
  where recall values of 1/159 came from.

V2 fixes both by construction:

    held experiment h
        -> fit on the training pool minus h, minus every calibration negative
        -> tau_h from the calibration negatives alone
        -> score h once, take the FIRST crossing of tau_h: ONE alarm time
        -> every onset definition is measured against that one alarm time
        -> the untouched test negatives report what tau_h actually costs

tau is per fold on purpose.  Each fold fits its own model, so the score scale
differs between folds and forcing a single number across them would be the
artificial choice.  "Fixed alarm" here means fixed WITHIN a held cell: changing
the onset definition cannot move tau, cannot move the alarm time, and cannot
move an experiment between the event and negative arms.

Two modes, because E1/E3 and E11 ask different questions:

  crossfit_loeo     leave one experiment out, tau recalibrated per fold.
                    For E1 and E3, where the question is about panels and
                    labels and every experiment should get a turn as the test.
  frozen_external   one model, one tau, applied once to a held-out source test
                    and then to ARC. For E11, where the question is what a
                    deployment would actually get.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import survival as SV       # noqa: E402
import make_windows as MW   # noqa: E402
import run_e3 as R          # noqa: E402

OUT = R.OUT
BUDGET = 0.10
TARGET = "ds12_arc"

# Onset definitions, as registry columns.  The alarm never moves between them.
LABELS = {"L2@1.0": "t_onset_L2_1.0", "L2@0.5": "t_onset_L2_0.5",
          "L2@2.0": "t_onset_L2_2.0", "L1 venting": "t_onset_L1",
          "L3 voltage": "t_onset_L3", "ISC 25mV": "t_isc"}
PRIMARY = "L2@1.0"


# ------------------------------------------------------------------- the plan
class Plan:
    """Which experiments play which role, fixed before anything is fitted.

    The calibration negatives are held out of TRAINING, not merely out of the
    test fold: a threshold read off experiments the model was fitted on is read
    off scores that have already been driven down, which is the same mistake as
    in-sample calibration and understates the false-alarm rate.
    """

    def __init__(self, d, meta, reg, panels):
        ds = np.asarray(d["experiment"]).astype(str)
        key2split = dict(zip(reg.dataset_id + "/" + reg.experiment_id, reg.split))
        is_src = ~np.char.startswith(ds, TARGET + "/")
        neg = d["y_tr"] == 0
        split = np.array([key2split.get(k, "?") for k in ds])
        observed = np.asarray(d.get("label_observed", np.ones(len(ds), bool))).astype(bool)

        self.ds = ds
        self.observed = observed
        self.cal_neg = np.flatnonzero(is_src & neg & observed & (split == "val"))
        self.test_neg = np.flatnonzero(is_src & neg & observed & (split == "test"))
        reserved = set(ds[self.cal_neg]) | set(ds[self.test_neg])

        elig = None
        for p in panels:
            e = set(MW.panel_experiments(p)["file"].map(
                lambda f: "%s/%s" % (f.split("/")[1], f.split("/")[2][:-8])))
            elig = e if elig is None else (elig & e)
        self.elig = elig & set(ds[is_src])
        # the training pool never contains a calibration or check negative
        self.pool = np.flatnonzero(is_src & observed & np.isin(ds, list(self.elig))
                                   & ~np.isin(ds, list(reserved)))
        # External evaluation uses the saved source training split only.
        # LOEO may pool source events across their original splits, but that
        # policy must not silently carry into the fixed external experiment.
        self.frozen_train = self.pool[split[self.pool] == "train"]
        onset = dict(zip(meta.key, meta.t_onset))
        self.folds = [k for k in sorted(self.elig)
                      if np.isfinite(onset.get(k, np.nan))]
        self.trig = dict(zip(meta.key, meta.t_trigger))
        self.onsets = {name: dict(zip(reg.dataset_id + "/" + reg.experiment_id,
                                      reg[col]))
                       for name, col in LABELS.items() if col in reg.columns}
        self._check()

    def _check(self):
        """Assertions 1 and 2: the three roles are disjoint and cal is unseen."""
        pool, cal, test = (set(self.ds[idx]) for idx in
                          (self.pool, self.cal_neg, self.test_neg))
        assert not (pool & cal), "training pool overlaps the calibration negatives"
        assert not (pool & test), "training pool overlaps the check negatives"
        assert not (cal & test), "calibration and check negatives overlap"
        assert len(cal), "no calibration negatives"
        assert len(test), "no independent check negatives"
        assert len(self.frozen_train), "no source training experiments"

    def train_idx(self, held):
        """Assertion 3: the held experiment never enters its own training fold."""
        return self.pool[self.ds[self.pool] != held]

    def describe(self):
        return ("pool %d windows / %d exp | cal-neg %d win / %d exp | "
                "check-neg %d win / %d exp | %d LOEO folds"
                % (len(self.pool), len(set(self.ds[self.pool])),
                   len(self.cal_neg), len(set(self.ds[self.cal_neg])),
                   len(self.test_neg), len(set(self.ds[self.test_neg])),
                   len(self.folds)))


# ------------------------------------------------------------------ the modes
def crossfit_loeo(d, meta, reg, models, panels, budget=BUDGET, seed=0,
                  verbose=True, horizon=R.HORIZON_S, curve_dir=None):
    """LOEO with a fold-specific, source-calibrated threshold.

    Returns one row per (model, panel, held, onset definition).  `tau` and
    `alarm_time` are constant within a (model, panel, held) group by
    construction -- assertions 4, 5 and 6 -- so any variation across labels is
    variation in the yardstick and nothing else.
    """
    plan = Plan(d, meta, reg, panels)
    if verbose:
        print("  " + plan.describe())
    rows = []
    for panel in panels:
        cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
        for held in plan.folds:
            te = np.flatnonzero((plan.ds == held) & plan.observed)
            tr = plan.train_idx(held)
            if not len(te) or not len(tr):
                continue
            for m in models:
                r_te, r_cal, r_chk = R.run_fold(
                    d, cols, m, tr, [te, plan.cal_neg, plan.test_neg], seed, horizon)

                # tau sees the calibration negatives and nothing else
                tau = SV.calibrate_threshold(
                    r_cal, d["experiment"][plan.cal_neg],
                    np.ones(len(plan.cal_neg), bool), far=budget)

                # one alarm, taken once, before any onset is consulted
                order = np.argsort(d["t_end"][te], kind="mergesort")
                t_h, r_h = d["t_end"][te][order], r_te[order]
                alarm = SV.first_alarm(t_h, np.maximum.accumulate(r_h), tau)

                # what that threshold actually costs, reported not optimised
                far_chk = SV.false_alarm_rate(
                    r_chk, d["experiment"][plan.test_neg],
                    np.ones(len(plan.test_neg), bool), tau)
                tg = plan.trig.get(held, np.nan)
                pre = bool(alarm is not None and np.isfinite(tg) and alarm < tg)
                if curve_dir is not None:
                    # Per-fold traces for a detection-FAR curve at matched
                    # check FAR (review 2026-09-24): the held cell's running-
                    # maximum risk and the per-experiment peak risks of the
                    # calibration and check negatives.  Nothing here changes
                    # the alarm above.
                    import os
                    os.makedirs(curve_dir, exist_ok=True)
                    cal_peak = pd.Series(np.asarray(r_cal, float)).groupby(
                        d["experiment"][plan.cal_neg]).max()
                    chk_peak = pd.Series(np.asarray(r_chk, float)).groupby(
                        d["experiment"][plan.test_neg]).max()
                    np.savez_compressed(
                        os.path.join(curve_dir, "%s_%s_s%d_%s.npz" % (m, panel, seed, held.replace("/", "__"))),
                        t=t_h, risk_running_max=np.maximum.accumulate(np.asarray(r_h, float)),
                        tau=np.float64(tau), t_trigger=np.float64(tg),
                        cal_peaks=cal_peak.to_numpy(float), cal_keys=np.asarray(cal_peak.index, str),
                        check_peaks=chk_peak.to_numpy(float), check_keys=np.asarray(chk_peak.index, str))

                for name, onset in plan.onsets.items():
                    on = onset.get(held, np.nan)
                    if not np.isfinite(on):
                        continue          # this cell has no such event: skip,
                        # never relabel it into the negative arm
                    det = alarm is not None and alarm < on and not pre
                    rows.append(dict(
                        model=m, panel=panel, held=held, label=name,
                        tau=tau, alarm_time=alarm, t_trigger=tg,
                        pre_trigger=pre, onset=float(on),
                        lead=float(on - alarm) if det else np.nan,
                        detected=bool(det), far_check=far_chk,
                        risk_max=float(r_h.max())))
        if verbose:
            print("    %s done (%d rows)" % (panel, len(rows)))
    res = pd.DataFrame(rows)
    _assert_one_alarm(res)
    return res


def _assert_one_alarm(res):
    """Assertions 4-6: one tau and one alarm per cell, shared by every label."""
    if not len(res):
        return
    g = res.groupby(["model", "panel", "held"])
    bad_tau = g["tau"].nunique()
    assert (bad_tau <= 1).all(), "tau varies within a fold: %s" % \
        bad_tau[bad_tau > 1].to_dict()
    bad_alarm = g["alarm_time"].nunique(dropna=False)
    assert (bad_alarm <= 1).all(), "alarm time varies within a fold: %s" % \
        bad_alarm[bad_alarm > 1].to_dict()


def frozen_external(d, meta, reg, models, panel, budget=BUDGET, seed=0,
                    verbose=True):
    """One fit, one threshold, applied once to a source test set and to ARC.

    This is the deployment question and it has to be answered with a single
    frozen operating point: anything recalibrated per target experiment is
    reporting a number the deployment could not have had.
    """
    plan = Plan(d, meta, reg, [panel])
    ds = plan.ds
    is_t = np.char.startswith(ds, TARGET + "/")
    cols = np.flatnonzero(MW.panel_mask(d["features"], panel))
    tgt = np.flatnonzero(is_t & plan.observed)
    onset_p = plan.onsets[PRIMARY]
    if verbose:
        print("  " + plan.describe() + " | ARC %d exp" % len(set(ds[tgt])))

    rows = []
    for m in models:
        r_chk, r_cal, r_tgt = R.run_fold(
            d, cols, m, plan.frozen_train, [plan.test_neg, plan.cal_neg, tgt], seed)
        tau = SV.calibrate_threshold(
            r_cal, d["experiment"][plan.cal_neg],
            np.ones(len(plan.cal_neg), bool), far=budget)

        far_src = SV.false_alarm_rate(r_chk, d["experiment"][plan.test_neg],
                                      np.ones(len(plan.test_neg), bool), tau)
        arc_neg = d["y_tr"][tgt] == 0
        far_arc = SV.false_alarm_rate(r_tgt, d["experiment"][tgt], arc_neg, tau)

        leads, det = [], 0
        ev = [k for k in sorted(set(ds[tgt])) if np.isfinite(onset_p.get(k, np.nan))]
        for k in ev:
            sel = np.flatnonzero(ds[tgt] == k)
            o = np.argsort(d["t_end"][tgt][sel], kind="mergesort")
            a = SV.first_alarm(d["t_end"][tgt][sel][o],
                               np.maximum.accumulate(r_tgt[sel][o]), tau)
            tg = plan.trig.get(k, np.nan)
            pre = a is not None and np.isfinite(tg) and a < tg
            if a is not None and a < onset_p[k] and not pre:
                leads.append(onset_p[k] - a)
                det += 1
        rows.append(dict(model=m, panel=panel, tau=tau,
                         n_arc_events=len(ev), n_detected=det,
                         recall=det / len(ev) if ev else np.nan,
                         lead_median=float(np.median(leads)) if leads else np.nan,
                         far_source_test=far_src, far_arc=far_arc))
        if verbose:
            print("    %-9s tau %.4g | recall %d/%d lead %s | FAR src %.3f arc %.3f"
                  % (m, tau, det, len(ev),
                     ("%.0fs" % np.median(leads)) if leads else "--",
                     far_src, far_arc))
    return pd.DataFrame(rows)
