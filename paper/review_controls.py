"""Re-scoring analyses added at the 2026-09-24 review (no model is fitted).

    paper/tables/S5c_d6_discrimination.{csv,md}
        Record-level discrimination on D6 from the stored risks: for every
        frozen model and seed, the maximum risk before L2 of each runaway
        record against the maximum risk of each non-runaway record over the
        full record and over an exposure matched to the positives.  Reports
        the AUC (probability that a random runaway record scores above a
        random non-runaway record) and the recall reachable at record-level
        false-alarm rates of 0, 0.10 and 0.20 when the threshold is free.
        Written because a narrow risk band does not by itself show that the
        ranking is uninformative (review item 2.10).
    paper/tables/S11b_d8_alarm_units.{csv,md}
        The D8 alarm counts in three units: vehicles with any exceedance,
        charging segments with any exceedance (one segment is one charging
        session, so it is the natural alarm episode), and the fraction of
        windows in an alarm state; plus the per-vehicle exposure spread
        (review item 2.9).

Sources: tr-corpus/results/validation_20260925_native and
validation_20260925_seq (predictions/*.npz), external_20260925_d8_field.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "trbench"), os.path.join(ROOT, "paper")]
import claims as C                      # noqa: E402
from make_figures import save_table, rng, MODEL_NAME   # noqa: E402

LEARNED = list(C.LEARNED)
D8 = os.path.join(ROOT, "tr-corpus", "results", "external_20260925_d8_field")


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def recall_at_far(pos, neg, far):
    """Recall when the threshold is the smallest value that keeps the record-level
    FAR on `neg` at or below `far` (free threshold; the deployment could not have it)."""
    neg = np.sort(np.asarray(neg, float))[::-1]
    k = int(np.floor(far * len(neg)))          # alarms allowed
    tau = np.nextafter(neg[k], np.inf) if k < len(neg) else -np.inf
    return float((np.asarray(pos, float) >= tau).mean())


def d6_discrimination():
    d, meta, reg, plan = C.load_meta()
    st = C.settings()
    st = st[st.run.isin(C.MAIN_RUNS) & (st.representation == "no_age") & st.model.isin(LEARNED)]
    onset = plan.onsets["L2@1.0"]
    rows = []
    for s in st.itertuples():
        z = np.load(s.file)
        idx, risk = z["arc_idx"], np.asarray(z["arc_risk"], float)
        e, t = d["experiment"][idx], d["t_end"][idx]
        pos_max, pre_span, neg_full, neg_rec = {}, {}, {}, {}
        for key in np.unique(e):
            sel = e == key
            tt, rr = t[sel], risk[sel]
            on = onset.get(key, np.nan)
            if np.isfinite(on):
                pre = tt < on
                if pre.any():
                    pos_max[key] = rr[pre].max()
                    pre_span[key] = float(on - tt.min())
            else:
                neg_full[key] = rr.max()
                neg_rec[key] = (tt - tt.min(), rr)
        spans = np.array(list(pre_span.values()))
        pos = np.array(list(pos_max.values()))
        negf = np.array(list(neg_full.values()))
        row = dict(run=s.run, model=s.model, seed=s.seed, tau=s.tau_stored,
                   n_pos=len(pos), n_neg=len(negf),
                   pos_preonset_span_median_s=float(np.median(spans)),
                   pos_preonset_span_min_s=float(spans.min()),
                   pos_max_risk_median=float(np.median(pos)),
                   neg_max_risk_full_median=float(np.median(negf)),
                   auc_full=auc(pos, negf))
        for name, T in (("matched_median", float(np.median(spans))),):
            negm = np.array([rr[tt <= T].max() for tt, rr in neg_rec.values()])
            row["neg_max_risk_%s_median" % name] = float(np.median(negm))
            row["auc_%s" % name] = auc(pos, negm)
            for far in (0.0, 0.10, 0.20):
                row["recall_at_far%.2f_%s" % (far, name)] = recall_at_far(pos, negm, far)
        # Pairwise exposure matching: runaway record i is compared with every
        # non-runaway record scored over the same span T_i that record i had
        # before its onset, so neither side of a pair has watched longer.
        keys = list(pos_max)
        gt = eq = 0.0
        for k in keys:
            T = pre_span[k]
            negm = np.array([rr[tt <= T].max() for tt, rr in neg_rec.values()])
            gt += (pos_max[k] > negm).sum()
            eq += (pos_max[k] == negm).sum()
        n_pairs = len(keys) * len(neg_rec)
        row["auc_pairwise_matched"] = float((gt + 0.5 * eq) / n_pairs)
        for far in (0.0, 0.10, 0.20):
            row["recall_at_far%.2f_full" % far] = recall_at_far(pos, negf, far)
        # what the frozen threshold actually did, for reference
        row["recall_frozen_tau"] = float((pos >= s.tau_stored).mean())
        row["far_full_frozen_tau"] = float((negf >= s.tau_stored).mean())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["model", "seed"])
    df.to_csv(os.path.join(ROOT, "paper", "tables", "S5c_d6_discrimination_cells.csv"), index=False)
    out = []
    for m in LEARNED:
        g = df[df.model == m]
        out.append({"model": MODEL_NAME[m],
                    "AUC, full record": rng(g.auc_full, "%.2f"),
                    "AUC, negatives cut at median pre-onset span": rng(g.auc_matched_median, "%.2f"),
                    "AUC, pairwise exposure-matched": rng(g.auc_pairwise_matched, "%.2f"),
                    "recall at record FAR 0 / 0.10 / 0.20 (matched, median)": " / ".join(
                        rng(g["recall_at_far%.2f_matched_median" % f], "%.2f") for f in (0.0, 0.10, 0.20)),
                    "recall at record FAR 0 / 0.10 / 0.20 (full)": " / ".join(
                        rng(g["recall_at_far%.2f_full" % f], "%.2f") for f in (0.0, 0.10, 0.20)),
                    "frozen tau: recall / full-record FAR": "%s / %s" % (rng(g.recall_frozen_tau, "%.2f"), rng(g.far_full_frozen_tau, "%.2f"))})
    med = df.pos_preonset_span_median_s.iloc[0]
    mn = df.pos_preonset_span_min_s.iloc[0]
    n_pos, n_neg = int(df.n_pos.iloc[0]), int(df.n_neg.iloc[0])
    save_table(pd.DataFrame(out), "S5c_d6_discrimination",
               "Sources: validation_20260925_native and validation_20260925_seq predictions, sensor values only; ranges over seeds 0–4. "
               "Score of a runaway record: maximum risk over its windows before L2; score of a non-runaway record: maximum risk over the full record "
               "or over its first %.0f s (median pre-onset span of the %d runaway records; shortest %.0f s); in the pairwise column each runaway "
               "record is compared with every non-runaway record scored over that runaway record's own pre-onset span. AUC is the probability that a "
               "runaway record outscores a non-runaway record (ties count one half). Recall at a record-level FAR uses the best threshold in hindsight "
               "on the same %d negatives; it bounds what any threshold could achieve on these records and is not a deployable number." % (med, n_pos, mn, n_neg))
    return df


def d8_alarm_units():
    pv = pd.read_csv(os.path.join(D8, "per_vehicle.csv"))
    ar = pd.read_csv(os.path.join(D8, "alarm_rates.csv"))
    hours = ar.exposure_h.iloc[0]
    rows = []
    for (arm, m), g in ar.groupby(["arm", "model"], sort=False):
        rows.append({"arm": arm, "model": MODEL_NAME.get(m, m), "seeds": len(g),
                     "vehicles with any exceedance (of 292)": rng(g.vehicles_alarm, "%d"),
                     "segments with any exceedance (of 8,611)": rng(g.snippets_alarm, "%d"),
                     "segments with an exceedance in their first 300 s": rng(g.snippets_alarm_head, "%d"),
                     "alarm episodes per 1,000 vehicle-h": rng(g.snippets_alarm / hours * 1000.0, "%.1f"),
                     "exceeding windows per 1,000 vehicle-h": rng(g.alarms_per_1000_vehicle_h, "%.0f"),
                     "fraction of windows in alarm state": rng(g.windows_alarm / g.windows, "%.3f")})
    one = pv[(pv.arm == "no_age") & (pv.model == "rule")]
    seg = one.snippets
    exp_h = one.exposure_s / 3600.0
    save_table(pd.DataFrame(rows), "S11b_d8_alarm_units",
               "Source: external_20260925_d8_field/alarm_rates.csv; %d normal vehicles, %d charging segments, %.0f vehicle-hours scored "
               "(the release holds 201,256 h; the scored sample is <= 30 segments per vehicle). One segment is one charging session of 128 samples at 10 s, "
               "so consecutive exceedances inside a segment form one episode and no merge rule across segments is needed. Per vehicle: %d–%d segments "
               "(median %d; %d of %d vehicles contribute 30) and %.1f–%.1f h (median %.1f h). Sequence models under the main representation are "
               "in an alarm state for a third of all windows when they alarm at all; the trees, when they alarm, do so in a few windows per segment."
               % (len(one), int(seg.sum()), hours, seg.min(), seg.max(), int(seg.median()), int((seg == 30).sum()), len(one),
                  exp_h.min(), exp_h.max(), exp_h.median()))
    per = pd.DataFrame(dict(segments=seg.describe(), exposure_h=exp_h.describe()))
    per.to_csv(os.path.join(ROOT, "paper", "tables", "S11b_d8_per_vehicle_exposure.csv"))
    return rows


if __name__ == "__main__":
    d8_alarm_units()
    df = d6_discrimination()
    cols = ["model", "seed", "auc_full", "auc_matched_median", "auc_pairwise_matched",
            "recall_at_far0.10_matched_median", "recall_frozen_tau", "far_full_frozen_tau"]
    print(df[cols].to_string(index=False))
