"""Reproducible E9/E11 supplement: independent calibration, equal exposure.

No target tuning. Each model/seed/age setting is fitted once on source train;
source validation negatives choose the threshold. Saved source test and ARC
are evaluation only. Short records are explicitly ineligible for fixed-length
head/tail windows. Seed repetitions are not additional independent cells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalv2 as V
import make_windows as MW
import deep as DEEP
import run_e3 as R
import survival as SV

HORIZONS = (300, 900, 1800, 3600)
REPRESENTATIONS = {"no_age", "with_age", "surface_max_only", "surface_mean_only",
                   "mask_only", "age_only"}
# Shortcut controls (2026-09-24, reviewer request): `mask_only` gives a model
# the channel-availability pattern of the M1 panel and no sensor value at all;
# `age_only` gives it the per-channel observation ages and nothing else.  Any
# separation they achieve is a domain cue, not battery physics.


def wilson(k, n):
    if not n:
        return np.nan, np.nan
    z = norm.ppf(.975)
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n)) / den
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0., mid-half), min(1., mid+half)


def negative_records(d, idx, risk, tau, window_s=None, anchor="tail"):
    """One cell per row; fixed windows require the entire requested exposure.

    Head/tail windows start at the first/last available prediction time, after
    the model's context warm-up. The same 10 s cadence applies to both pools.
    Events at either boundary count. A silent cell is right-censored, never
    assigned an invented first-alarm time.
    """
    if anchor not in ("head", "tail"):
        raise ValueError("anchor must be head or tail")
    e, t = d["experiment"][idx], d["t_end"][idx]
    rows = []
    for key in np.unique(e):
        sel = e == key
        order = np.argsort(t[sel], kind="stable")
        tt, rr = t[sel][order], np.asarray(risk)[sel][order]
        span = float(tt[-1] - tt[0])
        eligible = window_s is None or span >= window_s
        start, end = float(tt[0]), float(tt[-1])
        if window_s is not None and eligible:
            if anchor == "tail":
                start = end - window_s
            else:
                end = start + window_s
        a, peak = None, np.nan
        if eligible:
            keep = (tt >= start) & (tt <= end)
            a = SV.first_alarm(tt[keep], rr[keep], tau)
            peak = float(np.max(rr[keep]))
        rows.append(dict(held=str(key), eligible=eligible,
                         record_span_s=span, exposure_s=end-start if eligible else np.nan,
                         peak_risk=peak, fired=bool(a is not None) if eligible else None,
                         first_alarm_s=float(a-start) if a is not None else np.nan,
                         followup_s=float(a-start) if a is not None else
                         (end-start if eligible else np.nan)))
    return pd.DataFrame(rows)


def rate_summary(records):
    use = records[records.eligible]
    n = len(use)
    k = int(use.fired.eq(True).sum())
    lo, hi = wilson(k, n)
    return dict(n_total=len(records), n_eligible=n, n_excluded=len(records)-n,
                n_alarm=k, far=k/n if n else np.nan, far_ci_lo=lo, far_ci_hi=hi,
                exposure_min_s=use.exposure_s.min(), exposure_max_s=use.exposure_s.max(),
                first_alarm_conditional_median_s=use.first_alarm_s.median())


def positive_records(d, idx, risk, tau, plan):
    e, t = d["experiment"][idx], d["t_end"][idx]
    rows = []
    for key in np.unique(e):
        on = plan.onsets[V.PRIMARY].get(key, np.nan)
        if not np.isfinite(on):
            continue
        sel = e == key
        order = np.argsort(t[sel], kind="stable")
        tt, rr = t[sel][order], risk[sel][order]
        a = SV.first_alarm(tt, rr, tau)
        tg = plan.trig.get(key, np.nan)
        pre = a is not None and np.isfinite(tg) and a < tg
        det = a is not None and a < on and not pre
        rows.append(dict(held=str(key), onset=on, alarm_time=a,
                         pre_trigger=pre, detected=det,
                         lead=on-a if det else np.nan,
                         n_preonset_windows=int((tt < on).sum())))
    return pd.DataFrame(rows)


def median_ci(values, rng, n_boot):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    b = np.median(rng.choice(v, (n_boot, len(v)), replace=True), axis=1)
    return tuple(np.quantile(b, [.025, .975]))


def summary_features(d, cols):
    """Bound temporary arrays when full native records are long."""
    return np.concatenate([R.window_features(d["X"][i:i+2048],
                          d["mask"][i:i+2048], cols)
                          for i in range(0, len(d["X"]), 2048)])


def representation_mask(names, mode):
    """Predeclared M1 representations for domain-shortcut sensitivity."""
    if mode not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {mode!r}")
    names = np.asarray(names).astype(str)
    if mode == "surface_max_only":
        return names == "T_surface_max"
    if mode == "surface_mean_only":
        return names == "T_surface_mean"
    select = MW.panel_mask(names, "M1")
    age = np.array([s.startswith("age__") or s == MW.AGE_CH for s in names])
    if mode == "age_only":
        return select & age
    if mode == "mask_only":
        return select & ~age
    if mode == "no_age":
        select &= np.array([not s.startswith("age__") and s != MW.AGE_CH
                            for s in names])
    return select


def run(d, meta, reg, models, seeds, age_modes, out, n_boot=1000, budget=.1,
        horizon=R.HORIZON_S):
    plan = V.Plan(d, meta, reg, ["M1"])
    target = np.char.startswith(plan.ds, V.TARGET + "/")
    src_pos = np.flatnonzero((d["split"] == "test") & ~target & (d["y_tr"] == 1) & plan.observed)
    arc = np.flatnonzero(target & plan.observed)
    arc_neg = np.flatnonzero(target & (d["y_tr"] == 0) & plan.observed)
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], horizon)
    membership = []
    for role, idx in [("train", plan.frozen_train), ("calibration", plan.cal_neg),
                      ("source_test_negative", plan.test_neg),
                      ("source_test_positive", src_pos), ("arc", arc)]:
        membership.extend(dict(role=role, held=k) for k in np.unique(plan.ds[idx]))
    pd.DataFrame(membership).to_csv(out / "split_membership.csv", index=False)
    rates, all_neg, all_pos, positive_summaries, comparisons, calibration = [], [], [], [], [], []
    for age_mode in age_modes:
        names = np.array(d["features"])
        select = representation_mask(names, age_mode)
        age_cols = np.array([s.startswith("age__") or s == MW.AGE_CH for s in names])
        if age_mode == "no_age":
            select &= ~age_cols
        cols = np.flatnonzero(select)
        F = summary_features(d, cols)
        # window_features lays out nine blocks of len(cols): eight value
        # statistics and, last, the per-channel availability fraction.
        if age_mode == "mask_only":
            F = F[:, 8 * len(cols):]
        elif age_mode == "age_only":
            F = F[:, :8 * len(cols)]
        for model in models:
            # The physical rule must not differentiate the observation clock.
            if model == "rule" and age_mode != age_modes[0]:
                continue
            if model == "rule":
                Fm = summary_features(d, np.flatnonzero(select & ~age_cols))
            else:
                Fm = F
            for seed in (seeds[:1] if model == "rule" else seeds):
                started = time.time()
                if model in DEEP.SEQ_MODELS:
                    # Sequence models consume the window tensor, not the summary
                    # features; everything downstream is identical.
                    Xs, Ms = d["X"][:, :, cols], d["mask"][:, :, cols]
                    if age_mode == "mask_only":
                        Xs = np.zeros_like(Xs)
                    predict_seq = DEEP.fit_seq(model, Xs[plan.frozen_train], Ms[plan.frozen_train],
                                               y[plan.frozen_train], w[plan.frozen_train], seed)
                    score = lambda idx: np.asarray(predict_seq(Xs[idx], Ms[idx]), float)
                else:
                    predict = R.fit_model(model, Fm[plan.frozen_train],
                                          y[plan.frozen_train], w[plan.frozen_train], seed)
                    score = lambda idx: predict(Fm[idx])
                r_cal = score(plan.cal_neg)
                r_src = score(plan.test_neg)
                r_arc = score(arc)
                r_src_pos = score(src_pos)
                r_arc_neg = r_arc[(d["y_tr"][arc] == 0)]
                tau = SV.calibrate_threshold(r_cal, d["experiment"][plan.cal_neg],
                                            np.ones(len(r_cal), bool), budget)
                tag = dict(model=model, seed=seed, age_mode=age_mode, tau=tau)
                # Per-window predictions permit later analysis without refitting.
                pred_dir = out / "predictions"
                pred_dir.mkdir(exist_ok=True)
                np.savez_compressed(pred_dir / f"{model}_{age_mode}_s{seed}.npz",
                                    # Store risks as float64 so reloading cannot
                                    # round the nextafter threshold back onto a
                                    # tied float32 score.
                                    cal_idx=plan.cal_neg, cal_risk=np.asarray(r_cal, float),
                                    source_idx=plan.test_neg, source_risk=np.asarray(r_src, float),
                                    arc_idx=arc, arc_risk=np.asarray(r_arc, float),
                                    source_positive_idx=src_pos,
                                    source_positive_risk=np.asarray(r_src_pos, float),
                                    tau=np.float64(tau))
                # Preserve the float64 nextafter boundary in every comparison.
                # A float32 array would round a Python float threshold back to
                # float32 under NumPy's scalar promotion rules.
                cal_peaks = pd.Series(r_cal).groupby(plan.ds[plan.cal_neg]).max().to_numpy(dtype=float)
                rng = np.random.default_rng(20260914 + seed)
                boot_peaks = rng.choice(cal_peaks, (n_boot, len(cal_peaks)), replace=True)
                k = int(np.floor(budget * len(cal_peaks)))
                boot_tau = np.nextafter(np.sort(boot_peaks, axis=1)[:, ::-1][:, k], np.inf)
                calibration.append(dict(**tag, n_cal=len(cal_peaks),
                    n_cal_alarm=int((cal_peaks >= tau).sum()),
                    empirical_far=float((cal_peaks >= tau).mean()),
                    tau_boot_lo=float(np.quantile(boot_tau, .025)),
                    tau_boot_hi=float(np.quantile(boot_tau, .975))))
                for pool, idx, risk in [("source_test", src_pos, r_src_pos),
                                        ("arc", arc, r_arc)]:
                    pr = positive_records(d, idx, risk, tau, plan)
                    pr = pr.assign(**tag, pool=pool)
                    all_pos.append(pr)
                    nd = int(pr.detected.sum())
                    lo, hi = wilson(nd, len(pr))
                    llo, lhi = median_ci(pr.lead, rng, n_boot)
                    context = pr.n_preonset_windows > 0
                    n_context = int(context.sum())
                    nd_context = int(pr.loc[context, "detected"].sum())
                    clo, chi = wilson(nd_context, n_context)
                    positive_summaries.append(dict(**tag, pool=pool, n_events=len(pr),
                        n_detected=nd, recall=nd/len(pr) if len(pr) else np.nan,
                        recall_ci_lo=lo, recall_ci_hi=hi,
                        lead_median=pr.lead.median(), lead_ci_lo=llo, lead_ci_hi=lhi,
                        n_no_preonset_context=int((~context).sum()),
                        n_context_eligible=n_context, n_detected_context_eligible=nd_context,
                        recall_context_eligible=(nd_context/n_context if n_context else np.nan),
                        recall_context_ci_lo=clo, recall_context_ci_hi=chi))
                for horizon, anchor in [(None, "full")] + [(h, a) for h in HORIZONS
                                                                         for a in ("head", "tail")]:
                    cache = {}
                    for pool, idx, risk in [("source_mech", plan.test_neg, r_src),
                                           ("arc", arc_neg, r_arc_neg)]:
                        nr = negative_records(d, idx, risk, tau, horizon,
                                               "tail" if anchor == "full" else anchor)
                        label = "full" if horizon is None else horizon
                        stats = rate_summary(nr)
                        nr = nr.assign(**tag, pool=pool, window_s=label, anchor=anchor)
                        all_neg.append(nr)
                        peaks = nr.loc[nr.eligible, "peak_risk"].to_numpy()
                        boot_far = (peaks[None, :] >= boot_tau[:, None]).mean(axis=1) if len(peaks) else np.full(n_boot, np.nan)
                        blo, bhi = (np.quantile(boot_far, [.025, .975]) if len(peaks) else (np.nan, np.nan))
                        rates.append(dict(**tag, pool=pool, window_s=label, anchor=anchor,
                            **stats, far_cal_boot_lo=blo, far_cal_boot_hi=bhi))
                        cache[pool] = stats
                    s, a = cache["source_mech"], cache["arc"]
                    if s["n_eligible"] and a["n_eligible"]:
                        # Newcombe interval for difference of independent proportions.
                        delta = a["far"] - s["far"]
                        lo = delta - np.sqrt((a["far"]-a["far_ci_lo"])**2 + (s["far_ci_hi"]-s["far"])**2)
                        hi = delta + np.sqrt((a["far_ci_hi"]-a["far"])**2 + (s["far"]-s["far_ci_lo"])**2)
                        p = fisher_exact([[a["n_alarm"], a["n_eligible"]-a["n_alarm"]],
                                          [s["n_alarm"], s["n_eligible"]-s["n_alarm"]]]).pvalue
                    else:
                        delta = lo = hi = p = np.nan
                    comparisons.append(dict(**tag, window_s=label, anchor=anchor,
                        n_source=s["n_eligible"], n_arc=a["n_eligible"],
                        far_difference=delta, difference_ci_lo=lo, difference_ci_hi=hi,
                        fisher_p_exploratory=p))
                print(f"{model} seed={seed} {age_mode}: tau={tau:.8g}, {time.time()-started:.1f}s", flush=True)
                # Checkpoint after each complete configuration. Unique output dir
                # prevents partial runs overwriting previous experiment versions.
                for name, data in [("negative_rates", rates), ("positive_summary", positive_summaries),
                                   ("negative_comparisons", comparisons), ("calibration", calibration)]:
                    pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
                pd.concat(all_neg, ignore_index=True).to_csv(out / "negative_cells.csv", index=False)
                pd.concat(all_pos, ignore_index=True).to_csv(out / "positive_cells.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="W60_native")
    ap.add_argument("--models", default="rule,xgboost,lightgbm")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ages", default="no_age,with_age")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=float, default=R.HORIZON_S,
                    help="training horizon in seconds for the hazard target (default 60)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"use a fresh output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if set(a.models.split(",")) - ({"rule", "xgboost", "lightgbm"} | set(DEEP.SEQ_MODELS)):
        raise ValueError("models: rule, xgboost, lightgbm or a sequence model (%s)" % ",".join(DEEP.SEQ_MODELS))
    if set(a.ages.split(",")) - REPRESENTATIONS:
        raise ValueError("--ages must use: " + ",".join(sorted(REPRESENTATIONS)))
    d, meta = R.load_windows(a.windows, splits=("train", "val", "test", "test_zeroshot"))
    reg_path = Path(R.OUT) / "registry" / "experiments.csv"
    reg = pd.read_csv(reg_path)
    window_base = Path(R.OUT) / "windows" / a.windows
    manifest = dict(arguments=vars(a), git_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE.parent, text=True).strip(),
        registry_sha256=hashlib.sha256(reg_path.read_bytes()).hexdigest(),
        code_sha256={str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(HERE.rglob("*.py"))},
        environment_lock_sha256=hashlib.sha256((HERE.parent / "uv.lock").read_bytes()).hexdigest(),
        window_artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(window_base.glob("*.npz"))},
        analysis_status="exploratory supplement; target previously inspected",
        ci_scope="cells conditional on fitted model; calibration bootstrap reported separately",
        seeds_are_not_additional_cells=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run(d, meta, reg, a.models.split(","), [int(s) for s in a.seeds.split(",")],
        a.ages.split(","), out, n_boot=a.bootstrap, horizon=a.horizon)


if __name__ == "__main__":
    main()
