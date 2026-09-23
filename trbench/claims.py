"""Claim assessment for thermal-runaway early-warning results.

Implements reports/12_claim_assessment_protocol.md (committed before this code
was run).  No model is fitted: the stored window-level risks of the
held-dataset runs and the re-scored E3-A alarm rows are re-used.

    uv run python trbench/claims.py all
    uv run python trbench/claims.py report --H 300 --alpha 0.10 --events L2@1.0 --lead 60

Sub-commands: audit (support audit), profile (lead-requirement profiles, C),
support (post-hoc: cells in which a lead >= ell is observable at all),
rules (what the verdict rules return as a function of pool size, true rate and
clustering - a check that they discriminate, since every real pool here stops at rule 2),
segments (record-level first alarm vs alarm inside a watched window),
calibrate (fixed-horizon calibration, A), pairs (representation-pair
disagreement, B), report (one claim), all.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import beta, betabinom, binom

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_validation as RV   # noqa: E402  (scoring rules reused, not re-implemented)
import survival as SV         # noqa: E402

RES = ROOT / "tr-corpus" / "results"
OUT = RES / "claims_20260921"
WIN = ROOT / "tr-corpus" / "windows" / "W60_native"
SPLITS = ("train", "val", "test", "test_zeroshot")      # order of the stored indices
RUNS = {"native_final": RES / "validation_20260914_native_final",
        "common_surface": RES / "validation_20260914_common_surface",
        # the four sequence models on the same build and representations (protocol 20)
        "seq_matched": RES / "validation_20260921_seq_matched"}
MAIN_RUNS = ("native_final", "seq_matched")          # sensor values only, source max
LEARNED = ("xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer")
SEQ = ("gru", "mamba", "itransformer", "convtransformer")
# Internal design: the frozen tree files re-scored together with the matched
# sequence-model run, so the tree rows are byte-identical to the 2026-09-15 set.
E3A = [RES / "validation_20260921_all7_rescored" / f
       for f in ("validation_20260914_e3a_native.csv", "validation_20260914_e3a_native_xgb_s1to4.csv",
                 "validation_20260914_e3a_native_lgb_s1to4.csv", "e3a_seq_s0to4.csv")]
TARGET = "ds12_arc"

ALPHA = 0.10
H_PRIMARY = 300
H_SENS = (900, 1800)
H_AUDIT = (300, 900, 1800, 3600, 28800)
LEADS = (0, 30, 60, 120)
EVENT_SETS = {"L2@1.0": ["L2@1.0"],
              "E_TR": ["L2@0.5", "L2@1.0", "L2@2.0"],
              "E_elec": ["L3 voltage", "ISC 25mV"],
              "E_vent": ["L1 venting"]}

FROZEN = {
    "tr-corpus/results/validation_20260921_all7_rescored/validation_20260914_e3a_native.csv": "cd0cb4c826fa841ba6f0965636c94efee047cf954f67887b8fa0100fc8b22061",
    "tr-corpus/results/validation_20260921_all7_rescored/validation_20260914_e3a_native_xgb_s1to4.csv": "e7e58dde9577f51e5ebdc1c99c376780991fe0561d85cef78836d60d4d3b5ddb",
    "tr-corpus/results/validation_20260921_all7_rescored/validation_20260914_e3a_native_lgb_s1to4.csv": "300b3023a1db9b2753817f77a3069a2f13019b66622399e85c1b379bac7fad3f",
    "tr-corpus/results/validation_20260921_all7_rescored/e3a_seq_s0to4.csv": "f234bf3ffef7b6148846382b5665020080e6436fb818caf47c0df7627782e98a",
    "tr-corpus/registry/experiments.csv": "25f303024fa660cdb31fc2fef1db9515a30917a31223b7d2121778f8ad984e3d",
    "tr-corpus/splits/split_assignment.csv": "3109d69118e8fc0db0bc94857894a7e93475538f7d29377a014dbe1ee78a5ff6",
    "tr-corpus/results/validation_20260914_native_final/calibration.csv": "77a16effd747d45ca13df1f2c4095442807f495594fb3d4987e546a0b89a720f",
    "tr-corpus/results/validation_20260914_common_surface/calibration.csv": "3ecf5cae83db1bad8fbf48bc824527715ee3d7490c30d8114e22de9aad593422",
    "tr-corpus/windows/W60_native/val.npz": "c7dc9c08695ce5e2e774c7ba136f08e1383e493feb69b268ace17f8680b1c125",
    "tr-corpus/windows/W60_native/test.npz": "537ed728222e1c37e8fa62a66dc6f6d20a672af93b0456ce0b471240ac81e046",
    "tr-corpus/windows/W60_native/test_zeroshot.npz": "86e8d3a6f521437751088bbfb84bfc31fc37dbcef0f372c8b1f295d1f7931a3e",
    "tr-corpus/windows/W60_native/train.npz": "56b7e76b53ee0d5cfa6854266d0c66d25df9a3128a673eb491b15ecbf2551e7f",
    "tr-corpus/results/validation_20260915_e3a_d3_relabel/validation_20260914_e3a_native.csv": "9c1b526a062538168d1ec3fd78e28a76f36f5851be7a8ccf17487aface67d164",
    "tr-corpus/results/validation_20260915_e3a_d3_relabel/validation_20260914_e3a_native_xgb_s1to4.csv": "1adaaca183441a92bbec2e7343ce48a7be109d61e8508e4fb4d25139d65e653e",
    "tr-corpus/results/validation_20260915_e3a_d3_relabel/validation_20260914_e3a_native_lgb_s1to4.csv": "a003457b57032be23b34e65b69d6985e9d4a76eb72ffef567e66324d95be5bb1",
    "trbench/survival.py": "c4544d471c9d2f5b7309f3464c52ee788caa6f062c49e14a33920d5aa70c1f73",
    "trbench/run_validation.py": "9920a92d8448a750641d096ccf3de2939eddefb4af25e0e650478af811c8981c",
}


# ------------------------------------------------------------------ statistics
def zero_alarm_upper(n, conf=0.95):
    """One-sided exact binomial upper bound when none of n experiments alarm."""
    return float("nan") if n == 0 else 1.0 - (1.0 - conf) ** (1.0 / n)


def n_min(alpha, conf=0.95):
    """Smallest n whose zero-alarm upper bound is at most alpha."""
    return int(math.ceil(math.log(1.0 - conf) / math.log(1.0 - alpha)))


def cp_upper(k, n, conf=0.95):
    """One-sided Clopper-Pearson upper bound."""
    if n == 0:
        return float("nan")
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def verdict(k, n, alpha=ALPHA):
    """Protocol section 5, applied in order."""
    if n == 0:
        return "not assessable"
    if n < n_min(alpha):
        return "insufficient evidence (observed %s alpha)" % ("<=" if k / n <= alpha else ">")
    if k / n > alpha:
        return "observed fail"
    if cp_upper(k, n) <= alpha:
        return "supported"
    return "within budget, not supported"


def wilson(k, n):
    return RV.wilson(k, n)


# ------------------------------------------------------------------ inputs
def load_meta():
    """The window arrays the stored indices refer to, without X/mask."""
    parts, metas = [], []
    for sp in SPLITS:
        z = np.load(WIN / ("%s.npz" % sp), allow_pickle=True)
        parts.append({k: z[k] for k in ("experiment", "t_end", "y_tr")})
        parts[-1]["label_observed"] = z["label_observed"].astype(bool)
        metas.append(pd.read_csv(WIN / ("%s_experiments.csv" % sp)))
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    meta = pd.concat(metas, ignore_index=True)
    reg = pd.read_csv(ROOT / "tr-corpus" / "registry" / "experiments.csv")
    plan = SimpleNamespace(
        onsets={"L2@1.0": dict(zip(reg.dataset_id + "/" + reg.experiment_id, reg["t_onset_L2_1.0"]))},
        trig=dict(zip(meta.key, meta.t_trigger)))
    RV.V.PRIMARY = "L2@1.0"
    return d, meta, reg, plan


def settings():
    rows = []
    for run, path in RUNS.items():
        cal = pd.read_csv(path / "calibration.csv")
        for f in sorted(glob.glob(str(path / "predictions" / "*.npz"))):
            name = Path(f).stem
            model, rest = name.split("_", 1)
            rep, seed = rest.rsplit("_s", 1)
            seed = int(seed)
            tau = cal[(cal.model == model) & (cal.seed == seed) & (cal.age_mode == rep)].tau
            rows.append(dict(run=run, model=model, representation=rep, seed=seed, file=f,
                             tau_stored=float(tau.iloc[0])))
    return pd.DataFrame(rows)


def head_windows(exp, t, H):
    """Boolean mask of windows inside [t0, t0+H] for experiments spanning >= H."""
    keep = np.zeros(len(t), bool)
    for key in np.unique(exp):
        sel = np.flatnonzero(exp == key)
        tt = t[sel]
        if tt.max() - tt.min() >= H:
            keep[sel[tt <= tt.min() + H]] = True
    return keep


# ------------------------------------------------------------------ Part 1
def support_audit(d, reg):
    z = np.load(settings().file.iloc[0])
    pools = {"calibration": z["cal_idx"], "source check": z["source_idx"],
             "D6": z["arc_idx"][d["y_tr"][z["arc_idx"]] == 0]}
    rows = []
    for pool, idx in pools.items():
        e, t = d["experiment"][idx], d["t_end"][idx]
        spans = pd.Series(t).groupby(e).agg(lambda s: s.max() - s.min())
        for H in H_AUDIT:
            n = int((spans >= H).sum())
            rows.append(dict(pool=pool, n_experiments=len(spans), H_s=H, n_eligible=n,
                             span_min_s=spans.min(), span_median_s=spans.median(), span_max_s=spans.max(),
                             total_span_h=spans.sum() / 3600.0,
                             zero_alarm_upper95=zero_alarm_upper(n), n_min_alpha010=n_min(ALPHA)))
    audit = pd.DataFrame(rows)

    e3a = pd.concat([pd.read_csv(f) for f in E3A])
    cells = sorted(e3a.held.unique())
    r = reg.assign(key=reg.dataset_id + "/" + reg.experiment_id).set_index("key").loc[cells]
    need = {"L2@0.5": ("T", "t_onset_L2_0.5"), "L2@1.0": ("T", "t_onset_L2_1.0"), "L2@2.0": ("T", "t_onset_L2_2.0"),
            "L3 voltage": ("V_cell", "t_onset_L3"), "ISC 25mV": ("V_cell", "t_isc"), "L1 venting": ("P", "t_onset_L1")}
    ev = []
    for label, (ch, col) in need.items():
        present = r.channels_present.fillna("").str.split(",")
        if ch == "T":
            has = present.map(lambda c: any(x.startswith("T_") for x in c))
        elif ch == "P":
            has = present.map(lambda c: any(x in ("P_internal", "P_chamber") for x in c))
        else:
            has = present.map(lambda c: ch in c)
        lab = r[col].notna()
        ev.append(dict(event=label, cells=len(r), computable=int(lab.sum()),
                       channel_absent=int((~has).sum()), channel_present_rule_not_computable=int((has & ~lab).sum())))
    return audit, pd.DataFrame(ev)


# ------------------------------------------------------------------ Part 2 (C)
def event_profile():
    e3a = pd.concat([pd.read_csv(f) for f in E3A])
    rows = []
    for (model, panel, seed), g in e3a.groupby(["model", "panel", "seed"]):
        cells = g.held.unique()
        for name, events in EVENT_SETS.items():
            sub = g[g.label.isin(events) & g.onset.notna()]
            have = sub.groupby("held").label.nunique()
            cohort = have[have == len(events)].index
            s = sub[sub.held.isin(cohort)]
            for ell in LEADS:
                ok = s.detected.astype(bool) & ((ell == 0) | (s.lead >= ell))
                per = ok.groupby(s.held).agg(["all", "any"])
                n = len(per)
                k_all, k_any = int(per["all"].sum()), int(per["any"].sum())
                k_split = k_any - k_all
                k_none = n - k_any
                lo, hi = wilson(k_all, n)
                rows.append(dict(model=model, panel=panel, seed=seed, event_set=name, events="|".join(events),
                                 lead_required_s=ell, cells_scored=len(cells), cohort=n,
                                 not_assessable=len(cells) - n,
                                 pass_all=k_all, pass_all_rate=k_all / n if n else np.nan,
                                 pass_all_wilson_lo=lo, pass_all_wilson_hi=hi,
                                 split=k_split, split_rate=k_split / n if n else np.nan,
                                 fail_all=k_none, fail_all_rate=k_none / n if n else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rule behaviour
def decision_map(n_max=300, alpha=ALPHA):
    """Which verdict each (k, n) produces, summarised by its k boundaries.

    Every pool in this corpus has n <= 26, where rule 2 stops the sequence, so
    the other verdicts are never reached.  This tabulates where they would be.
    """
    rows = []
    for n in range(1, n_max + 1):
        v = [verdict(k, n, alpha) for k in range(n + 1)]
        sup = [k for k, x in enumerate(v) if x == "supported"]
        wit = [k for k, x in enumerate(v) if x == "within budget, not supported"]
        fail = [k for k, x in enumerate(v) if x == "observed fail"]
        rows.append(dict(n=n, verdict_at_k0=v[0],
                         k_supported_max=max(sup) if sup else np.nan,
                         k_within_budget_max=max(wit) if wit else np.nan,
                         k_observed_fail_min=min(fail) if fail else np.nan,
                         reachable="|".join(sorted(set(x.split(" (")[0] for x in v)))))
    return pd.DataFrame(rows)


def _verdict_probabilities(pmf, n, alpha=ALPHA):
    out = {}
    for k, w in enumerate(pmf):
        out[verdict(k, n, alpha).split(" (")[0]] = out.get(verdict(k, n, alpha).split(" (")[0], 0.0) + float(w)
    return out


def operating_characteristics(alpha=ALPHA):
    """Exact probability of each verdict when records are independent.

    The pool is n independent non-runaway records, each alarming with the same
    unknown probability p; k ~ Binomial(n, p).  No simulation: the verdict is a
    deterministic function of k, so its probability is a sum of binomial terms.
    """
    ns = [10, 21, 25, 26, 28, 29, 35, 40, 50, 75, 100, 150, 200, 300]
    ps = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows = []
    for n in ns:
        for p in ps:
            pmf = binom.pmf(np.arange(n + 1), n, p)
            v = _verdict_probabilities(pmf, n, alpha)
            rows.append(dict(n=n, true_p=p, expected_alarms=n * p,
                             p_supported=v.get("supported", 0.0),
                             p_within_budget=v.get("within budget, not supported", 0.0),
                             p_observed_fail=v.get("observed fail", 0.0),
                             p_insufficient=v.get("insufficient evidence", 0.0)))
    return pd.DataFrame(rows)


def sample_size_table(targets=(0.5, 0.8, 0.95), ps=(0.00, 0.01, 0.02, 0.05), alpha=ALPHA, n_cap=2000):
    """Smallest pool that reaches a given probability of the verdict 'supported'."""
    rows = []
    for p in ps:
        row = dict(true_p=p)
        for t in targets:
            need = np.nan
            for n in range(1, n_cap + 1):
                pmf = binom.pmf(np.arange(n + 1), n, p)
                if _verdict_probabilities(pmf, n, alpha).get("supported", 0.0) >= t:
                    need = n
                    break
            row["n_for_P(supported)>=%.2f" % t] = need
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_sensitivity(alpha=ALPHA):
    """What clustering does to the rule, since replicates of one cell design are not independent.

    Records come in equal clusters (cell designs).  Within a cluster the alarm
    probability is drawn from a Beta with mean p and intra-cluster correlation
    rho, so k is a sum of independent beta-binomials; its pmf is their
    convolution, computed exactly.  rho = 0 reduces to the independent case.
    """
    rows = []
    for n, m in ((30, 5), (60, 5), (60, 10), (120, 10)):
        for p in (0.02, 0.05, 0.10, 0.15):
            for rho in (0.0, 0.05, 0.10, 0.25, 0.50):
                if rho == 0:
                    pmf = binom.pmf(np.arange(n + 1), n, p)
                else:
                    s = (1.0 - rho) / rho
                    single = betabinom.pmf(np.arange(m + 1), m, p * s, (1.0 - p) * s)
                    pmf = np.array([1.0])
                    for _ in range(n // m):
                        pmf = np.convolve(pmf, single)
                v = _verdict_probabilities(pmf, n, alpha)
                rows.append(dict(n=n, cluster_size=m, clusters=n // m, true_p=p, icc=rho,
                                 p_supported=v.get("supported", 0.0),
                                 p_observed_fail=v.get("observed fail", 0.0)))
    return pd.DataFrame(rows)


def rule_illustration(alpha=ALPHA):
    """The rule applied to real counts, including pools large enough to leave rule 2.

    No pool in the corpus reaches n_min, so the two pools are also combined -
    an illustration of the rule's behaviour, not a claim about either pool:
    they come from different laboratories and are not exchangeable.
    """
    far = pd.read_csv(OUT / "claims_far.csv")
    audit = pd.read_csv(OUT / "support_audit.csv")
    rows = []
    for _, a in audit[audit.n_eligible == 0].head(1).iterrows():
        rows.append(dict(pool="%s, H=%d s" % (a.pool, a.H_s), setting="any", k=0, n=0,
                         observed=np.nan, cp_upper95=np.nan, verdict=verdict(0, 0, alpha), status="real pool"))
    f = far[(far.run.isin(MAIN_RUNS)) & (far.representation == "no_age") & (far.tau_kind == "full")
            & (far.H_s == H_PRIMARY) & (far.segment.isin(["head", "full"]))]
    for seg in ("head", "full"):
        g = f[f.segment == seg]
        for (model, seed), gg in g.groupby(["model", "seed"]):
            for pool in ("source check", "D6"):
                r = gg[gg.pool == pool]
                if len(r):
                    rows.append(dict(pool=pool, setting="%s s%d, %s 300 s" % (model, seed, seg),
                                     k=int(r.k.iloc[0]), n=int(r.n.iloc[0]),
                                     observed=float(r.far.iloc[0]), cp_upper95=float(r.cp_upper95.iloc[0]),
                                     verdict=verdict(int(r.k.iloc[0]), int(r.n.iloc[0]), alpha), status="real pool"))
            k = int(gg.k.sum())
            n = int(gg.n.sum())
            rows.append(dict(pool="source check + D6 (illustration only)",
                             setting="%s s%d, %s 300 s" % (model, seed, seg), k=k, n=n,
                             observed=k / n, cp_upper95=cp_upper(k, n), verdict=verdict(k, n, alpha),
                             status="pooled for illustration; not exchangeable"))
    return pd.DataFrame(rows)


def split_sensitivity_by_design(d):
    """Evaluation negatives of the alternative split, split by design overlap with training.

    A design counts as present in training if any training record of that cell
    design is there, positive or negative.  Under protocol 16 only negatives were
    re-assigned, so the overlap runs through positive training records.
    """
    import cluster_bootstrap as CB
    cm = CB.cluster_map().set_index("key")
    mem = pd.read_csv(SPLIT_SENS / "split_membership.csv")
    mem["design"] = [cm.loc[k, "cluster"] if k in cm.index else "?" for k in mem.held]
    train_designs = set(mem[mem.role == "train"].design)
    rows = []
    for f in sorted((SPLIT_SENS / "predictions").glob("*.npz")):
        z = np.load(f)
        model, rest = Path(f).stem.split("_", 1)
        seed = int(rest.rsplit("_s", 1)[1])
        rec = RV.negative_records(d, z["source_idx"], np.asarray(z["source_risk"], float),
                                  float(z["tau"]), H_PRIMARY, "head")
        rec["design"] = [cm.loc[k, "cluster"] if k in cm.index else "?" for k in rec.held]
        rec["in_training"] = rec.design.isin(train_designs)
        for (design, shared), g in rec[rec.eligible].groupby(["design", "in_training"]):
            rows.append(dict(model=model, seed=seed, design=design, design_in_training=bool(shared),
                             n=len(g), k=int(g.fired.eq(True).sum())))
    return pd.DataFrame(rows)


def lead_support(d, plan):
    """Post-hoc (protocol section 9, deviation 2): can a lead >= ell occur at all?

    An alarm counts only from max(first prediction endpoint, trigger), so a cell
    whose reference event comes sooner than ell after that point cannot pass at
    lead ell whatever the detector does.  For each event set the earliest event
    of the set bounds the cell.  Pass counts are re-stated on the eligible cells.
    """
    t0 = pd.Series(d["t_end"]).groupby(d["experiment"]).min()
    e3a = pd.concat([pd.read_csv(f) for f in E3A])
    start = e3a.drop_duplicates("held").set_index("held")
    start = pd.Series({k: max(t0[k], tg) if np.isfinite(tg) else t0[k] for k, tg in start.t_trigger.items()})
    cells, rows = [], []
    for name, events in EVENT_SETS.items():
        on = e3a[e3a.label.isin(events) & e3a.onset.notna()].drop_duplicates(["held", "label"])
        have = on.groupby("held").label.nunique()
        cohort = have[have == len(events)].index
        span = (on[on.held.isin(cohort)].groupby("held").onset.min() - start[cohort]).rename("span_s")
        for k, v in span.items():
            cells.append(dict(event_set=name, held=k, dataset=k.split("/")[0], start_s=start[k], span_s=v))
        for ell in LEADS:
            elig = span.index[(span > 0) if ell == 0 else (span >= ell)]
            for (model, panel, seed), g in e3a[e3a.label.isin(events) & e3a.held.isin(cohort)].groupby(["model", "panel", "seed"]):
                ok = (g.detected.astype(bool) & ((ell == 0) | (g.lead >= ell))).groupby(g.held).all()
                k_el, k_out = int(ok[elig].sum()), int(ok.drop(elig).sum())
                lo, hi = wilson(k_el, len(elig))
                rows.append(dict(event_set=name, lead_required_s=ell, model=model, panel=panel, seed=seed,
                                 cohort=len(span), eligible=len(elig), pass_all_eligible=k_el,
                                 pass_all_rate_eligible=k_el / len(elig) if len(elig) else np.nan,
                                 wilson_lo=lo, wilson_hi=hi, pass_outside_eligible=k_out))
    st = settings()
    z = np.load(st.file.iloc[0])
    held = []
    for pool, key in (("source test", "source_positive_idx"), ("D6", "arc_idx")):
        for k in np.unique(d["experiment"][z[key]]):
            on = plan.onsets["L2@1.0"].get(k, np.nan)
            if np.isfinite(on):
                tg = plan.trig.get(k, np.nan)
                held.append(dict(event_set="L2@1.0 (held-dataset design)", pool=pool, held=k, dataset=k.split("/")[0],
                                 start_s=max(t0[k], tg) if np.isfinite(tg) else t0[k],
                                 span_s=on - (max(t0[k], tg) if np.isfinite(tg) else t0[k])))
    cells = pd.concat([pd.DataFrame(cells).assign(pool="internal E3-A"), pd.DataFrame(held)], ignore_index=True)
    assert (pd.DataFrame(rows).pass_outside_eligible == 0).all()
    return cells, pd.DataFrame(rows)


def segment_definitions(d):
    """Two readings of a tail-window false alarm, which the manuscript must separate.

    `negative_records` re-applies the threshold inside the watched window, so a
    tail alarm is what a monitor starting at the window would raise.  The
    record's own first alarm may lie much earlier.  Both are reported.
    """
    rows = []
    for s in settings().itertuples():
        z = np.load(s.file)
        pools = {"D6": (z["arc_idx"][d["y_tr"][z["arc_idx"]] == 0],
                        np.asarray(z["arc_risk"], float)[d["y_tr"][z["arc_idx"]] == 0]),
                 "source check": (z["source_idx"], np.asarray(z["source_risk"], float))}
        tau = float(z["tau"])
        for pool, (idx, risk) in pools.items():
            e, t = d["experiment"][idx], d["t_end"][idx]
            full = RV.negative_records(d, idx, risk, tau, None, "tail")
            tail = RV.negative_records(d, idx, risk, tau, H_PRIMARY, "tail")
            head = RV.negative_records(d, idx, risk, tau, H_PRIMARY, "head")
            in_tail = 0
            for key in np.unique(e):
                sel = e == key
                order = np.argsort(t[sel], kind="stable")
                tt, rr = t[sel][order], risk[sel][order]
                a = SV.first_alarm(tt, rr, tau)
                in_tail += int(a is not None and a >= tt[-1] - H_PRIMARY)
            rows.append(dict(run=s.run, model=s.model, representation=s.representation, seed=s.seed,
                             pool=pool, n=len(full),
                             full_record_alarms=int(full.fired.eq(True).sum()),
                             tail_window_alarms=int(tail[tail.eligible].fired.eq(True).sum()),
                             first_alarm_inside_tail=in_tail,
                             head_window_alarms=int(head[head.eligible].fired.eq(True).sum()),
                             n_eligible_tail=int(tail.eligible.sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part 3 (A)
def _neg_rates(d, idx, risk, tau, H):
    out = {}
    for seg, (w, anchor) in {"head": (H, "head"), "tail": (H, "tail"), "full": (None, "tail")}.items():
        rec = RV.negative_records(d, idx, risk, tau, w, anchor)
        use = rec[rec.eligible]
        out[seg] = (int(use.fired.eq(True).sum()), len(use))
    return out


def _detection(d, idx, risk, tau, plan):
    pr = RV.positive_records(d, idx, np.asarray(risk, float), tau, plan)
    pre_obs = {}
    e, t = d["experiment"][idx], d["t_end"][idx]
    for key in pr.held:
        pre_obs[key] = float(plan.onsets["L2@1.0"][key] - t[e == key].min())
    return pr, float(np.median(list(pre_obs.values()))) if pre_obs else np.nan


def calibrate_all(d, plan):
    st = settings()
    gate, far_rows, det_rows, tau_rows = [], [], [], []
    for s in st.itertuples():
        z = np.load(s.file)
        cal_idx, cal_risk = z["cal_idx"], np.asarray(z["cal_risk"], float)
        ce, ct = d["experiment"][cal_idx], d["t_end"][cal_idx]
        tau_full = SV.calibrate_threshold(cal_risk, ce, np.ones(len(cal_risk), bool), ALPHA)
        ok = bool(np.isclose(tau_full, s.tau_stored, rtol=1e-9, atol=0))
        gate.append(dict(run=s.run, model=s.model, representation=s.representation, seed=s.seed,
                         tau_recomputed=tau_full, tau_stored=s.tau_stored, reproduced=ok))
        taus = {"full": (tau_full, len(np.unique(ce)))}
        for H in (H_PRIMARY,) + H_SENS:
            keep = head_windows(ce, ct, H)
            n = len(np.unique(ce[keep]))
            taus["H%d" % H] = (SV.calibrate_threshold(cal_risk[keep], ce[keep], np.ones(keep.sum(), bool), ALPHA), n)
        for kind, (tau, n) in taus.items():
            tau_rows.append(dict(run=s.run, model=s.model, representation=s.representation, seed=s.seed,
                                 tau_kind=kind, tau=tau, n_calibration=n, k_allowed=int(np.floor(ALPHA * n))))

        arc_idx, arc_risk = z["arc_idx"], np.asarray(z["arc_risk"], float)
        neg_arc = d["y_tr"][arc_idx] == 0
        pools = {"source check": (z["source_idx"], np.asarray(z["source_risk"], float)),
                 "D6": (arc_idx[neg_arc], arc_risk[neg_arc])}
        pos = {"source test": (z["source_positive_idx"], np.asarray(z["source_positive_risk"], float)),
               "D6": (arc_idx, arc_risk)}
        for kind, (tau, _) in taus.items():
            for H in (H_PRIMARY,) + H_SENS:
                for pool, (idx, risk) in pools.items():
                    rates = _neg_rates(d, idx, risk, tau, H)
                    for seg, (k, n) in rates.items():
                        far_rows.append(dict(run=s.run, model=s.model, representation=s.representation, seed=s.seed,
                                             tau_kind=kind, tau=tau, H_s=H, pool=pool, segment=seg, n=n, k=k,
                                             far=k / n if n else np.nan, cp_upper95=cp_upper(k, n),
                                             verdict=(verdict(k, n) if seg == "head" else ""),
                                             exploratory=pool == "D6"))
            for pool, (idx, risk) in pos.items():
                pr, pre_med = _detection(d, idx, risk, tau, plan)
                for ell in LEADS:
                    okd = pr.detected & ((ell == 0) | (pr.lead >= ell))
                    k, n = int(okd.sum()), len(pr)
                    lo, hi = wilson(k, n)
                    det_rows.append(dict(run=s.run, model=s.model, representation=s.representation, seed=s.seed,
                                         tau_kind=kind, tau=tau, pool=pool, lead_required_s=ell, n=n, k=k,
                                         rate=k / n if n else np.nan, wilson_lo=lo, wilson_hi=hi,
                                         pre_event_observation_median_s=pre_med, exploratory=pool == "D6"))
    return (pd.DataFrame(gate), pd.DataFrame(tau_rows), pd.DataFrame(far_rows), pd.DataFrame(det_rows))


def h_contrast(d, plan):
    """Separate "watched longer" from "only the long records survive" (protocol 16, experiment B).

    Two contrasts on the same stored risks: the same calibration cohort scored
    over a longer horizon, and the same horizon applied to a smaller cohort.
    """
    st = settings()
    st = st[(st.run.isin(MAIN_RUNS)) & (st.representation == "no_age")]
    rows = []
    for s in st.itertuples():
        z = np.load(s.file)
        cal_idx, cal_risk = z["cal_idx"], np.asarray(z["cal_risk"], float)
        ce, ct = d["experiment"][cal_idx], d["t_end"][cal_idx]
        spans = pd.Series(ct).groupby(ce).agg(lambda x: x.max() - x.min())
        arc_idx, arc_risk = z["arc_idx"], np.asarray(z["arc_risk"], float)
        neg = d["y_tr"][arc_idx] == 0
        pools = {"source check": (z["source_idx"], np.asarray(z["source_risk"], float)),
                 "D6": (arc_idx[neg], arc_risk[neg])}
        pos = {"source test": (z["source_positive_idx"], np.asarray(z["source_positive_risk"], float)),
               "D6": (arc_idx, arc_risk)}
        for cohort_H in (900, 1800):
            cohort = set(spans.index[spans >= cohort_H])
            in_cohort = np.isin(ce, list(cohort))
            arms = [("all 25, first 300 s", np.ones(len(ce), bool), H_PRIMARY),
                    ("cohort, first 300 s", in_cohort, H_PRIMARY),
                    ("cohort, first %d s" % cohort_H, in_cohort, cohort_H)]
            for name, keep_exp, H in arms:
                keep = keep_exp & head_windows(ce, ct, H)
                n = len(np.unique(ce[keep]))
                tau = SV.calibrate_threshold(cal_risk[keep], ce[keep], np.ones(keep.sum(), bool), ALPHA)
                row = dict(run=s.run, model=s.model, seed=s.seed, cohort="span >= %d s" % cohort_H,
                           cohort_n=len(cohort), arm=name, n_calibration=n,
                           k_allowed=int(np.floor(ALPHA * n)), tau=tau)
                for pool, (idx, risk) in pools.items():
                    rates = _neg_rates(d, idx, risk, tau, H_PRIMARY)
                    for seg, (k, nn) in rates.items():
                        row["%s %s k" % (pool, seg)] = k
                        row["%s %s n" % (pool, seg)] = nn
                for pool, (idx, risk) in pos.items():
                    pr, _ = _detection(d, idx, risk, tau, plan)
                    row["%s detected" % pool] = int(pr.detected.sum())
                    row["%s positives" % pool] = len(pr)
                rows.append(row)
    return pd.DataFrame(rows)


SPLIT_SENS = RES / "validation_20260918_split_sensitivity"


def split_sensitivity(d, plan):
    """Score the pre-specified alternative split (protocol 16, experiment A).

    The split separates cell designs among the NEGATIVE roles only: positives
    keep their roles, so two of the five evaluation designs are still present in
    training as positive records.  `split_sensitivity_by_design` stratifies the
    evaluation pool by that overlap, because the alarms are not spread over it.
    """
    import cluster_bootstrap as CB
    cm = CB.cluster_map().set_index("key")
    rows = []
    for f in sorted((SPLIT_SENS / "predictions").glob("*.npz")):
        z = np.load(f)
        model, rest = Path(f).stem.split("_", 1)
        seed = int(rest.rsplit("_s", 1)[1])
        tau = float(z["tau"])
        row = dict(model=model, seed=seed, tau=tau)
        peaks = pd.Series(np.asarray(z["cal_risk"], float)).groupby(d["experiment"][z["cal_idx"]]).max()
        row.update(calibration_n=len(peaks), calibration_alarms=int((peaks >= tau).sum()))
        idx, risk = z["source_idx"], np.asarray(z["source_risk"], float)
        row["evaluation_designs"] = len({cm.loc[k, "cluster"] for k in np.unique(d["experiment"][idx]) if k in cm.index})
        for seg, (w, anchor_) in {"head": (H_PRIMARY, "head"), "full": (None, "tail")}.items():
            rec = RV.negative_records(d, idx, risk, tau, w, anchor_)
            use = rec[rec.eligible]
            k, n = int(use.fired.eq(True).sum()), len(use)
            row["eval_%s_k" % seg], row["eval_%s_n" % seg] = k, n
            row["eval_%s_far" % seg] = k / n if n else np.nan
            row["eval_%s_cp95" % seg] = cp_upper(k, n)
            row["eval_%s_verdict" % seg] = verdict(k, n)
        arc_idx, arc_risk = z["arc_idx"], np.asarray(z["arc_risk"], float)
        neg = d["y_tr"][arc_idx] == 0
        for seg, (k_, n_) in _neg_rates(d, arc_idx[neg], arc_risk[neg], tau, H_PRIMARY).items():
            row["D6_%s_k" % seg], row["D6_%s_n" % seg] = k_, n_
        for pool, idx_, risk_ in (("source_test", z["source_positive_idx"], np.asarray(z["source_positive_risk"], float)),
                                  ("D6", arc_idx, arc_risk)):
            pr, _ = _detection(d, idx_, risk_, tau, plan)
            for ell in LEADS:
                ok = pr.detected & ((ell == 0) | (pr.lead >= ell))
                row["%s_detected_l%d" % (pool, ell)] = int(ok.sum())
            row["%s_positives" % pool] = len(pr)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "seed"])


# ------------------------------------------------------------------ Part 5 (B)
def pair_disagreement(d, plan):
    st = settings()
    pairs = [("common_surface", "surface_max_only", "surface_mean_only"), ("native_final", "no_age", "with_age")]
    rows = []
    for tree_run, ra, rb in pairs:
        for model in LEARNED:
            # the sequence models hold all four representations in one run directory
            run = "seq_matched" if model in SEQ else tree_run
            for seed in range(5):
                za = np.load(st[(st.run == run) & (st.model == model) & (st.representation == ra) & (st.seed == seed)].file.iloc[0])
                zb = np.load(st[(st.run == run) & (st.model == model) & (st.representation == rb) & (st.seed == seed)].file.iloc[0])
                row = dict(run=run, pair="%s vs %s" % (ra, rb), model=model, seed=seed)
                for label, key in (("source check", "source"), ("D6 negatives", "arc")):
                    status, first = {}, {}
                    for tag, z in (("a", za), ("b", zb)):
                        idx, risk = z["%s_idx" % key], np.asarray(z["%s_risk" % key], float)
                        if key == "arc":
                            m = d["y_tr"][idx] == 0
                            idx, risk = idx[m], risk[m]
                        full = RV.negative_records(d, idx, risk, float(z["tau"]), None, "tail").set_index("held")
                        head = RV.negative_records(d, idx, risk, float(z["tau"]), H_PRIMARY, "head").set_index("held")
                        status[tag] = (full.fired.astype(bool), head.fired[head.eligible].astype(bool))
                        first[tag] = full.first_alarm_s
                    row["%s full disagreement" % label] = float((status["a"][0] != status["b"][0]).mean())
                    row["%s head300 disagreement" % label] = float((status["a"][1] != status["b"][1]).mean())
                    both = first["a"].notna() & first["b"].notna()
                    row["%s median |d first alarm| (s)" % label] = float((first["a"][both] - first["b"][both]).abs().median()) if both.any() else np.nan
                for label, key in (("source test positives", "source_positive"), ("D6 positives", "arc")):
                    det = {}
                    for tag, z in (("a", za), ("b", zb)):
                        pr = RV.positive_records(d, z["%s_idx" % key], np.asarray(z["%s_risk" % key], float), float(z["tau"]), plan)
                        det[tag] = pr.set_index("held").detected.astype(bool)
                    row["%s detection disagreement" % label] = float((det["a"] != det["b"]).mean())
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def report(args):
    far = pd.read_csv(OUT / "claims_far.csv")
    det = pd.read_csv(OUT / "claims_detection.csv")
    prof = pd.read_csv(OUT / "event_profile.csv")
    sel = (far.run.isin(MAIN_RUNS)) & (far.representation == args.representation) & (far.segment == "head") \
        & (far.H_s == args.H) & (far.tau_kind == args.tau_kind)
    print("Claim: head-%d s false-alarm probability <= %.2f (tau: %s, representation: %s)" %
          (args.H, args.alpha, args.tau_kind, args.representation))
    print(far[sel][["model", "seed", "pool", "k", "n", "far", "cp_upper95", "verdict"]].to_string(index=False))
    events = args.events
    name = next((k for k, v in EVENT_SETS.items() if "|".join(v) == events or k == events), None)
    if name is None:
        print("event set %r is not pre-specified; use one of %s" % (events, list(EVENT_SETS)))
        return 1
    p = prof[(prof.event_set == name) & (prof.lead_required_s == args.lead)]
    print("\nInternal design, event set %s, lead >= %d s (pass under every event / split / fail under every event):" % (name, args.lead))
    print(p[["model", "panel", "seed", "cohort", "pass_all", "split", "fail_all", "not_assessable"]].to_string(index=False))
    if args.min_detection is not None:
        lo_ok = p.pass_all_wilson_lo >= args.min_detection
        print("\nDetection claim >= %.2f supported (Wilson lower bound) in %d of %d settings" % (args.min_detection, lo_ok.sum(), len(p)))
    dsel = (det.run.isin(MAIN_RUNS)) & (det.representation == args.representation) & (det.tau_kind == args.tau_kind) \
        & (det.lead_required_s == args.lead)
    print("\nHeld-dataset design, L2@1.0 detection at lead >= %d s:" % args.lead)
    print(det[dsel][["model", "seed", "pool", "k", "n", "rate", "wilson_lo", "wilson_hi"]].to_string(index=False))
    return 0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(extra):
    frozen = {k: dict(expected=v, actual=sha(ROOT / k)) for k, v in FROZEN.items()}
    for v in frozen.values():
        v["unchanged"] = v["expected"] == v["actual"]
    man = dict(protocol="reports/12_claim_assessment_protocol.md", protocol_commit="fe9ecf3",
               seven_model_protocol="reports/20_sequence_tier_internal_design_protocol.md", seven_model_commit="eed8297",
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               code_sha256=sha(__file__), frozen_inputs=frozen,
               constants=dict(alpha=ALPHA, H_primary=H_PRIMARY, H_sensitivity=H_SENS, H_audit=H_AUDIT,
                              leads=LEADS, event_sets=EVENT_SETS), **extra)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    assert all(v["unchanged"] for v in frozen.values()), frozen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["audit", "profile", "support", "rules", "segments", "hcontrast", "splitsens", "calibrate", "pairs", "report", "all"])
    ap.add_argument("--H", type=int, default=H_PRIMARY)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--events", default="E_TR")
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--tau-kind", default="H300")
    ap.add_argument("--representation", default="no_age")
    ap.add_argument("--min-detection", type=float, default=None)
    a = ap.parse_args(argv)
    if a.command == "report":
        if a.alpha != ALPHA or a.H not in (H_PRIMARY,) + H_SENS or a.lead not in LEADS:
            print("only the pre-specified alpha=%.2f, H in %s and lead in %s are computed" % (ALPHA, (H_PRIMARY,) + H_SENS, LEADS))
            return 1
        return report(a)
    OUT.mkdir(parents=True, exist_ok=True)
    extra = {}
    d, meta, reg, plan = load_meta()
    if a.command in ("audit", "all"):
        audit, ev = support_audit(d, reg)
        audit.to_csv(OUT / "support_audit.csv", index=False)
        ev.to_csv(OUT / "event_support.csv", index=False)
        print(audit.to_string(index=False)); print(ev.to_string(index=False))
    if a.command in ("profile", "all"):
        prof = event_profile()
        prof.to_csv(OUT / "event_profile.csv", index=False)
    if a.command in ("support", "all"):
        cells, elig = lead_support(d, plan)
        cells.to_csv(OUT / "lead_support_cells.csv", index=False)
        elig.to_csv(OUT / "event_profile_eligible.csv", index=False)
    if a.command in ("rules", "all"):
        decision_map().to_csv(OUT / "rule_decision_map.csv", index=False)
        oc = operating_characteristics()
        oc.to_csv(OUT / "rule_operating_characteristics.csv", index=False)
        sample_size_table().to_csv(OUT / "rule_sample_size.csv", index=False)
        cluster_sensitivity().to_csv(OUT / "rule_cluster_sensitivity.csv", index=False)
        if (OUT / "claims_far.csv").exists():
            rule_illustration().to_csv(OUT / "rule_illustration.csv", index=False)
    if a.command in ("segments", "all"):
        segment_definitions(d).to_csv(OUT / "segment_definitions.csv", index=False)
    if a.command in ("splitsens", "all") and SPLIT_SENS.exists():
        split_sensitivity(d, plan).to_csv(OUT / "split_sensitivity.csv", index=False)
        split_sensitivity_by_design(d).to_csv(OUT / "split_sensitivity_by_design.csv", index=False)
    if a.command in ("hcontrast", "all"):
        h_contrast(d, plan).to_csv(OUT / "h_contrast.csv", index=False)
    if a.command in ("calibrate", "all"):
        gate, taus, far, det = calibrate_all(d, plan)
        gate.to_csv(OUT / "tau_reproduction_gate.csv", index=False)
        extra["tau_gate_reproduced"] = "%d/%d" % (gate.reproduced.sum(), len(gate))
        print(gate.to_string(index=False))
        if not gate.reproduced.all():
            print("STOP: stored tau not reproduced; fixed-horizon calibration is not reported (protocol section 5)")
        else:
            taus.to_csv(OUT / "calibration_H.csv", index=False)
            far.to_csv(OUT / "claims_far.csv", index=False)
            det.to_csv(OUT / "claims_detection.csv", index=False)
    if a.command in ("pairs", "all"):
        pair_disagreement(d, plan).to_csv(OUT / "pair_disagreement.csv", index=False)
    manifest(extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
