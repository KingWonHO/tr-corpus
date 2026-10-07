"""Claim-assessment figure and tables (protocol reports/12_claim_assessment_protocol.md).

Reads the outputs of trbench/claims.py and writes

    paper/figures/F7_claim_support.{png,pdf}
    paper/tables/S8a_support_audit.{csv,md}
    paper/tables/S8b_lead_profile.{csv,md}
    paper/tables/S8c_fixed_horizon.{csv,md}
    paper/tables/S8d_pair_disagreement.{csv,md}
    paper/figures/SF3_rule_behaviour.{png,pdf}
    paper/tables/S9a_rule_behaviour.{csv,md}
    paper/tables/S9b_rule_clustering.{csv,md}
    paper/tables/S4f_segment_definitions.{csv,md}
    paper/tables/S8e_horizon_contrast.{csv,md}
    paper/tables/S8f_split_sensitivity.{csv,md}
    paper/tables/S8g_split_by_design.{csv,md}

    uv run python paper/case_claims.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "paper")]
from make_figures import save_fig, save_table, rng, GREY, RED, BLUE, MODEL_NAME  # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results", "claims_20260925")
LEARNED = ["xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer"]
MAIN_RUNS = ["native_final", "seq_matched"]
PALETTE = {"xgboost": BLUE, "lightgbm": "#63b3ed", "gru": "#b8332f", "mamba": "#dd6b20", "itransformer": "#805ad5", "convtransformer": "#2f855a"}
SRC = os.path.relpath(RES, ROOT).replace(os.sep, "/")
POOL_NAME = {"calibration": "calibration negatives", "source check": "source check negatives", "D6": "D6 negatives"}
POOL_COL = {"calibration": BLUE, "source check": GREY, "D6": RED}
VERDICT = lambda v: v.replace("insufficient evidence (observed <= alpha)", "insufficient, ≤ α").replace(
    "insufficient evidence (observed > alpha)", "insufficient, > α").replace(
    "inconclusive (observed <= alpha)", "inconclusive, ≤ α").replace("inconclusive (observed > alpha)", "inconclusive, > α")
SET_NAME = {"L2@1.0": "L2", "E_TR": "E_TR (L2 at 0.5, 1, 2 °C/s)",
            "E_elec": "E_elec (L3, 25 mV)", "E_vent": "E_vent (L1)"}


def read(name):
    return pd.read_csv(os.path.join(RES, name))


# ------------------------------------------------------------------ F7
def fig7(audit, prof):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 5.0))
    fig.subplots_adjust(left=.09, right=.98, bottom=.32, top=.84, wspace=.50)

    ax = axes[0]
    Hs = sorted(audit.H_s.unique())
    x = np.arange(len(Hs)); w = 0.26
    for k, pool in enumerate(["calibration", "source check", "D6"]):
        a = audit[audit.pool == pool].set_index("H_s").loc[Hs]
        ax.bar(x + (k - 1) * w, a.n_eligible, w, color=POOL_COL[pool], linewidth=0,
               label="%s (%d)" % (POOL_NAME[pool], a.n_experiments.iloc[0]))
    nmin = int(audit.n_min_alpha010.iloc[0])
    ax.axhline(nmin, color="#2d3748", lw=0.8, ls="--")
    ax.text(len(Hs) - 0.55, nmin + 0.6, "n = %d needed if none alarm\n(one-sided 95 %%, α = 0.10)" % nmin,
            fontsize=7.5, ha="right", va="bottom", color="#2d3748")
    ax.set_xticks(x); ax.set_xticklabels(["%g" % (h / 60) for h in Hs])
    ax.set_xlabel("Observation horizon H\nfrom record start (min)", fontsize=9)
    ax.set_ylabel("Negative records observed\nfor the full horizon H", fontsize=9)
    ax.set_ylim(0, 36)
    ax.set_title("(a) Negative records available\nover the observation horizon", loc="left",
                 fontsize=10, fontweight="bold", y=1.18, pad=0, va="top", linespacing=1.2)
    ax.legend(fontsize=7.7, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, -.32), borderaxespad=0, labelspacing=.6)
    ax.grid(axis="y", lw=0.4, alpha=.5)

    ax = axes[1]
    p = prof[prof.event_set == "L2@1.0"]
    leads = sorted(p.lead_required_s.unique())
    styles = {(m, panel): (PALETTE[m], "-" if panel == "M1" else "--") for m in LEARNED for panel in ("M1", "M2")}
    for (m, panel), (c, ls) in styles.items():
        g = p[(p.model == m) & (p.panel == panel)].groupby("lead_required_s")
        med, lo, hi = g.pass_all_rate_eligible.median(), g.pass_all_rate_eligible.min(), g.pass_all_rate_eligible.max()
        ax.plot(leads, med.loc[leads], color=c, ls=ls, lw=1.3, marker="o", ms=2.5,
                label="%s, %s" % (MODEL_NAME[m], panel))
        ax.fill_between(leads, lo.loc[leads], hi.loc[leads], color=c, alpha=.10, lw=0)
    ax.set_xticks(leads); ax.set_xlim(-8, 128); ax.set_ylim(0, 0.4)
    ax.set_yticks([0, .1, .2, .3, .4])
    ax.set_xlabel("required lead ℓ before L2 (s)")
    ax.set_ylabel("Fraction of observable cells\nwith lead ≥ ℓ", fontsize=9)
    ax.set_title("(b) Required-lead profile\n(internal design)", loc="left",
                 fontsize=10, fontweight="bold", y=1.18, pad=0, va="top", linespacing=1.2)
    handles = [Line2D([], [], color=PALETTE[m], lw=1.6, label=MODEL_NAME[m]) for m in LEARNED]
    handles += [Line2D([], [], color="#334155", lw=1.5, ls=ls, label=panel)
                for panel, ls in (("M1", "-"), ("M2", "--"))]
    ax.legend(handles=handles, fontsize=7.3, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, -.32), ncol=2, columnspacing=1.2,
              handlelength=2.2, borderaxespad=0, labelspacing=.6)
    ax.grid(lw=0.4, alpha=.5)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(os.path.join(ROOT, "paper", "figures", "F7_claim_support.svg"), bbox_inches="tight")
    save_fig(fig, "F7_claim_support")


# ------------------------------------------------------------------ S8
def table_s8a(audit):
    rows = []
    for r in audit.itertuples():
        rows.append({"pool": POOL_NAME[r.pool], "experiments": r.n_experiments, "H (s)": r.H_s,
                     "observed for all of H": r.n_eligible,
                     "zero-alarm 95 % upper bound": "" if pd.isna(r.zero_alarm_upper95) else "%.3f" % r.zero_alarm_upper95,
                     "record span min / median / max (s)": "%g / %g / %g" % (r.span_min_s, r.span_median_s, r.span_max_s),
                     "summed span (h)": "%.1f" % r.total_span_h})
    save_table(pd.DataFrame(rows), "S8a_support_audit",
               "Source: %s/support_audit.csv. The summed span is a sum of separate records, not continuous exposure. "
               "Zero alarms among n records bound the false-alarm probability at 0.10 (one-sided 95 %%) only for n >= %d."
               % (SRC, int(audit.n_min_alpha010.iloc[0])))


def table_s8b(prof, raw):
    rows = []
    for (es, ell), g in prof.groupby(["event_set", "lead_required_s"], sort=False):
        r0 = raw[(raw.event_set == es) & (raw.lead_required_s == ell)]
        rows.append({"event set": SET_NAME[es], "ℓ (s)": ell,
                     "cohort": g.cohort.iloc[0], "lead ≥ ℓ observable": g.eligible.iloc[0],
                     "pass under every event": rng(g.pass_all_eligible, "%d"),
                     "rate among observable": rng(g.pass_all_rate_eligible, "%.2f"),
                     "split between events": rng(r0.split, "%d"),
                     "Wilson 95 % (min lo – max hi)": "%.2f–%.2f" % (g.wilson_lo.min(), g.wilson_hi.max())})
    save_table(pd.DataFrame(rows), "S8b_lead_profile",
               "Sources: %s/event_profile.csv (pre-specified) and event_profile_eligible.csv (post-hoc observability restriction). "
               "Ranges over the six learned models, panels M1 and M2, seeds 0–4. A cell is observable at ℓ if the earliest event "
               "of the set falls at least ℓ after max(first prediction endpoint, trigger); no cell outside that set passes." % SRC)


def table_s8c(taus, far, det):
    rows = []
    nf = far[(far.run.isin(MAIN_RUNS)) & (far.representation == "no_age")]
    nd = det[(det.run.isin(MAIN_RUNS)) & (det.representation == "no_age") & (det.pool == "source test")]
    dd = det[(det.run.isin(MAIN_RUNS)) & (det.representation == "no_age") & (det.pool == "D6")]
    tt = taus[(taus.run.isin(MAIN_RUNS)) & (taus.representation == "no_age")]
    for kind in ["full", "H300", "H900", "H1800"]:
        for m in ["rule"] + LEARNED:
            t = tt[(tt.tau_kind == kind) & (tt.model == m)]
            H = 300 if kind in ("full", "H300") else int(kind[1:])
            f = nf[(nf.tau_kind == kind) & (nf.model == m) & (nf.H_s == H)]
            src = f[(f.pool == "source check") & (f.segment == "head")]
            d6h = f[(f.pool == "D6") & (f.segment == "head")]
            d6t = f[(f.pool == "D6") & (f.segment == "tail")]
            d6f = f[(f.pool == "D6") & (f.segment == "full")]
            s0 = nd[(nd.tau_kind == kind) & (nd.model == m)]
            d0 = dd[(dd.tau_kind == kind) & (dd.model == m)]
            rows.append({"threshold set on": "full records" if kind == "full" else "first %d s" % H,
                         "calibration n": t.n_calibration.iloc[0], "model": MODEL_NAME[m],
                         "scored H (s)": H,
                         "source check, head": "%s/%d" % (rng(src.k, "%d"), src.n.iloc[0]),
                         "D6 head": "%s/%d" % (rng(d6h.k, "%d"), d6h.n.iloc[0]),
                         "D6 tail": "%s/%d" % (rng(d6t.k, "%d"), d6t.n.iloc[0]),
                         "D6 full": "%s/%d" % (rng(d6f.k, "%d"), d6f.n.iloc[0]),
                         "verdict, source head": " / ".join(sorted(set(src.verdict.map(VERDICT)))),
                         "source test detected, ℓ=0 / 60 s": "%s / %s of %d" % (
                             rng(s0[s0.lead_required_s == 0].k, "%d"), rng(s0[s0.lead_required_s == 60].k, "%d"), s0.n.iloc[0]),
                         "D6 detected, ℓ=0 / 60 s": "%s / %s of %d" % (
                             rng(d0[d0.lead_required_s == 0].k, "%d"), rng(d0[d0.lead_required_s == 60].k, "%d"), d0.n.iloc[0])})
    save_table(pd.DataFrame(rows), "S8c_fixed_horizon",
               "Sources: %s/calibration_H.csv, claims_far.csv, claims_detection.csv; native no-age runs, rule seed 0, trees seeds 0–4. "
               "Verdicts follow protocol 12 section 5; D6 is exploratory. Thresholds set on the first 900 or 1,800 s use only the 13 or 11 "
               "calibration records that long, which exclude every high-risk calibration record." % SRC)


def table_s8d(pairs):
    cols = [c for c in pairs.columns if "disagreement" in c or "first alarm" in c]
    rows = []
    for (run, pair, m), g in pairs.groupby(["run", "pair", "model"]):
        row = {"pair": pair, "model": MODEL_NAME[m]}
        for c in cols:
            row[c] = rng(g[c], "%.2f") if "disagreement" in c else rng(g[c], "%g")
        rows.append(row)
    save_table(pd.DataFrame(rows), "S8d_pair_disagreement",
               "Source: %s/pair_disagreement.csv; each row spans seeds 0–4 of one model. Status disagreement is scale-free; "
               "the 20 pairs are not independent samples and no correlation is tested." % SRC)


# ------------------------------------------------------------------ S9 (rule behaviour)
def fig_s3(dm, oc, cs):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axes[0]
    for p, c, ls in ((0.00, BLUE, "-"), (0.02, BLUE, "--"), (0.05, GREY, "-.")):
        g = oc[oc.true_p == p].sort_values("n") if p else None
        if g is None:
            ns = sorted(oc.n.unique())
            y = [1.0 if n >= 29 else 0.0 for n in ns]
            ax.plot(ns, y, color=c, ls=ls, lw=1.3, marker="o", ms=2.5, label="true rate 0")
        else:
            ax.plot(g.n, g.p_supported, color=c, ls=ls, lw=1.3, marker="o", ms=2.5, label="true rate %.2f" % p)
    g = oc[oc.true_p == 0.20].sort_values("n")
    ax.plot(g.n, g.p_exceeded, color=RED, lw=1.3, marker="s", ms=2.5, label="budget exceeded, true rate 0.20")
    g = oc[oc.true_p == 0.30].sort_values("n")
    ax.plot(g.n, g.p_exceeded, color=RED, lw=1.0, ls="--", marker="s", ms=2.0, label="budget exceeded, true rate 0.30")
    ax.axvline(29, color="#2d3748", lw=0.8, ls=":")
    ax.text(30, 0.03, "n = 29", fontsize=6.4, color="#2d3748")
    for n, lab in ((13, "D6"), (27, "source")):      # the primary check pools
        ax.axvline(n, color="#cbd5e0", lw=0.8)
        ax.text(n * 0.94, 0.90, lab, fontsize=6.2, color=GREY, ha="right", rotation=90, va="top")
    ax.set_xscale("log"); ax.set_xlim(9, 320); ax.set_ylim(0, 1.02)
    ax.set_xlabel("independent non-runaway records in the pool")
    ax.set_ylabel("probability of the verdict")
    ax.set_title("(a) What the rules return, by pool size", loc="left", fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="center right")
    ax.grid(lw=0.4, alpha=.5)

    ax = axes[1]
    for (n, m), g in cs[cs.true_p == 0.10].groupby(["n", "cluster_size"]):
        g = g.sort_values("icc")
        ax.plot(g.icc, g.p_supported, lw=1.3, marker="o", ms=2.5,
                label="n = %d, %d per design" % (n, m))
    ax.axhline(0.05, color="#2d3748", lw=0.8, ls="--")
    ax.text(0.5, 0.062, "nominal 0.05", fontsize=6.4, color="#2d3748", ha="right")
    ax.set_xlabel("intra-design correlation of the alarm indicator")
    ax.set_ylabel("probability of the verdict 'supported'")
    ax.set_title("(b) When the true rate equals α = 0.10", loc="left", fontsize=8)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    ax.grid(lw=0.4, alpha=.5)
    fig.text(0.5, -0.10, "Exact binomial (a) and beta-binomial (b) computation of the assessment rules; no model and no data enter. "
             "(a) 'Supported' needs at least 29 zero-alarm records; 'budget exceeded' is reachable at any pool size once enough records alarm.\n(b) Replicates of one cell design are "
             "not independent; with correlated records the rule calls a claim supported more often than its nominal 5 % when the true rate is exactly α.",
             ha="center", fontsize=6.4, color=GREY)
    save_fig(fig, "SF3_rule_behaviour")


def table_s9(dm, oc, ss, cs, ill):
    rows = []
    for n in [10, 13, 25, 27, 28, 29, 40, 50, 75, 100, 150, 200]:
        d = dm[dm.n == n].iloc[0]
        o = oc[oc.n == n].set_index("true_p")
        rows.append({"pool size n": n,
                     "alarms for 'supported'": "none" if pd.isna(d.k_supported_max) else ("0" if d.k_supported_max == 0 else "0–%d" % d.k_supported_max),
                     "alarms for 'budget exceeded'": "none" if pd.isna(d.k_exceeded_min) else "≥ %d" % d.k_exceeded_min,
                     "P(supported), true 0.02": "%.3f" % o.loc[0.02, "p_supported"],
                     "P(supported), true 0.05": "%.3f" % o.loc[0.05, "p_supported"],
                     "P(supported), true 0.10": "%.3f" % o.loc[0.10, "p_supported"],
                     "P(exceeded), true 0.20": "%.3f" % o.loc[0.20, "p_exceeded"],
                     "P(exceeded), true 0.30": "%.3f" % o.loc[0.30, "p_exceeded"]})
    save_table(pd.DataFrame(rows), "S9a_rule_behaviour",
               "Source: %s/rule_decision_map.csv and rule_operating_characteristics.csv. Exact binomial computation for independent records; "
               "α = 0.10; assessment rule (supported: one-sided 95 %% upper bound ≤ α; budget exceeded: one-sided 95 %% lower bound > α; otherwise inconclusive). "
               "Smallest pool reaching P(supported) ≥ 0.8: %s records at a true rate of 0.01, %s at 0.02 and %s at 0.05; "
               "29 suffice only if the true rate is 0. The primary check pools contain 13 and 27 records; the alternative allocation has 71 and the field case 292 vehicles."
               % (SRC, *[int(ss.loc[ss.true_p == p, "n_for_P(supported)>=0.80"].iloc[0]) for p in (0.01, 0.02, 0.05)]))
    rows = []
    for (n, m), g in cs[cs.true_p == 0.10].groupby(["n", "cluster_size"]):
        g = g.set_index("icc")
        row = {"records n": n, "records per cell design": m, "designs": n // m}
        for r in (0.0, 0.05, 0.10, 0.25, 0.50):
            row["ICC %.2f" % r] = "%.3f" % g.loc[r, "p_supported"]
        rows.append(row)
    v = ill[ill.pool.str.startswith("source check + D6")]
    counts = "; ".join("%s: %d of %d settings" % (k.split(" (")[0], c, len(v)) for k, c in v.verdict.str.split(" \\(").str[0].value_counts().items())
    save_table(pd.DataFrame(rows), "S9b_rule_clustering",
               "Source: %s/rule_cluster_sensitivity.csv. Probability of the verdict 'supported' when the true rate is exactly α = 0.10 and records "
               "of one cell design are correlated (beta-binomial, exact). The rule assumes independence, so correlated records make it optimistic. "
               "Applying the rules to the two real pools combined — an illustration only, since they are not exchangeable — gives %s."
               % (SRC, counts))


def table_s4f(seg):
    rows = []
    g = seg[(seg.run.isin(MAIN_RUNS)) & (seg.representation == "no_age")]
    for (pool, model), gg in g.groupby(["pool", "model"]):
        rows.append({"negative pool": pool, "model": MODEL_NAME[model], "records": gg.n.iloc[0],
                     "first alarm anywhere in the record": rng(gg.full_record_alarms, "%d"),
                     "alarm inside the last 300 s (reported)": rng(gg.tail_window_alarms, "%d"),
                     "record's first alarm falls in the last 300 s": rng(gg.first_alarm_inside_tail, "%d"),
                     "alarm inside the first 300 s": rng(gg.head_window_alarms, "%d")})
    save_table(pd.DataFrame(rows), "S4f_segment_definitions",
               "Source: %s/segment_definitions.csv; native runs without observation age, ranges over seeds 0–4 (rule: seed 0). "
               "A window alarm is what a monitor starting at that window would raise; the record's first alarm is a different "
               "quantity and is given next to it. Head and full-record columns are the ones used elsewhere in the paper." % SRC)


def table_s8e(hc):
    rows = []
    hc = hc[hc.model != "rule"]
    for (cohort, model, arm), g in hc.groupby(["cohort", "model", "arm"], sort=False):
        rows.append({"calibration cohort": cohort.replace(">=", "≥"), "model": MODEL_NAME[model], "arm": arm,
                     "records": g.n_calibration.iloc[0], "alarms allowed": g.k_allowed.iloc[0],
                     "τ": rng(g.tau, "%.4f"),
                     "source check, first 300 s": "%s/%d" % (rng(g["source check head k"], "%d"), g["source check head n"].iloc[0]),
                     "D6, full record": "%s/%d" % (rng(g["D6 full k"], "%d"), g["D6 full n"].iloc[0]),
                     "source test detected": "%s/%d" % (rng(g["source test detected"], "%d"), g["source test positives"].iloc[0]),
                     "D6 detected": "%s/%d" % (rng(g["D6 detected"], "%d"), g["D6 positives"].iloc[0])})
    save_table(pd.DataFrame(rows), "S8e_horizon_contrast",
               "Source: %s/h_contrast.csv; native runs without observation age, seeds 0–4, no model refitted. "
               "Within a cohort the first two rows change only the pool (25 records vs the subset long enough for the horizon) "
               "and the last two change only the watched horizon on that same subset." % SRC)


def table_s8f(ss):
    rows = []
    for model, g in ss.groupby("model"):
        rows.append({"model": MODEL_NAME[model],
                     "calibration": "%s/%d alarms" % (rng(g.calibration_alarms, "%d"), g.calibration_n.iloc[0]),
                     "evaluation records": g.eval_head_n.iloc[0],
                     "cell designs": g.evaluation_designs.iloc[0],
                     "alarming, first 300 s": rng(g.eval_head_k, "%d"),
                     "false-alarm rate": rng(g.eval_head_far, "%.2f"),
                     "CP 95 % upper": rng(g.eval_head_cp95, "%.2f"),
                     "verdict": "; ".join(sorted(set(g.eval_head_verdict))),
                     "source test detected, ℓ=0": "%s/%d" % (rng(g.source_test_detected_l0, "%d"), g.source_test_positives.iloc[0]),
                     "D6 first 300 s": "%s/%d" % (rng(g.D6_head_k, "%d"), g.D6_head_n.iloc[0]),
                     "D6 full record": "%s/%d" % (rng(g.D6_full_k, "%d"), g.D6_full_n.iloc[0]),
                     "D6 detected, ℓ=0": "%s/%d" % (rng(g.D6_detected_l0, "%d"), g.D6_positives.iloc[0])})
    save_table(pd.DataFrame(rows), "S8f_split_sensitivity",
               "Source: %s/split_sensitivity.csv (protocol reports/16). Source negatives were re-assigned by whole cell design, "
               "outcome-blind, giving 67 evaluation / 42 calibration / 60 training records; positives and D6 keep their roles and "
               "the two trees were refitted (seeds 0–4, no observation age). Because only negatives were re-assigned, two of the "
               "five evaluation designs remain in training through positive records (S8g)." % SRC)


def table_s8g(bd):
    rows = []
    for (shared, design), g in bd.groupby(["design_in_training", "design"]):
        rows.append({"cell design": design.replace("D5:", ""),
                     "in training (as positives)": "yes" if shared else "no",
                     "evaluation negatives": g.n.iloc[0],
                     "alarming, first 300 s": rng(g.k, "%d")})
    tot = bd.groupby(["model", "seed", "design_in_training"])[["k", "n"]].sum().reset_index()
    for shared, g in tot.groupby("design_in_training"):
        rows.append({"cell design": "all designs " + ("in" if shared else "absent from") + " training",
                     "in training (as positives)": "yes" if shared else "no",
                     "evaluation negatives": g.n.iloc[0], "alarming, first 300 s": rng(g.k, "%d")})
    save_table(pd.DataFrame(rows), "S8g_split_by_design",
               "Source: %s/split_sensitivity_by_design.csv; ranges over the two trees and seeds 0–4. A design is in training if any "
               "training record of that cell design exists; under protocol 16 that overlap runs through positive records only. "
               "Post-hoc stratification: the absence of alarms in the three absent designs is not evidence about unseen designs in general." % SRC)


def main():
    audit = read("support_audit.csv")
    prof = read("event_profile_eligible.csv")
    fig7(audit, prof)
    table_s8a(audit)
    table_s8b(prof, read("event_profile.csv"))
    table_s8c(read("calibration_H.csv"), read("claims_far.csv"), read("claims_detection.csv"))
    if os.path.exists(os.path.join(RES, "pair_disagreement.csv")):
        table_s8d(read("pair_disagreement.csv"))
    if os.path.exists(os.path.join(RES, "split_sensitivity.csv")):
        table_s8f(read("split_sensitivity.csv"))
        table_s8g(read("split_sensitivity_by_design.csv"))
    if os.path.exists(os.path.join(RES, "h_contrast.csv")):
        table_s8e(read("h_contrast.csv"))
    if os.path.exists(os.path.join(RES, "segment_definitions.csv")):
        table_s4f(read("segment_definitions.csv"))
    if os.path.exists(os.path.join(RES, "rule_decision_map.csv")):
        dm, oc = read("rule_decision_map.csv"), read("rule_operating_characteristics.csv")
        ss, cs, ill = read("rule_sample_size.csv"), read("rule_cluster_sensitivity.csv"), read("rule_illustration.csv")
        fig_s3(dm, oc, cs)
        table_s9(dm, oc, ss, cs, ill)


if __name__ == "__main__":
    main()
