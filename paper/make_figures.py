"""Regenerate every manuscript figure and table from the canonical artefacts.

Nothing here fits a model.  Each output is a view of a file that a trbench
script already wrote (registry, validation supplement, uncertainty summaries),
plus three model-independent analyses that need only the registry and the
standardized corpus: the pairwise onset-interval matrix, the L2 persistence
sensitivity (2 s vs 3 s), and the dataset-1 label check against the source
report.  The manuscript can be rebuilt from the repository alone and any
number in it traces to one CSV.

    uv run python paper/make_figures.py

Numbering follows reports/07_manuscript_plan_jes.md section 5:

    F1 framework · T1 datasets (hand) · F2 co-observation · F3 reference
    events · T2 increments · F4 false-alarm timing · F5 representation ·
    F6 model-family robustness · T3 checklist (hand)
    S1 literature audit · S2 onset definitions and validation · S3 model
    configuration (hand) · S4 fold-level FAR · S5 sequence tier · S6 Lin case
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "tr-corpus", "registry")
RES = os.path.join(ROOT, "tr-corpus", "results")
COR = os.path.join(ROOT, "tr-corpus")
FIG = os.path.join(ROOT, "paper", "figures")
TAB = os.path.join(ROOT, "paper", "tables")
sys.path[:0] = [os.path.join(ROOT, "trbench")]
import co_occurrence as CO  # noqa: E402
import common as C          # noqa: E402

# canonical inputs -- one place to change if a supplement is superseded
# Internal design, all six learned models on identical inputs (protocol 20):
# the three tree files are the frozen originals re-scored together with the
# matched sequence-model run, so the tree numbers are unchanged.
E3A = os.path.join(RES, "summary_20260925_all7")
E3A_RELABEL = os.path.join(RES, "validation_20260925_all7_rescored")
E3A_RAW = [os.path.join(E3A_RELABEL, f) for f in
           ("e3a_trees_s0to4.csv", "e3a_seq_s0to4.csv")]
LEARNED = ["xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer"]
E3B = os.path.join(RES, "summary_20260925_e3b")
TREES = os.path.join(RES, "validation_20260925_native")
SURFACE = os.path.join(RES, "validation_20260925_common_surface")
SEQ = os.path.join(RES, "validation_20260925_seq")      # four sequence models, same build and representations as TREES/SURFACE
MARKER = {"xgboost": "o", "lightgbm": "s", "gru": "^", "mamba": "v", "itransformer": "D", "convtransformer": "P"}
PALETTE = {"xgboost": "#2b6cb0", "lightgbm": "#63b3ed", "gru": "#b8332f", "mamba": "#dd6b20", "itransformer": "#805ad5", "convtransformer": "#2f855a"}
SHORT = {"xgboost": "XGB", "lightgbm": "LGBM", "gru": "GRU", "mamba": "Mamba", "itransformer": "iTr", "convtransformer": "ConvTr"}
DS01_REPORT = os.path.join(ROOT, "reports", "ds01_event_reference", "event_timing_summary.json")
CLUSTER = os.path.join(RES, "summary_20260925_cluster_bootstrap")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})
MODEL_NAME = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "rule": "Rule",
              "gru": "GRU", "mamba": "Mamba", "itransformer": "iTransformer",
              "convtransformer": "ConvTransformer"}
GROUP_NAME = {"T_surface": "Surface T", "T_internal": "Internal T", "T_vent": "Vent T",
              "P_internal": "In-cell P", "P_chamber": "Chamber P", "V": "Voltage",
              "I": "Current", "gas_speciated": "Speciated gas",
              "gas_derived": "Derived gas", "F_expansion": "Expansion F"}
DNAME = {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3",
         "ds04_osf": "D4", "ds09_mech": "D5", "ds12_arc": "D6"}
LABELS = [("t_onset_L1", "L1 main release"), ("t_onset_L2_0.5", "L2$_{0.5}$"),
          ("t_onset_L2_1.0", "L2"), ("t_onset_L2_2.0", "L2$_{2}$"),
          ("t_onset_L3", "L3 voltage"), ("t_isc", "25 mV drop")]
GREY, RED, BLUE = "#718096", "#b8332f", "#2b6cb0"


# ------------------------------------------------------------------ helpers
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("  %s.png/.pdf" % name)


def save_table(df, name, note=None):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    body = "\n".join(lines)
    if note:
        body = note.rstrip() + "\n\n" + body
    with open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8") as f:
        f.write(body + "\n")
    print("  %s.csv/.md (%d rows)" % (name, len(df)))


def rng(s, fmt="%g"):
    s = pd.Series(s).dropna()
    if not len(s):
        return ""
    lo, hi = s.min(), s.max()
    return fmt % lo if lo == hi else (fmt % lo) + "–" + (fmt % hi)


def srng(s, fmt="%+g"):
    s = pd.Series(s).dropna()
    if not len(s):
        return ""
    lo, hi = s.min(), s.max()
    return (fmt % lo) if lo == hi else (fmt % lo) + " to " + (fmt % hi)


def registry():
    reg = pd.read_csv(os.path.join(REG, "experiments.csv"))
    return reg[reg.role != "pretrain_only"].copy()


# ------------------------------------------------------------------ F1
def fig1_framework():
    # Retain the asset name; LaTeX assigns the final manuscript figure number.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "framework_figure", os.path.join(ROOT, "paper", "framework_figure.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.draw_framework(FIG)


# ------------------------------------------------------------------ F2
def fig2_cooccurrence():
    reg = pd.read_csv(os.path.join(REG, "experiments.csv"))
    matrix, pairs = CO.compute(reg)
    names = list(matrix.index)
    n = len(names)
    M = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    cmap = matplotlib.colormaps["Blues"]
    gap = .06                                    # white gutter between cells
    for i in range(n):
        for j in range(i + 1):
            v = int(M[i, j])
            # never co-observed: white cell with 0; otherwise shaded on a log scale
            face = "white" if v == 0 else cmap(np.log10(v + 1) / np.log10(400))
            ax.add_patch(plt.Rectangle((j - .5 + gap, i - .5 + gap), 1 - 2 * gap, 1 - 2 * gap,
                                       facecolor=face, edgecolor="#5F6F7D" if i == j else "#9AAAB8",
                                       lw=.9 if i == j else .6))
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > 30 else "black", fontweight="bold" if i == j else "normal")
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5); ax.set_aspect("equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([GROUP_NAME[g] for g in names], rotation=90, ha="center")
    ax.set_yticklabels([GROUP_NAME[g] for g in names]); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save_fig(fig, "F2_cooccurrence")
    save_table(pairs, "F2_cooccurrence_pairs", "Source: tr-corpus/registry/experiments.csv via trbench/co_occurrence.py")


# ------------------------------------------------------------------ F3 / S2
def onset_intervals(reg):
    """Pairwise (row − column) onset-time intervals over cells carrying both."""
    rows = []
    for ci, ni in LABELS:
        for cj, nj in LABELS:
            if ci == cj:
                continue
            d = (reg[ci] - reg[cj]).dropna()
            rows.append(dict(event_a=ni, event_b=nj, n=len(d),
                             median_s=d.median() if len(d) else np.nan,
                             p10_s=d.quantile(.1) if len(d) else np.nan,
                             p90_s=d.quantile(.9) if len(d) else np.nan))
    return pd.DataFrame(rows)


def fig3_reference_events(reg):
    iv = onset_intervals(reg)
    save_table(iv.round(1), "S2_onset_intervals",
               "Model-independent pairwise onset intervals (event A − event B, seconds) over the "
               "measured cells that carry both labels. Source: tr-corpus/registry/experiments.csv.")
    names = [n for _, n in LABELS]
    k = len(names)
    med = np.full((k, k), np.nan); cnt = np.zeros((k, k), int)
    for _, r in iv.iterrows():
        i, j = names.index(r.event_a), names.index(r.event_b)
        med[i, j], cnt[i, j] = r.median_s, r.n

    ls = pd.read_csv(os.path.join(E3A, "label_sensitivity.csv"))
    ls = ls[(ls.scope == "overall") & ls.model.isin(LEARNED)]      # the rate rule is not one of the six learned models
    order = ["L2@0.5", "L2@2.0", "L1 venting", "L3 voltage", "ISC 25mV"]
    disp = {"ISC 25mV": "25 mV drop", "L2@0.5": "L2$_{0.5}$", "L2@2.0": "L2$_{2}$", "L1 venting": "L1 main release"}
    flips = [(disp.get(l, l), ls[ls.comparison_label == l].n_flip, int(ls[ls.comparison_label == l].n_unique_cells.iloc[0])) for l in order]

    # wspace leaves room for the (b) row labels, which otherwise run into the (a) heat map
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.62))
    ax = axes[0]
    lim = 400
    cmap = matplotlib.colormaps["RdBu_r"].copy(); cmap.set_bad("#f4f6f8")
    ax.imshow(np.clip(med, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="equal")
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            if cnt[i, j] == 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color=GREY)
            else:
                ax.text(j, i, "%+.0f\n(n=%d)" % (med[i, j], cnt[i, j]), ha="center", va="center",
                        fontsize=5.6, color="white" if abs(med[i, j]) > 200 else "black")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(names, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax = axes[1]
    y = np.arange(len(flips))
    for yi, (name, s, n) in zip(y, flips):
        ax.plot([s.min(), s.max()], [yi, yi], color=BLUE, lw=2.2, solid_capstyle="round")
        ax.scatter(s, [yi] * len(s), s=10, color=BLUE, zorder=3, linewidths=0)
        ax.text(max(s.max(), 0.4) + 0.4, yi, "of %d" % n, va="center", fontsize=6.8, color=GREY)
    ax.set_yticks(y); ax.set_yticklabels([f[0] for f in flips], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(-0.3, max(23, float(max(f[1].max() for f in flips)) + 2))  # never clip the largest range
    ax.set_xlabel("cells whose detection status differs from L2 (same alarm)")
    ax.grid(axis="x", lw=0.4, alpha=.5)
    # the square heat map ends lower than (b); put both titles on one line (row − column is in the caption)
    fig.canvas.draw()
    top = max(a.get_position().y1 for a in axes)
    for a, title in zip(axes, ("(a) Median onset interval (s)", "(b) Status flips of a fixed alarm")):
        box = a.get_position()
        a.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8, y=(top - box.y0) / box.height)
    save_fig(fig, "F3_reference_events")  # the (b) design note is in the caption


def table_s2_definitions(reg):
    """Persistence sensitivity, guard effect, and the dataset-1 check."""
    rows, shifts = [], []
    for _, r in reg.iterrows():
        if not np.isfinite(r["t_onset_L2_1.0"]):
            continue
        df = pd.read_parquet(os.path.join(COR, r.file), columns=["time_s", "T_surface_max"])
        ceiling = r.T_censored_at if np.isfinite(r.T_censored_at) else None
        t3 = C.onset_L2(df.time_s.values, df.T_surface_max.values, 1.0, sustain_s=3.0, ceiling=ceiling)
        shifts.append(dict(key=r.dataset_id + "/" + r.experiment_id, dataset=DNAME[r.dataset_id],
                           t2=r["t_onset_L2_1.0"], t3=t3,
                           shift=(t3 - r["t_onset_L2_1.0"]) if t3 is not None else np.nan,
                           unguarded=r.t_onset_L2_unguarded,
                           guard_shift=r["t_onset_L2_1.0"] - r.t_onset_L2_unguarded))
    sh = pd.DataFrame(shifts)
    for ds, g in list(sh.groupby("dataset")) + [("all", sh)]:
        s = g["shift"]
        rows.append(dict(dataset=ds, n=len(g), lost_at_3s=int(s.isna().sum()),
                         shift_median_s=s.median(), shift_p90_s=s.quantile(.9), shift_max_s=s.max(),
                         guard_effect_median_s=g.guard_shift.median(), guard_effect_max_s=g.guard_shift.max()))
    save_table(pd.DataFrame(rows).round(1), "S2_l2_persistence",
               "L2@1.0 with the persistence requirement raised from 2 s (registry) to 3 s, recomputed on the "
               "standardized 1 Hz corpus with the same smoothing, floor (60 °C), rise guard (50 °C within 60 s) "
               "and censoring ceiling. 'shift' = t(3 s) − t(2 s); 'guard effect' = guarded − unguarded crossing.")
    sh.to_csv(os.path.join(TAB, "S2_l2_persistence_cells.csv"), index=False)

    rep = json.load(open(DS01_REPORT, encoding="utf-8"))
    rows = []
    for e in rep:
        r = reg[(reg.dataset_id == "ds01_bak") & (reg.experiment_id == e["test"])].iloc[0]
        rows.append({"cell": e["test"], "report: heater on": e["heater_on_s"], "registry t_trigger": round(r.t_trigger, 1),
                     "report: first vent": e["first_vent_s"], "report: main gas release (90 % start)": e["main_gas_90_start_s"],
                     "report: pressure peak": e["pressure_peak_s"],
                     "registry L1 (chamber P)": r.t_onset_L1, "registry L2@1.0": r["t_onset_L2_1.0"],
                     "L1 − main release": round(r.t_onset_L1 - e["main_gas_90_start_s"], 1),
                     "L1 − first vent": round(r.t_onset_L1 - e["first_vent_s"], 1)})
    save_table(pd.DataFrame(rows), "S2_ds01_label_check",
               "Registry labels for D1 against the event times tabulated in the source report "
               "(Golubkov 2026, report pages 40–44; extracted in reports/ds01_event_reference). The chamber-pressure "
               "L1 rule lands on the main gas release, not on the first (minor) vent, which the pressure trace does "
               "not resolve at 1 Hz.")


# ------------------------------------------------------------------ T2
def table2_increments():
    a = pd.read_csv(os.path.join(E3A, "panel_pairs.csv"))
    a = a[(a.scope == "overall") & (a.label == "L2@1.0") & (a.panel == "M2")]
    raw = pd.concat([pd.read_csv(p) for p in E3A_RAW])
    raw = raw[raw.model.isin(LEARNED) & (raw.label == "L2@1.0")]
    rows = []
    for m in LEARNED:
        s = a[a.model == m]
        fc = raw[raw.model == m].groupby(["panel", "seed"]).far_check.median()
        rows.append({
            "comparison": "M2 − M1 (add voltage)", "cells": str(int(s.n_unique_cells.iloc[0])), "model": MODEL_NAME[m],
            "M1 detected": rng(s.baseline_detected, "%d"), "panel detected": rng(s.panel_detected, "%d"),
            "rescued / lost": rng(s.rescue, "%d") + " / " + rng(s.lost, "%d"),
            "Δ detection rate": srng(s.detection_rate_difference, "%+.3f"),
            "95% CI covers 0": "%d of %d seeds" % (int(((s.detection_diff_ci_low <= 0) & (s.detection_diff_ci_high >= 0)).sum()), len(s)),
            "McNemar p": rng(s.mcnemar_exact_p, "%.2f"),
            "Δlead median (s), both detected": srng(s.conditional_lead_delta_median_s),
            "Δlead CI covers 0": "%d of %d seeds" % (int(((s.conditional_lead_delta_ci_low_s <= 0) & (s.conditional_lead_delta_ci_high_s >= 0)).sum()), len(s)),
            "fold-median check FAR, M1 / M2": rng(fc.loc["M1"], "%.2f") + " / " + rng(fc.loc["M2"], "%.2f"),
        })
    b = pd.read_csv(os.path.join(E3B, "panel_pairs.csv"))
    b = b[(b.scope == "overall") & (b.label == "L2@1.0")]
    names = {"M2": "M2 − M1 (add voltage)", "M3": "M3 − M1 (add chamber pressure)",
             "M4": "M4 − M1 (add derived gas)", "M5": "M5 − M1 (add both)"}
    for p in ["M2", "M3", "M4", "M5"]:
        for m in ["xgboost", "lightgbm"]:
            s = b[(b.panel == p) & (b.model == m)]
            rows.append({
                "comparison": names[p] + ", D1 only", "cells": "8", "model": MODEL_NAME[m],
                "M1 detected": rng(s.baseline_detected, "%d"), "panel detected": rng(s.panel_detected, "%d"),
                "rescued / lost": rng(s.rescue, "%d") + " / " + rng(s.lost, "%d"),
                "Δ detection rate": "saturated", "95% CI covers 0": "—", "McNemar p": "—",
                "Δlead median (s), both detected": srng(s.conditional_lead_delta_median_s),
                "Δlead CI covers 0": "%d of %d seeds" % (int(((s.conditional_lead_delta_ci_low_s <= 0) & (s.conditional_lead_delta_ci_high_s >= 0)).sum()), len(s)),
                "fold-median check FAR, M1 / M2": "",
            })
    note = ("Source: %s/panel_pairs.csv (adjudicated internal cohort) and %s/panel_pairs.csv (D1, corrected gas time origin); "
            "fold-level check FAR from the E3-A evaluator rows. Ranges over seeds 0–4; CIs from 10,000 "
            "held-experiment bootstrap resamples; individual folds' check FAR spans 0–0.31."
            % (os.path.relpath(E3A, ROOT).replace(os.sep, "/"), os.path.relpath(E3B, ROOT).replace(os.sep, "/")))
    save_table(pd.DataFrame(rows), "T2_sensor_increments", note)


# ------------------------------------------------------------------ F4 / S4
def held_frames():
    """Held-dataset outputs of all seven models on one build (W60_native).

    Trees and rule from TREES (no_age / with_age) and SURFACE (source max / mean);
    the four sequence models from SEQ, which carries all four representations.
    """
    frames = {}
    for name in ("positive_summary", "negative_rates", "negative_cells"):
        parts = [pd.read_csv(os.path.join(d, name + ".csv")) for d in (TREES, SURFACE, SEQ)]
        frames[name] = pd.concat(parts, ignore_index=True)
    return frames


def pool_n(frame, pool, col):
    """The single pool size behind a figure label, read from the plotted rows (never hard-coded)."""
    vals = frame.loc[frame.pool == pool, col].dropna().astype(int).unique()
    assert len(vals) == 1, "pool %s has sizes %s" % (pool, vals)
    return int(vals[0])


def fig4_false_alarm_timing():
    hf = held_frames()
    cells = hf["negative_cells"]; rates_all = hf["negative_rates"]
    cells = cells[(cells.age_mode == "no_age") & (cells.window_s.astype(str) == "full") & cells.model.isin(LEARNED)]
    rates = rates_all[(rates_all.age_mode == "no_age") & rates_all.model.isin(LEARNED)]
    full_rates = rates[rates.window_s.astype(str) == "full"]
    n_src, n_d6 = pool_n(full_rates, "source_mech", "n_eligible"), pool_n(full_rates, "arc", "n_eligible")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.42))
    ax = axes[0]
    grid = np.logspace(1, np.log10(200000), 300)

    def curves(g):
        out = []
        for _, gs in g.groupby("seed"):
            fa = np.nan_to_num(gs.first_alarm_s.values.astype(float), nan=np.inf)
            out.append([(fa <= t).sum() / len(gs) for t in grid])
        return np.array(out)

    # source negatives: one band over all six models and their seeds
    src = np.vstack([curves(g) for _, g in cells[cells.pool == "source_mech"].groupby("model")])
    ax.fill_between(grid, src.min(axis=0), src.max(axis=0), color=GREY, alpha=.18, lw=0)
    ax.plot(grid, np.median(src, axis=0), color=GREY, lw=1.4, label="source negatives, all models (n = %d)" % n_src)
    # ARC negatives: the median curve of each model
    for m in LEARNED:
        c = curves(cells[(cells.pool == "arc") & (cells.model == m)])
        ax.plot(grid, np.median(c, axis=0), color=PALETTE[m], lw=1.2, label="%s, D6 (n = %d)" % (MODEL_NAME[m], n_d6))
    ax.axvline(300, color="#cbd5e0", lw=0.8); ax.text(280, 0.30, "300 s", fontsize=6.5, color=GREY, va="top", ha="right")
    ax.axvspan(300, 3800, color="#edf2f7", zorder=0)
    ax.text(1100, 0.62, "source records end\n(300–3,800 s)", fontsize=6.2, color=GREY, va="top", ha="center")
    ax.set_xscale("log"); ax.set_xlim(10, 2e5); ax.set_ylim(0, 1.0)
    ax.set_xlabel("time since start of record (s)"); ax.set_ylabel("cumulative fraction alarmed")
    ax.set_title("(a) Time to first false alarm, full record", loc="left", fontsize=10, fontweight="bold", pad=8)
    # legends sit below the panels so no curve or marker crosses their text
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              columnspacing=1.2, handlelength=1.6)
    ax.grid(lw=0.4, alpha=.5)

    ax = axes[1]
    segs = [("300", "head", "first 300 s"), ("300", "tail", "last 300 s"), ("full", "full", "full record")]
    x = np.arange(len(segs)); step = 0.11
    for k, m in enumerate(LEARNED):
        off = (k - 2.5) * step
        for pool, filled in (("source_mech", False), ("arc", True)):
            vals, lo, hi = [], [], []
            for ws, anc, _ in segs:
                r = rates[(rates.model == m) & (rates.pool == pool) & (rates.window_s.astype(str) == ws) & (rates.anchor == anc)]
                vals.append(r.far.median()); lo.append(r.far.min()); hi.append(r.far.max())
            vals, lo, hi = map(np.array, (vals, lo, hi))
            ax.errorbar(x + off, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=PALETTE[m], elinewidth=0.6, capsize=0, alpha=.8)
            ax.scatter(x + off, vals, marker=MARKER[m], s=18, zorder=3, linewidths=0.7,
                       facecolors=PALETTE[m] if filled else "white", edgecolors=PALETTE[m],
                       label=("%s" % MODEL_NAME[m]) if filled else None)
    ax.scatter([], [], marker="o", facecolors="white", edgecolors=GREY, s=18, label="hollow: source (n = %d)" % n_src)
    ax.scatter([], [], marker="o", facecolors=GREY, edgecolors=GREY, s=18, label="filled: D6 (n = %d)" % n_d6)
    ax.set_xticks(x); ax.set_xticklabels([s[2] for s in segs]); ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("fraction of negatives alarming"); ax.set_title("(b) Matched exposure", loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              columnspacing=1.2)
    ax.grid(axis="y", lw=0.4, alpha=.5)
    save_fig(fig, "F4_false_alarm_timing")  # the design note is in the caption

    rows = []
    for m in ["rule"] + LEARNED:
        for pool, pname in [("source_mech", "source check negatives"), ("arc", "ARC negatives")]:
            for ws, anc, sname in segs:
                r = rates_all[(rates_all.age_mode == "no_age") & (rates_all.model == m) & (rates_all.pool == pool)
                              & (rates_all.window_s.astype(str) == ws) & (rates_all.anchor == anc)]
                if not len(r):
                    continue
                rows.append({"model": MODEL_NAME[m], "negative pool": pname, "segment": sname,
                             "eligible": rng(r.n_eligible, "%d"), "alarming": rng(r.n_alarm, "%d"), "FAR": rng(r.far, "%.3f"),
                             "Wilson 95% (min lo – max hi)": "%.3f – %.3f" % (r.far_ci_lo.min(), r.far_ci_hi.max()),
                             "first alarm, median s": rng(r.first_alarm_conditional_median_s, "%g")})
    save_table(pd.DataFrame(rows), "S4_far_pool_segment",
               "Sources: %s and %s, negative_rates.csv, sensor values only, seeds 0–4 (rule: seed 0)."
               % (os.path.relpath(TREES, ROOT).replace(os.sep, "/"), os.path.relpath(SEQ, ROOT).replace(os.sep, "/")))


# ------------------------------------------------------------------ F5
def fig5_representation():
    """Change in held-out scores when one source-side setting changes, drawn like Fig. F4b.

    Marker: mean over seeds 0-4 of the paired change (second setting minus first,
    same seed); whisker: the seed range.  Hollow: source pools, filled: D6.
    """
    from matplotlib.lines import Line2D

    hf = held_frames()
    pos = hf["positive_summary"]
    neg = hf["negative_rates"]
    pos = pos[pos.model.isin(LEARNED)]
    neg = neg[neg.model.isin(LEARNED)]
    full = neg[neg.window_s.astype(str) == "full"]
    tail = neg[(neg.window_s.astype(str) == "300") & (neg.anchor == "tail")]
    # x categories; each holds a (source, D6) pair of pools
    cats = [("recall", pos, "n_events", "n_detected", ("source_test", "arc")),
            ("false alarms,\nfull record", full, "n_eligible", "n_alarm", ("source_mech", "arc")),
            ("false alarms,\nlast 300 s", tail, "n_eligible", "n_alarm", ("source_mech", "arc"))]
    changes = [(("surface_max_only", "surface_mean_only"), "(a) Max → mean"),
               (("no_age", "with_age"), "(b) Adding observation age")]

    def scores(frame, model, rep, pool, ncol, kcol):
        sub = frame[(frame.model == model) & (frame.age_mode == rep) & (frame.pool == pool)].sort_values("seed")
        assert len(sub) == 5, (model, rep, pool, len(sub))
        y = (sub[kcol] / sub[ncol]).to_numpy()
        assert np.isfinite(y).all() and ((0 <= y) & (y <= 1)).all()
        return sub.seed.to_numpy(), y

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    x = np.arange(len(cats)); step = 0.12; split = 0.024
    for ax, ((rep_a, rep_b), title) in zip(axes, changes):
        for k, m in enumerate(LEARNED):
            off = (k - 2.5) * step
            for pool_i, (filled, dx) in enumerate(((False, -split), (True, split))):
                means, lo, hi = [], [], []
                for _, frame, ncol, kcol, pools in cats:
                    sa, ya = scores(frame, m, rep_a, pools[pool_i], ncol, kcol)
                    sb, yb = scores(frame, m, rep_b, pools[pool_i], ncol, kcol)
                    assert (sa == sb).all(), (m, pools[pool_i])      # paired by seed
                    d = yb - ya
                    means.append(d.mean()); lo.append(d.min()); hi.append(d.max())
                means, lo, hi = map(np.array, (means, lo, hi))
                xs = x + off + dx
                ax.errorbar(xs, means, yerr=[means - lo, hi - means], fmt="none", ecolor=PALETTE[m],
                            elinewidth=0.6, capsize=0, alpha=.8)
                ax.scatter(xs, means, marker=MARKER[m], s=18, zorder=3, linewidths=0.7,
                           facecolors=PALETTE[m] if filled else "white", edgecolors=PALETTE[m])
        ax.axhline(0, color="#4a5568", lw=0.8, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cats])
        ax.set_xlim(-0.5, len(cats) - 0.5); ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, -.5, 0, .5, 1]); ax.set_yticklabels(["−1", "−0.5", "0", "+0.5", "+1"])
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
        ax.grid(axis="y", lw=0.4, alpha=.5)
        for b in (0.5, 1.5):
            ax.axvline(b, color="#edf0f3", lw=0.6, zorder=0)
    axes[0].set_ylabel("change in fraction (5-seed mean)")
    handles = [Line2D([], [], marker=MARKER[m], color=PALETTE[m], linestyle="none", markersize=5, label=MODEL_NAME[m])
               for m in LEARNED]
    handles += [Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=GREY, color=GREY,
                       linestyle="none", markersize=5, label="hollow: source"),
                Line2D([], [], marker="o", color=GREY, linestyle="none", markersize=5, label="filled: D6")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False,
               fontsize=7, columnspacing=1.4, handletextpad=.4)
    # SVG keeps labels editable in Inkscape. No explanatory footer in the art.
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(os.path.join(FIG, "F5_representation.svg"), bbox_inches="tight")
    save_fig(fig, "F5_representation")  # the design note is in the caption


# ------------------------------------------------------------------ F6 / S5
def fig6_robustness():
    hf = held_frames()
    pos = hf["positive_summary"]; neg = hf["negative_rates"]
    pos = pos[(pos.age_mode == "no_age") & pos.model.isin(LEARNED)]; neg = neg[(neg.age_mode == "no_age") & neg.model.isin(LEARNED)]
    models = LEARNED
    markers, colours = MARKER, PALETTE
    n_st, n_d6p = pool_n(pos, "source_test", "n_events"), pool_n(pos, "arc", "n_events")
    n_d6 = pool_n(neg[neg.window_s.astype(str) == "full"], "arc", "n_eligible")

    # Average the two plotted coordinates across the same five seeds.
    # Fits and thresholds stay seed-specific; only the displayed scores are averaged.
    mean_scores = {}
    for m in models:
        src = pos[(pos.model == m) & (pos.pool == "source_test")].set_index("seed").sort_index()
        arc = pos[(pos.model == m) & (pos.pool == "arc")].set_index("seed").sort_index()
        tail = neg[(neg.model == m) & (neg.pool == "arc") &
                   (neg.window_s.astype(str) == "300") & (neg.anchor == "tail")].set_index("seed").sort_index()
        for frame in (src, arc, tail):
            assert frame.index.tolist() == list(range(5)), (m, frame.index.tolist())
        mean_scores[m] = (float(src.recall.mean()), float(arc.recall.mean()), float(tail.far.mean()))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    fig.subplots_adjust(left=.09, right=.98, bottom=.26, top=.77, wspace=.48)
    ax = axes[0]
    for m in models:
        src_mean, rec_mean, far_mean = mean_scores[m]
        ax.scatter(far_mean, rec_mean, marker=markers[m], color=colours[m],
                   s=38, linewidths=0, alpha=.95, label=MODEL_NAME[m])
    ax.set_xlabel("D6 negatives alarming in the last 300 s\n(fraction of %d)" % n_d6)
    ax.set_ylabel("D6 events detected (fraction of %d)" % n_d6p)
    ax.set_title("(a) External recall against\nmatched-window false alarms", loc="left",
                 fontsize=10, fontweight="bold", y=1.20, pad=0, va="top", linespacing=1.2)
    ax.set_xlim(-.03, 1.05); ax.set_ylim(-.03, 1.05); ax.grid(lw=0.4, alpha=.5)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=7, frameon=False,
               loc="upper center", bbox_to_anchor=(.52, .06), ncol=3,
               columnspacing=1.8, handletextpad=.5)

    ax = axes[1]
    for m in models:
        src_mean, rec_mean, far_mean = mean_scores[m]
        ax.scatter(src_mean, rec_mean, marker=markers[m], color=colours[m],
                   s=38, linewidths=0, alpha=.95)
    ax.plot([0, 1], [0, 1], color="#cbd5e0", lw=0.6)
    ax.set_xlabel("Source test events detected\n(fraction of %d)" % n_st); ax.set_ylabel("D6 events detected (fraction of %d)" % n_d6p)
    ax.set_title("(b) Source vs external recall", loc="left",
                 fontsize=10, fontweight="bold", y=1.20, pad=0, va="top")
    ax.set_xlim(-.03, 1.05); ax.set_ylim(-.03, 1.05); ax.grid(lw=0.4, alpha=.5)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(os.path.join(FIG, "F6_robustness.svg"), bbox_inches="tight")
    save_fig(fig, "F6_robustness")

    rows = []
    for m in models:
        p = pos[pos.model == m]; n = neg[neg.model == m]
        src, arc = p[p.pool == "source_test"], p[p.pool == "arc"]

        def far(pool, ws, anc):
            s = n[(n.pool == pool) & (n.window_s.astype(str) == ws) & (n.anchor == anc)]
            return rng(s.n_alarm, "%d") + "/" + str(int(s.n_eligible.iloc[0]))
        rows.append({"model": MODEL_NAME[m], "seeds": len(src),
                     "source test detected / %d" % int(src.n_events.iloc[0]): rng(src.n_detected, "%d"), "ARC detected / %d" % int(arc.n_events.iloc[0]): rng(arc.n_detected, "%d"),
                     "ARC conditional lead, median s": rng(arc.lead_median, "%g"),
                     "ARC FAR full": far("arc", "full", "full"), "ARC FAR first 300 s": far("arc", "300", "head"),
                     "ARC FAR last 300 s": far("arc", "300", "tail"), "source FAR full": far("source_mech", "full", "full")})
    save_table(pd.DataFrame(rows), "S5_model_families",
               "Sources: %s (trees) and %s (sequence models), sensor values only, same window build. Ranges over seeds 0–4."
               % (os.path.relpath(TREES, ROOT).replace(os.sep, "/"), os.path.relpath(SEQ, ROOT).replace(os.sep, "/")))


# ------------------------------------------------------------------ S1 / S2b / S4b
def table_s1_literature_audit():
    """Coding of the studies in Supplementary Table S1, from the full texts (re-verified 2026-10-06, reports/37).

    lead: the study states a time by which its signal or alarm precedes an adverse event.
    rule: "yes" when that lead is measured to an event defined by a computable rule stated in the
    paper; "stated, not tied" when a rule is stated but the reported lead is not measured to it.
    """
    rows = [  # row, lead stated, lead measured to, rule, negatives / false alarms, FAR reported, data
        ("A", "yes", "TR (tabulated time not defined)", "stated, not tied", "none", "no", "on request"),
        ("B", "yes", "venting (force peak) and TR at 10 °C/s", "yes", "none", "no", "on request"),
        ("C", "yes", "TR, qualitative", "no", "none", "no", "on request"),
        ("D", "yes", "TR (in situ peaks over 100 s before TR)", "no", "none", "no", "not stated"),
        ("E", "no", "venting, no number", "no", "none", "no", "in article / SI, rest on request"),
        ("F", "yes", "failure = venting or explosion; fraction of overcharge time", "no", "none", "no", "unavailable (confidentiality)"),
        ("G", "yes", "not stated", "no", "normal-operation windows", "yes", "no public file; inquiries"),
        ("H", "yes", "TR onset t_TR (rule for t_TR not specified)", "stated, not tied", "miss rate only", "no", "not stated"),
        ("I-a", "yes", "visible bulging (not TR)", "no", "none", "no", "no data used"),
        ("I-b", "yes", "temperature signal / TR onset, qualitative", "no", "none", "no", "on request"),
        ("I-c", "yes", "TR marked on the traces", "no", "none", "no", "on request"),
        ("J", "no", "sensor response time to a short (not TR)", "no", "none", "no", "on request"),
        ("K", "no", "venting stages (no lead)", "no", "none", "no", "public (data descriptor)"),
        ("L", "yes", "TR by dT/dt >= 1 °C/s for > 3 s", "yes", "sample-level FAR", "yes", "on request"),
        ("M", "yes", "venting / TR (simulated)", "no", "none", "no", "on request"),
        ("N", "yes", "voltage or temperature meeting the GB 38031 criterion", "yes", "one non-runaway test, no rate", "no", "on request"),
        ("O", "yes", "CID / valve / ISC failure trigger (not TR)", "no", "qualitative", "no", "on request"),
        ("P", "no", "no lead (regression)", "no", "none", "no", "on request"),
        ("Q", "yes", "TR, qualitative; four-event timeline", "no", "none", "no", "on request"),
        ("R", "yes", "TR by T >= 150 °C or dT/dt >= 3 °C/s", "yes", "non-runaway mechanical-abuse experiments", "yes", "on request"),
        ("S", "yes", "TR, qualitative", "no", "intervention-censored records", "no", "public (Figshare)"),
        ("T", "yes", "TR by voltage plunge with 1 °C/s, or 25 % voltage drop", "yes", "none", "no", "on request"),
        ("U", "yes", "TR by T2 (> 1 °C/s) under ARC; qualitative in overcharge", "yes", "none", "no", "on request"),
    ]
    df = pd.DataFrame(rows, columns=["row", "lead stated", "lead measured to", "computable rule",
                                     "negatives / false alarms", "FAR reported", "data"])
    req = {"F": "5 min (UN GTR 20) printed with its own warning times",
           "G": "5 min alarm deadline (GB 38031-2025) cited, not compared",
           "Q": "5 min pre-warning (GTR 20) cited and compared",
           "R": "> 10 s, no source given"}
    beside_regulation = {"F", "Q"}
    df["required lead"] = [req.get(r, "not stated") for r in df.row]
    lead = df["lead stated"] == "yes"
    summary = pd.DataFrame([
        {"criterion": "lead time stated", "studies": int(lead.sum()), "of": len(df)},
        {"criterion": "lead measured to an event defined by a computable rule", "studies": int((lead & (df["computable rule"] == "yes")).sum()), "of": int(lead.sum())},
        {"criterion": "computable rule stated but the lead not measured to it", "studies": int((lead & (df["computable rule"] == "stated, not tied")).sum()), "of": int(lead.sum())},
        {"criterion": "any false-alarm rate reported", "studies": int((df["FAR reported"] == "yes").sum()), "of": len(df)},
        {"criterion": "false alarms priced on non-runaway abuse tests", "studies": int((df["negatives / false alarms"] == "non-runaway mechanical-abuse experiments").sum()), "of": len(df)},
        {"criterion": "required lead stated (any source)", "studies": int((~df["required lead"].str.startswith("not stated")).sum()), "of": len(df)},
        {"criterion": "warning time placed beside a regulatory requirement", "studies": len(beside_regulation), "of": len(df)},
        {"criterion": "entries read in full", "studies": len(df), "of": len(df)},
        {"criterion": "data public", "studies": int(df["data"].str.startswith("public").sum()), "of": len(df)},
    ])
    save_table(df, "S1_literature_coding", "Coding of the studies in Supplementary Table S1, from the full texts "
               "(re-verified 2026-10-06, reports/37). 'computable rule' = the stated lead is measured to an event defined "
               "by a computable rule given in the paper.")
    save_table(summary, "S1_literature_summary", "Counts derived from S1_literature_coding.csv.")


def table_s2b_flips():
    raw = pd.concat([pd.read_csv(p) for p in E3A_RAW]); raw = raw[raw.model.isin(LEARNED)]
    ls = pd.read_csv(os.path.join(E3A, "label_sensitivity.csv")); ls = ls[(ls.scope == "overall") & ls.model.isin(LEARNED)]
    rows = []
    for lab in ["L2@1.0", "L2@0.5", "L2@2.0", "L1 venting", "L3 voltage", "ISC 25mV"]:
        d = raw[raw.label == lab].groupby(["model", "panel", "seed"]).agg(n=("held", "nunique"), k=("detected", "sum"))
        s = ls[ls.comparison_label == lab]
        rows.append({"onset definition": "25 mV drop" if lab == "ISC 25mV" else lab, "cells": rng(d.n, "%d"),
                     "detected by the same alarm": rng(d.k, "%d"), "detection rate": rng(d.k / d.n, "%.2f"),
                     "common cohort with L2@1.0": "—" if lab == "L2@1.0" else rng(s.n_unique_cells, "%d"),
                     "status flips": "reference" if lab == "L2@1.0" else rng(s.n_flip, "%d"),
                     "only this": "" if lab == "L2@1.0" else rng(s.comparison_only_detected, "%d"),
                     "only L2@1.0": "" if lab == "L2@1.0" else rng(s.l2_only_detected, "%d")})
    coh = pd.read_csv(os.path.join(E3A, "all_label_cohort.csv")); coh = coh[(coh.scope == "overall") & coh.model.isin(LEARNED)]
    save_table(pd.DataFrame(rows), "S2_fixed_alarm_flips",
               "Same alarm, six reference events; ranges over 6 learned models × 2 panels × 5 seeds. Cells carrying all six "
               "definitions: %s; of these, %s change status under at least one." % (rng(coh.n_unique_cells, "%d"), rng(coh.n_cells_with_any_detection_flip, "%d")))


def table_s4_fold_far():
    raw = pd.concat([pd.read_csv(p) for p in E3A_RAW]); raw = raw[raw.model.isin(LEARNED) & (raw.label == "L2@1.0")]
    g = raw.groupby(["model", "panel", "seed"]).far_check.agg(["median", "min", "max"]).reset_index()
    g["model"] = g.model.map(MODEL_NAME)
    save_table(g.round(3), "S4_fold_check_far",
               "Realised false-alarm rate on the adjudicated check negatives per leave-one-experiment-out fold (E3-A): "
               "median, min and max over the folds of each model–panel–seed. Pooled fold range %.2f–%.2f." % (raw.far_check.min(), raw.far_check.max()))


def table_s4_cluster():
    """Experiment-level vs cell-model cluster bootstrap (trbench/cluster_bootstrap.py)."""
    rel = os.path.relpath(CLUSTER, ROOT).replace(os.sep, "/")
    rates = pd.read_csv(os.path.join(CLUSTER, "negative_rates_cluster.csv"))
    wil = pd.read_csv(os.path.join(TREES, "negative_rates.csv"))
    wil = wil[wil.age_mode == "no_age"]
    segs = [("full", "full", "full record"), ("300", "head", "first 300 s"), ("300", "tail", "last 300 s")]
    rows = []
    for m in ["xgboost", "lightgbm"]:
        for pool, pname in [("source_mech", "source"), ("arc", "D6")]:
            for ws, anc, sname in segs:
                r = rates[(rates.model == m) & (rates.pool == pool) & (rates.window_s.astype(str) == ws) & (rates.anchor == anc)]
                w = wil[(wil.model == m) & (wil.pool == pool) & (wil.window_s.astype(str) == ws) & (wil.anchor == anc)]
                rows.append({"model": MODEL_NAME[m], "pool": pname, "segment": sname,
                             "experiments / cell models": "%d / %d" % (r.n_experiments.iloc[0], r.n_clusters.iloc[0]),
                             "alarming": rng(r.n_alarm, "%d"),
                             "Wilson 95% (experiment)": "%.3f–%.3f" % (w.far_ci_lo.min(), w.far_ci_hi.max()),
                             "cluster bootstrap 95%": "%.3f–%.3f" % (r.cluster_ci_lo.min(), r.cluster_ci_hi.max())})
    save_table(pd.DataFrame(rows), "S4_cluster_far",
               "Source: %s/negative_rates_cluster.csv. Interval columns give the lowest lower and highest upper bound across "
               "seeds 0–4. Cluster = cell model (10,000 resamples of whole cell models). A cluster interval of 0–0 means every "
               "resample had no alarm; the Wilson interval is then the informative one." % rel)

    d = pd.read_csv(os.path.join(CLUSTER, "negative_difference_cluster.csv"))
    rows = []
    for m in ["xgboost", "lightgbm"]:
        for ws, anc, sname in segs:
            g = d[(d.model == m) & (d.window_s.astype(str) == ws) & (d.anchor == anc)]
            rows.append({"model": MODEL_NAME[m], "segment": sname, "D6 − source FAR": srng(g.far_difference, "%+.3f"),
                         "experiment-level 95% (Newcombe)": "%+.3f to %+.3f" % (g.cell_ci_lo.min(), g.cell_ci_hi.max()),
                         "cluster bootstrap 95%": "%+.3f to %+.3f" % (g.cluster_ci_lo.min(), g.cluster_ci_hi.max()),
                         "cluster lower bound > 0": "%d of %d seeds" % (int((g.cluster_ci_lo > 0).sum()), len(g))})
    save_table(pd.DataFrame(rows), "S4_cluster_far_difference",
               "Source: %s/negative_difference_cluster.csv; each pool resampled by its own cell models "
               "(source 8, D6 6)." % rel)

    p = pd.read_csv(os.path.join(CLUSTER, "e3a_m2_m1_cluster.csv"))
    rows = []
    for _, r in p.sort_values(["model", "seed"]).iterrows():
        rows.append({"model": MODEL_NAME[r.model], "seed": int(r.seed), "M2 − M1 detection rate": "%+.3f" % r.detection_rate_difference,
                     "experiment bootstrap 95%": "%+.3f to %+.3f" % (r.cell_ci_lo, r.cell_ci_hi),
                     "cluster bootstrap 95%": "%+.3f to %+.3f" % (r.cluster_ci_lo, r.cluster_ci_hi),
                     "cluster interval excludes 0": "yes" if r.cluster_ci_lo > 0 else "no"})
    save_table(pd.DataFrame(rows), "S4_cluster_m2_m1",
               "Source: %s/e3a_m2_m1_cluster.csv; held cells of the adjudicated internal cohort resampled by cell model." % rel)


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    reg = registry()
    print("figures")
    fig1_framework(); fig2_cooccurrence(); fig3_reference_events(reg)
    fig4_false_alarm_timing(); fig5_representation(); fig6_robustness()
    print("tables")
    table2_increments(); table_s1_literature_audit(); table_s2_definitions(reg)
    table_s2b_flips(); table_s4_fold_far(); table_s4_cluster()


if __name__ == "__main__":
    main()
