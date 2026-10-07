"""Main-text physical timeline of one abuse test (review 2026-09-25, item 7.2).

    paper/figures/F8_timeline.{png,pdf} and F8_timeline_[a-d].{png,pdf}
    paper/tables/F8_timeline_events.{csv,md}   the event and alarm times drawn

One D1 record (BAK N21700CG-50, one-sided heating) on a single time axis:
surface and heater temperature, cell voltage, chamber pressure and derived gas
amount, and the running-maximum risk of the internal-design tree models with
their calibrated thresholds.  Vertical lines mark the trigger and the four
reference events; the shaded band is the range of first alarms over seeds.
The same issued alarm is early against L1/L2 and late against L3/25 mV, which
is the anatomy of every reference-event flip in Section 3.1.

Sources: tr-corpus/measured/ds01_bak/<record>.parquet, registry/experiments.csv,
results/validation_20260925_e3a_trees/traces (label v1.1, adjudicated pools).
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "paper")]
from make_figures import save_fig, save_table, PALETTE, MODEL_NAME, GREY, RED   # noqa: E402

RECORD = ("ds01_bak", "TS0330A")
TRACES = os.path.join(ROOT, "tr-corpus", "results", "validation_20260925_e3a_trees", "traces")
T_END = 1000.0
DARK = "#1a202c"


def load():
    ds, exp = RECORD
    d = pd.read_parquet(os.path.join(ROOT, "tr-corpus", "measured", ds, exp + ".parquet"))
    d = d[d.time_s <= T_END]
    reg = pd.read_csv(os.path.join(ROOT, "tr-corpus", "registry", "experiments.csv"), low_memory=False)
    row = reg[(reg.dataset_id == ds) & (reg.experiment_id == exp)].iloc[0]
    ev = {"trigger": row.t_trigger, "L1": row.t_onset_L1, "L2": row.t_onset_L2, "L2_0.5": row["t_onset_L2_0.5"],
          "L3": row.t_onset_L3, "25mV": row.t_isc}
    traces = {}
    for m in ("xgboost", "lightgbm", "rule"):
        traces[m] = []
        for f in sorted(glob.glob(os.path.join(TRACES, "%s_M1_s*_%s__%s.npz" % (m, ds, exp)))):
            z = np.load(f, allow_pickle=True)
            t, r, tau = z["t"], z["risk_running_max"], float(z["tau"])
            hit = t[r >= tau]
            traces[m].append(dict(t=t, r=r, tau=tau, alarm=float(hit[0]) if len(hit) else np.nan, seed=int(f.split("_s")[1].split("_")[0])))
    return d, ev, traces


def draw_panel(ax, index, d, ev, traces, a_lo, a_hi, rule_alarm):
    """Four independent coordinate systems; no shared or hidden time axes."""
    titles = ["(a) Temperature", "(b) Cell voltage",
              "(c) Pressure and gas", "(d) Model risk and rule score"]
    ax.set_title(titles[index], loc="left", fontsize=11, fontweight="bold", pad=52)
    ax.set_xlim(0, T_END)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel("Record time (s)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.axvspan(a_lo, a_hi, color="#2b6cb0", alpha=0.20, lw=0, zorder=0)
    for t, c, ls in [(ev["trigger"], GREY, ":"), (ev["L3"], RED, "-"),
                     (ev["L2_0.5"], "#4a5568", ":"), (ev["L2"], DARK, "-")]:
        ax.axvline(t, color=c, lw=0.85, ls=ls, zorder=1)
    ax.axvline(rule_alarm, color=GREY, lw=0.8, ls="--", zorder=1)
    right = None
    if index == 0:
        ax.plot(d.time_s, d.T_heater, color=GREY, lw=1.15, ls="--", label="Heater")
        ax.plot(d.time_s, d.T_vent, color="#a0aec0", lw=1.15, ls=":", label="Vent")
        ax.plot(d.time_s, d.T_surface_max, color=DARK, lw=1.6, label="Surface max")
        ax.set_ylabel("Temperature (°C)", fontsize=10)
        ax.set_ylim(0, 830)
        # Minimal event keys live above the plotting area, below the curve legend.
        for event, label, colour in (("trigger", "Heater on", GREY),
                                     ("L3", "L3 / 25 mV", RED), ("L2", "L2", DARK)):
            ax.text(ev[event], 1.025, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=colour,
                    clip_on=False, zorder=10)
    elif index == 1:
        ax.plot(d.time_s, d.V_cell, color=DARK, lw=1.6)
        ax.set_ylabel("Cell voltage (V)", fontsize=10)
        ax.set_ylim(-0.3, 5.6);ax.set_yticks([0, 2, 4])
    elif index == 2:
        ax.plot(d.time_s, d.P_chamber, color=DARK, lw=1.6, label="Chamber pressure")
        ax.set_ylabel("Chamber pressure (kPa)", fontsize=10)
        ax.set_ylim(122, 160)
        right=ax.twinx()
        right.plot(d.time_s, d.gas_total_mol.interpolate(limit_area="inside"), color=GREY,
                   lw=1.3, ls="--", label="Gas amount")
        right.set_ylabel("Gas amount (mol)", color=GREY, fontsize=10)
        right.set_ylim(5.65, 7.35)
    else:
        for m in ("xgboost", "lightgbm"):
            for k, x in enumerate(traces[m]):
                ax.plot(x["t"], x["r"], color=PALETTE[m], lw=1.1, alpha=0.85,
                        label=MODEL_NAME[m] if k==0 else None)
            taus=[x["tau"] for x in traces[m]]
            ax.axhspan(min(taus), max(taus), color=PALETTE[m], alpha=0.18, lw=0)
        ax.set_ylabel("Risk (running maximum)", fontsize=10)
        ax.set_ylim(-0.04, 1.32);ax.set_yticks([0, 0.5, 1])
        right=ax.twinx();rule=traces["rule"][0]
        right.plot(rule["t"], rule["r"], color=GREY, lw=1.2, ls="--", label="Rate rule")
        right.axhline(rule["tau"], color=GREY, lw=0.8, ls=":")
        right.set_ylabel("Rule score (°C per step)", color=GREY, fontsize=10)
        right.set_ylim(-4,184)
    if right is not None:
        right.tick_params(axis="y",colors=GREY,labelsize=9)
        right.spines["top"].set_visible(False)
    handles,labels=ax.get_legend_handles_labels()
    if right is not None:
        h,l=right.get_legend_handles_labels();handles+=h;labels+=l
    if handles:
        ax.legend(handles,labels,loc="lower left",bbox_to_anchor=(0,1.15),
                  borderaxespad=0,frameon=False,fontsize=8.2,ncol=len(handles),
                  handlelength=1.6,columnspacing=0.9)


def main():
    d, ev, traces = load()
    tree_alarms = [x["alarm"] for m in ("xgboost", "lightgbm") for x in traces[m]]
    a_lo, a_hi = float(np.nanmin(tree_alarms)), float(np.nanmax(tree_alarms))
    rule_alarm = float(np.nanmin([x["alarm"] for x in traces["rule"]]))
    for index, letter in enumerate("abcd"):
        fig, ax=plt.subplots(figsize=(4.8,3.5))
        fig.subplots_adjust(left=0.16,right=0.83,bottom=0.18,top=0.77)
        draw_panel(ax,index,d,ev,traces,a_lo,a_hi,rule_alarm)
        save_fig(fig,"F8_timeline_"+letter)
    fig, axes=plt.subplots(2,2,figsize=(10.4,7.2))
    fig.subplots_adjust(left=0.085,right=0.915,bottom=0.085,top=0.86,wspace=0.61,hspace=0.97)
    for index,ax in enumerate(axes.flat):
        draw_panel(ax,index,d,ev,traces,a_lo,a_hi,rule_alarm)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(os.path.join(ROOT, "paper", "figures", "F8_timeline.svg"), bbox_inches="tight")
    save_fig(fig,"F8_timeline")

    rows = [dict(item="trigger (heater on)", time_s=ev["trigger"]), dict(item="L3 voltage collapse", time_s=ev["L3"]), dict(item="25 mV event", time_s=ev["25mV"]),
            dict(item="L2_0.5", time_s=ev["L2_0.5"]), dict(item="L1 main release", time_s=ev["L1"]), dict(item="L2 thermal onset", time_s=ev["L2"])]
    for m in ("xgboost", "lightgbm", "rule"):
        for x in traces[m]:
            rows.append(dict(item="first alarm %s M1 seed %d (tau %.3f)" % (MODEL_NAME[m], x["seed"], x["tau"]), time_s=x["alarm"]))
    save_table(pd.DataFrame(rows), "F8_timeline_events",
               "Source: %s/%s, registry/experiments.csv, %s. Internal design, panel M1, label v1.1." % (RECORD[0], RECORD[1], os.path.relpath(TRACES, ROOT).replace(os.sep, "/")))
    print("tree alarms %.0f–%.0f s; L2 %.0f; L3 %.0f; rule %.0f" % (a_lo, a_hi, ev["L2"], ev["L3"], rule_alarm))


if __name__ == "__main__":
    main()
