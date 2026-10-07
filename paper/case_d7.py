"""Supplementary D7 case (Gardner et al. 2025): event timeline and frozen alarms.

Reads the audit outputs written by trbench/external_d7.py (rules committed in
reports/11_d7_gardner_audit_protocol.md before download) and writes

    paper/figures/SF2_d7_timeline.{png,pdf}
    paper/tables/S7_d7_events.{csv,md}
    paper/tables/S7_d7_alarms.{csv,md}
    paper/tables/S7_d7_interventions.{csv,md}

    uv run python paper/case_d7.py
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
sys.path[:0] = [os.path.join(ROOT, "trbench", "adapters"), os.path.join(ROOT, "paper")]
import d7_gardner as G                                        # noqa: E402
from make_figures import save_fig, save_table, GREY, RED, BLUE  # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results", "external_20260925_d7_gardner")
GREEN, ORANGE = "#2f855a", "#dd6b20"


def _s(x):
    return "" if pd.isna(x) else "%d" % round(float(x))


def events_table(ev, recs):
    rows = []
    P = {r.experiment_id: r for r in recs}
    for rid in ("3A", "3B"):
        e, r = ev.loc[rid], P[rid]
        t, p = r.t, r.channels["P_chamber"]
        base = np.median(p[t <= 600])
        i = int(np.nanargmax(p))
        h = r.channels["gas_H2"]
        first = t[np.flatnonzero(h > 0.5)[0]]
        v_status = "not assessable (parallel pair)" if rid in G.PARALLEL_PAIR else ""
        items = [
            ("H2 > 0.5 ppm, first sample (descriptive)", first, ""),
            ("H2 onset, 10 ppm held 5 s (pre-specified)", e["H2 onset (computed)"], ""),
            ("chamber pressure maximum (descriptive)", t[i], "+%.2f hPa (baseline sd %.3f)" % (p[i] - base, np.std(p[t <= 600]))),
            ("L2 unguarded 1 degC/s crossing", e["L2 unguarded"], ""),
            ("L2@1.0 (guarded, registry rule)", e["L2@1.0"], "" if pd.notna(e["L2@1.0"]) else "not computable: probe lost at %s s, peak %.1f degC" % (_s(e["rtd_lost_s"]), e["T peak (degC)"])),
            ("L3 voltage collapse", e["L3 (computed)"], v_status),
            ("25 mV drop", e["25 mV (computed)"], (v_status or "fires within board voltage noise (sd %.0f mV)" % e["V noise sd, first 120 s (mV)"])),
            ("L1 chamber-pressure rule", e["L1 chamber (computed)"], ("rule output on a %.2f hPa trace" % e["P max rise over baseline (hPa)"]) if e["P max rise over baseline (hPa)"] < 1 else "pressure burst at TR; the first vent was not resolved (<0.1 hPa, source paper)"),
            ("RTD signal lost", e["rtd_lost_s"], ""),
            ("TR, author lead added to pre-specified H2 onset (protocol anchor)", e["TR author-derived (s)"], "after file end (%d s); not observable" % e["duration_s"] if e["TR author-derived (s)"] > e["duration_s"] else ""),
            ("record end", e["duration_s"], ""),
        ]
        if rid == "3B":
            # The source measures its 3.6 min from the first venting, which coincides with the case-temperature
            # discontinuity (source paper p. 3; reports/37): unguarded crossing to first H2 sample.
            lead_s = 60 * G.AUTHOR_H2_LEAD_MIN[rid]
            lo, hi = float(e["L2 unguarded"]), float(first)
            items.insert(-1, ("TR, author lead added to the authors' anchor (first venting)", lo + lead_s,
                              "first venting %s-%s s (unguarded crossing to first H2 sample) + %.1f min = %s-%s s; "
                              "coincides with RTD loss and L3" % (_s(lo), _s(hi), G.AUTHOR_H2_LEAD_MIN[rid], _s(lo + lead_s), _s(hi + lead_s))))
        for name, val, note in items:
            rows.append({"record": rid, "event": name, "time (s)": _s(val),
                         "minus H2 onset (s)": ("%+d" % round(float(val) - float(e["H2 onset (computed)"]))) if pd.notna(val) else "",
                         "note": note})
    return pd.DataFrame(rows)


def alarm_table(al):
    refs = ["H2 onset (computed)", "L1 chamber (computed)", "L2@1.0", "L3 (computed)", "25 mV (computed)", "TR author-derived (s)"]
    rows = []
    for (model, rid), g in al.groupby(["model", "record"], sort=False):
        row = {"model": model, "record": rid, "seeds": len(g),
               "tau reproduced": "%d/%d" % (g.tau_reproduced.sum(), len(g)),
               "first alarm (s)": "–".join(sorted({_s(x) if pd.notna(x) else "none" for x in g.first_alarm_s}, key=lambda s: (s == "none", float(s) if s != "none" else 0)))}
        for ref in refs:
            vals = g[ref].astype(str)
            det = vals.str.startswith("detected").sum()
            if vals.str.startswith("not").all():
                row[ref] = vals.iloc[0]
            else:
                leads = [float(v.split("lead ")[1].split(" s")[0]) for v in vals if v.startswith("detected")]
                row[ref] = "%d/%d%s" % (det, len(g), (" (lead %s s)" % ("%d" % leads[0] if min(leads) == max(leads) else "%d to %d" % (min(leads), max(leads)))) if leads else "")
        rows.append(row)
    return pd.DataFrame(rows)


def figure(ev, recs, al):
    from matplotlib.ticker import LogLocator, NullLocator
    P = {r.experiment_id: r for r in recs}
    fig = plt.figure(figsize=(7.4, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], hspace=0.62, wspace=0.62)
    for row, (rid, xlim) in enumerate((("3A", (2800, 5000)), ("3B", (1150, 1560)))):
        r, e = P[rid], ev.loc[rid]
        t = r.t
        ax = fig.add_subplot(gs[row, 0])
        T = r.channels["T_case"].astype(float).copy()
        T[T >= 880] = np.nan
        ax.plot(t, T, color="black", lw=1.0)
        ax.set_ylabel("case T (°C)", fontsize=7.5)
        ax.set_xlim(*xlim)
        sel = (t >= xlim[0]) & (t <= xlim[1])
        ax.set_ylim(0, np.nanmax(T[sel]) * 1.08)
        a2 = ax.twinx()
        a2.plot(t, r.channels["V_cell"], color=BLUE, lw=0.5, alpha=0.8)
        a2.set_ylim(-0.2, 5.6)
        a2.set_ylabel("V%s" % (" (parallel pair)" if rid in G.PARALLEL_PAIR else ""), fontsize=7, color=BLUE)
        a2.tick_params(labelsize=6.5, colors=BLUE)
        a3 = ax.twinx()
        a3.spines["right"].set_position(("axes", 1.10))
        a3.plot(t, np.clip(r.channels["gas_H2"], 0.1, None), color=GREEN, lw=0.8)
        a3.set_yscale("log")
        a3.set_ylim(0.1, 3e4)
        a3.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        a3.yaxis.set_minor_locator(NullLocator())
        a3.set_ylabel("H$_2$ (ppm)", fontsize=7, color=GREEN)
        a3.tick_params(labelsize=6.5, colors=GREEN)
        handles = [plt.Line2D([], [], color="black", lw=1.0, label="case RTD"),
                   plt.Line2D([], [], color=BLUE, lw=0.8, label="voltage"),
                   plt.Line2D([], [], color=GREEN, lw=0.8, label="H$_2$")]
        marks = [("H2 onset (computed)", GREEN, "--", "H$_2$ 10 ppm (pre-specified)"),
                 ("L2 unguarded", GREY, ":", "1 °C/s unguarded"),
                 ("L2@1.0", RED, "-", "L2@1.0"),
                 ("L3 (computed)", BLUE, "--", "L3"),
                 ("rtd_lost_s", "#805ad5", ":", "RTD lost"),
                 ("TR author-derived (s)", "#4a5568", "-.", "H$_2$ onset + author lead")]
        for key, c, ls, lab in marks:
            if pd.notna(e[key]) and xlim[0] <= e[key] <= xlim[1]:
                ax.axvline(e[key], color=c, ls=ls, lw=1.0)
                handles.append(plt.Line2D([], [], color=c, ls=ls, lw=1.0, label=lab))
        g = al[(al.record == rid) & al.first_alarm_s.notna()]
        for model, c in (("xgboost", "#d69e2e"), ("rule", "#b7791f")):
            a = g[g.model == model].first_alarm_s
            if len(a):
                ax.plot([a.iloc[0]], [ax.get_ylim()[1] * 0.97], marker="v", color=c, ms=6, ls="none")
                handles.append(plt.Line2D([], [], marker="v", color=c, ls="none",
                                          label="frozen alarm, %s" % ("trees (all seeds)" if model == "xgboost" else "rule")))
        if rid == "3B":
            handles.append(plt.Line2D([], [], ls="none", label="frozen alarms: none"))
        ax.set_title("(%s) %s" % ("ab"[row], {"3A": "3A: 2170 NCA, heated cell of a parallel pair, 12.5 W heating", "3B": "3B: NMC pouch, 1 C overcharge"}[rid]), fontsize=7.3, loc="left")
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("time from record start (s)", fontsize=7)
        ax.legend(handles=handles, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  frameon=False, ncol=4)

        axp = fig.add_subplot(gs[row, 1])
        p = r.channels["P_chamber"]
        base = np.median(p[t <= 600])
        axp.plot(t, p - base, color=ORANGE, lw=0.6)
        axp.set_xlim(*xlim)
        axp.axvline(e["H2 onset (computed)"], color=GREEN, ls="--", lw=0.9)
        axp.set_title("chamber ΔP (hPa)", fontsize=7, loc="left")
        axp.tick_params(labelsize=6)
        axp.set_xlabel("s", fontsize=6.5)
    save_fig(fig, "SF2_d7_timeline")


def main():
    ev = pd.read_csv(os.path.join(RES, "event_times.csv")).set_index("record")
    recs = G.read_all()
    note_src = ("D7 = Gardner et al. 2025 Figshare v2 (CC BY 4.0), audited after the D1–D6 freeze under "
                "reports/11_d7_gardner_audit_protocol.md (committed before download). ")
    save_table(events_table(ev, recs), "S7_d7_events",
               note_src + "Event rules are the registry rules of trbench/common.py applied unchanged; the H2 rule "
               "(10 ppm above a 600 s baseline, held 5 s) was fixed in the protocol. Times from record start.")
    al = pd.read_csv(os.path.join(RES, "alarm_status.csv"))
    save_table(alarm_table(al), "S7_d7_alarms",
               note_src + "Held-dataset M1 no-age alarms (validation_20260925_native) re-fitted with the stored "
               "code, source split and seeds; applied only if the re-fitted tau reproduced the stored tau. Cell entries "
               "are detections over seeds under each reference event (detected = first alarm before the event, t_a < t_e).")
    inter = pd.read_csv(os.path.join(RES, "interventions.csv"))
    save_table(inter, "S7_d7_interventions",
               note_src + "Intervention records: heating or charging was stopped when H2 was detected, so the outcome is "
               "conditioned on the alarm. They are described, not scored, and excluded from every false-alarm denominator. "
               "Descriptors stop at the first open-circuit RTD reading or voltage-lead dropout after the H2 onset.")
    figure(ev, recs, al)


if __name__ == "__main__":
    main()
