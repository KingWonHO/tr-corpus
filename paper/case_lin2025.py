"""Re-interpretation case for the Discussion: Lin et al. 2025, J. Energy
Storage 109:115085 (10.1016/j.est.2024.115085).

The paper reports that the expansion-force signal leads voltage and
temperature "by up to 110 s".  Its Table 3 gives, per chemistry x trigger,
the absolute time at which each of three signals meets its own criterion:

    expansion force : its peak, which the authors equate with valve opening
    voltage         : 25 % drop            (GB 38031)
    temperature     : dT/dt > 1 degC/s for 3 s, any of five thermocouples

So every "lead time" in that paper is the interval between two event
DEFINITIONS on the same cell, not the lead of a detector over an event.
This script (a) recomputes the paper's numbers under each choice of
reference event, and (b) places the same interval -- voltage criterion
minus temperature criterion -- next to its value in the public corpus,
where the corresponding labels are L3 (20 % sustained drop) and L2@1.0.

    uv run python paper/case_lin2025.py

Writes paper/tables/S6_case_lin2025_*.csv/.md.  Table 3 values are
transcribed from the paper; the corpus values come from the frozen registry.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "tr-corpus", "registry", "experiments.csv")
TAB = os.path.join(ROOT, "paper", "tables")
sys.path[:0] = [os.path.join(ROOT, "paper")]
from make_figures import save_table  # noqa: E402

# Lin et al. 2025, Table 3 ("Comparison of warning advance times", seconds).
# Single value per cell; the paper does not state repeats.
LIN = pd.DataFrame([
    dict(chemistry="NCM523 51 Ah", trigger="overcharge 1C", force=1358.10, voltage=1359.00, temperature=1361.50),
    dict(chemistry="NCM523 51 Ah", trigger="penetration 2 mm s⁻¹", force=6.59, voltage=13.50, temperature=9.20),
    dict(chemistry="NCM523 51 Ah", trigger="external heating 330 W", force=410.80, voltage=419.50, temperature=413.40),
    dict(chemistry="LFP 52 Ah", trigger="overcharge 1C", force=421.60, voltage=np.nan, temperature=np.nan),
    dict(chemistry="LFP 52 Ah", trigger="penetration 2 mm s⁻¹", force=27.50, voltage=28.50, temperature=47.36),
    dict(chemistry="LFP 52 Ah", trigger="external heating 330 W", force=787.00, voltage=897.40, temperature=825.20),
])


def part_a():
    """The same six tests under each admissible reference event."""
    rows = []
    for _, r in LIN.iterrows():
        tr = "no TR (valve opened, T 39.7 °C)" if np.isnan(r.voltage) else ""
        rows.append({
            "chemistry": r.chemistry, "trigger": r.trigger,
            "force peak (s)": "%.2f" % r.force,
            "voltage −25 % (s)": "—" if np.isnan(r.voltage) else "%.2f" % r.voltage,
            "dT/dt > 1 °C/s (s)": "—" if np.isnan(r.temperature) else "%.2f" % r.temperature,
            "force lead vs voltage (s)": "—" if np.isnan(r.voltage) else "%+.2f" % (r.voltage - r.force),
            "force lead vs temperature (s)": "—" if np.isnan(r.temperature) else "%+.2f" % (r.temperature - r.force),
            "force lead vs venting (s)": "0 by construction",
            "voltage − temperature (s)": "—" if np.isnan(r.voltage) else "%+.2f" % (r.voltage - r.temperature),
            "note": tr,
        })
    note = ("Source: Lin et al. 2025, J. Energy Storage 109:115085, Table 3, transcribed. "
            "Each column is the time at which that signal meets its own criterion; the paper's "
            "'lead time' is a difference between two such columns. The abstract's 'up to 110 s' is "
            "the LFP-heating force-vs-voltage entry (897.40 − 787.00 = 110.40 s); against temperature "
            "the same test gives 38.20 s, and against venting — which the authors equate with the "
            "force peak — the lead is zero by definition. No non-runaway experiment prices false "
            "alarms; the one test that did not reach runaway (LFP overcharge) is reported as a "
            "success of the force signal, i.e. the same event is a true warning under a venting "
            "reference and a false alarm under a runaway reference.")
    save_table(pd.DataFrame(rows), "S6_case_lin2025_reference_event", note)


def part_b():
    """Voltage-criterion minus temperature-criterion, Lin 2025 vs the corpus."""
    reg = pd.read_csv(REG)
    m = reg[(reg.role != "pretrain_only") & reg["t_onset_L2_1.0"].notna()].copy()
    m["dV"] = m.t_onset_L3 - m["t_onset_L2_1.0"]        # voltage collapse − dT/dt 1 °C/s
    m["dP"] = m.t_onset_L1 - m["t_onset_L2_1.0"]        # venting (pressure) − dT/dt 1 °C/s
    rows = []
    for _, r in LIN.dropna().iterrows():
        rows.append({"source": "Lin 2025", "cell": r.chemistry, "trigger": r.trigger, "n": 1,
                     "voltage event − temperature event (s)": "%+.1f" % (r.voltage - r.temperature),
                     "venting event − temperature event (s)": "%+.1f" % (r.force - r.temperature),
                     "voltage rule": "−25 %", "venting rule": "force peak"})
    names = {"ds01_bak": "#1 BAK 21700 NMC", "ds03_warwick": "#3 21700", "ds04_osf": "#4",
             "ds02_overcharge": "#2", "ds09_mech": "#9 pouch"}
    trig = {"external_heating": "external heating", "overcharge": "overcharge",
            "mechanical_indentation": "mechanical indentation"}
    for (ds, tg), g in m.groupby(["dataset_id", "trigger"]):
        if ds not in names:
            continue
        dv, dp = g.dV.dropna(), g.dP.dropna()
        rows.append({"source": "this corpus", "cell": names[ds], "trigger": trig.get(tg, tg), "n": len(g),
                     "voltage event − temperature event (s)":
                         ("%+.0f (median; %+.0f to %+.0f, n = %d)" % (dv.median(), dv.min(), dv.max(), len(dv))) if len(dv) else "no voltage channel",
                     "venting event − temperature event (s)":
                         ("%+.0f (median; %+.0f to %+.0f, n = %d)" % (dp.median(), dp.min(), dp.max(), len(dp))) if len(dp) else "no pressure channel",
                     "voltage rule": "L3: −20 % sustained 3 s" if len(dv) else "—",
                     "venting rule": "L1: pressure-defined main release" if len(dp) else "—"})
    heat_dv = m[m.dataset_id.isin(["ds01_bak", "ds03_warwick"])].dV.dropna()
    before_lo = abs(float(heat_dv.max()))
    before_hi = abs(float(heat_dv.min()))
    note = ("Lin 2025 rows: Table 3 differences (voltage −25 %% criterion and force peak, each minus the "
            "1 °C/s temperature criterion). Corpus rows: registry labels t_onset_L3 and t_onset_L1 minus "
            "t_onset_L2_1.0 (dT/dt ≥ 1 °C/s sustained 2 s with heater-ramp and glitch guards). "
            "The sign of the voltage−temperature interval is the point: in Lin's heating tests the "
            "voltage criterion fires 6–72 s AFTER the temperature criterion; in the corpus's heating "
            "tests the voltage criterion fires %.0f–%.0f s BEFORE it. A 'lead over voltage' therefore "
            "has no fixed meaning across public tests of the same trigger." % (before_lo, before_hi))
    save_table(pd.DataFrame(rows), "S6_case_lin2025_corpus", note)


if __name__ == "__main__":
    part_a(); part_b()
