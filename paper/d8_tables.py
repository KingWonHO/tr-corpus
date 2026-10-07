"""Supplementary Tables S10 and S11 from the D8 field case outputs.

    paper/tables/S10_d8_exposure.{csv,md}     what the two field releases supply as negatives
    paper/tables/S11_d8_alarm_rates.{csv,md}  frozen models and thresholds applied to the normal vehicles

Sources: tr-corpus/results/external_20260925_d8_field/{exposure_accounting,alarm_rates,refit_gate}.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "paper")]
from make_figures import save_table, rng, MODEL_NAME   # noqa: E402

RES = os.path.join(ROOT, "tr-corpus", "results", "external_20260925_d8_field")
ARM = {"no_age": "values", "with_age": "with observation age", "surface_max_only": "source max", "surface_mean_only": "source mean"}
ORDER = ["rule", "xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer"]


def s10():
    e = pd.read_csv(os.path.join(RES, "exposure_accounting.csv"))
    out = pd.DataFrame({"source": e.source.str.replace(r" \(.*\)", "", regex=True), "subset": e.subset, "vehicles": e.vehicles,
                        "normal": e.vehicles_normal.fillna("").apply(lambda v: "" if v == "" else str(int(v))),
                        "records": e.records, "record_s": e.record_s.round().astype(int), "exposure_h": e.exposure_h.round().astype(int), "note": e.note.fillna("")})
    save_table(out, "S10_d8_exposure",
               "Source: %s/exposure_accounting.csv. A Zhang record is one charging segment of 128 samples at 10 s; exposure is the summed record span. "
               "The Cao release carries no timestamps; its duration is the frame count times the 30 s nominal rate its authors state." % os.path.relpath(RES, ROOT).replace(os.sep, "/"))
    return out


def s11():
    a = pd.read_csv(os.path.join(RES, "alarm_rates.csv"))
    g = pd.read_csv(os.path.join(RES, "refit_gate.csv"))
    rows = []
    for arm in ARM:
        for m in ORDER:
            x = a[(a.arm == arm) & (a.model == m)]
            gate = g[(g.arm == arm) & (g.model == m)]
            if not len(x):
                if len(gate) and not gate.reproduced.all():
                    rows.append({"arm": ARM[arm], "model": MODEL_NAME.get(m, m), "seeds": len(gate), "vehicles_alarm": "not assessable (refit did not reproduce tau)",
                                 "snippets_head": "", "snippets_full": "", "per_1000_h": ""})
                continue
            rows.append({"arm": ARM[arm], "model": MODEL_NAME.get(m, m), "seeds": len(x), "vehicles_alarm": rng(x.vehicles_alarm, "%d"),
                         "snippets_head": rng(x.snippets_alarm_head, "%d"), "snippets_full": rng(x.snippets_alarm, "%d"),
                         "per_1000_h": rng(x.alarms_per_1000_vehicle_h, "%.1f")})
    out = pd.DataFrame(rows)
    hours = a.exposure_h.iloc[0]
    save_table(out, "S11_d8_alarm_rates",
               "Source: %s/alarm_rates.csv and refit_gate.csv. %d normal vehicles, %d charging segments (at most 30 per vehicle, evenly spaced), %.0f vehicle-hours; "
               "panel M1; frozen models and thresholds of the held-out design (label v1.1), applied only where the refit reproduced the stored threshold. "
               "'per_1000_h' counts threshold-exceeding windows per 1,000 vehicle-hours." % (os.path.relpath(RES, ROOT).replace(os.sep, "/"), int(a.vehicles.iloc[0]), int(a.snippets.iloc[0]), hours))
    return out


if __name__ == "__main__":
    print(s10().to_string(index=False)); print(s11().to_string(index=False))
