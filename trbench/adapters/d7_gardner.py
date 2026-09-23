"""D7 -- Gardner et al. 2025 trace-H2 abuse records (Figshare 28677122 v2).

Registered after the D1--D6 analysis freeze as an external audit
(reports/11_d7_gardner_audit_protocol.md).  It is NOT part of
tr-corpus/registry/experiments.csv, so nothing here changes a D1--D6 number.

Each figure workbook holds one test.  The header row (the one containing
``RTD``) sits on the first or second spreadsheet row; a group caption
("commercial sensor") may sit above it.  Channels are mapped from header
strings only:

    ppmo / ppma          -> gas_H2        H2-CSFET, ppm (the paper's sensor)
    RTD                  -> T_case        case RTD, degC (single probe)
    Volts                -> V_cell        cell voltage; in Fig. 3A across a
                                          parallel pair, see PARALLEL_PAIR
    T_C / Ambient_T      -> T_ambient     evaluation-board SHT-35, degC
    RH (first)           -> RH_ambient    board, %
    P_hPa                -> P_chamber     board LPS22HB, hPa; chamber sealed
    H2 (after caption)   -> gas_H2_tc     commercial thermocatalytic sensor
    MOX                  -> gas_MOX       commercial metal-oxide sensor

Record clock: ``mins_raw`` if present, otherwise ``mins`` when it starts at 0,
otherwise ``time`` (s).  ``mins_offset`` (3A/3B) or an offset ``mins``
(4C/5A/5C) is kept as ``offset_zero_s`` -- its meaning is not labelled in the
files, so it is not used as the trigger (protocol section 3.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Dataset" / "16_gardner"
DOI = "10.6084/m9.figshare.28677122.v2"
LICENSE = "CC BY 4.0"

# file stem -> (record id, role, cell, stressor), from the paper's figure captions
RECORDS = {
    "Figure3a": ("3A", "untreated_tr", "cylindrical 2170 NCA, 100% SOC (heated cell in a parallel pair)", "external heating 12.5 W"),
    "Figure3b": ("3B", "untreated_tr", "NMC pouch, 100% SOC", "overcharge 1 C"),
    "Figure4a": ("4A", "intervention", "cylindrical 2170 NCA, 10% SOC", "external heating 12.5 W"),
    "Figure4b": ("4B", "intervention", "cylindrical 2170 NCA, 50% SOC", "external heating 12.5 W"),
    "Figure4c": ("4C", "intervention", "cylindrical 2170 NCA, 100% SOC", "external heating 12.5 W"),
    "Figure5a": ("5A", "intervention", "NMC pouch, 10% SOC", "external heating 110 W"),
    "Figure5b": ("5B", "intervention", "NMC pouch, 50% SOC", "external heating 110 W"),
    "Figure5c": ("5C", "intervention", "NMC pouch, 100% SOC", "external heating 110 W"),
    "Figure5d": ("5D", "intervention", "NMC pouch", "overcharge 0.25 C"),
    "Figure6": ("6", "intervention", "prismatic LFP, 100% SOC", "external heating 115/230 W"),
    "FigureS23": ("S23", "intervention", "NMC pouch, 100% SOC", "external heating 220 W"),
}
PARALLEL_PAIR = {"3A", "4A", "4B", "4C"}      # "Voltage is measured across two cells in parallel"
AUTHOR_H2_LEAD_MIN = {"3A": 23.9, "3B": 3.6}   # H2 first signal -> TR, paper text


@dataclass
class Record:
    dataset_id: str
    experiment_id: str
    role: str
    cell: str
    stressor: str
    source_file: str
    sheet: str
    t: np.ndarray
    channels: dict
    header_map: dict
    offset_zero_s: float | None
    t_trigger: float | None = None
    notes: list = field(default_factory=list)

    @property
    def series(self):          # native_windows.causal_record interface
        return {k: (self.t, v) for k, v in self.channels.items()}


def _sheet(path):
    x = pd.ExcelFile(path)
    for sh in x.sheet_names:
        raw = x.parse(sh, header=None)
        if raw.empty:
            continue
        for h in range(3):
            if "RTD" in [str(v).strip() for v in raw.iloc[h]]:
                cols = [str(v).strip() for v in raw.iloc[h]]
                body = raw.iloc[h + 1:].reset_index(drop=True)
                body.columns = range(len(cols))
                return sh, cols, body
    raise ValueError("%s: no header row containing RTD" % path.name)


def read(stem):
    path = RAW / (stem + ".xlsx")
    rid, role, cell, stressor = RECORDS[stem]
    sh, cols, body = _sheet(path)
    num = body.apply(pd.to_numeric, errors="coerce")

    def first(*names):
        for n in names:
            if n in cols:
                return cols.index(n)
        return None

    # clock
    offset = None
    if "mins_raw" in cols:
        clock = num[cols.index("mins_raw")] * 60.0
        offset = num[cols.index("mins")] * 60.0
        clock_name = "mins_raw"
    elif "mins" in cols and np.isclose(num[cols.index("mins")].dropna().iloc[0], 0.0):
        clock = num[cols.index("mins")] * 60.0
        clock_name = "mins"
        if "mins_offset" in cols:
            offset = num[cols.index("mins_offset")] * 60.0
    else:
        clock = num[cols.index("time")].astype(float)
        clock_name = "time"
    ok = clock.notna().to_numpy()
    t = np.round(clock.to_numpy(float)[ok], 3)
    offset_zero = None
    if offset is not None:
        o = offset.to_numpy(float)[ok]
        offset_zero = float(np.round(np.nanmedian(t - o), 3))

    hmap = {"time_s": clock_name}
    ch = {}
    spec = [("gas_H2", ("ppmo", "ppma")), ("T_case", ("RTD",)), ("V_cell", ("Volts",)),
            ("T_ambient", ("T_C", "Ambient_T")), ("RH_ambient", ("RH",)), ("P_chamber", ("P_hPa",)),
            ("gas_MOX", ("MOX",))]
    for name, heads in spec:
        j = first(*heads)
        if j is not None:
            ch[name] = num[j].to_numpy(float)[ok]
            hmap[name] = cols[j]
    # the commercial block's H2 is the second "H2"-like header after the board block
    hj = [j for j, c in enumerate(cols) if c == "H2"]
    if hj:
        ch["gas_H2_tc"] = num[hj[0]].to_numpy(float)[ok]
        hmap["gas_H2_tc"] = "H2 (commercial block, col %d)" % hj[0]
    unmapped = [c for c in cols if c not in hmap.values() and not c.startswith("H2")
                and c not in ("mins", "mins_offset", "mins_raw", "time")]
    rec = Record("d7_gardner", rid, role, cell, stressor, path.name, sh, t, ch, hmap,
                 offset_zero, t_trigger=None)
    if unmapped:
        rec.notes.append("unmapped columns: %s" % ", ".join(unmapped))
    if rid in PARALLEL_PAIR:
        rec.notes.append("V_cell measured across a parallel pair (paper)")
    return rec


def read_all():
    return [read(s) for s in RECORDS]
