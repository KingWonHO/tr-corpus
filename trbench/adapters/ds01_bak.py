"""#1 BAK N21700CG-50, 60% SOC, one-sided heating (8 cells, A..H).

Source of truth is compiled_data2_TS0330X.mat (100 Hz, MATLAB v7.3 inside a
zip), not export_TS0330X.xlsx -- the xlsx is decimated to ~0.6 Hz, below the
1 Hz common grid, so using it would force upsampling of every channel.

Gas: the released-gas amount `n` is derived from the cassette pressure, so it
is NOT an independent gas channel.  Speciated gas (H2/CO/CO2/HF/...) exists only
as a single post-test GC measurement in the report workbook and is carried as
experiment metadata, never as a time series.  This is what makes panel M4/M5
partly degenerate for #1 and belongs in the E0 gap matrix.
"""
import os
import zipfile
import numpy as np
import pandas as pd

import h5py
from common import RawExperiment

DS = "ds01_bak"
ROOT = ("Dataset/1/report_BAK_N21700CG-50_60SOC_TS0330")
RAW = os.path.join(ROOT, "Plots_and_raw_data")
REPORT = os.path.join(ROOT, "report_BAK_N21700CG-50_60SOC_TS0330.xlsx")

# Map by the mat file's own semantic GROUP, not by thermocouple number: the TC
# numbering is per-cell (cell A uses TC1..TC4, B uses TC5..TC8, ... H uses
# TC29..TC32), so a fixed TC-name table silently drops seven of the eight
# experiments' cell-can and vent channels.
GROUP_MAP = {
    "cell_can": ["T_surface_mid"],
    "heater": ["T_heater"],
    "vent_gas": ["T_vent", "T_vent_2"],      # ordered by z, nearest first
}


def _s(o):
    return "".join(chr(c) for c in np.array(o).ravel())


def _report_meta(base):
    df = pd.read_excel(base, header=None)
    hdr = [str(x) for x in df.iloc[0].tolist()]
    out = {}
    for i in range(3, 11):
        exp = str(df.iloc[i, 0])
        row = {}
        for c, name in enumerate(hdr):
            v = df.iloc[i, c]
            if name in ("nan", ".") or pd.isna(v):
                continue
            row[name] = v.item() if hasattr(v, "item") else v
        out[exp] = row
    return out


def load(root):
    base = os.path.join(root, REPORT)
    rep = _report_meta(base) if os.path.exists(base) else {}
    rawdir = os.path.join(root, RAW)
    for exp in sorted(os.listdir(rawdir)):
        zp = os.path.join(rawdir, exp, "compiled_data2_%s.zip" % exp)
        if not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as z:
            name = z.namelist()[0]
            with z.open(name) as fh, h5py.File(fh, "r") as f:
                d = f["d"]
                t = np.array(d["time"]).ravel()
                t = t - t[0]
                ser = {}

                for grp, chans in GROUP_MAP.items():
                    probes = []
                    for ref in np.array(d["temperature"][grp]).ravel():
                        o = f[ref]
                        z = float(np.array(o["z"]).ravel()[0])
                        probes.append((z if np.isfinite(z) else 0.0,
                                       _s(o["name"]), np.array(o["data"]).ravel()))
                    probes.sort(key=lambda p: p[0])
                    for ch, (_z, _nm, v) in zip(chans, probes):
                        ser[ch] = (t, v)

                # four reactor-gas probes measure the chamber, not the cell.
                # Keep the quietest one as T_ambient and drop the rest rather
                # than mislabelling them as cell surface.
                gas = [(np.ptp(np.array(f[r]["data"]).ravel()),
                        np.array(f[r]["data"]).ravel())
                       for r in np.array(d["temperature"]["gas"]).ravel()]
                if gas:
                    ser["T_ambient"] = (t, min(gas, key=lambda g: g[0])[1])

                # Both transducers sit OUTSIDE the cell: the reactor (I61) and
                # the sample cassette around it (I81).  Both start near 123 kPa
                # absolute in the N2 atmosphere and RISE to ~162 kPa as vent gas
                # arrives, so neither is an in-cell pressure and neither belongs
                # in P_internal.  #1 therefore has no in-cell pressure at all.
                ser["P_chamber"] = (t, np.array(d["pressure"]["data"]).ravel() / 1000.0)
                ser["P_chamber_2"] = (t, np.array(d["pressure_cassette"]["data"]).ravel() / 1000.0)
                ser["V_cell"] = (t, np.array(d["voltage"]["data"]).ravel())

                # Heater current/voltage are already scaled to A and V despite
                # the channel `unit` field reading "Volt".  They are the abuse
                # actuation: used to locate t_trigger and then discarded, never
                # mapped to `I`, which means cell current -- a heater trace in
                # panel M2 would leak the trigger straight into the model.
                hp = None
                if "heater_current" in d and "heater_voltage" in d:
                    hi = np.array(d["heater_current"]["data"]).ravel()
                    hv = np.array(d["heater_voltage"]["data"]).ravel()
                    hp = hi * hv

        notes = ["no cell current channel: panel M2 is voltage-only for #1",
                 "P_chamber = reactor (I61), P_chamber_2 = sample cassette "
                 "(I81), both absolute kPa; no in-cell pressure sensor",
                 "gas_total_mol derived from cassette pressure, not an "
                 "independent gas sensor",
                 "speciated gas is a single post-test GC value, not a time series"]

        # released-gas amount `n` exists only in the decimated export workbook
        # (~0.6 Hz); it is upsampled onto the 1 Hz grid and every filled sample
        # is reported through is_interpolated.
        #
        # The workbook's `time` column is ALREADY elapsed time on the MAT
        # origin: its first numeric row is 22-97 s, not 0.  Subtracting that
        # first value (as this adapter did until 2026-09-14) pulled the gas
        # channel 22-97 s earlier than every other channel -- a look-ahead
        # that only the gas-bearing panels M4/M5 could exploit.  The audit in
        # reports/ds01_alignment_audit/export_origin_check.json shows the
        # workbook's own temperature/pressure/voltage columns match the MAT
        # to rounding error when the time column is used unchanged, and are
        # off by 10-37 degC / 1.3 V when the first value is subtracted.
        exp_xlsx = os.path.join(rawdir, exp, "export_%s.xlsx" % exp)
        if os.path.exists(exp_xlsx):
            e = pd.read_excel(exp_xlsx, skiprows=[1, 2])
            et = pd.to_numeric(e["time"], errors="coerce").to_numpy(float) / 1000.0
            en = pd.to_numeric(e["n"], errors="coerce").to_numpy(float) / 1000.0
            ok = np.isfinite(et) & np.isfinite(en)
            if ok.sum() > 5:
                ser["gas_total_mol"] = (et[ok], en[ok])
                notes.append("gas_total_mol upsampled from 0.6 Hz export "
                             "workbook to the 1 Hz grid")

        # t_trigger = heater switched on
        t_trig = None
        if hp is not None and np.isfinite(hp).any():
            on = np.flatnonzero(hp > max(1.0, 0.05 * np.nanmax(hp)))
            if len(on):
                t_trig = float(t[on[0]])

        m = dict(rep.get(exp, {}))
        meta = dict(
            cell_type=m.get("cell_type_id", "BAK_N21700CG-50"),
            chemistry="NMC", form_factor="21700",
            capacity_ah=m.get("measured cell capacity"), soc_pct=m.get("soc", 60),
            trigger="external_heating", scale="cell", soh="fresh", soh_pct=100,
            gas_composition_molpct={k: m[k] for k in
                                    ("H2", "CH4", "CO", "CO2", "HF", "C2H4", "H2O")
                                    if k in m},
            gas_total_mol_post=m.get("amount of produced not condensated gas"),
            max_cell_temperature=m.get("max_cell_temperature"),
            max_overpressure_bar=m.get("max over pressure"),
            source_doi="10.5281/zenodo.18849418",
        )
        yield RawExperiment(DS, exp, ser, meta, t_trig, notes)
