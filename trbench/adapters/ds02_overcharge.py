"""#2 Overcharge-induced TR in prismatic LFP cells.

Phase 0 question answered: the "9,795 records" are TIMESTEPS, not experiments.
There are exactly FIVE experiments, one per overcharge current (16/24/32/40/48 A),
10,598 rows in total at 1 Hz.  This is the single biggest input to the gate
decision in plan section 4.

Jun Shen, "overcharge-induced Thermal Runaway data in Prismatic Lithium Iron
Phosphate Battery", Mendeley Data V1 (2025), doi:10.17632/8zjttd77my.1, CC BY 4.0.

The repository description confirms the count question above -- it says "9,795
valid experimental records", i.e. rows -- and pins down two things the files
alone do not:

  * "a five-stage current protocol (0.5C -> 0.75C -> 1C -> 1.25C -> 1.5C)".
    The five files are 16/24/32/40/48 A, and 16/0.5 = 24/0.75 = 32/1 = 40/1.25
    = 48/1.5 = 32, so the CELL IS 32 Ah and each file is one C-rate stage.
  * "surface/gas temperatures at five locations" -- two on the cell and three
    on the flue gas, which is the independent confirmation that yqs/yqz/yqx are
    gas probes and not cell surface.

COD.  The record has no companion paper, so this was settled from the data.
The description names CO/CO2/HF while the files carry H2, COD, HF and CH4 and no
CO2 column at all.  COD is read as CO on three grounds:

  * its baseline is 16-28 ppm across the five files.  Atmospheric CO2 sits near
    400-450 ppm, so a CO2 sensor could not idle here; an electrochemical CO cell
    idles near zero with a small offset, which is exactly this.
  * it peaks at 5,200-7,500 ppm 40-90 s BEFORE the cell temperature peak, in
    step with H2 -- the behaviour of a vent-gas species, not of an ambient one.
  * the repository description names CO explicitly, and no other column could
    carry it.

The baseline argument is the discriminating one.  It is recorded here because
the mapping rests on inference rather than on a stated column definition.

Headers are Chinese and differ between files; 24A carries minute-resolution
timestamps, so the index column is used as the seconds axis throughout.
Gas is genuinely instrumented here (H2, CO/COD, HF, CH4) -- unlike #1, where the
gas trace is pressure-derived.
"""
import glob
import os
import numpy as np
import pandas as pd

from common import RawExperiment

DS = "ds02_overcharge"
ROOT = "Dataset/2"

COL_MAP = {
    "H2(PPM)": ("gas_H2", 1.0),
    "COD(PPM)": ("gas_CO", 1.0),      # COD probe = CO detector on this rig
    "cod(ppm)": ("gas_CO", 1.0),
    "HF(PPM)": ("gas_HF", 1.0),
    "CH4(%)": ("gas_CH4", 1e4),       # vol% -> ppm
    "V(V)": ("V_cell", 1.0),
    "电池上部温度(℃)": ("T_surface_pos", 1.0),   # cell upper
    "电池中部温度(℃)": ("T_surface_mid", 1.0),   # cell middle
    # yq* = 烟气 (flue gas) probes at three heights, present only in 16A.
    # They are NOT cell surface: they start at ambient 26-30 degC while the cell
    # thermocouples already read 39-41 degC, they peak far lower (58/89/206 vs
    # 262/344 degC) and they peak ~60 s EARLIER than the cell.  That is a vent
    # gas signature, so feeding them into T_surface_max/mean would both dilute
    # the surface aggregate and mix two response times the plan forbids merging.
    "yqx(℃)": ("T_vent", 1.0),        # lower probe, nearest the cell, hottest
    "yqz(℃)": ("T_vent_2", 1.0),      # middle probe
    "yqs(℃)": ("T_ambient", 1.0),     # upper probe, coolest, closest to ambient
}


def load(root):
    files = sorted(glob.glob(os.path.join(root, ROOT, "**", "*.csv"), recursive=True))
    for f in files:
        amps = float(os.path.basename(f).replace("A.csv", ""))
        df = pd.read_csv(f, encoding="utf-8-sig")
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
        t = t - t[0]                       # index column == seconds at 1 Hz

        ser = {}
        for c in df.columns:
            key = COL_MAP.get(c.strip())
            if key is None:
                continue
            ch, k = key
            ser[ch] = (t, pd.to_numeric(df[c], errors="coerce").to_numpy(float) * k)

        # charger current is constant by design and named by the file
        ser["I"] = (t, np.full(len(t), amps))

        # t_trigger = overcharge start: first sustained rise of cell voltage
        t_trig = 0.0
        if "V_cell" in ser:
            v = ser["V_cell"][1]
            base = np.nanmedian(v[:min(30, len(v))])
            up = np.flatnonzero(np.isfinite(v) & (v > base + 0.1))
            if len(up):
                t_trig = float(t[up[0]])

        meta = dict(
            cell_type="prismatic LFP", chemistry="LFP", form_factor="prismatic",
            capacity_ah=32.0, soc_pct=100, soh="fresh", soh_pct=100,
            trigger="overcharge", scale="cell", overcharge_current_a=amps,
            c_rate=round(amps / 32.0, 2),
            source_doi="10.17632/8zjttd77my.1",
        )
        notes = ["index column used as the 1 Hz time axis (timestamps in 24A.csv "
                 "have minute resolution only)",
                 "COD read as gas_CO: its 16-28 ppm baseline rules out CO2 "
                 "(atmospheric ~400 ppm) and it tracks H2 into the vent event; "
                 "inferred from the data, no column definition is published",
                 "yq* flue-gas probes (16A only) mapped to T_vent/T_vent_2/"
                 "T_ambient, not to cell surface",
                 "no pressure channel: panels M3/M5/M6 are not expressible"]
        yield RawExperiment(DS, "%dA" % int(amps), ser, meta, t_trig, notes)
