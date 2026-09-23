"""#4 OSF multi-modal TR: cell and module, fresh and aged.

Kwak, Jeong, Kim, Shin, Park & Oh, "Multi-modal thermal runaway dataset of
fresh and aged lithium-ion battery cells and modules", Scientific Data 13:1084
(2026), doi:10.1038/s41597-026-07857-1.  Dataset: doi:10.17605/OSF.IO/C2HNQ.
The paper ships with the data and is the ground truth used below; three of its
statements resolve problems that the file layout alone leaves ambiguous.

1.  "The temperature and pressure measurements were collected using a DAQ system
    (GL860, GRAPHTEC) with a 10-Hz frequency ... all measurement channels were
    inherently synchronized within the same acquisition framework."
    So the two records of one test must cover the same span.  They do not, for
    the aged 2170 pair -- see C1/C2 below.

2.  Fig. 9(b): "Temperature and force measurements for the aged cell (D1) with
    86%% SOH", and Data Record: "the NMC pouch folders -- at both the cell and
    module levels -- feature Temperature.csv and Force.csv".
    So D1_Pressure.xlsx is a FORCE record, despite its name.

3.  Technical Validation: A1 has no force record and B1 no pressure record
    because "2 cases exhibited pressure or force sensor malfunctions, which were
    excluded from the published dataset"; B1's transducer is named explicitly.
    Those two absences are intentional, not missing files.

Units.  The Data Record says pressure is in kPa and the SOH section quotes
"peak pressures of 8.68 and 8.00 kPa", which are exactly the raw column maxima.
Read literally that cannot be right: the transducer has a 3 MPa range and a
stated uncertainty of +/-0.06 MPa = 60 kPa, so an 8.68 kPa peak would sit seven
times below the sensor's own uncertainty; the same paper reports the fresh-cell
"peak pressure of 900 kPa" where the raw maximum is 9.73; and ~0.33 mol of vent
gas (the measured figure for a comparable 21700 in dataset #1) released into the
2.829 L canister gives several hundred kPa, not single digits.  The raw column
is therefore read as BAR and converted to kPa, which makes all four statements
agree.  The "kPa" in that one sentence is taken to be a unit slip for bar.
"""
import os
import numpy as np
import pandas as pd

from common import RawExperiment

DS = "ds04_osf"
ROOT = "Dataset/4/Dataset_TR"

BAR_TO_KPA = 100.0

# D1's value column stops being a measurement at t=7731 s: from there it holds
# a copy of the row counter (offset +63) for 1,450 rows, then 63 rows of the
# literal string "--".  Only the physical part is kept.
D1_VALID_UNTIL_S = 7730.0

# Explicit layout, from the paper.  Guessing a channel from the filename is what
# put a force record into P_internal in the first place.
#   role: (path, channel)   scale/chem/soh from Tables 1-2 and Technical Validation
LAYOUT = {
    "A1": dict(dir="Cell/NMC pouch/Fresh", files={"A1_Temperature.xlsx": "T"},
               scale="cell", chem="NMC pouch", soh_pct=100),
    "A2": dict(dir="Cell/NMC pouch/Fresh",
               files={"A2_Temperature.xlsx": "T", "A2_Force.xlsx": "F"},
               scale="cell", chem="NMC pouch", soh_pct=100),
    "B1": dict(dir="Cell/NMC 2170/Fresh", files={"B1_Temperature.xlsx": "T"},
               scale="cell", chem="NMC 2170", soh_pct=100),
    "B2": dict(dir="Cell/NMC 2170/Fresh",
               files={"B2_Temperature.xlsx": "T", "B2_Pressure.xlsx": "P"},
               scale="cell", chem="NMC 2170", soh_pct=100),
    # C1/C2: the pressure and temperature records are cross-paired in the
    # published files.  Durations are C1_T 2637 s, C2_T 5373 s, C1_P 5373 s,
    # C2_P 2599 s.  C1_P matches C2_T to the second and C2_P matches C1_T to
    # within 38 s, while the published pairing mismatches by ~2740 s.  For
    # reference, correctly paired B2 differs by 80 s, so a sub-minute offset is
    # normal here and a 46-minute one is not.  Re-paired accordingly; the
    # pressure peaks then land 10-12 s from their temperature peaks instead of
    # 35-37 s.  Which of the two names is wrong cannot be determined and does
    # not matter: C1 and C2 are two repeats of one condition (2170, 86% SOH).
    "C1": dict(dir="Cell/NMC 2170/Aged",
               files={"C1_Temperature.xlsx": "T", "C2_Pressure.xlsx": "P"},
               scale="cell", chem="NMC 2170", soh_pct=86),
    "C2": dict(dir="Cell/NMC 2170/Aged",
               files={"C2_Temperature.xlsx": "T", "C1_Pressure.xlsx": "P"},
               scale="cell", chem="NMC 2170", soh_pct=86),
    # D1_Pressure.xlsx holds force, per Fig. 9(b).
    "D1": dict(dir="Cell/NMC pouch/Aged",
               files={"D1_Temperature.xlsx": "T", "D1_Pressure.xlsx": "F"},
               scale="cell", chem="NMC pouch", soh_pct=86),
    "M1": dict(dir="Module/NMC pouch/Fresh",
               files={"M1_Temperature.csv": "T", "M1_Force.csv": "F"},
               scale="module", chem="NMC pouch", soh_pct=100),
    # Technical Validation: "1 NMC pouch module at 100% SOH and 1 NMC pouch
    # module at 76% SOH" -- the aged module is 76%, not the 86% of the aged
    # cells, even though it sits in a folder named "Aged".
    "M2": dict(dir="Module/NMC pouch/Aged",
               files={"M2_Temperature.csv": "T", "M2_Force.csv": "F"},
               scale="module", chem="NMC pouch", soh_pct=76),
}

SPEC = {                       # Table 2
    "NMC 2170": dict(model="INR21700-50E", form_factor="21700",
                     capacity_ah=4.9, nominal_v=3.6),
    "NMC pouch": dict(model="CTS", form_factor="pouch",
                      capacity_ah=10.0, nominal_v=3.7),
}


def _read2(path):
    df = (pd.read_csv(path, header=None) if path.lower().endswith(".csv")
          else pd.read_excel(path, header=None))
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def load(root):
    base = os.path.join(root, ROOT)
    for exp, g in LAYOUT.items():
        ser, notes = {}, []
        spec = SPEC[g["chem"]]

        for fname, role in g["files"].items():
            path = os.path.join(base, g["dir"], fname)
            if not os.path.exists(path):
                notes.append("expected file missing: " + fname)
                continue
            df = _read2(path)
            t = df.iloc[:, 0].to_numpy(float)

            if role == "T" and g["scale"] == "module":
                # The `ms` column is the millisecond field only: it cycles
                # 0,200,...,800 and restarts every second, so it is a sub-second
                # offset, not elapsed time.  Reading it literally squeezes a
                # 4,400 s test into 0.8 s.  Rebuild from the row index.
                d = np.diff(t[np.isfinite(t)])
                pos = d[d > 0]
                step = float(np.median(pos)) / 1000.0 if len(pos) else 0.2
                t = np.arange(len(df), dtype=float) * step
                for j in range(1, df.shape[1]):
                    ser["T_surface_x%02d" % j] = (t, df.iloc[:, j].to_numpy(float))
                notes.append("module time axis rebuilt as row_index * %.3f s; "
                             "the raw `ms` column is a cycling sub-second field"
                             % step)
                notes.append("%d module thermocouples; the paper gives their cell "
                             "grouping but not per-channel positions, so they feed "
                             "T_surface_max/mean only" % (df.shape[1] - 1))
                continue

            v = df.iloc[:, 1].to_numpy(float)

            if exp == "D1" and role == "F":
                keep = np.isfinite(t) & (t <= D1_VALID_UNTIL_S)
                notes.append("D1_Pressure.xlsx holds FORCE, not pressure "
                             "(paper Fig. 9(b)); mapped to F_expansion")
                notes.append("D1 truncated at t=%.0f s: beyond it the value "
                             "column is a copy of the row counter (+63) for "
                             "1450 rows, then 63 rows of '--'"
                             % D1_VALID_UNTIL_S)
                t, v = t[keep], v[keep]

            if role == "T":
                ser["T_surface_mid"] = (t, v)
            elif role == "P":
                # The transducer is on the canister head, not inside the cell:
                # it starts at atmospheric and JUMPS when vent gas arrives.  It
                # is a chamber pressure, so venting is a rise, not a collapse.
                ser["P_chamber"] = (t, v * BAR_TO_KPA)
                notes.append("P_chamber read as bar and converted to kPa; see "
                             "the module docstring for why the paper's 'kPa' is "
                             "treated as a unit slip")
                notes.append("canister pressure, gauge, starting at atmospheric; "
                             "mapped to P_chamber not P_internal")
            elif role == "F":
                ser["F_expansion"] = (t, v)

            if fname.split("_")[0] != exp:
                notes.append("channel taken from %s: published temperature and "
                             "pressure records for C1/C2 are cross-paired, "
                             "re-paired here by record duration" % fname)

        meta = dict(
            cell_type=spec["model"], chemistry="NMC",
            form_factor=spec["form_factor"], capacity_ah=spec["capacity_ah"],
            soc_pct=100, soh="fresh" if g["soh_pct"] == 100 else "aged",
            soh_pct=g["soh_pct"], trigger="external_heating", scale=g["scale"],
            heating_rate_c_per_min=5.5, daq_native_hz=10.0,
            canister_volume_m3=0.002829 if role == "P" else None,
            source_doi="10.1038/s41597-026-07857-1",
        )
        notes.append("TR triggered by a constant 5.5 degC/min heating ramp from "
                     "the start of the record, so t_trigger = 0")
        yield RawExperiment(DS, exp, ser, meta, 0.0, notes)
