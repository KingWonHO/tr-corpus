"""#15 simulated TR events -- PRETRAINING ONLY, never evaluated (plan E8).

Kriston, A., Pfrang, A., Podias, A. & Ibtissam, A., "Analysis of the effect of
thermal runaway initiation conditions on the severity of thermal runaway --
numerical simulation and machine learning study", Mendeley Data V2 (2020),
doi:10.17632/9cykcc3svn.2, CC BY 4.0.  European Commission Joint Research Centre.

A coupled electrical-thermal model generating 780 thermal runaway events for
Graphite-NMC(111) cells, spanning initial energy input, anode and cathode
decomposition, and internal-short electrical energy.  The authors' own machine
learning analysis groups the events into five clusters from no thermal runaway
to severe thermal runaway.

This is a MODEL, not a measurement.  It is written to simulated/ and assigned
the pretrain split so it cannot reach an evaluation set by accident.

The detailed traces live in a 3.5 GB Deflate64 zip that Python's zipfile cannot
open, so 7-Zip is used to stream each member to stdout.  Only the columns that
map onto the measured schema are kept (Temp, V, I, Time, ID) and everything is
binned to 1 Hz, which turns 3.5 GB into a few MB.

Output goes to simulated/ , physically separate from measured/ , so that a later
user cannot mix synthetic and measured events by accident.
"""
import csv
import io
import os
import subprocess
import numpy as np
import pandas as pd

from common import RawExperiment

DS = "ds15_sim"
ROOT = "Dataset/15/TR_numerical simul"
ZIP = "Detailed_results_Double3_5_TR_Latin_1_13_2020_A_V_Fine_300.zip"
SUMMARY = "Evaluation_FReez_1_13_2020_A_V_Fine_new_cluster_Published.xlsx"

SEVEN_ZIP = [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe",
             "7z", "7za"]
# Only Temp is carried, and the source paper (J. Electrochem. Soc. 167, 090555)
# says why the other two must not be:
#   V  is the OVERPOTENTIAL V_OP of the equivalent-circuit model, the unknown in
#      its Kirchhoff solution -- not a cell terminal voltage.
#   I  is a CURRENT RATE normalised by cell capacity, units 1/s -- not amperes.
# Mapping either onto V_cell / I would put incompatible scales in one column and
# corrupt any globally normalised pretraining run.
#
# Temp is exported in degC even though the model itself works in kelvin: the
# per-severity mean of Max_Temp in the summary workbook is 130/257/367/566/936,
# against the paper's cluster maxima of ~127/253/363/670/928 degC.
KEEP = {"Temp": "T_surface_mid"}

SEVERITY_CLUSTER = {
    1: "no TR, no ISC", 2: "no TR, ISC (soft short)", 3: "mild TR",
    4: "severe ISC TR (hard short dominates)", 5: "severe TR",
}


def _sevenzip():
    for c in SEVEN_ZIP:
        if os.path.exists(c):
            return c
    return None


def _members(zpath, exe):
    out = subprocess.run([exe, "l", "-ba", zpath], capture_output=True, text=True)
    names = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".txt"):
            names.append(parts[-1])
    return sorted(names)


def _stream(zpath, member, exe):
    """Yield (sim_id, DataFrame) for each simulation inside one member file."""
    p = subprocess.Popen([exe, "e", "-so", zpath, member],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rd = csv.reader(io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace"))
    header = next(rd)
    idx = {c: i for i, c in enumerate(header)}
    need = [idx[c] for c in ("Time", "ID")] + [idx[c] for c in KEEP if c in idx]
    cols = ["Time", "ID"] + [c for c in KEEP if c in idx]

    cur_id, buf = None, []
    for row in rd:
        if len(row) <= max(need):
            continue
        try:
            vals = [float(row[i]) for i in need]
        except ValueError:
            continue
        sid = vals[1]
        if cur_id is not None and sid != cur_id:
            yield int(cur_id), pd.DataFrame(buf, columns=cols)
            buf = []
        cur_id = sid
        buf.append(vals)
    if buf:
        yield int(cur_id), pd.DataFrame(buf, columns=cols)
    p.stdout.close()
    p.wait()


def load(root, limit=None):
    exe = _sevenzip()
    zpath = os.path.join(root, ROOT, ZIP)
    if exe is None or not os.path.exists(zpath):
        print("  [ds15] 7-Zip or archive unavailable; skipping detailed traces")
        return

    spath = os.path.join(root, ROOT, SUMMARY)
    summ = {}
    if os.path.exists(spath):
        sdf = pd.read_excel(spath)
        for _, r in sdf.iterrows():
            if pd.notna(r.get("ID")):
                # These columns hold the STRINGS "TR"/"No TR" and "Severe TR"/
                # "Mild TR".  bool("No TR") is True, so a plain bool() marked
                # every one of the 781 events as a thermal runaway, including
                # ones whose peak temperature never leaves ambient.
                sev = pd.to_numeric(r.get("Severity"), errors="coerce")
                summ[int(r["ID"])] = dict(
                    tr=str(r.get("TR")).strip() == "TR",
                    severe_tr=str(r.get("Severe TR")).strip() == "Severe TR",
                    severity=None if pd.isna(sev) else int(sev),
                    severity_cluster=(None if pd.isna(sev)
                                      else SEVERITY_CLUSTER.get(int(sev))),
                    tr_initiation=(None if pd.isna(r.get("TR_Initiation"))
                                   else float(r["TR_Initiation"])),
                    max_temp=(None if pd.isna(r.get("Max_Temp"))
                              else float(r["Max_Temp"])),
                    init_type=(None if pd.isna(r.get("Init_Type"))
                               else str(r["Init_Type"])),
                )

    n = 0
    for member in _members(zpath, exe):
        for sid, df in _stream(zpath, member, exe):
            t = df["Time"].to_numpy(float)
            ser = {ch: (t, df[src].to_numpy(float))
                   for src, ch in KEEP.items() if src in df.columns}
            meta = dict(
                cell_type="coupled electrical-thermal model",
                chemistry="NMC111", anode="graphite",
                source_doi="10.17632/9cykcc3svn.2",
                form_factor="simulated", capacity_ah=None, soc_pct=None,
                soh="n/a", trigger="simulated_heating", scale="cell",
                simulated=True, **summ.get(sid, {}))
            notes = ["SIMULATED, not measured -- pretraining only, "
                     "never included in any evaluation split",
                     "only the temperature column is carried; per the source "
                     "paper V is the model overpotential and I a "
                     "capacity-normalised current rate (1/s), neither of which "
                     "is the physical quantity its name suggests",
                     "Temp is exported in degC although the model works in "
                     "kelvin; the summary workbook mixes the two (IniT is in K, "
                     "Max_Temp in degC)"]
            yield RawExperiment(DS, "sim_%04d" % sid, ser, meta, 0.0, notes)
            n += 1
            if limit and n >= limit:
                return
