"""#3 Warwick internal temperature & pressure (3 experiments).

Stored as a MATLAB table inside a v7 .mat, which scipy exposes only as an opaque
MCOS blob.  The table is recovered by reading the `__function_workspace__`
subsystem stream directly; cell 9 of the FileWrapper holds the variable names and
cell 4 holds one column per variable x one entry per test.

Gulsoy, Briggs, Ngo, Faraji Niri & Marco, "Dataset of internal temperature and
gas pressure in cylindrical lithium-ion cells during thermal runaway", Data in
Brief 63, 112190 (2025), doi:10.1016/j.dib.2025.112190.  Data:
doi:10.17632/rgfhdhcd9k.2.  Method paper: Gulsoy et al., J. Power Sources 617,
235147 (2024), doi:10.1016/j.jpowsour.2024.235147.

The paper confirms three things this adapter had been assuming, and corrects a
fourth:

  * IntPre really is in BAR (measured as a voltage signal, converted via a
    calibration transfer function);
  * the trigger is a 50x50 mm flexible heating pad at a constant 40 W applied
    from the start of the record until runaway, so t_trigger = record start is
    correct here rather than merely a fallback;
  * cells are at 100% SOC (CC-CV to 4.25 V, 1C, taper to C/20);
  * the cell is a Sony VTC6A -- an NCA cathode with a graphite/SiOx anode, not
    the NMC this adapter previously recorded.

Rates.  The paper states 1 kHz for the voltage-side channels and 10 Hz for the
temperature side.  The files carry 10 kHz on the voltage side, and 10 Hz
temperature for tests 1 and 2 but 100 Hz for test 3.  Resampling is driven by
each channel's measured interval, not by these figures, so the discrepancy does
not affect the output; it is recorded because a reader comparing the paper with
the files will hit it.  Test 3 has no vent thermocouples.
"""
import io
import os
import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

from common import RawExperiment

DS = "ds03_warwick"
MAT = "Dataset/3/TR_dataTable.mat"

VAR_MAP = {
    "IntPre": ("P_internal", 100.0),        # bar gauge -> kPa gauge
    "CellVoltage": ("V_cell", 1.0),
    "MidIntTemp": ("T_internal_core", 1.0),
    "MidSurfTemp": ("T_surface_mid", 1.0),
    "NegSurfTemp": ("T_surface_neg", 1.0),
    "PosSurfTemp": ("T_surface_pos", 1.0),
    "VentPos5mmAway": ("T_vent", 1.0),
    "VentPos10mmAway": ("T_vent_2", 1.0),
}
FAST = {"IntPre", "CellVoltage"}


def _read_table(path):
    ws = sio.loadmat(path, variable_names=["__function_workspace__"])
    b = ws["__function_workspace__"].tobytes()
    r = MatFile5Reader(io.BytesIO(b[8:]), byte_order="<", struct_as_record=True)
    r.initialize_read()
    hdr, _ = r.read_var_header()
    arr = r.read_var_array(hdr, process=False)
    opaque = arr["MCOS"][0, 0]
    # SciPy <=1.15 exposed this FileWrapper payload as `_ObjectMetadata`;
    # newer releases preserve the four MatlabOpaque fields and call the object
    # array `arr`.  Both contain the same 14-cell table metadata block.
    fields = opaque.dtype.names or ()
    payload = "_ObjectMetadata" if "_ObjectMetadata" in fields else "arr"
    if payload not in fields:
        raise ValueError("unsupported MATLAB MCOS FileWrapper fields: %r" %
                         (fields,))
    cells = opaque[payload].ravel()[0].ravel()
    names = [str(x[0]) for x in cells[9].ravel()]
    test_ids = [str(x[0]) for x in cells[2].ravel()]
    cols = cells[4].ravel()
    return names, test_ids, cols


def load(root):
    names, test_ids, cols = _read_table(os.path.join(root, MAT))
    name_idx = {n: i for i, n in enumerate(names)}

    for k, tid in enumerate(test_ids):
        t_fast = cols[name_idx["ExpTime"]].ravel()[k].ravel()
        t_slow = cols[name_idx["ExpTimeTemp"]].ravel()[k].ravel()

        ser, missing = {}, []
        for var, (ch, scale) in VAR_MAP.items():
            v = cols[name_idx[var]].ravel()[k].ravel()
            if v.size == 0:
                missing.append(var)
                continue
            t = t_fast if var in FAST else t_slow
            n = min(len(t), len(v))
            ser[ch] = (np.asarray(t[:n], float), np.asarray(v[:n], float) * scale)

        notes = ["pressure/voltage decimated from 10 kHz by bin mean "
                 "(anti-aliased), temperatures at %.0f Hz "
                 "(the paper states 1 kHz / 10 Hz; the files differ)"
                 % (len(t_slow) / max(t_slow[-1], 1.0)),
                 "trigger is a 50x50 mm heating pad at a constant 40 W from the "
                 "start of the record, so t_trigger = 0 is the real trigger",
                 "P_internal is gauge pressure; negative excursions after "
                 "rupture are a real sensor artefact, kept and flagged"]
        if missing:
            notes.append("absent in this test: " + ", ".join(missing))

        meta = dict(
            cell_type="Sony VTC6A", chemistry="NCA", anode="graphite-SiOx",
            form_factor="21700", capacity_ah=4.0, nominal_v=3.6,
            soc_pct=100, soh="fresh", soh_pct=100,
            trigger="external_heating", scale="cell", has_internal_sensor=True,
            heater_power_w=40.0, heater_pad_mm="50x50", ambient_c="15-20",
            source_doi="10.1016/j.dib.2025.112190",
            data_doi="10.17632/rgfhdhcd9k.2",
        )
        yield RawExperiment(DS, "test_%s" % tid, ser, meta, 0.0, notes)
