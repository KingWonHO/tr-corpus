"""#12 KIT ARC 18650 / 21700 / 4680 -- zero-shot holdout only (plan E11).

Three Zenodo records from the same KIT group, all CC BY 4.0:

  18650  Ohneseit, Schoeberl, Lienkamp, Seifert & Ziebert (2025)
         doi:10.5281/zenodo.14956641  -- LFP, NMC811 and SODIUM-ION, 100% SOC
  21700  Ohneseit, Finster, Floras, Lubenau, Uhlmann, Seifert & Ziebert (2023)
         doi:10.5281/zenodo.7707929   -- LFP, NMC, NCA (HEI/HEII/HP), SOC 0-100
  4680   Ohneseit, Schoeberl, Lienkamp, Seifert & Ziebert (2025)
         doi:10.5281/zenodo.14956635  -- NMC811 and LFP, 100% SOC

What the records settle:

  * the temperature channel is the SURFACE AT THE CENTRE of the cell, so
    T_surface_mid is the right target;
  * M1/M2/M3/M4 are separate cells, i.e. replicates of one condition, not
    measurement channels;
  * 18650 and 4680 are 100% SOC throughout (their filenames carry no SOC);
  * recording starts ONLY once the cell is exothermic above 0.02 degC/min, so
    the file does not begin at the start of the ARC run -- the first row of
    these files sits 600-2000 minutes in.  t=0 is the ARC run start; the
    record is the exothermic tail of it.

The 18650 set includes two SODIUM-ION cells.  They are kept, because a
zero-shot holdout that contains an out-of-family chemistry is a harder and more
honest test, but any per-chemistry breakdown must treat them separately.

Local copy note: the 21700 record lists 60 files and this tree has 55.  Every
chemistry x SOC combination still has at least two replicates, so nothing is
structurally missing, but the set is incomplete against the published record.

66 experiments, temperature only, from an accelerating-rate calorimeter running
heat-wait-seek.  The time axis is in MINUTES and is event-driven: samples appear
when the cell crosses a temperature step, so the mean interval is minutes early
on and sub-second during runaway.

Putting this on a 1 Hz grid means heavy upsampling.  That is exactly the
"realistic worst case" role the plan assigns it, and every filled sample is
reported through is_interpolated so a model cannot mistake interpolation for
measurement.  This dataset is never used for training.
"""
import glob
import os
import numpy as np
import pandas as pd

from common import RawExperiment

DS = "ds12_arc"
ROOT = "Dataset/12/TR_KIT ARC"

DOI_BY_FORMAT = {
    "18650": "10.5281/zenodo.14956641",
    "21700": "10.5281/zenodo.7707929",
    "4680": "10.5281/zenodo.14956635",
}


def _read(path):
    """Positional parse.

    The .EXO files name four columns in the header but write six numeric fields
    per row, so reading with header=0 makes pandas silently promote the two
    extra leading fields into an index -- the time axis then comes out
    non-monotonic and spanning negative values.  Column names are taken from the
    header line for documentation only; the data is read positionally.
    """
    with open(path, encoding="latin-1") as fh:
        names = fh.readline().replace("\t", " ").split()
    df = pd.read_csv(path, sep=r"[\t ]+", engine="python", header=None,
                     skiprows=1, encoding="latin-1", index_col=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = [names[i] if i < len(names) else "col%d" % i
                  for i in range(df.shape[1])]
    df = df.dropna(subset=list(df.columns[:2]))
    return df.sort_values(df.columns[0])


def load(root):
    base = os.path.join(root, ROOT)
    files = sorted(glob.glob(os.path.join(base, "**", "*.txt"), recursive=True) +
                   glob.glob(os.path.join(base, "**", "*.EXO"), recursive=True))
    for f in files:
        form = os.path.basename(os.path.dirname(f))
        stem = os.path.splitext(os.path.basename(f))[0]
        df = _read(f)
        if len(df) < 20:
            continue

        t = df.iloc[:, 0].to_numpy(float) * 60.0          # minutes -> seconds
        T = df.iloc[:, 1].to_numpy(float)
        ser = {"T_surface_mid": (t, T)}

        notes = ["ARC heat-wait-seek: native sampling is event-driven "
                 "(median interval %.1f s); upsampled to 1 Hz, every filled "
                 "sample flagged in is_interpolated"
                 % float(np.median(np.diff(t))),
                 "temperature is the cell surface at its centre (per the Zenodo "
                 "record), mapped to T_surface_mid",
                 "recording begins only once the cell is exothermic above "
                 "0.02 degC/min, so the file starts %.0f min into the ARC run"
                 % (t[0] / 60.0),
                 "temperature only: expressible in panel M1 only",
                 "zero-shot holdout, never used for training"]

        if df.shape[1] >= 4:
            hp = df.iloc[:, 3].to_numpy(float)
            if np.isfinite(hp).any() and np.nanmax(np.abs(hp)) > 0:
                ser["T_heater"] = (t, hp)
                notes.append("column 4 (heater power, W) carried in T_heater as "
                             "a trigger-side channel")

        chem = next((c for c in ("LFP", "NMC811", "NMC", "NCA", "SIB")
                     if c in stem.upper()), "unknown")
        soc = None
        for tok in stem.replace("-", "_").split("_"):
            if tok.upper().startswith("SOC"):
                try:
                    soc = float(tok[3:])
                except ValueError:
                    pass
        # 18650 and 4680 carry no SOC in the filename; both records state 100%
        if soc is None and form in ("18650", "4680"):
            soc = 100.0
            notes.append("SOC not in filename; both the 18650 and 4680 Zenodo "
                         "records state 100% SOC")
        if chem == "SIB":
            notes.append("SODIUM-ION cell, not lithium-ion: out-of-family for "
                         "this corpus, keep separate in any chemistry breakdown")
        meta = dict(
            cell_type=stem, chemistry=chem, form_factor=form,
            capacity_ah=None, soc_pct=soc, soh="fresh", soh_pct=100,
            trigger="arc_hws", scale="cell",
            arc_instrument="Thermal Hazard Technology ES ARC",
            arc_seek_threshold_c_per_min=0.02,
            source_doi=DOI_BY_FORMAT.get(form),
        )
        yield RawExperiment(DS, "%s__%s" % (form, stem), ser, meta, 0.0, notes)
