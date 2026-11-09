# native-rate KIT ARC records (D6) read from the original Zenodo text files
import glob
import os
import warnings

import numpy as np
import pandas as pd

DOI_BY_FORMAT = {"18650": "10.5281/zenodo.14956641", "21700": "10.5281/zenodo.7707929", "4680": "10.5281/zenodo.14956635"}


# one raw record: channel -> (time s, values) at the native sampling
class RawExperiment:
    def __init__(self, dataset_id, experiment_id, series, meta=None, t_trigger=None, notes=None):
        self.dataset_id = dataset_id
        self.experiment_id = experiment_id
        self.series = series
        self.meta = meta or {}
        self.t_trigger = t_trigger
        self.notes = notes or []


# positional parse: the files name four columns but write six fields per row
def read(path):
    with open(path, encoding="latin-1") as fh:
        names = fh.readline().replace("\t", " ").split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.ParserWarning)
        df = pd.read_csv(path, sep=r"[\t ]+", engine="python", header=None, skiprows=1, encoding="latin-1", index_col=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = [names[i] if i < len(names) else "col%d" % i for i in range(df.shape[1])]
    df = df.dropna(subset=list(df.columns[:2]))
    return df.sort_values(df.columns[0])


# all D6 records under raw_dir/{18650,21700,4680}
def load(raw_dir):
    base = str(raw_dir)
    files = sorted(glob.glob(os.path.join(base, "**", "*.txt"), recursive=True) +
                   glob.glob(os.path.join(base, "**", "*.EXO"), recursive=True))
    for f in files:
        form = os.path.basename(os.path.dirname(f))
        stem = os.path.splitext(os.path.basename(f))[0]
        df = read(f)
        if len(df) < 20:
            continue
        t = df.iloc[:, 0].to_numpy(float) * 60.0
        T = df.iloc[:, 1].to_numpy(float)
        ser = {"T_surface_mid": (t, T)}
        if df.shape[1] >= 4:
            hp = df.iloc[:, 3].to_numpy(float)
            if np.isfinite(hp).any() and np.nanmax(np.abs(hp)) > 0:
                ser["T_heater"] = (t, hp)
        chem = next((c for c in ("LFP", "NMC811", "NMC", "NCA", "SIB") if c in stem.upper()), "unknown")
        soc = None
        for tok in stem.replace("-", "_").split("_"):
            if tok.upper().startswith("SOC"):
                try:
                    soc = float(tok[3:])
                except ValueError:
                    pass
        if soc is None and form in ("18650", "4680"):
            soc = 100.0
        meta = dict(cell_type=stem, chemistry=chem, form_factor=form, capacity_ah=None, soc_pct=soc, soh="fresh",
                    soh_pct=100, trigger="arc_hws", scale="cell", source_doi=DOI_BY_FORMAT.get(form))
        yield RawExperiment("ds12_arc", "%s__%s" % (form, stem), ser, meta, 0.0)


# finite, time-sorted observations with the first duplicate kept
def clean_series(t, v):
    t = np.asarray(t, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    n = min(len(t), len(v))
    t, v = t[:n], v[:n]
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    if not len(t):
        return t, v
    order = np.argsort(t, kind="mergesort")
    t, v = t[order], v[order]
    unique = np.r_[True, np.diff(t) > 0]
    return t[unique], v[unique]
