# record-level helpers: cell design labels, adjudicated evaluation pools and record roles
import os
import re

import numpy as np
import pandas as pd

from core import windows

TARGET = "ds12_arc"
DS_OF = {"D1": "ds01_bak", "D2": "ds02_overcharge", "D3": "ds03_warwick",
         "D4": "ds04_osf", "D5": "ds09_mech", "D6": "ds12_arc"}
LABELS = {"L2@1.0": "t_onset_L2_1.0", "L2@0.5": "t_onset_L2_0.5", "L2@2.0": "t_onset_L2_2.0",
          "L1 venting": "t_onset_L1", "L3 voltage": "t_onset_L3", "ISC 25mV": "t_isc"}


# nominal capacity from a D5 record name
def nominal_ah(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*mAh", name, re.I)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*Ah", name, re.I)
    if m:
        return float(m.group(1))
    raise ValueError("no capacity in " + name)


# nominal cell design; records sharing it are replicates
def cell_model(row):
    ds, ct = row["dataset_id"], str(row["cell_type"])
    chem, cap = str(row["chemistry"]), row["capacity_ah"]
    src = row["cell_source"] if isinstance(row["cell_source"], str) else ""
    if ds == "ds12_arc":
        fmt = str(row["experiment_id"]).split("__")[0]
        design = re.sub(r"_SOC\d+_M\d+$", "", ct)
        design = re.sub(r"_cell\d+$", "", design)
        return "D6:%s:%s" % (fmt, design)
    if ds == "ds09_mech":
        if src == "Soteria":
            return "D5:Soteria NMC811"
        if src in ("Chevrolet Volt", "Nissan Leaf"):
            return "D5:" + src
        cap = float(cap) if pd.notna(cap) else nominal_ah(str(row["experiment_id"]))
        if chem == "LCO" and 6.2 <= cap <= 6.6:
            return "D5:LCO 6.4 Ah"
        return "D5:%s %g Ah" % (chem, cap)
    return {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3", "ds04_osf": "D4"}[ds] + ":" + chem


# windows whose adjudicated outcome agrees with the rule label
def adjudicated_mask(ds, y_tr, path):
    if not os.path.exists(path):
        return np.ones(len(ds), bool)
    adj = pd.read_csv(path)
    fa = {"%s/%s" % (DS_OF[a], e): f for a, e, f in zip(adj.dataset, adj.experiment_id, adj.final_assignment)}
    y = np.asarray(y_tr)
    want = np.where(y == 1, "positive", "negative")
    got = np.array([fa.get(k, "unlisted") for k in np.asarray(ds).astype(str)])
    return (got == want) | (got == "unlisted")


# source records eligible for the paired panels, and the fixed calibration/check negatives
class Plan:
    def __init__(self, d, meta, reg, panels, adjudication):
        ds = np.asarray(d["experiment"]).astype(str)
        key2split = dict(zip(reg.dataset_id + "/" + reg.experiment_id, reg.split))
        is_src = ~np.char.startswith(ds, TARGET + "/")
        neg = d["y_tr"] == 0
        split = np.array([key2split.get(k, "?") for k in ds])
        observed = np.asarray(d.get("label_observed", np.ones(len(ds), bool))).astype(bool)
        evaluable = observed & adjudicated_mask(ds, d["y_tr"], adjudication)
        self.ds = ds
        self.observed = observed
        self.evaluable = evaluable
        self.cal_neg = np.flatnonzero(is_src & neg & evaluable & (split == "val"))
        self.test_neg = np.flatnonzero(is_src & neg & evaluable & (split == "test"))
        reserved = set(ds[is_src & neg & observed & np.isin(split, ["val", "test"])])
        elig = None
        for p in panels:
            e = set(windows.panel_experiments(reg, p)["file"].map(lambda f: "%s/%s" % (f.split("/")[1], f.split("/")[2][:-8])))
            elig = e if elig is None else (elig & e)
        self.elig = elig & set(ds[is_src])
        self.pool = np.flatnonzero(is_src & observed & np.isin(ds, list(self.elig)) & ~np.isin(ds, list(reserved)))
        self.frozen_train = self.pool[split[self.pool] == "train"]
        onset = dict(zip(meta.key, meta.t_onset))
        evaluable_keys = set(ds[evaluable])
        self.folds = [k for k in sorted(self.elig) if np.isfinite(onset.get(k, np.nan)) and k in evaluable_keys]
        self.trig = dict(zip(meta.key, meta.t_trigger))
        self.onsets = {name: dict(zip(reg.dataset_id + "/" + reg.experiment_id, reg[col]))
                       for name, col in LABELS.items() if col in reg.columns}
        pool, cal, test = (set(self.ds[idx]) for idx in (self.pool, self.cal_neg, self.test_neg))
        assert not (pool & cal) and not (pool & test) and not (cal & test)
        assert len(cal) and len(test) and len(self.frozen_train)
