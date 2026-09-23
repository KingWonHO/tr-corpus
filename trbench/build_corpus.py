"""Run every adapter, standardize, and write the distributable corpus.

    raw/{dataset}  ->  adapter  ->  standardize()  ->  tr-corpus/

Usage:  python build_corpus.py [--only ds01_bak,ds03_warwick] [--sim-limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "adapters")]

import schema as S                                    # noqa: E402
from common import standardize, baseline_scaler_stats  # noqa: E402

import ds01_bak, ds02_overcharge, ds03_warwick, ds04_osf   # noqa: E402
import ds09_mech, ds12_arc, ds15_sim                        # noqa: E402

ADAPTERS = {
    "ds01_bak": ds01_bak, "ds02_overcharge": ds02_overcharge,
    "ds03_warwick": ds03_warwick, "ds04_osf": ds04_osf,
    "ds09_mech": ds09_mech, "ds12_arc": ds12_arc, "ds15_sim": ds15_sim,
}

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "tr-corpus")
MIN_SAMPLES = 30          # an experiment shorter than this is unusable


def _dirs():
    for d in ("measured", "simulated", "registry", "splits", "scalers"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)


def build(only=None, sim_limit=None):
    _dirs()
    rows, scalers, all_notes = [], {}, []

    for ds_id, mod in ADAPTERS.items():
        if only and ds_id not in only:
            continue
        info = S.DATASETS[ds_id]
        sub = "simulated" if info["role"] == "pretrain_only" else "measured"
        outdir = os.path.join(OUT, sub, ds_id)
        os.makedirs(outdir, exist_ok=True)

        t0, n_ok, n_fail = time.time(), 0, 0
        kwargs = {"limit": sim_limit} if ds_id == "ds15_sim" and sim_limit else {}
        print("[%s] %s" % (ds_id, info["name"]))
        try:
            it = mod.load(ROOT, **kwargs)
        except Exception:
            traceback.print_exc()
            continue

        while True:
            try:
                raw = next(it)
            except StopIteration:
                break
            except Exception:
                n_fail += 1
                traceback.print_exc()
                continue
            try:
                df, meta = standardize(raw)
            except Exception as e:
                n_fail += 1
                print("   ! %s: %s" % (raw.experiment_id, e))
                continue
            if len(df) < MIN_SAMPLES:
                n_fail += 1
                print("   ! %s: only %d samples on the 1 Hz grid, dropped"
                      % (raw.experiment_id, len(df)))
                continue

            safe = "".join(c if c.isalnum() or c in "-_." else "_"
                           for c in meta["experiment_id"])
            df.to_parquet(os.path.join(outdir, safe + ".parquet"), index=False)

            scalers["%s/%s" % (ds_id, safe)] = baseline_scaler_stats(df, meta)
            all_notes.extend("[%s/%s] %s" % (ds_id, safe, n) for n in meta["notes"])

            row = {k: v for k, v in meta.items() if k != "notes"}
            row["file"] = "%s/%s/%s.parquet" % (sub, ds_id, safe)
            row["experiment_id"] = safe
            row["role"] = info["role"]
            row["license"] = info["license"]
            row["dataset_no"] = info["no"]
            row["channels_present"] = ",".join(meta["channels_present"])
            row["n_channels"] = len(meta["channels_present"])
            rows.append(row)
            n_ok += 1

        print("   %d experiments, %d failed, %.1fs" % (n_ok, n_fail, time.time() - t0))

    if not rows:
        print("nothing built")
        return None

    exp = pd.DataFrame(rows)
    exp.to_csv(os.path.join(OUT, "registry", "experiments.csv"), index=False)

    with open(os.path.join(OUT, "scalers", "baseline_stats.json"), "w") as f:
        json.dump(scalers, f, indent=1)
    with open(os.path.join(OUT, "registry", "preprocessing_notes.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(sorted(set(all_notes))))

    _write_registry(exp)
    return exp


def _write_registry(exp):
    """Table 1 (dataset registry) and Table 2 (label expressibility matrix)."""
    reg = []
    for ds_id, info in S.DATASETS.items():
        sub = exp[exp["dataset_id"] == ds_id]
        if not len(sub):
            continue
        chans = sorted({c for s in sub["channels_present"] for c in s.split(",") if c})
        reg.append(dict(
            dataset_id=ds_id, no=info["no"], name=info["name"],
            license=info["license"], role=info["role"], scale=info["scale"],
            chemistry=info["chemistry"], form_factor=info["form_factor"],
            trigger=info["trigger"], native_hz=info["native_hz"],
            n_experiments=len(sub),
            total_duration_s=round(float(sub["duration_s"].sum()), 1),
            median_duration_s=round(float(sub["duration_s"].median()), 1),
            n_channels=len(chans), channels=",".join(chans),
            L1_computable=int(sub["t_onset_L1"].notna().sum()),
            L2_computable=int(sub["t_onset_L2"].notna().sum()),
            L3_computable=int(sub["t_onset_L3"].notna().sum()),
            redistributable=info["redistributable"],
        ))
    pd.DataFrame(reg).to_csv(os.path.join(OUT, "registry", "datasets.csv"), index=False)

    # Table 2: label definitions and sensor panels each dataset can express.
    # Expressibility is per EXPERIMENT and requires the panel's full channel
    # set, so a dataset counts only for the experiments that actually carry it.
    exp = exp.copy()
    exp["_panels"] = exp["channels_present"].apply(
        lambda s: S.panel_expressible(s.split(",")))
    mat = []
    for ds_id in exp["dataset_id"].unique():
        sub = exp[exp["dataset_id"] == ds_id]
        row = dict(dataset_id=ds_id, n_experiments=len(sub))
        for lab in ("L1", "L2", "L3"):
            n = int(sub["t_onset_%s" % lab].notna().sum())
            row[lab] = "%d/%d" % (n, len(sub))
        for panel in S.PANEL_REQUIRES:
            k = int(sub["_panels"].apply(lambda p: panel in p).sum())
            row[panel] = "%d/%d" % (k, len(sub))
        mat.append(row)
    pd.DataFrame(mat).to_csv(os.path.join(OUT, "registry", "label_matrix.csv"),
                             index=False)

    # Panel statistical power -- the table that decides what E3 may claim.
    # The binding constraint is not the experiment count (277 core, gate passed)
    # but how many experiments carry BOTH sides of a comparison.
    meas = exp[exp["role"] != "pretrain_only"]
    power = []
    for panel in S.PANEL_REQUIRES:
        hit = meas[meas["_panels"].apply(lambda p: panel in p)]
        tr = hit[hit["t_onset_L2"].notna()]
        power.append(dict(
            panel=panel, role=S.PANEL_ROLE.get(panel, "?"),
            n_experiments=len(hit), n_with_tr=len(tr),
            n_datasets=hit["dataset_id"].nunique(),
            datasets=",".join(sorted(hit["dataset_id"].unique())),
            triggers=",".join(sorted(hit["trigger"].unique())),
            channels=",".join(c for c in S.SENSOR_PANELS.get(panel, [])
                              if c not in S.TRIGGER_SIDE),
        ))
    pd.DataFrame(power).to_csv(os.path.join(OUT, "registry", "panel_power.csv"),
                               index=False)

    # Channel co-occurrence -- the gap in its quantitative form.
    # Channel availability alone understates the problem: a corpus can carry
    # every channel somewhere and still support no comparison, because a
    # comparison needs BOTH channels in the SAME experiment.  Sixteen of these
    # pairs are zero across all public data.
    groups = {
        "T_surface": S.GROUP_T_SURF, "T_internal": S.GROUP_TINT,
        "T_vent": ["T_vent", "T_vent_2"],
        "P_internal": ["P_internal"], "P_chamber": ["P_chamber", "P_chamber_2"],
        "V": S.GROUP_V, "I": ["I"],
        "gas_speciated": ["gas_H2", "gas_CO", "gas_CO2", "gas_HF", "gas_CH4"],
        "gas_derived": ["gas_total_mol"], "F_expansion": ["F_expansion"],
    }
    meas = exp[exp["role"] != "pretrain_only"]
    sets = [set(str(c).split(",")) for c in meas["channels_present"].fillna("")]
    dsid = list(meas["dataset_id"])
    names = list(groups)
    have = {g: [any(c in s_ for c in groups[g]) for s_ in sets] for g in names}
    co = []
    for a in names:
        row = {"group": a}
        for b in names:
            hit = [i for i in range(len(sets)) if have[a][i] and have[b][i]]
            row[b] = len(hit)
            if a < b:
                row.setdefault("_", None)
        co.append(row)
    pd.DataFrame(co).drop(columns=["_"], errors="ignore").to_csv(
        os.path.join(OUT, "registry", "co_occurrence.csv"), index=False)

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hit = [k for k in range(len(sets)) if have[a][k] and have[b][k]]
            pairs.append(dict(group_a=a, group_b=b, n_experiments=len(hit),
                              datasets=",".join(sorted({dsid[k] for k in hit}))))
    pd.DataFrame(pairs).sort_values("n_experiments").to_csv(
        os.path.join(OUT, "registry", "co_occurrence_pairs.csv"), index=False)

    # per-channel availability, experiment counts
    av = []
    for ds_id in exp["dataset_id"].unique():
        sub = exp[exp["dataset_id"] == ds_id]
        r = dict(dataset_id=ds_id, n_experiments=len(sub))
        for ch in S.CHANNELS:
            r[ch] = int(sum(ch in s.split(",") for s in sub["channels_present"]))
        av.append(r)
    pd.DataFrame(av).to_csv(
        os.path.join(OUT, "registry", "channel_availability.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--sim-limit", type=int, default=None)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    exp = build(only, a.sim_limit)
    if exp is not None:
        print("\n%d experiments ->" % len(exp), OUT)
        print(exp.groupby("dataset_id").size().to_string())
