"""Cluster-aware bootstrap for the false-alarm and paired-detection results.

Every interval elsewhere treats an experiment as the sampling unit.  That is
too optimistic when experiments repeat one cell design: D6's non-runaway
cells are replicates (M1, M2, M3) of six designs, and D5's are drawn from
about a dozen commercial cell models tested at several states of charge.
Replicates of one design share whatever makes a model alarm on it, so they
are not independent evidence.

This module resamples CELL MODELS instead of experiments.  It fits nothing:
it reads the per-experiment outcomes that run_validation.py and run_v2.py
already wrote, groups them by a deterministic cell-model rule, and reports
percentile intervals from a cluster bootstrap next to the experiment-level
ones.  With 6-12 clusters a percentile bootstrap is itself approximate; it is
reported as a dependence sensitivity, not as a replacement interval.

    uv run python trbench/cluster_bootstrap.py

Writes tr-corpus/results/summary_20260915_cluster_bootstrap/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "tr-corpus" / "results"
REG = ROOT / "tr-corpus" / "registry" / "experiments.csv"

# Held-dataset negatives of the trees and of the four sequence models (same build,
# sensor values only); concatenated by _read_neg().
NEG_RUNS = (RES / "validation_20260914_native_final", RES / "validation_20260921_seq_matched")
NEG_CELLS = [d / "negative_cells.csv" for d in NEG_RUNS]
NEG_COMPARE = [d / "negative_comparisons.csv" for d in NEG_RUNS]


def _read_neg(paths):
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)
# Internal design, all six learned models on identical inputs (protocol 20).
E3A_RAW = [RES / "validation_20260921_all7_rescored" / f
           for f in ("validation_20260914_e3a_native.csv",
                     "validation_20260914_e3a_native_xgb_s1to4.csv",
                     "validation_20260914_e3a_native_lgb_s1to4.csv",
                     "e3a_seq_s0to4.csv")]
E3A_PAIRS = RES / "summary_20260921_all7" / "panel_pairs.csv"
LEARNED = ("xgboost", "lightgbm", "gru", "mamba", "itransformer", "convtransformer")
OUT = RES / "summary_20260921_cluster_bootstrap"

N_BOOT = 10_000
SEED = 20260915


# ------------------------------------------------------------------ clusters
def _nominal_ah(name: str) -> float:
    """Nominal capacity from a D5 experiment name when the registry field is empty."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*mAh", name, re.I)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*Ah", name, re.I)
    if m:
        return float(m.group(1))
    raise ValueError("no capacity in " + name)


def cell_model(row) -> str:
    """Deterministic cell-model label; experiments sharing it are replicates."""
    ds, ct = row["dataset_id"], str(row["cell_type"])
    # D5 rule: named programme / vehicle cells are one model each; otherwise
    # chemistry x nominal capacity (merging more is the conservative choice).
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
        cap = float(cap) if pd.notna(cap) else _nominal_ah(str(row["experiment_id"]))
        if chem == "LCO" and 6.2 <= cap <= 6.6:      # measured 6.27-6.56 Ah, nominal 6.4 Ah
            return "D5:LCO 6.4 Ah"
        return "D5:%s %g Ah" % (chem, cap)
    # D1-D4 each test a single cell design
    return {"ds01_bak": "D1", "ds02_overcharge": "D2",
            "ds03_warwick": "D3", "ds04_osf": "D4"}[ds] + ":" + chem


def cluster_map() -> pd.DataFrame:
    reg = pd.read_csv(REG)
    reg = reg[reg.role != "pretrain_only"].copy()
    reg["key"] = reg.dataset_id + "/" + reg.experiment_id.astype(str)
    reg["cluster"] = reg.apply(cell_model, axis=1)
    return reg[["key", "dataset_id", "cell_type", "chemistry", "capacity_ah", "cluster", "split"]]


# ------------------------------------------------------------------ bootstrap
def cluster_ratio_ci(k, n, groups, rng, n_boot=N_BOOT, alpha=0.05):
    """Percentile CI of sum(k)/sum(n) resampling whole groups with replacement."""
    k, n, groups = np.asarray(k, float), np.asarray(n, float), np.asarray(groups)
    labels, inv = np.unique(groups, return_inverse=True)
    kg = np.bincount(inv, weights=k, minlength=len(labels))
    ng = np.bincount(inv, weights=n, minlength=len(labels))
    draw = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    ks, ns = kg[draw].sum(axis=1), ng[draw].sum(axis=1)
    est = np.where(ns > 0, ks / np.where(ns > 0, ns, 1), np.nan)
    lo, hi = np.nanquantile(est, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), len(labels)


def cluster_diff_ci(a, b, groups, rng, n_boot=N_BOOT, alpha=0.05):
    """Paired mean(b − a) per experiment, resampling whole groups."""
    d = np.asarray(b, float) - np.asarray(a, float)
    lo, hi, g = cluster_ratio_ci(d, np.ones_like(d), groups, rng, n_boot, alpha)
    return lo, hi, g


def cluster_two_pool_diff_ci(k1, g1, k2, g2, rng, n_boot=N_BOOT, alpha=0.05):
    """Rate(pool 2) − rate(pool 1), each pool resampled by its own groups."""
    def boot(k, g):
        k = np.asarray(k, float)
        labels, inv = np.unique(np.asarray(g), return_inverse=True)
        kg = np.bincount(inv, weights=k, minlength=len(labels))
        ng = np.bincount(inv, minlength=len(labels)).astype(float)
        draw = rng.integers(0, len(labels), size=(n_boot, len(labels)))
        return kg[draw].sum(axis=1) / ng[draw].sum(axis=1), len(labels)
    r1, n1 = boot(k1, g1)
    r2, n2 = boot(k2, g2)
    lo, hi = np.quantile(r2 - r1, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), n1, n2


# ------------------------------------------------------------------ analyses
def negative_rates(cmap, rng):
    cells = _read_neg(NEG_CELLS)
    cells = cells[(cells.age_mode == "no_age") & cells.eligible.astype(bool)]
    cells = cells.merge(cmap[["key", "cluster"]], left_on="held", right_on="key", how="left")
    assert cells.cluster.notna().all(), "unmapped negative experiment"
    cells["fired"] = cells.fired.astype(str).str.lower().eq("true").astype(int)
    rows = []
    for (m, seed, pool, w, anc), g in cells.groupby(["model", "seed", "pool", "window_s", "anchor"]):
        lo, hi, G = cluster_ratio_ci(g.fired, np.ones(len(g)), g.cluster, rng)
        rows.append(dict(model=m, seed=seed, pool=pool, window_s=w, anchor=anc,
                         n_experiments=len(g), n_clusters=G, n_alarm=int(g.fired.sum()),
                         far=g.fired.mean(), cluster_ci_lo=lo, cluster_ci_hi=hi))
    rates = pd.DataFrame(rows)

    diffs = []
    comp = _read_neg(NEG_COMPARE)
    comp = comp[comp.age_mode == "no_age"]
    for (m, seed, w, anc), g in cells.groupby(["model", "seed", "window_s", "anchor"]):
        s, a = g[g.pool == "source_mech"], g[g.pool == "arc"]
        if not len(s) or not len(a):
            continue
        lo, hi, gs, ga = cluster_two_pool_diff_ci(s.fired, s.cluster, a.fired, a.cluster, rng)
        c = comp[(comp.model == m) & (comp.seed == seed) & (comp.window_s.astype(str) == str(w))
                 & (comp.anchor == anc)]
        diffs.append(dict(model=m, seed=seed, window_s=w, anchor=anc,
                          n_source=len(s), n_source_clusters=gs, n_arc=len(a), n_arc_clusters=ga,
                          far_difference=a.fired.mean() - s.fired.mean(),
                          cell_ci_lo=c.difference_ci_lo.iloc[0] if len(c) else np.nan,
                          cell_ci_hi=c.difference_ci_hi.iloc[0] if len(c) else np.nan,
                          cluster_ci_lo=lo, cluster_ci_hi=hi))
    return rates, pd.DataFrame(diffs)


def paired_detection(cmap, rng):
    raw = pd.concat([pd.read_csv(p) for p in E3A_RAW], ignore_index=True)
    raw = raw[raw.model.isin(LEARNED) & (raw.label == "L2@1.0")]
    wide = raw.pivot_table(index=["model", "seed", "held"], columns="panel",
                           values="detected", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["M1", "M2"])
    wide = wide.merge(cmap[["key", "cluster"]], left_on="held", right_on="key", how="left")
    assert wide.cluster.notna().all(), "unmapped held experiment"
    pairs = pd.read_csv(E3A_PAIRS)
    pairs = pairs[(pairs.scope == "overall") & (pairs.label == "L2@1.0") & (pairs.panel == "M2")]
    rows = []
    for (m, seed), g in wide.groupby(["model", "seed"]):
        a, b = g.M1.astype(float), g.M2.astype(float)
        lo, hi, G = cluster_diff_ci(a, b, g.cluster, rng)
        p = pairs[(pairs.model == m) & (pairs.seed == seed)]
        rows.append(dict(model=m, seed=seed, n_cells=len(g), n_clusters=G,
                         detection_rate_difference=b.mean() - a.mean(),
                         cell_ci_lo=p.detection_diff_ci_low.iloc[0], cell_ci_hi=p.detection_diff_ci_high.iloc[0],
                         cluster_ci_lo=lo, cluster_ci_hi=hi))
    return pd.DataFrame(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cmap = cluster_map()
    rates, diffs = negative_rates(cmap, rng)
    paired = paired_detection(cmap, rng)

    cmap.to_csv(out / "cluster_map.csv", index=False)
    rates.to_csv(out / "negative_rates_cluster.csv", index=False)
    diffs.to_csv(out / "negative_difference_cluster.csv", index=False)
    paired.to_csv(out / "e3a_m2_m1_cluster.csv", index=False)
    manifest = dict(status="exploratory dependence sensitivity", n_bootstrap=N_BOOT, rng_seed=SEED,
                    sampling_unit="cell model (see cell_model() and cluster_map.csv)",
                    inputs={str(p.relative_to(ROOT)).replace(os.sep, "/"): sha(p)
                            for p in [REG, *NEG_CELLS, *NEG_COMPARE, E3A_PAIRS, *E3A_RAW]},
                    code_sha256=sha(__file__))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    neg = cmap[cmap.key.isin(_read_neg(NEG_CELLS).held.unique())]
    print("negative clusters:", neg.groupby(neg.key.str.startswith("ds12")).cluster.nunique().to_dict())
    print(rates.shape, diffs.shape, paired.shape, "->", out)


if __name__ == "__main__":
    main()
