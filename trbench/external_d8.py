"""D8 field-negative external case, executed under the rules committed in
reports/24_d8_field_negative_protocol.md before any analysis was run.

    uv run python trbench/external_d8.py integrity     # licences, MD5 against the archives
    uv run python trbench/external_d8.py index         # one pass over the Zhang snippets
    uv run python trbench/external_d8.py accounting    # protocol 6.1
    uv run python trbench/external_d8.py score         # protocol 6.2
    uv run python trbench/external_d8.py manifest      # frozen-hash re-check

Sources (both CC BY 4.0, verified through the provider APIs on 2026-09-22):
  Zhang et al., Nat. Commun. 14:5940 (2023), figshare 10.6084/m9.figshare.23659323
  Cao et al., Nat. Commun. 16:1651 (2025), Zenodo 10.5281/zenodo.10656500

D8 supplies negatives only.  The release carries no runaway onset time, so no
lead, recall or detection number is computed here -- see protocol section 6.3.
Nothing in this module reads or writes tr-corpus/registry/experiments.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(HERE / "adapters")]

OUT = ROOT / "tr-corpus" / "results" / "external_20260922_d8_field"
EXT = ROOT / "tr-corpus" / "external" / "d8_field"
ZHANG = ROOT / "Dataset" / "BMS" / "Tsinghua" / "23659323"
CAO = ROOT / "Dataset" / "BMS"

# protocol section 1 -- must be unchanged at the end of the case
FROZEN = {
    "tr-corpus/registry/experiments.csv": "25f303024fa660cdb31fc2fef1db9515a30917a31223b7d2121778f8ad984e3d",
    "tr-corpus/splits/split_assignment.csv": "3109d69118e8fc0db0bc94857894a7e93475538f7d29377a014dbe1ee78a5ff6",
    "tr-corpus/results/validation_20260914_native_final/calibration.csv": "77a16effd747d45ca13df1f2c4095442807f495594fb3d4987e546a0b89a720f",
    "tr-corpus/results/validation_20260914_common_surface/calibration.csv": "3ecf5cae83db1bad8fbf48bc824527715ee3d7490c30d8114e22de9aad593422",
    "tr-corpus/results/validation_20260921_seq_matched/calibration.csv": "2032fcc5458b73f8061a19a01c9833927ed428600bbadfd6140d9e165672e50b",
    "trbench/common.py": "624ca4be7c7cb9e1290560525c456ed90b4b69e62e20644170ba8c482a0aa09b",
    "trbench/native_windows.py": "ff8faa62beac44de281deb2530ae26cabfa4462774dc8bf884d3762a9857177a",
    # run_validation.py re-pinned 2026-09-24 after adding the --horizon option and the
    # mask_only / age_only shortcut-control arms; the scoring functions reused here are unchanged.
    "trbench/run_validation.py": "c27f0e680fe4b991a54815a8c4485dd8fe44f71c82f9e91d79d79366e21b5f18",
    "trbench/schema.py": "56393ca4fcba95e081a145d46735e072e066c4717e859516b0bd132645654a90",
}

# protocol section 2.1: figshare supplied_md5, read from the API before the run
FIGSHARE_MD5 = {
    "battery_brand1.tar.gz": "8d32af34063c0475587c07f6021861a1",
    "battery_brand2.tar.gz": "8e16ac45307edc13630bcf8b833f361e",
    "battery_brand3.tar.gz": "038e953f199bc86f356e03c88098df99",
}
LICENCES = {"figshare 23659323": "CC BY 4.0", "zenodo 10656500": "cc-by-4.0"}

# protocol section 2.1: the release layout
BRANDS = {
    "battery_brand1": (("train", "test"), "label/%s_label.csv"),
    "battery_brand2": (("train", "test"), "label/%s_label.csv"),
    "battery_brand3": (("data",), "label/all_label.csv"),
}
COLUMNS = ["volt", "current", "soc", "max_single_volt",
           "min_single_volt", "max_temp", "min_temp", "timestamp"]

# protocol section 5
MAX_SNIPPETS = 30
# protocol section 6.2
HEAD_S = 300.0
# protocol section 2.2: Cao frames carry no timestamp; the paper's nominal rate
CAO_NOMINAL_FRAME_S = 30.0

ARMS = ("no_age", "with_age", "surface_max_only", "surface_mean_only")
TREE_CAL = {
    "no_age": "tr-corpus/results/validation_20260914_native_final/calibration.csv",
    "with_age": "tr-corpus/results/validation_20260914_native_final/calibration.csv",
    "surface_max_only": "tr-corpus/results/validation_20260914_common_surface/calibration.csv",
    "surface_mean_only": "tr-corpus/results/validation_20260914_common_surface/calibration.csv",
}
SEQ_CAL = "tr-corpus/results/validation_20260921_seq_matched/calibration.csv"
SEEDS = (0, 1, 2, 3, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def frozen_check():
    return {k: dict(expected=v, actual=sha(ROOT / k), unchanged=sha(ROOT / k) == v)
            for k, v in FROZEN.items()}


# --------------------------------------------------------------- integrity

def run_integrity():
    EXT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, supplied in FIGSHARE_MD5.items():
        p = ZHANG / name
        actual = md5(p) if p.exists() else None
        rows.append(dict(source="figshare 23659323", file=name, bytes=p.stat().st_size if p.exists() else 0,
                         supplied_md5=supplied, local_md5=actual,
                         status="ok" if actual == supplied else "integrity_failed",
                         licence=LICENCES["figshare 23659323"]))
    for name in ("DTI", "GIS", "QAS"):
        d = CAO / name
        rows.append(dict(source="zenodo 10656500", file=name + "/ (extracted)",
                         bytes=0, supplied_md5="", local_md5="",
                         status="extracted copy, archive checksums not applicable",
                         licence=LICENCES["zenodo 10656500"]))
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "integrity.csv", index=False)
    print(df.to_string(index=False))
    if (df.status == "integrity_failed").any():
        raise SystemExit("integrity failed -- protocol section 2.1 stops the case here")


# ------------------------------------------------------------------- index

def _snippet_paths(brand):
    subdirs, _ = BRANDS[brand]
    for sub in subdirs:
        d = ZHANG / brand / sub
        for p in sorted(d.glob("*.pkl")):
            yield sub, p


def _read_meta(args):
    """Metadata of one snippet.  Returns None when the file cannot be read."""
    import torch
    brand, sub, path = args
    try:
        arr, meta = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:                                  # noqa: BLE001
        return dict(brand=brand, split=sub, file=Path(path).name, error=repr(exc)[:120])
    t = np.asarray(arr[:, COLUMNS.index("timestamp")], float)
    return dict(brand=brand, split=sub, file=Path(path).name,
                car=int(meta["car"]), label=int(str(meta["label"])[0]),
                charge_segment=str(meta["charge_segment"]),
                mileage=float(meta["mileage"]), n_rows=int(arr.shape[0]),
                t_first=float(t[0]), t_last=float(t[-1]), error="")


def run_index(workers=8):
    EXT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, sub, str(p)) for b in BRANDS for sub, p in _snippet_paths(b)]
    print(f"{len(jobs)} snippets", flush=True)
    rows, started = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_read_meta, jobs, chunksize=256), 1):
            rows.append(row)
            if i % 50000 == 0:
                print(f"  {i}/{len(jobs)}  {time.time()-started:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(EXT / "snippet_index.csv.gz", index=False, compression="gzip")
    bad = df.error.astype(str).ne("").sum()
    print(f"indexed {len(df)} snippets in {time.time()-started:.0f}s, {bad} unreadable")
    print(df.groupby(["brand", "label"]).size().to_string())


def _index():
    df = pd.read_csv(EXT / "snippet_index.csv.gz")
    return df[df.error.isna() | (df.error.astype(str) == "")].copy()


def _vehicle_labels():
    """car -> label from the release's own label files, per brand."""
    out = {}
    for brand, (_, pattern) in BRANDS.items():
        frames = []
        if "%s" in pattern:
            for split in ("train", "test"):
                frames.append(pd.read_csv(ZHANG / brand / (pattern % split)))
        else:
            frames.append(pd.read_csv(ZHANG / brand / pattern))
        lab = pd.concat(frames, ignore_index=True)
        out[brand] = dict(zip(lab.car.astype(int), lab.label.astype(int)))
    return out


# -------------------------------------------------------------- accounting

def _cao_counts():
    import torch
    rows = []
    for name in ("DTI", "GIS", "QAS"):
        d = CAO / name
        if not d.exists():
            continue
        labels = {}
        xls = d / "Labels.xls"
        if xls.exists():
            lab = pd.read_excel(xls)
            labels = dict(zip(lab.iloc[:, 0].astype(int), lab.iloc[:, 1].astype(int)))
        vehicles = sorted(p for p in d.iterdir() if p.is_dir())
        frames, unreadable = 0, 0
        for v in vehicles:
            f = v / "vin_1.pkl"
            if not f.exists():
                unreadable += 1
                continue
            t = None
            for loader in (lambda p: pickle.load(open(p, "rb")),
                           lambda p: torch.load(p, weights_only=False, map_location="cpu")):
                try:
                    t = loader(f)
                    break
                except Exception:                              # noqa: BLE001,PERF203
                    continue
            if t is None:
                unreadable += 1
            else:
                frames += int(np.asarray(getattr(t, "cpu", lambda: t)()).shape[0])
        rows.append(dict(
            source="Cao 2025 (Zenodo 10656500)", subset=name,
            vehicles=len(vehicles),
            vehicles_normal=sum(1 for v in vehicles if labels.get(int(v.name), None) == 0) if labels else np.nan,
            vehicles_abnormal=sum(1 for v in vehicles if labels.get(int(v.name), None) == 1) if labels else np.nan,
            records=frames, record_s=CAO_NOMINAL_FRAME_S,
            exposure_h=frames * CAO_NOMINAL_FRAME_S / 3600.0,
            note=("no timestamps in the release; 1 frame ~ 30 s is the paper's nominal rate"
                  + (f"; {unreadable} vehicles unreadable" if unreadable else ""))))
    return rows


def run_accounting():
    OUT.mkdir(parents=True, exist_ok=True)
    idx, labels = _index(), _vehicle_labels()
    idx["label"] = [labels[b].get(c, -1) for b, c in zip(idx.brand, idx.car)]
    idx["span_s"] = idx.t_last - idx.t_first
    rows = []
    for brand, g in idx.groupby("brand"):
        for lab, name in ((0, "normal"), (1, "abnormal")):
            h = g[g.label == lab]
            if not len(h):
                continue
            rows.append(dict(source="Zhang 2023 (figshare 23659323)", subset=f"{brand} / {name}",
                             vehicles=h.car.nunique(), vehicles_normal=h.car.nunique() if lab == 0 else 0,
                             vehicles_abnormal=h.car.nunique() if lab == 1 else 0,
                             records=len(h), record_s=float(h.span_s.median()),
                             exposure_h=float(h.span_s.sum()) / 3600.0, note=""))
    z = idx[idx.label == 0]
    rows.append(dict(source="Zhang 2023 (figshare 23659323)", subset="all normal vehicles",
                     vehicles=z.groupby(["brand", "car"]).ngroups,
                     vehicles_normal=z.groupby(["brand", "car"]).ngroups, vehicles_abnormal=0,
                     records=len(z), record_s=float(z.span_s.median()),
                     exposure_h=float(z.span_s.sum()) / 3600.0,
                     note="charging segments only; driving and parked time are not in the release"))
    rows += _cao_counts()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exposure_accounting.csv", index=False)
    print(df.to_string(index=False))
    print("\nheld-out negatives in this paper's frozen split: 21 (D6) and 26 (source check);"
          "\n29 independent records with no alarm would bound the false-alarm probability at 0.10.")


# ------------------------------------------------------------------ sample

def sample_snippets():
    """Protocol section 5.2: <= 30 snippets per normal vehicle, evenly spaced
    over its charging history, no randomness."""
    idx, labels = _index(), _vehicle_labels()
    idx["label"] = [labels[b].get(c, -1) for b, c in zip(idx.brand, idx.car)]
    idx = idx[idx.label == 0]
    idx["snippet_id"] = idx.file.str.replace(".pkl", "", regex=False).astype(int)
    keep = []
    for (brand, car), g in idx.groupby(["brand", "car"], sort=True):
        g = g.sort_values(["charge_segment", "snippet_id"], kind="mergesort")
        n = len(g)
        if n <= MAX_SNIPPETS:
            take = np.arange(n)
        else:
            take = np.unique(np.round(np.arange(MAX_SNIPPETS) * (n - 1) / (MAX_SNIPPETS - 1)).astype(int))
        keep.append(g.iloc[take])
    out = pd.concat(keep, ignore_index=True)
    return out.sort_values(["brand", "car", "charge_segment", "snippet_id"], kind="mergesort")


# ------------------------------------------------------------------- score

def _d8_windows(brand, split, name):
    """One snippet -> (window end times, window tensor, window mask).

    The channel mapping is protocol section 3; the resampling is the corpus's
    own causal rule, unchanged.
    """
    import torch
    import native_windows as NW
    arr, _ = torch.load(ZHANG / brand / split / name, weights_only=False, map_location="cpu")
    a = np.asarray(arr, float)
    t = a[:, COLUMNS.index("timestamp")]
    # The two pack probes enter as unnamed extra surface members, which is what
    # T_surface_x* is for: causal_record then derives T_surface_max and
    # T_surface_mean with the corpus's own aggregation rule, and no claim is
    # made about where on the pack either probe sits.
    series = {"T_surface_x1": (t, a[:, COLUMNS.index("max_temp")]),
              "T_surface_x2": (t, a[:, COLUMNS.index("min_temp")])}
    raw = type("Raw", (), dict(series=series, dataset_id="d8_field",
                               experiment_id=f"{brand}/{name}"))
    grid, f, m, _t0, _, _ = NW.causal_record(raw)
    starts = NW._window_starts(len(grid))
    if not len(starts):
        return None
    take = starts[:, None] + np.arange(NW.WINDOW_S)[None, :]
    # Seconds since the start of this charging segment: D8 has no event axis,
    # and the head window of protocol 6.2 is the first 300 s of the record.
    tend = grid[starts + NW.WINDOW_S - 1]
    return tend, f[take], m[take]


def _predictors(arms, models):
    """Fit every (arm, model, seed) once and gate each on its stored tau."""
    import evalv2 as V
    import make_windows as MW
    import run_e3 as R
    import run_validation as RV
    import survival as SV
    import deep as DEEP

    d, meta = R.load_windows("W60_native", splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(ROOT / "tr-corpus" / "registry" / "experiments.csv")
    plan = V.Plan(d, meta, reg, ["M1"])
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], R.HORIZON_S)
    names = np.array(d["features"])
    age_cols = np.array([s.startswith("age__") or s == MW.AGE_CH for s in names])

    cal = {}
    for arm in ARMS:
        cal[("tree", arm)] = pd.read_csv(ROOT / TREE_CAL[arm])
    cal[("seq", None)] = pd.read_csv(ROOT / SEQ_CAL)

    out, gate = {}, []
    for arm in arms:
        select = RV.representation_mask(names, arm)
        if arm == "no_age":
            select &= ~age_cols
        cols = np.flatnonzero(select)
        F = RV.summary_features(d, cols)
        Frule = RV.summary_features(d, np.flatnonzero(select & ~age_cols))
        # One copy of the window tensor per arm, not per seed.
        seq_in = None
        if any(m in DEEP.SEQ_MODELS for m in models):
            seq_in = (d["X"][:, :, cols], d["mask"][:, :, cols])
        for model in models:
            # The rule has no seed and, as in the main runs, is not differentiated
            # by the observation clock: it is scored once, under no_age.
            if model == "rule" and arm != "no_age":
                continue
            seq = model in DEEP.SEQ_MODELS
            stored = cal[("seq", None)] if seq else cal[("tree", arm)]
            for seed in ((0,) if model == "rule" else SEEDS):
                row = stored[(stored.model == model) & (stored.seed == seed)
                             & (stored.age_mode == arm)]
                if not len(row):
                    gate.append(dict(arm=arm, model=model, seed=seed, tau_stored=np.nan,
                                     tau_refit=np.nan, reproduced=False,
                                     status="no stored threshold for this arm"))
                    continue
                started = time.time()
                if seq:
                    Xs, Ms = seq_in
                    fn = DEEP.fit_seq(model, Xs[plan.frozen_train], Ms[plan.frozen_train],
                                      y[plan.frozen_train], w[plan.frozen_train], seed)
                    r_cal = np.asarray(fn(Xs[plan.cal_neg], Ms[plan.cal_neg]), float)
                else:
                    Fm = Frule if model == "rule" else F
                    fn = R.fit_model(model, Fm[plan.frozen_train], y[plan.frozen_train],
                                     w[plan.frozen_train], seed)
                    r_cal = np.asarray(fn(Fm[plan.cal_neg]), float)
                tau = SV.calibrate_threshold(r_cal, d["experiment"][plan.cal_neg],
                                             np.ones(len(r_cal), bool), 0.1)
                tau_stored = float(row.tau.iloc[0])
                ok = bool(np.isclose(tau, tau_stored, rtol=1e-9, atol=0))
                gate.append(dict(arm=arm, model=model, seed=seed, tau_stored=tau_stored,
                                 tau_refit=float(tau), reproduced=ok,
                                 status="ok" if ok else "not assessable (refit did not reproduce tau)",
                                 fit_s=round(time.time() - started, 1)))
                print(f"  {arm:18s} {model:15s} seed={seed} tau={tau:.8g} "
                      f"{'ok' if ok else 'TAU MISMATCH'} ({time.time()-started:.0f}s)", flush=True)
                if ok:
                    out[(arm, model, seed)] = ("seq" if seq else "tree",
                                               cols if seq else (cols if model != "rule"
                                                                 else np.flatnonzero(select & ~age_cols)),
                                               fn, tau)
    return out, pd.DataFrame(gate)


def run_score(models, arms, chunk=200, limit=0):
    import run_e3 as R
    OUT.mkdir(parents=True, exist_ok=True)
    sample = sample_snippets()
    if limit:
        sample = sample.groupby(["brand", "car"], sort=False).head(1).head(limit)
    sample.to_csv(EXT / "scored_snippets.csv", index=False)
    print(f"{len(sample)} snippets from {sample.groupby(['brand','car']).ngroups} normal vehicles",
          flush=True)

    preds, gate = _predictors(arms, models)
    gate.to_csv(OUT / "refit_gate.csv", index=False)
    if not preds:
        raise SystemExit("no model reproduced its stored threshold; nothing scored")

    # accumulators, one row per (key, brand, car)
    acc, first_rows = {}, []
    rows = list(sample.itertuples(index=False))
    started = time.time()
    for i in range(0, len(rows), chunk):
        block = rows[i:i + chunk]
        built = []
        for r in block:
            got = _d8_windows(r.brand, r.split, r.file)
            if got is not None:
                built.append((r, got))
        if not built:
            continue
        tend = np.concatenate([g[0] for _, g in built])
        Wf = np.concatenate([g[1] for _, g in built])
        Wm = np.concatenate([g[2] for _, g in built])
        edges = np.cumsum([0] + [len(g[0]) for _, g in built])
        feat = {}     # tree summary features are shared by every model on the same columns
        for key, (kind, cols, fn, tau) in preds.items():
            if kind == "tree":
                ck = cols.tobytes()
                if ck not in feat:
                    feat[ck] = R.window_features(Wf, Wm, cols)
                risk = np.asarray(fn(feat[ck]), float)
            else:
                risk = np.asarray(fn(Wf[:, :, cols], Wm[:, :, cols]), float)
            for j, (r, _) in enumerate(built):
                s, e = edges[j], edges[j + 1]
                rr, tt = risk[s:e], tend[s:e]
                hit = rr >= tau
                head = tt <= HEAD_S
                a = acc.setdefault((key, r.brand, int(r.car)), dict(
                    snippets=0, snippets_alarm=0, snippets_alarm_head=0,
                    windows=0, windows_alarm=0, exposure_s=0.0))
                a["snippets"] += 1
                a["snippets_alarm"] += int(hit.any())
                a["snippets_alarm_head"] += int((hit & head).any())
                a["windows"] += len(rr)
                a["windows_alarm"] += int(hit.sum())
                a["exposure_s"] += float(r.t_last - r.t_first)
                if hit.any():
                    first_rows.append(dict(arm=key[0], model=key[1], seed=key[2],
                                           brand=r.brand, car=int(r.car), file=r.file,
                                           first_alarm_s=float(tt[np.flatnonzero(hit)[0]])))
        done = min(i + chunk, len(rows))
        print(f"  scored {done}/{len(rows)} snippets  {time.time()-started:.0f}s", flush=True)

    per_vehicle = pd.DataFrame([dict(arm=k[0][0], model=k[0][1], seed=k[0][2],
                                     brand=k[1], car=k[2], **v) for k, v in acc.items()])
    per_vehicle.to_csv(OUT / "per_vehicle.csv", index=False)
    pd.DataFrame(first_rows).to_csv(OUT / "first_alarm_times.csv", index=False)

    agg = []
    for (arm, model, seed), g in per_vehicle.groupby(["arm", "model", "seed"]):
        hours = g.exposure_s.sum() / 3600.0
        agg.append(dict(
            arm=arm, model=model, seed=seed,
            vehicles=len(g), vehicles_alarm=int((g.snippets_alarm > 0).sum()),
            vehicles_alarm_head=int((g.snippets_alarm_head > 0).sum()),
            snippets=int(g.snippets.sum()), snippets_alarm=int(g.snippets_alarm.sum()),
            snippets_alarm_head=int(g.snippets_alarm_head.sum()),
            windows=int(g.windows.sum()), windows_alarm=int(g.windows_alarm.sum()),
            exposure_h=round(hours, 2),
            alarms_per_1000_vehicle_h=round(g.windows_alarm.sum() / hours * 1000.0, 3) if hours else np.nan))
    out = pd.DataFrame(agg).sort_values(["arm", "model", "seed"])
    out.to_csv(OUT / "alarm_rates.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False))


# ---------------------------------------------------------------- manifest

def run_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    man = dict(
        case="D8 field negatives",
        protocol="reports/24_d8_field_negative_protocol.md",
        git_head=head,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        python=platform.python_version(),
        sources={"zhang_2023": "10.6084/m9.figshare.23659323",
                 "cao_2025": "10.5281/zenodo.10656500"},
        licences=LICENCES,
        sampling=dict(max_snippets_per_vehicle=MAX_SNIPPETS, rule="even spacing, no randomness",
                      abnormal_vehicles="excluded (no fault onset time)"),
        panel="M1", head_window_s=HEAD_S, budget=0.1,
        computed=["exposure accounting", "false alarms on normal vehicles"],
        not_computed=["lead time", "recall", "AUROC"],
        frozen=frozen_check(),
    )
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    bad = [k for k, v in man["frozen"].items() if not v["unchanged"]]
    print(json.dumps({k: v["unchanged"] for k, v in man["frozen"].items()}, indent=2))
    print("frozen inputs unchanged" if not bad else f"CHANGED: {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["integrity", "index", "accounting", "score", "manifest"])
    ap.add_argument("--models", default="rule,xgboost,lightgbm,gru,mamba,itransformer,convtransformer")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="debug: score only N snippets")
    a = ap.parse_args()
    if a.command == "integrity":
        run_integrity()
    elif a.command == "index":
        run_index(a.workers)
    elif a.command == "accounting":
        run_accounting()
    elif a.command == "score":
        run_score(a.models.split(","), [x for x in a.arms.split(",") if x], a.chunk, a.limit)
    else:
        run_manifest()


if __name__ == "__main__":
    main()
