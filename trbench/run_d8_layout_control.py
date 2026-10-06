"""D8 layout control: hold the D8 target tensor fixed, change only the source.

External review of 2026-10-06.  In ``external_d8.py`` the M1 arm ("no_age")
and the single-channel aggregate arms ("surface_max_only",
"surface_mean_only") differ on BOTH sides of the transfer: each has its own
source fit, and the D8 tensor each model reads also differs (six M1 slots
with availability pattern 000110 versus one temperature channel).  This
script adds arms whose D8 input is the unchanged M1 tensor of the "no_age"
arm, while the source side is re-presented in that same layout:

    slot             neg pos mid max  mean ambient
    D8 mask (M1)      0   0   0   1    1     0       (verified on every window)

    d8layout_agg     source windows: neg/pos/mid/ambient values 0, mask 0;
                     max slot <- source T_surface_max, mean slot <- source
                     T_surface_mean (M1 restricted to the slots D8 carries)
    d8layout_max     as above, both populated slots <- source T_surface_max
    d8layout_mean    as above, both populated slots <- source T_surface_mean

The populated slots keep the source aggregate's own availability mask (the
max and mean masks are identical by construction in native_windows).  Train
and calibration negatives are both re-presented; nothing else changes:
same frozen train split, same 60 s hazard target, same 10 % record-level
calibration on the 25 validation negatives, same D8 snippets, same alarm
accounting as ``external_d8.run_score``.

Two contrasts follow, each on one D8 input:
  * target fixed (D8 M1 tensor):   no_age vs d8layout_agg vs d8layout_max
                                   vs d8layout_mean
  * source representation fixed:   surface_max_only (1 slot) vs d8layout_max
                                   (6 slots), likewise for the mean

The three existing arms are refitted here, gated on their stored tau exactly
as ``external_d8._predictors`` does, and re-scored; their D8 rows must equal
external_20260925_d8_field/alarm_rates.csv (reproduction_check.csv).  The
new arms have no stored tau: their gate is that an independent refit with the
same seed reproduces tau (rtol 1e-9), and the taus are written to
calibration.csv so a later run can gate against them.

    uv run --no-sync python trbench/run_d8_layout_control.py \\
        --out tr-corpus/results/review_20261006/d8_layout

Nothing here writes outside --out; existing modules are imported, not edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(HERE / "adapters")]

import external_d8 as EXT     # noqa: E402  constants and the D8 window builder; no I/O on import
import run_e3 as R            # noqa: E402

DEFAULT_OUT = ROOT / "tr-corpus" / "results" / "review_20261006" / "d8_layout"
STORED_D8 = EXT.OUT           # external_20260925_d8_field, read only

M1_SLOTS = ("T_surface_neg", "T_surface_pos", "T_surface_mid",
            "T_surface_max", "T_surface_mean", "T_ambient")
D8_POPULATED = ("T_surface_max", "T_surface_mean")
D8_PATTERN = "".join("1" if s in D8_POPULATED else "0" for s in M1_SLOTS)   # 000110

EXISTING_ARMS = ("no_age", "surface_max_only", "surface_mean_only")
# populated D8 slot -> source channel copied into it
LAYOUT_ARMS = {
    "d8layout_agg": {"T_surface_max": "T_surface_max", "T_surface_mean": "T_surface_mean"},
    "d8layout_max": {"T_surface_max": "T_surface_max", "T_surface_mean": "T_surface_max"},
    "d8layout_mean": {"T_surface_max": "T_surface_mean", "T_surface_mean": "T_surface_mean"},
}
ALL_ARMS = EXISTING_ARMS + tuple(LAYOUT_ARMS)
# which D8 column set each arm reads; every layout arm reads the no_age tensor
D8_COLSET = {"no_age": "m1", "surface_max_only": "max", "surface_mean_only": "mean",
             **{a: "m1" for a in LAYOUT_ARMS}}

ARM_DEFINITIONS = {
    "no_age": dict(
        source="M1 sensor values as recorded: 6 slots (neg,pos,mid,max,mean,ambient), "
               "each with its own availability mask (mixed patterns across datasets)",
        d8_target="M1 tensor: 6 slots, mask 000110; max slot = max(max_temp,min_temp), "
                  "mean slot = mean(max_temp,min_temp), other slots value 0 mask 0",
        tau="stored, validation_20260925_native/calibration.csv"),
    "surface_max_only": dict(
        source="1 slot: source T_surface_max with its mask",
        d8_target="1 slot: D8 T_surface_max (= max_temp probe), mask 1",
        tau="stored, validation_20260925_common_surface/calibration.csv"),
    "surface_mean_only": dict(
        source="1 slot: source T_surface_mean with its mask",
        d8_target="1 slot: D8 T_surface_mean (= mean of the two probes), mask 1",
        tau="stored, validation_20260925_common_surface/calibration.csv"),
    "d8layout_agg": dict(
        source="6 M1 slots in the D8 layout: neg/pos/mid/ambient value 0 mask 0; "
               "max slot = source T_surface_max, mean slot = source T_surface_mean",
        d8_target="identical to no_age (D8 M1 tensor, mask 000110)",
        tau="new; gate = refit reproduces tau"),
    "d8layout_max": dict(
        source="6 M1 slots in the D8 layout; max AND mean slot = source T_surface_max",
        d8_target="identical to no_age (D8 M1 tensor, mask 000110)",
        tau="new; gate = refit reproduces tau"),
    "d8layout_mean": dict(
        source="6 M1 slots in the D8 layout; max AND mean slot = source T_surface_mean",
        d8_target="identical to no_age (D8 M1 tensor, mask 000110)",
        tau="new; gate = refit reproduces tau"),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------- source side

def arm_columns(names, arm):
    """Feature columns an arm reads, from the same rule external_d8 uses."""
    import make_windows as MW
    import run_validation as RV
    rep = "no_age" if arm in LAYOUT_ARMS else arm
    select = RV.representation_mask(names, rep)
    if rep == "no_age":
        select &= ~np.array([s.startswith("age__") or s == MW.AGE_CH for s in names])
    return np.flatnonzero(select)


def layout_view(X, M, names, arm):
    """Copy of a window batch with the six M1 slots rewritten into the D8 layout."""
    pos = {n: i for i, n in enumerate(names)}
    X2, M2 = X.copy(), M.copy()
    for slot in M1_SLOTS:
        X2[:, :, pos[slot]] = 0.0
        M2[:, :, pos[slot]] = 0
    for slot, src in LAYOUT_ARMS[arm].items():
        X2[:, :, pos[slot]] = X[:, :, pos[src]]
        M2[:, :, pos[slot]] = M[:, :, pos[src]]
    return X2, M2


def source_features(d, idx, cols, names, layout=None, chunk=2048):
    """window_features on the rows `idx`, optionally after the layout rewrite."""
    parts = []
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        X, M = d["X"][j], d["mask"][j]
        if layout is not None:
            X, M = layout_view(X, M, names, layout)
        parts.append(R.window_features(X, M, cols))
    return np.concatenate(parts) if parts else np.zeros((0, 9 * len(cols)), np.float32)


def fit_predictors(arms, models, seeds, budget=0.1):
    import evalv2 as V
    import run_validation as RV
    import survival as SV

    d, meta = R.load_windows("W60_native", splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(ROOT / "tr-corpus" / "registry" / "experiments.csv")
    plan = V.Plan(d, meta, reg, ["M1"])
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], R.HORIZON_S)
    names = np.array(d["features"]).astype(str)
    m1 = arm_columns(names, "no_age")
    if tuple(names[m1]) != M1_SLOTS:
        raise RuntimeError("M1 slot order changed: %s" % list(names[m1]))
    for a in arms:
        if a in LAYOUT_ARMS:
            pos = {n: i for i, n in enumerate(names)}
            mmax, mmean = d["mask"][:, :, pos["T_surface_max"]], d["mask"][:, :, pos["T_surface_mean"]]
            if not np.array_equal(mmax, mmean):
                raise RuntimeError("source max and mean masks differ; the layout arms assume equality")
            del mmax, mmean
            break

    target = np.char.startswith(plan.ds, V.TARGET + "/")
    src_pos = np.flatnonzero((d["split"] == "test") & ~target & (d["y_tr"] == 1) & plan.evaluable)
    roles = dict(train=plan.frozen_train, cal=plan.cal_neg, test_neg=plan.test_neg, test_pos=src_pos)
    print("source: train %d win / %d exp | cal-neg %d / %d | check-neg %d / %d | test-pos %d / %d"
          % tuple(x for idx in roles.values() for x in (len(idx), len(set(plan.ds[idx])))), flush=True)
    stored = {a: pd.read_csv(ROOT / EXT.TREE_CAL[a]) for a in EXISTING_ARMS}

    preds, gate, calib, sanity = {}, [], [], []
    tr = plan.frozen_train
    for arm in arms:
        cols = arm_columns(names, arm)
        layout = arm if arm in LAYOUT_ARMS else None
        F = {role: source_features(d, idx, cols, names, layout) for role, idx in roles.items()}
        for model in models:
            for seed in seeds:
                started = time.time()
                fn = R.fit_model(model, F["train"], y[tr], w[tr], seed)
                r_cal = np.asarray(fn(F["cal"]), float)
                tau = SV.calibrate_threshold(r_cal, d["experiment"][plan.cal_neg],
                                             np.ones(len(r_cal), bool), budget)
                if arm in EXISTING_ARMS:
                    s = stored[arm]
                    row = s[(s.model == model) & (s.seed == seed) & (s.age_mode == arm)]
                    ref = float(row.tau.iloc[0]) if len(row) else np.nan
                    ok = bool(len(row)) and bool(np.isclose(tau, ref, rtol=1e-9, atol=0))
                    kind = "stored tau"
                else:
                    fn2 = R.fit_model(model, F["train"], y[tr], w[tr], seed)
                    r2 = np.asarray(fn2(F["cal"]), float)
                    ref = SV.calibrate_threshold(r2, d["experiment"][plan.cal_neg],
                                                 np.ones(len(r2), bool), budget)
                    ok = bool(np.isclose(tau, ref, rtol=1e-9, atol=0))
                    kind = "independent refit (no stored tau exists)"
                    del fn2, r2
                gate.append(dict(arm=arm, model=model, seed=seed, gate=kind, tau_reference=ref,
                                 tau_refit=float(tau), reproduced=ok,
                                 status="ok" if ok else "not assessable (refit did not reproduce tau)",
                                 fit_s=round(time.time() - started, 1)))
                peaks = pd.Series(r_cal).groupby(plan.ds[plan.cal_neg]).max().to_numpy(float)
                calib.append(dict(model=model, seed=seed, age_mode=arm, tau=float(tau),
                                  n_cal=len(peaks), n_cal_alarm=int((peaks >= tau).sum()),
                                  empirical_far=float((peaks >= tau).mean())))
                # Source-side sanity: is the refitted detector a working detector?
                r_neg = np.asarray(fn(F["test_neg"]), float)
                r_pos = np.asarray(fn(F["test_pos"]), float)
                nr = RV.negative_records(d, plan.test_neg, r_neg, tau, None, "tail")
                nh = RV.negative_records(d, plan.test_neg, r_neg, tau, EXT.HEAD_S, "head")
                pr = RV.positive_records(d, src_pos, r_pos, tau, plan)
                sanity.append(dict(arm=arm, model=model, seed=seed, tau=float(tau),
                                   check_neg_records=int(nr.eligible.sum()),
                                   check_neg_alarm_full=int(nr.fired.eq(True).sum()),
                                   check_neg_eligible_head300=int(nh.eligible.sum()),
                                   check_neg_alarm_head300=int(nh.fired.eq(True).sum()),
                                   test_pos_records=len(pr), test_pos_detected=int(pr.detected.sum()),
                                   test_pos_pretrigger=int(pr.pre_trigger.sum()),
                                   test_pos_lead_median_s=float(pr.lead.median())))
                print("  %-18s %-9s seed=%d tau=%.10g %s [%s] (%.1fs)"
                      % (arm, model, seed, tau, "ok" if ok else "TAU MISMATCH", kind,
                         time.time() - started), flush=True)
                if ok:
                    preds[(arm, model, seed)] = (D8_COLSET[arm], fn, float(tau))
        del F
    colsets = {"m1": tuple(int(c) for c in m1),
               "max": tuple(int(c) for c in arm_columns(names, "surface_max_only")),
               "mean": tuple(int(c) for c in arm_columns(names, "surface_mean_only"))}
    del d
    return preds, colsets, pd.DataFrame(gate), pd.DataFrame(calib), pd.DataFrame(sanity)


# ----------------------------------------------------------------- D8 side

_COLSETS = None


def _worker_init(colsets):
    global _COLSETS
    _COLSETS = {k: np.asarray(v, dtype=np.int64) for k, v in colsets.items()}
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                          # noqa: BLE001
        pass


def _d8_job(job):
    """One snippet -> window end times, features per column set, layout counts."""
    brand, split, name = job
    got = EXT._d8_windows(brand, split, name)
    if got is None:
        return None
    tend, Wf, Wm = got
    feats = {k: R.window_features(Wf, Wm, c) for k, c in _COLSETS.items()}
    m1 = _COLSETS["m1"]
    pat = Wm[:, :, m1].astype(bool).any(axis=1).astype(np.uint8)
    keys, counts = np.unique(["".join(map(str, p)) for p in pat], return_counts=True)
    jmax, jmean = m1[M1_SLOTS.index("T_surface_max")], m1[M1_SLOTS.index("T_surface_mean")]
    check = dict(patterns=dict(zip(keys.tolist(), counts.tolist())),
                 populated_mask_steps=int(Wm[:, :, [jmax, jmean]].sum()),
                 populated_steps=int(Wm.shape[0] * Wm.shape[1] * 2),
                 windows_max_eq_mean=int(np.all(Wf[:, :, jmax] == Wf[:, :, jmean], axis=1).sum()))
    return tend, feats, check


def score_d8(preds, colsets, out, workers=8, chunk=400, limit=0):
    sample = EXT.sample_snippets()
    if limit:
        sample = sample.groupby(["brand", "car"], sort=False).head(1).head(limit)
    sample.to_csv(out / "scored_snippets.csv", index=False)
    stored_sample = EXT.EXT / "scored_snippets.csv"
    same_sample = (not limit and stored_sample.exists()
                   and sha(out / "scored_snippets.csv") == sha(stored_sample))
    print("%d snippets from %d normal vehicles (identical to the stored D8 sample: %s)"
          % (len(sample), sample.groupby(["brand", "car"]).ngroups, same_sample), flush=True)

    rows = list(sample.itertuples(index=False))
    jobs = [(r.brand, r.split, r.file) for r in rows]
    acc, first_rows = {}, []
    layout = dict(patterns={}, populated_mask_steps=0, populated_steps=0,
                  windows_max_eq_mean=0, windows=0, snippets_without_windows=0)
    started = time.time()

    def flush(block):
        if not block:
            return
        tend = np.concatenate([g[0] for _, g in block])
        edges = np.cumsum([0] + [len(g[0]) for _, g in block])
        feat = {k: np.concatenate([g[1][k] for _, g in block]) for k in colsets}
        for key, (ck, fn, tau) in preds.items():
            risk = np.asarray(fn(feat[ck]), float)
            for j, (r, _) in enumerate(block):
                s, e = edges[j], edges[j + 1]
                rr, tt = risk[s:e], tend[s:e]
                hit = rr >= tau
                head = tt <= EXT.HEAD_S
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

    ctx = mp.get_context("spawn")
    block = []
    with ctx.Pool(workers, initializer=_worker_init, initargs=(colsets,)) as pool:
        for i, (r, got) in enumerate(zip(rows, pool.imap(_d8_job, jobs, chunksize=8)), 1):
            if got is None:
                layout["snippets_without_windows"] += 1
            else:
                block.append((r, got))
                c = got[2]
                for k, v in c["patterns"].items():
                    layout["patterns"][k] = layout["patterns"].get(k, 0) + v
                for k in ("populated_mask_steps", "populated_steps", "windows_max_eq_mean"):
                    layout[k] += c[k]
                layout["windows"] += len(got[0])
            if len(block) >= chunk or i == len(rows):
                flush(block)
                block = []
                print("  scored %d/%d snippets  %.0fs" % (i, len(rows), time.time() - started), flush=True)
    flush(block)

    per_vehicle = pd.DataFrame([dict(arm=k[0][0], model=k[0][1], seed=k[0][2],
                                     brand=k[1], car=k[2], **v) for k, v in acc.items()])
    per_vehicle.to_csv(out / "per_vehicle.csv", index=False)
    pd.DataFrame(first_rows).to_csv(out / "first_alarm_times.csv", index=False)

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
            alarms_per_1000_vehicle_h=round(g.windows_alarm.sum() / hours * 1000.0, 3) if hours else np.nan,
            # paper/review_controls.py: an episode is a charging segment with any exceedance
            episodes_per_1000_vehicle_h=round(g.snippets_alarm.sum() / hours * 1000.0, 3) if hours else np.nan))
    rates = pd.DataFrame(agg).sort_values(["arm", "model", "seed"])
    rates.to_csv(out / "alarm_rates.csv", index=False)

    layout["fraction_populated_steps_available"] = (layout["populated_mask_steps"] / layout["populated_steps"]
                                                    if layout["populated_steps"] else np.nan)
    layout["fraction_windows_with_d8_pattern"] = (layout["patterns"].get(D8_PATTERN, 0) / layout["windows"]
                                                  if layout["windows"] else np.nan)
    layout["d8_pattern"] = D8_PATTERN
    layout["slot_order"] = list(M1_SLOTS)
    (out / "d8_layout_check.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")
    print("D8 M1 availability patterns:", layout["patterns"], flush=True)
    return rates, same_sample


def reproduction_check(rates, out):
    old = pd.read_csv(STORED_D8 / "alarm_rates.csv")
    cols = [c for c in old.columns if c not in ("arm", "model", "seed")]
    mine = rates[rates.arm.isin(EXISTING_ARMS)]
    m = mine.merge(old, on=["arm", "model", "seed"], how="left", suffixes=("", "_stored"),
                   indicator=True)
    m["identical"] = (m["_merge"] == "both") & np.logical_and.reduce(
        [np.isclose(m[c].astype(float), m[c + "_stored"].astype(float), rtol=0, atol=1e-9)
         for c in cols])
    m.drop(columns="_merge").to_csv(out / "reproduction_check.csv", index=False)
    return bool(len(m)) and bool(m.identical.all())


def summarise(rates, gate, out):
    order = {a: i for i, a in enumerate(ALL_ARMS)}
    rows = []
    for (arm, model), g in rates.groupby(["arm", "model"]):
        gg = gate[(gate.arm == arm) & (gate.model == model)]
        r = dict(arm=arm, model=model, seeds_scored=len(g),
                 seeds_gated_ok="%d/%d" % (int(gg.reproduced.sum()), len(gg)),
                 d8_target=ARM_DEFINITIONS[arm]["d8_target"])
        for c in ("vehicles_alarm", "vehicles_alarm_head", "snippets_alarm", "windows_alarm",
                  "alarms_per_1000_vehicle_h", "episodes_per_1000_vehicle_h"):
            lo, hi = g[c].min(), g[c].max()
            fmt = "%d" if c in ("vehicles_alarm", "vehicles_alarm_head", "snippets_alarm",
                                "windows_alarm") else "%.1f"
            r[c] = (fmt % lo) if lo == hi else ("%s-%s" % (fmt % lo, fmt % hi))
        rows.append(r)
    s = pd.DataFrame(rows)
    s = s.iloc[sorted(range(len(s)), key=lambda i: (s.model.iloc[i], order.get(s.arm.iloc[i], 99)))]
    s.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 30):
        print(s.drop(columns="d8_target").to_string(index=False))
    return s


# ---------------------------------------------------------------- manifest

def used_code():
    files = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "trbench" and p.suffix == ".py":
            files[str(rel).replace("\\", "/")] = sha(p)
    return dict(sorted(files.items()))


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def write_manifest(out, a, extra):
    window_base = ROOT / "tr-corpus" / "windows" / "W60_native"
    code = used_code()
    man = dict(
        purpose="D8 layout control (external review 2026-10-06): D8 target tensor held fixed, "
                "source representation changed",
        arguments=vars(a), git_head=git("rev-parse", "HEAD"),
        git_status_of_used_code={f: (git("status", "--porcelain", "--", f) or "clean") for f in code},
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), python=platform.python_version(),
        code_sha256_used=code,
        code_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(HERE.rglob("*.py"))},
        registry_sha256=sha(ROOT / "tr-corpus" / "registry" / "experiments.csv"),
        environment_lock_sha256=sha(ROOT / "uv.lock"),
        window_artifact_sha256={p.name: sha(p) for p in sorted(window_base.glob("*.npz"))},
        stored_calibration_sha256={k: sha(ROOT / v) for k, v in sorted(set(EXT.TREE_CAL.items()))},
        stored_d8_results_sha256={p.name: sha(p) for p in sorted(STORED_D8.glob("*.csv"))},
        d8_snippet_index_sha256=sha(EXT.EXT / "snippet_index.csv.gz"),
        external_d8_frozen_check=EXT.frozen_check(),
        arm_definitions=ARM_DEFINITIONS, d8_pattern_slot_order=list(M1_SLOTS), d8_pattern=D8_PATTERN,
        head_window_s=EXT.HEAD_S, budget=a.budget,
        analysis_status="exploratory sensitivity requested by external review; D8 previously inspected",
        **extra)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--models", default="xgboost,lightgbm")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="debug: score only N snippets")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    models = [x for x in a.models.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    if set(arms) - set(ALL_ARMS):
        raise ValueError("--arms must use: " + ",".join(ALL_ARMS))
    if set(models) - {"xgboost", "lightgbm"}:
        raise ValueError("this control covers the tree models only (xgboost, lightgbm)")
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("use a fresh output directory: %s" % out)
    for protected in (STORED_D8.resolve(), EXT.EXT.resolve()):
        if out == protected or protected in out.parents:
            raise ValueError("refusing to write inside %s" % protected)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    write_manifest(out, a, dict(status="running"))

    preds, colsets, gate, calib, sanity = fit_predictors(arms, models, seeds, a.budget)
    gate.to_csv(out / "refit_gate.csv", index=False)
    calib.to_csv(out / "calibration.csv", index=False)
    sanity.to_csv(out / "source_sanity.csv", index=False)
    pd.DataFrame([dict(arm=k, **v) for k, v in ARM_DEFINITIONS.items() if k in arms]).to_csv(
        out / "arm_definitions.csv", index=False)
    t_fit = time.time() - t0
    if not preds:
        raise SystemExit("no setting passed its gate; nothing scored")

    rates, same_sample = score_d8(preds, colsets, out, a.workers, a.chunk, a.limit)
    reproduced = (reproduction_check(rates, out)
                  if not a.limit and set(arms) & set(EXISTING_ARMS) else None)
    summarise(rates, gate, out)
    runtime = time.time() - t0
    print("existing-arm D8 rows identical to the stored run: %s" % reproduced)
    print("runtime %.0fs (fit+gate %.0fs)" % (runtime, t_fit))
    write_manifest(out, a, dict(status="complete", runtime_s=round(runtime, 1),
                                fit_and_gate_s=round(t_fit, 1),
                                d8_sample_identical_to_stored=same_sample,
                                existing_arm_rows_identical_to_stored=reproduced,
                                settings_passing_gate=int(gate.reproduced.sum()),
                                settings_total=len(gate)))


if __name__ == "__main__":
    main()
