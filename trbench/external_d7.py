"""D7 external audit (Gardner et al. 2025), executed under the rules committed
in reports/11_d7_gardner_audit_protocol.md before the files were downloaded.

    uv run python trbench/external_d7.py events      # event table, interventions
    uv run python trbench/external_d7.py alarms      # frozen held-dataset alarms

Nothing here reads or writes tr-corpus/registry/experiments.csv.  Outputs go
to tr-corpus/results/external_20260915_d7_gardner/ and
tr-corpus/external/d7_gardner/.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(HERE / "adapters")]
import common as C            # noqa: E402
import schema as S            # noqa: E402
import d7_gardner as G        # noqa: E402

OUT = ROOT / "tr-corpus" / "results" / "external_20260915_d7_gardner"
EXT = ROOT / "tr-corpus" / "external" / "d7_gardner"

# protocol section 1 -- must be unchanged at the end of the audit
FROZEN = {
    "tr-corpus/registry/experiments.csv": "25f303024fa660cdb31fc2fef1db9515a30917a31223b7d2121778f8ad984e3d",
    "tr-corpus/splits/split_assignment.csv": "3109d69118e8fc0db0bc94857894a7e93475538f7d29377a014dbe1ee78a5ff6",
    "tr-corpus/results/validation_20260914_native_final/calibration.csv": "77a16effd747d45ca13df1f2c4095442807f495594fb3d4987e546a0b89a720f",
    "trbench/common.py": "624ca4be7c7cb9e1290560525c456ed90b4b69e62e20644170ba8c482a0aa09b",
    "trbench/native_windows.py": "ff8faa62beac44de281deb2530ae26cabfa4462774dc8bf884d3762a9857177a",
    "trbench/run_validation.py": "139ce380ef2b39eb1e2b3957a816ffe2b9a62c45f90e905e2baedee4e034d95a",
    "trbench/schema.py": "56393ca4fcba95e081a145d46735e072e066c4717e859516b0bd132645654a90",
}

# protocol section 3.3 constants
H2_BASELINE_S = 600.0
H2_K_SIGMA = 5.0
H2_MIN_PPM = 10.0
H2_SUSTAIN_S = 5.0
RTD_STUCK_N = 10          # operationalisation of "constant" (logged in protocol section 5)
RTD_OPEN_DEGC = 880.0     # the RTD's open-circuit reading (883.0/883.39) seen in 8 of 11 files


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_check():
    return {k: dict(expected=v, actual=sha(ROOT / k), unchanged=sha(ROOT / k) == v)
            for k, v in FROZEN.items()}


def _clean(t, v):
    ok = np.isfinite(v)
    return t[ok], v[ok]


def rtd_lost_at(t, T):
    """First sample of a terminal run of >= RTD_STUCK_N identical readings,
    or of a terminal all-NaN run; None if the probe reads to the end."""
    fin = np.isfinite(T)
    if not fin.any():
        return float(t[0])
    last = np.flatnonzero(fin)[-1]
    if last < len(T) - RTD_STUCK_N:
        return float(t[last + 1])
    j = last
    while j > 0 and fin[j - 1] and T[j - 1] == T[last]:
        j -= 1
    return float(t[j]) if last - j + 1 >= RTD_STUCK_N else None


def h2_onset(t, h):
    t, h = _clean(t, h)
    span = H2_BASELINE_S if t[-1] - t[0] > H2_BASELINE_S else 0.1 * (t[-1] - t[0])
    b = h[t <= t[0] + span]
    m = float(np.median(b))
    sigma = 1.4826 * float(np.median(np.abs(b - m)))
    thr = m + max(H2_K_SIGMA * sigma, H2_MIN_PPM)
    n = max(1, int(round(H2_SUSTAIN_S / float(np.median(np.diff(t))))))
    above = (h > thr).astype(int)
    run = np.convolve(above, np.ones(n, int), mode="valid")
    hit = np.flatnonzero(run >= n)
    return (float(t[hit[0]]) if len(hit) else None), m, sigma, thr


def _r(x):
    return None if x is None else round(float(x), 1)


def events(rec):
    t = rec.t
    T = rec.channels["T_case"].copy()
    lost = rtd_lost_at(t, T)
    if lost is not None:
        T[t >= lost] = np.nan
    tT, TT = _clean(t, T)
    row = dict(record=rec.experiment_id, role=rec.role, cell=rec.cell, stressor=rec.stressor,
               duration_s=float(t[-1] - t[0]), rtd_lost_s=lost,
               offset_zero_s=rec.offset_zero_s)
    for th in S.L2_THRESHOLDS:
        row["L2@%s" % th] = _r(C.onset_L2(tT, TT, th, ceiling=None))
    row["L2 unguarded"] = _r(C.onset_L2(tT, TT, S.L2_PRIMARY, guarded=False))
    i_pk = int(np.argmax(TT))
    row["T peak (degC)"], row["T peak time (s)"] = round(float(TT[i_pk]), 1), float(tT[i_pk])

    tv, vv = _clean(t, rec.channels["V_cell"])
    first = vv[(tv <= tv[0] + 120) & (vv > 0.05)]
    row["V noise sd, first 120 s (mV)"] = round(1000 * float(np.std(first)), 1)
    l3, isc = C.onset_L3(tv, vv, 0.0), C.onset_isc(tv, vv, 0.0)
    pair = rec.experiment_id in G.PARALLEL_PAIR
    row["L3 (computed)"], row["25 mV (computed)"] = _r(l3), _r(isc)
    row["voltage events scored"] = "not assessable (parallel pair)" if pair else "yes"

    tp, pp = _clean(t, rec.channels["P_chamber"])
    row["L1 chamber (computed)"] = _r(C.onset_L1(tp, pp, 0.0, "chamber"))
    base = pp[tp <= tp[0] + H2_BASELINE_S]
    row["P max rise over baseline (hPa)"] = round(float(pp.max() - np.median(base)), 2)

    on, m, sigma, thr = h2_onset(t, rec.channels["gas_H2"])
    row.update({"H2 onset (computed)": _r(on), "H2 baseline median (ppm)": round(m, 3),
                "H2 baseline sigma (ppm)": round(sigma, 3), "H2 threshold (ppm)": round(thr, 3),
                "H2 onset (author, in file)": "not in file"})
    lead = G.AUTHOR_H2_LEAD_MIN.get(rec.experiment_id)
    row["TR author-derived (s)"] = _r(on + 60 * lead) if (lead is not None and on is not None) else None
    return row


def intervention_row(rec, ev):
    t = rec.t
    on = ev["H2 onset (computed)"]
    out = dict(record=rec.experiment_id, cell=rec.cell, stressor=rec.stressor,
               duration_s=ev["duration_s"], **{"H2 onset (computed, s)": on},
               **{"stop time": "not in file"},
               **{"pre-onset record (s)": on})
    if on is None:
        return out
    # Post-test handling: the RTD reads its open-circuit value (883 degC) and
    # the voltage lead drops to zero or chatters.  Descriptors stop at the first
    # such sample after the H2 onset (protocol section 5, deviation 2).
    T = rec.channels["T_case"]
    V = rec.channels["V_cell"]
    art = (t > on) & ((T >= RTD_OPEN_DEGC) | (V < 0.05))
    end = float(t[np.flatnonzero(art)[0]]) if art.any() else float(t[-1])
    out["descriptor window end (s)"] = end
    post = (t >= on) & (t < end)
    out["T_case at H2 onset (degC)"] = round(float(np.interp(on, *_clean(t, T))), 1)
    out["max T_case after onset (degC)"] = round(float(np.nanmax(T[post])), 1)
    out["min V after onset (V)"] = round(float(np.nanmin(V[post])), 3)
    out["max T_case over whole file (degC, excl. 883 open-circuit)"] = round(float(np.nanmax(np.where(T >= RTD_OPEN_DEGC, np.nan, T))), 1)
    P = rec.channels["P_chamber"]
    out["max P rise after onset (hPa)"] = round(float(np.nanmax(P[post]) - np.nanmedian(P[t <= t[0] + H2_BASELINE_S])), 2)
    out["L2@1.0 (s)"] = ev["L2@1.0"]
    return out


def run_events():
    OUT.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    recs = G.read_all()
    evs = [events(r) for r in recs]
    ev = pd.DataFrame(evs)
    ev.to_csv(OUT / "event_times.csv", index=False)
    inter = pd.DataFrame([intervention_row(r, e) for r, e in zip(recs, evs) if r.role == "intervention"])
    inter.to_csv(OUT / "interventions.csv", index=False)
    pd.DataFrame([dict(record=r.experiment_id, role=r.role, cell=r.cell, stressor=r.stressor,
                       source_file=r.source_file, sheet=r.sheet, n_samples=len(r.t),
                       duration_s=float(r.t[-1]), dt_median_s=float(np.median(np.diff(r.t))),
                       dt_max_s=float(np.diff(r.t).max()), offset_zero_s=r.offset_zero_s,
                       notes="; ".join(r.notes), doi=G.DOI, license=G.LICENSE)
                  for r in recs]).to_csv(EXT / "registry.csv", index=False)
    pd.DataFrame([dict(record=r.experiment_id, channel=k, source_header=v)
                  for r in recs for k, v in r.header_map.items()]).to_csv(EXT / "channel_map.csv", index=False)
    for r in recs:   # local standardized copies (parquet is gitignored)
        df = pd.DataFrame({"time_s": r.t, **r.channels})
        df.to_parquet(EXT / ("%s.parquet" % r.experiment_id), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(ev.T.to_string())
        print(inter.to_string(index=False))


# ------------------------------------------------------------------ alarms
def run_alarms():
    import evalv2 as V
    import native_windows as NW
    import run_e3 as R
    import run_validation as RV
    import survival as SV
    import make_windows as MW

    d, meta = R.load_windows("W60_native", splits=("train", "val", "test", "test_zeroshot"))
    reg = pd.read_csv(ROOT / "tr-corpus" / "registry" / "experiments.csv")
    assert list(d["features"]) == list(NW.OUTPUT_FEATURES)
    plan = V.Plan(d, meta, reg, ["M1"])
    y, w = SV.discrete_hazard_targets(d["y_time"], d["y_event"], R.HORIZON_S)
    names = np.array(d["features"])
    select = RV.representation_mask(names, "no_age")
    age_cols = np.array([s.startswith("age__") or s == MW.AGE_CH for s in names])
    select &= ~age_cols
    cols = np.flatnonzero(select)
    F = RV.summary_features(d, cols)
    stored = pd.read_csv(ROOT / FROZEN_CAL)

    ev = pd.read_csv(OUT / "event_times.csv").set_index("record")
    d7 = {}
    for rec in G.read_all():
        if rec.role != "untreated_tr":
            continue
        T = rec.channels["T_case"].copy()
        lost = rtd_lost_at(rec.t, T)
        if lost is not None:
            T[rec.t >= lost] = np.nan
        series = {"T_surface_mid": (rec.t, T), "T_ambient": (rec.t, rec.channels["T_ambient"])}
        raw = type("Raw", (), dict(series=series, dataset_id="d7_gardner", experiment_id=rec.experiment_id))
        grid, f, m, t0, _, _ = NW.causal_record(raw)
        starts = NW._window_starts(len(grid))
        take = starts[:, None] + np.arange(NW.WINDOW_S)[None, :]
        Fx = R.window_features(f[take], m[take], cols)
        d7[rec.experiment_id] = (grid[starts + NW.WINDOW_S - 1] + t0, Fx)

    rows, gate = [], []
    refs = ["L2@0.5", "L2@1.0", "L2@2.0", "25 mV (computed)", "L3 (computed)",
            "L1 chamber (computed)", "H2 onset (computed)", "TR author-derived (s)"]
    for model, seeds in (("rule", [0]), ("xgboost", [0, 1, 2, 3, 4]), ("lightgbm", [0, 1, 2, 3, 4])):
        for seed in seeds:
            predict = R.fit_model(model, F[plan.frozen_train], y[plan.frozen_train], w[plan.frozen_train], seed)
            r_cal = predict(F[plan.cal_neg])
            tau = SV.calibrate_threshold(r_cal, d["experiment"][plan.cal_neg], np.ones(len(r_cal), bool), RV_BUDGET(RV))
            s = stored[(stored.model == model) & (stored.seed == seed) & (stored.age_mode == "no_age")].tau.iloc[0]
            ok = bool(np.isclose(tau, s, rtol=1e-9, atol=0))
            gate.append(dict(model=model, seed=seed, tau_refit=tau, tau_stored=s, reproduced=ok))
            for rid, (tend, Fx) in d7.items():
                base = dict(model=model, seed=seed, record=rid, tau=tau, tau_reproduced=ok)
                if not ok:
                    rows.append(dict(base, status="not assessable (refit did not reproduce tau)"))
                    continue
                risk = np.asarray(predict(Fx), float)
                a = SV.first_alarm(tend, risk, tau)
                out = dict(base, first_alarm_s=a, peak_risk=float(risk.max()), n_windows=len(risk))
                for ref in refs:
                    e = ev.loc[rid, ref]
                    assessable = not (rid in G.PARALLEL_PAIR and ref in ("25 mV (computed)", "L3 (computed)"))
                    if not assessable or pd.isna(e):
                        out[ref] = "not assessable" if not assessable else "not computable"
                        continue
                    e = float(e)
                    if a is None:
                        out[ref] = "miss (no alarm)"
                    elif a < e:     # t_a < t_e, the manuscript's detection rule
                        out[ref] = "detected, lead %.0f s" % (e - a)
                    else:
                        out[ref] = "miss, alarm %.0f s after" % (a - e)
                rows.append(out)
    pd.DataFrame(gate).to_csv(OUT / "refit_gate.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "alarm_status.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(pd.DataFrame(gate).to_string(index=False))
        print(res.to_string(index=False))


FROZEN_CAL = "tr-corpus/results/validation_20260914_native_final/calibration.csv"


def RV_BUDGET(RV):
    return RV.V.BUDGET if hasattr(RV, "V") else 0.1


def write_manifest():
    man = dict(protocol="reports/11_d7_gardner_audit_protocol.md",
               protocol_commit="cf165aa",
               git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               source=dict(doi=G.DOI, license=G.LICENSE,
                           integrity="tr-corpus/external/d7_gardner/integrity.csv"),
               frozen_inputs=frozen_check(),
               code_sha256={p: sha(ROOT / p) for p in ("trbench/external_d7.py", "trbench/adapters/d7_gardner.py")},
               constants=dict(H2_BASELINE_S=H2_BASELINE_S, H2_K_SIGMA=H2_K_SIGMA, H2_MIN_PPM=H2_MIN_PPM,
                              H2_SUSTAIN_S=H2_SUSTAIN_S, RTD_STUCK_N=RTD_STUCK_N),
               statistics="none pooled; n = 2 untreated TR records")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    assert all(v["unchanged"] for v in man["frozen_inputs"].values()), man["frozen_inputs"]


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "events"
    if what == "events":
        run_events()
    elif what == "alarms":
        run_alarms()
    write_manifest()
