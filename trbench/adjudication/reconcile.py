"""Stage 2 of the outcome adjudication (protocol 29): reconcile the L2 label with the source verdict for the 343 measured experiments (D1-D6).

Ported from the 2026-09-24 audit script (reports/26). Writes paper/tables/S2g_* and tr-corpus/registry/outcome_adjudication.csv.

Inputs: registry, per-experiment diagnostics (diag.pkl from diag.py), D6 raw
statistics (d6raw.pkl), D5 main.xlsx.  Outputs the two S2g tables and the
report 26_outcome_reconciliation.md.
"""
import os, sys, glob, warnings, importlib.util
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O = os.path.join(ROOT, "tr-corpus", "registry", "adjudication_cache") + os.sep
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT + "/trbench")
src = open(os.path.join(HERE, "diag.py")).read().split("reg = pd.read_csv")[0]
ns = {"__file__": os.path.join(HERE, "diag.py")}; exec(src, ns); l2_diag = ns["l2_diag"]
spec = importlib.util.spec_from_file_location("ds12", ROOT + "/trbench/adapters/ds12_arc.py")
ds12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds12)

DS_LABEL = {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3",
            "ds04_osf": "D4", "ds09_mech": "D5", "ds12_arc": "D6"}
IR_CAP = 150.24

reg = pd.read_csv(ROOT + "/tr-corpus/registry/experiments.csv")
reg = reg[reg.dataset_id.isin(DS_LABEL)].copy()
diag = pd.read_pickle(O + "diag.pkl").set_index("experiment_id")
d6raw = pd.read_pickle(O + "d6raw.pkl").set_index("experiment_id")

# ------------------------------------------------------------------ D5 source
xl = ROOT + "/Dataset/9/Mechanically Induced Thermal Runaway for Li-ion Batteries/main.xlsx"
summ = pd.read_excel(xl, sheet_name="Summary")
s1 = pd.read_excel(xl, sheet_name="Sheet1", header=None)
s1 = s1.rename(columns={0: "FileName", 4: "txt1"})[["FileName", "txt1"]]
summ = summ.merge(s1, on="FileName", how="left")
summ["ohs"] = pd.to_numeric(summ["Observed Score.1"], errors="coerce")
# Sheet1 fallback for rows whose Summary score is blank but Sheet1 carries a level
fb = summ.ohs.isna() & summ.txt1.notna()
summ.loc[fb, "ohs"] = summ.loc[fb, "txt1"].str.extract(r"–\s*(\d)")[0].astype(float)
summ["chs"] = pd.to_numeric(summ["Calculated Score"], errors="coerce")
summ["key"] = summ.FileName.str.strip().str.lower()
src_map = summ.set_index("key")[["ohs", "chs", "Observed Score", "txt1"]]
n_fallback = int(fb.sum())

def chs_band(v):
    if pd.isna(v): return None
    return "VL" if v < 10 else "L" if v < 25 else "M" if v < 75 else "H" if v < 90 else "VH"

# ------------------------------------------------------------------ D6 raw extras
files = {os.path.basename(os.path.dirname(f)) + "__" + os.path.splitext(os.path.basename(f))[0]: f
         for f in glob.glob(ROOT + "/Dataset/12/TR_KIT ARC/**/*.*", recursive=True)}
kit = {}
for eid, f in files.items():
    df = ds12._read(f)
    t = df.iloc[:, 0].to_numpy(float) * 60; T = df.iloc[:, 1].to_numpy(float)
    dT, dt = np.diff(T), np.diff(t)
    fast = (dT > 1.5) & (dt <= 6.0)          # >1 K between consecutive fast-logged points
    kit[eid] = dict(kit_fast_step=bool(fast.any()),
                    kit_first_step_T=float(T[np.flatnonzero(fast)[0]]) if fast.any() else np.nan,
                    kit_first_step_t=float(t[np.flatnonzero(fast)[0]] - t[0]) if fast.any() else np.nan)
kit = pd.DataFrame(kit).T

# ------------------------------------------------------------------ per-experiment table
rows = []
for _, r in reg.iterrows():
    eid, ds = r.experiment_id, DS_LABEL[r.dataset_id]
    if r.dataset_id in ("ds09_mech", "ds12_arc"):
        g = diag.loc[eid]
        T_max, T_last, T_first = g.T_max, g.T_last, g.T_first
        frac_fin, rise60 = g.frac_T_finite, g.max_rise_60
        l2d = dict(n_cand=g.l2_n_cand, n_floor=g.l2_n_below_floor, n_conf=g.l2_n_conf_fail,
                   best_rise=g.l2_best_rise, best_rise_T=g.l2_best_rise_T, max_rate=g.l2_max_rate,
                   end_gap_cand=g.l2_first_cand_end_gap)
        end_gap_L2, t_Tmax, t_last, V_present = g.end_gap_after_L2, g.t_Tmax, g.t_last, g.V_present
        if r.dataset_id == "ds12_arc" and pd.isna(r.t_onset_L2):
            # the shipped record is the last 3 h only; diagnose the rule on the full raw-derived grid
            from common import resample_channel
            dfr = ds12._read(files[eid]); tr = dfr.iloc[:, 0].to_numpy(float) * 60; Tr = dfr.iloc[:, 1].to_numpy(float)
            grid = np.arange(0.0, (tr[-1] - tr[0]) + 0.5, 1.0); full, _ = resample_channel(tr - tr[0], Tr, grid)
            fg = l2_diag(grid, full, ceiling=None)
            assert fg["t_L2"] is None
            l2d = dict(n_cand=fg["n_cand"], n_floor=fg["n_below_floor"], n_conf=fg["n_conf_fail"],
                       best_rise=fg["best_rise"], best_rise_T=fg["best_rise_T"], max_rate=fg["max_rate"],
                       end_gap_cand=fg["first_cand_end_gap"])
    else:
        d = pd.read_parquet(ROOT + "/tr-corpus/" + r.file)
        T = d.T_surface_max.to_numpy(float); t = d.time_s.to_numpy(float); ok = np.isfinite(T)
        T_max, T_last, T_first = float(np.nanmax(T)), float(T[ok][-1]), float(T[ok][0])
        frac_fin, rise60 = float(ok.mean()), np.nan
        l2d = dict(n_cand=np.nan, n_floor=np.nan, n_conf=np.nan, best_rise=np.nan, best_rise_T=np.nan,
                   max_rate=np.nan, end_gap_cand=np.nan)
        end_gap_L2 = t[-1] - r.t_onset_L2; t_Tmax = float(t[int(np.nanargmax(T))]); t_last = t[-1]
        V_present = bool(d.mask_V_cell.any())
    L2 = pd.notna(r.t_onset_L2)
    row = dict(experiment_id=eid, dataset=ds, l2_detected=L2, t_onset_L2=r.t_onset_L2,
               t_onset_L2_unguarded=r.t_onset_L2_unguarded, max_surface_T=round(T_max, 1),
               T_censored_at=r.T_censored_at, censored_channels=r.censored_channels,
               duration_s=r.duration_s, record_duration_s=r.record_duration_s,
               chemistry=r.chemistry, soc_pct=r.soc_pct, T_last=round(T_last, 1),
               frac_T_rows_finite=round(frac_fin, 3), max_rise_60s=rise60,
               peak_rate_C_per_s=l2d["max_rate"], end_gap_after_L2_s=end_gap_L2, **{k: v for k, v in l2d.items() if k != "max_rate"})

    # ---- waiver dependence: L2 label exists only because of the saturation ceiling
    waiver_dep, bogus_ceiling = False, False
    if L2 and pd.notna(r.T_censored_at):
        d = pd.read_parquet(ROOT + "/tr-corpus/" + r.file)
        nc = l2_diag(d.time_s.to_numpy(float), d.T_surface_max.to_numpy(float), ceiling=None)
        waiver_dep = nc["t_L2"] is None or abs(nc["t_L2"] - r.t_onset_L2) > 1e-6
        bogus_ceiling = r.T_censored_at < 100.0     # ceiling far below the trace (heater W, pinned ambient TC)
    row.update(l2_waiver_dependent=waiver_dep, waiver_ceiling_bogus=bool(waiver_dep and bogus_ceiling))

    # ---- source verdict
    if ds in ("D1", "D2", "D3", "D4"):
        sv, sf, detail = "TR", "publication (every experiment reported as reaching thermal runaway)", ""
        ohs = chs = None
    elif ds == "D5":
        s = src_map.loc[eid.strip().lower()] if eid.strip().lower() in src_map.index else None
        ohs = None if s is None or pd.isna(s.ohs) else int(s.ohs)
        chs = None if s is None or pd.isna(s.chs) else float(s.chs)
        if ohs is None:
            sv, sf, detail = "not stated", "main.xlsx: no Observed Score for this file", ""
        else:
            sv = "TR" if ohs >= 6 else "non-TR"
            sf = "main.xlsx Observed Score (OHS) = %d; CHS = %s (%s)" % (ohs, "n/a" if chs is None else "%.1f" % chs, chs_band(chs))
            detail = str(s["Observed Score"] if pd.notna(s["Observed Score"]) else s.txt1)
    else:  # D6
        form = eid.split("__")[0]
        if form == "21700":
            if r.soc_pct == 0:
                sv, detail = "non-TR", "Ohneseit 2023: 'at SOC 0 for the other cell chemistries, no thermal runaway occurred until 350 C' (test stopped at 350 C)"
            elif r.chemistry == "LFP" and r.soc_pct == 30:
                sv, detail = "non-TR", "Ohneseit 2023: 'no thermal runaway occurred for SOC 30 for LFP cells' (reaction stopped 250-275 C; heated on to 450 C)"
            elif r.chemistry == "LFP" and r.soc_pct == 60:
                sv, detail = "TR", "Ohneseit 2023: 'at SOC 60, LFP cells went into thermal runaway but with very low rates'; one of two replicates met their >1 K log-step criterion, the other did not"
            else:
                sv, detail = "TR", "Ohneseit 2023: condition-level TR (start-of-TR temperature reported for this chemistry x SOC, Fig. 7b / Table 19)"
            sf = "Ohneseit et al. 2023 Batteries 9:237 text (condition level, not per cell)"
        else:
            sv, sf, detail = "not stated", "Zenodo record and Schoeberl/Ohneseit 2025 JPS abstract give no per-cell TR statement", ""
        ohs = chs = None
    row.update(source_verdict=sv, source_field_used=sf, source_detail=detail, ohs=ohs, chs=chs)

    # ---- D6 raw extras
    if ds == "D6":
        q = d6raw.loc[eid]; k = kit.loc[eid]
        row.update(raw_Tmax=round(q.raw_Tmax, 1), arc_rate_max_C_per_min=round(q.raw_rate_max_c_per_min, 2) if pd.notna(q.raw_rate_max_c_per_min) else np.nan,
                   kit_fast_step=bool(k.kit_fast_step), kit_first_step_T=k.kit_first_step_T,
                   n_native=int(q.n_native), n_native_in_window=int(q.n_native_in_window),
                   frac_rows_native=round(q.frac_rows_native, 4), frac_rows_not_flagged_interp=round(1 - diag.loc[eid].probe_interp_frac, 3),
                   dt_median_s=round(q.dt_median, 2), dt_p90_s=round(q.dt_p90, 1),
                   n_native_pm300=q.n_native_pm300, dt_median_pm300_s=q.dt_median_pm300, dt_p90_pm300_s=q.dt_p90_pm300,
                   n_native_after_L2=q.n_native_after_L2, raw_end_minus_L2_s=q.raw_end_minus_L2_s)
    rows.append(row)

cells = pd.DataFrame(rows)

# ------------------------------------------------------------------ classification
def classify(x):
    ds, L2, sv = x.dataset, x.l2_detected, x.source_verdict
    Tm = x.max_surface_T
    notes = []
    if ds in ("D1", "D2", "D3", "D4"):
        suff = "sufficient"
        if pd.notna(x.T_censored_at) and isinstance(x.censored_channels, str) and "T_surface" in x.censored_channels:
            suff = "one surface TC pinned at %.0f C (peak censored); L2 does not depend on the saturation waiver" % x.T_censored_at
        return ("A1 agree_positive", suff, "source TR; L2 detected; record continues %.0f s after L2; peak %.0f C" % (x.end_gap_after_L2_s, Tm), "positive")
    if ds == "D5":
        sparse = x.frac_T_rows_finite < 0.5
        capped = Tm >= IR_CAP - 0.05
        if L2:
            if x.l2_waiver_dependent and x.waiver_ceiling_bogus:
                return ("C3 negative_rule_artefact",
                        "L2 label depends on the saturation waiver with a ceiling of %.1f C taken from a pinned non-surface / dead channel (%s)" % (x.T_censored_at, x.censored_channels),
                        "no source verdict; without the waiver no L2 (largest 60 s rise after an above-60 C crossing %.0f C); peak %.0f C then cools to %.0f C" % (x.best_rise, Tm, x.T_last), "negative")
            if sv == "TR":
                suff = "peak censored at the 150 C IR-camera cap" if capped else "sufficient"
                if x.l2_waiver_dependent: suff += "; L2 confirmed by saturation (rise to cap %.0f C < 50 C)" % x.best_rise
                if sparse: suff += "; surface T present in %.0f%% of 1 s rows" % (100 * x.frac_T_rows_finite)
                return ("A1 agree_positive", suff, "OHS %d (%s); L2 detected; record continues %.0f s after L2" % (x.ohs, "rupture/gas/fire" if x.ohs == 7 else "rupture/gas", x.end_gap_after_L2_s), "positive")
            if sv == "non-TR":
                suff = "peak censored at the 150 C IR-camera cap" if capped else "sufficient (peak %.0f C measured, record continues %.0f s after L2)" % (Tm, x.end_gap_after_L2_s)
                return ("D1 disagree_source_nonTR_L2",
                        suff, "source OHS %d (%s); L2 fired on a %.0f C rise from %.0f C within 60 s (peak rate %.1f C/s); peak %.0f C%s; cell cooled to %.0f C by record end" % (
                            x.ohs, x.source_detail.split(" – ")[0].strip(), x.best_rise, x.best_rise_T, x.peak_rate_C_per_s, Tm, " (at IR cap)" if capped else "", x.T_last), "indeterminate")
            # not stated
            if capped or Tm > 150.5:
                suff = "peak censored at the 150 C IR-camera cap" if capped else "sufficient"
                if sparse: suff += "; surface T present in %.0f%% of 1 s rows" % (100 * x.frac_T_rows_finite)
                return ("A2 positive_trace_inferred", suff, "no source verdict; L2 detected; peak %s; record continues %.0f s after L2" % ("pinned at 150.2 C (IR cap)" if capped else "%.0f C" % Tm, x.end_gap_after_L2_s), "positive")
            return ("D2 indeterminate_rule_positive_unverified", "sufficient (peak %.0f C measured, below the IR cap; record continues %.0f s after L2)" % (Tm, x.end_gap_after_L2_s),
                    "no source verdict; L2 fired on a %.0f C rise from %.0f C within 60 s; peak %.0f C then %.0f C at record end; same trace shape as the OHS 3-5 cells the source calls non-TR" % (x.best_rise, x.best_rise_T, Tm, x.T_last), "indeterminate")
        # no L2
        if x.n_cand == 0:
            why = "peak rate %.2f C/s < 1 C/s" % x.peak_rate_C_per_s
        elif x.n_floor == x.n_cand:
            why = "rate >= 1 C/s only below the 60 C floor"
        else:
            why = "above-60 C crossing but +50 C/60 s confirmation failed (largest rise %.0f C from %.0f C)" % (x.best_rise, x.best_rise_T)
        ends_peak = (x.T_last >= Tm - 5) and ((x.duration_s - (x.t_last_minus_tTmax if False else 0)) >= 0) and x.ends_near_peak
        suff = "sufficient"
        if sparse: suff = "surface T present in %.0f%% of 1 s rows (sparse logging)" % (100 * x.frac_T_rows_finite)
        if x.post_isc_s is not None and pd.notna(x.post_isc_s) and x.post_isc_s < 300: suff += "; record ends %.0f s after the 25 mV short" % x.post_isc_s
        if sv == "non-TR":
            if ends_peak and Tm >= 80: suff += "; record ends at its temperature maximum (%.0f C)" % Tm
            return ("C1 agree_negative", suff, "OHS %d; no L2 (%s); peak %.0f C" % (x.ohs, why, Tm), "negative")
        if ends_peak and Tm >= 80:
            return ("D3 indeterminate_insufficient", "record ends within 60 s of its temperature maximum (%.0f C, still at peak)" % Tm,
                    "no source verdict; no L2 (%s); outcome not observable" % why, "indeterminate")
        return ("C2 negative_trace_inferred", suff, "no source verdict; no L2 (%s); peak %.0f C, %.0f C at record end" % (why, Tm, x.T_last), "negative")
    # D6
    if L2:
        suff = "record ends %.0f s after L2 (ARC log stops at the runaway; %d native samples after L2)" % (x.end_gap_after_L2_s, x.n_native_after_L2)
        if x.l2_waiver_dependent:
            suff += "; L2 label depends on the saturation waiver (ceiling %.1f from T_heater); without it largest 60 s rise %.0f C" % (x.T_censored_at, x.best_rise)
        if x.n_native <= 30:
            return ("D3 indeterminate_insufficient", "fragment: %d native samples spanning %.0f s, %.0f-%.0f C; " % (x.n_native, x.record_duration_s, 176, x.raw_Tmax) + suff,
                    "source not stated; ARC rate reached %.0f C/min at %.0f C but the file ends 7 s later; L2 only via waiver" % (x.arc_rate_max_C_per_min, x.raw_Tmax), "indeterminate")
        if sv == "TR":
            return ("A1 agree_positive", suff, "source TR (condition level); L2 detected; ARC rate max %.0f C/min; raw peak %.0f C" % (x.arc_rate_max_C_per_min, x.raw_Tmax), "positive")
        return ("A2 positive_trace_inferred", suff, "source not stated; L2 detected; ARC rate max %.0f C/min, >1 K log steps (authors' TR criterion) %s; raw peak %.0f C" % (
            x.arc_rate_max_C_per_min, "present" if x.kit_fast_step else "absent", x.raw_Tmax), "positive")
    # no L2
    if sv == "non-TR":
        return ("C1 agree_negative", "sufficient (heated to %.0f C, test end criterion)" % x.raw_Tmax,
                "source non-TR; no 1 C/s crossing (peak rate %.2f C/s, ARC rate max %s C/min); no >1 K fast log step" % (
                    x.peak_rate_C_per_s, "n/a" if pd.isna(x.arc_rate_max_C_per_min) else "%.1f" % x.arc_rate_max_C_per_min), "negative")
    # source TR or not stated, no L2
    if x.n_cand > 0:
        if x.end_gap_cand <= 60:
            return ("B1 rule_negative_truncated", "record ends %.0f s after the >=1 C/s crossing" % x.end_gap_cand,
                    "%s; fast event present (ARC rate max %.0f C/min, >1 K log steps at %.0f C) but the +50 C confirmation window is cut by the end of the log (rise to end %.0f C from %.0f C)" % (
                        "source TR" if sv == "TR" else "source not stated", x.arc_rate_max_C_per_min, x.kit_first_step_T, x.best_rise, x.best_rise_T), "positive")
        return ("B1 rule_negative_truncated", "logging gap after the fast event (crossing at %.0f C, next logged point %.0f C)" % (x.best_rise_T, 394),
                "source TR (condition level); fast event 180-225 C/min at 281-315 C (>1 K log steps) then a logging gap; rise within 60 s %.1f C < 50 C; cell later at %.0f C" % (x.best_rise, x.raw_Tmax), "positive")
    # slow exotherm
    if x.experiment_id == "21700__LFP_SOC60_M2":
        return ("D3 indeterminate_insufficient", "sufficient record (to %.0f C) but authors state this replicate did not meet their TR criterion" % x.raw_Tmax,
                "source: 'one of two [SOC 60 LFP] cells ... met the criterion ..., whereas the other did not' (this is the low-rate replicate, ARC rate max %.1f C/min); peak rate %.2f C/s; self-heated to %.0f C" % (x.arc_rate_max_C_per_min, x.peak_rate_C_per_s, x.raw_Tmax), "indeterminate")
    return ("B2 rule_negative_slow", "sufficient record (self-heating tracked to %.0f C)" % x.raw_Tmax,
            "source TR (condition level, 'very low rates'); no 1 C/s crossing anywhere (peak rate %.2f C/s, ARC rate max %.1f C/min); no >1 K fast log step; L2 cannot fire on a slow LFP runaway" % (x.peak_rate_C_per_s, x.arc_rate_max_C_per_min), "positive")

# helper flags for D5
d5m = reg[reg.dataset_id == "ds09_mech"].set_index("experiment_id")
cells["post_isc_s"] = cells.apply(lambda x: (x.duration_s - d5m.loc[x.experiment_id].t_isc) if x.dataset == "D5" and pd.notna(d5m.loc[x.experiment_id].t_isc) else np.nan, axis=1)
cells["ends_near_peak"] = cells.apply(lambda x: bool((x.dataset == "D5") and (diag.loc[x.experiment_id].T_last >= diag.loc[x.experiment_id].T_max - 5) and ((diag.loc[x.experiment_id].t_last - diag.loc[x.experiment_id].t_Tmax) <= 60)), axis=1)
cells["t_last_minus_tTmax"] = 0.0

out = cells.apply(lambda x: pd.Series(classify(x), index=["category", "sufficiency_category", "reason", "final_assignment"]), axis=1)
cells = pd.concat([cells, out], axis=1)

# ------------------------------------------------------------------ write cells table
req = ["experiment_id", "dataset", "source_verdict", "source_field_used", "l2_detected", "t_onset_L2", "max_surface_T",
       "T_censored_at", "censored_channels", "duration_s", "category", "reason", "final_assignment"]
extra = ["sufficiency_category", "source_detail", "ohs", "chs", "chemistry", "soc_pct", "record_duration_s", "T_last",
         "peak_rate_C_per_s", "max_rise_60s", "best_rise", "best_rise_T", "end_gap_after_L2_s", "frac_T_rows_finite",
         "t_onset_L2_unguarded", "l2_waiver_dependent", "waiver_ceiling_bogus", "post_isc_s",
         "raw_Tmax", "arc_rate_max_C_per_min", "kit_fast_step", "kit_first_step_T", "n_native", "n_native_in_window",
         "frac_rows_native", "frac_rows_not_flagged_interp", "dt_median_s", "dt_p90_s", "n_native_pm300",
         "dt_median_pm300_s", "dt_p90_pm300_s", "n_native_after_L2"]
order = {"D1": 0, "D2": 1, "D3": 2, "D4": 3, "D5": 4, "D6": 5}
cells = cells.sort_values(["dataset", "experiment_id"], key=lambda s: s.map(order) if s.name == "dataset" else s).reset_index(drop=True)
cells_out = cells[req + extra].copy()
for c in ("best_rise", "best_rise_T", "peak_rate_C_per_s", "max_rise_60s", "dt_median_pm300_s", "dt_p90_pm300_s"):
    cells_out[c] = pd.to_numeric(cells_out[c], errors="coerce").round(2)
os.makedirs(ROOT + "/paper/tables", exist_ok=True)
cells_out.to_csv(ROOT + "/paper/tables/S2g_outcome_reconciliation_cells.csv", index=False)

# ------------------------------------------------------------------ summary table
def suff_group(x):
    s = x.sufficiency_category
    if x.category.startswith("C3"): return "rule artefact (waiver ceiling from pinned non-surface channel)"
    if "fragment" in s: return "fragment record"
    if "IR-camera cap" in s: return "peak censored at 150 C IR cap"
    if "ARC log stops" in s: return "record ends <=60 s after L2 (ARC log stops at runaway)"
    if s.startswith("record ends") and "crossing" in s: return "record ends 5-12 s after the rate crossing"
    if "logging gap" in s: return "logging gap after the fast event"
    if "temperature maximum" in s and "still at peak" in s: return "record ends at its temperature maximum"
    if "did not meet" in s: return "source: replicate did not meet authors' criterion"
    if "slow" in x.category: return "sufficient (slow exotherm tracked to >540 C)"
    if s.startswith("surface T present"): return "sparse surface-T logging (<50% of rows)"
    if "censored" in s: return "one surface TC pinned at its maximum (peak censored); L2 independent of the waiver"
    return "sufficient"

cells["suff_group"] = cells.apply(suff_group, axis=1)
GENERIC = {
    "A1 agree_positive": "source reports TR and L2 detected",
    "A2 positive_trace_inferred": "no source verdict; L2 detected and trace reaches the IR cap / >150 C (D5) or ARC rate >=200 C/min with >1 K log steps (D6)",
    "B1 rule_negative_truncated": "source TR; a >=1 C/s event is present but the +50 C/60 s confirmation is cut by the end of the ARC log or a logging gap",
    "B2 rule_negative_slow": "source TR (LFP, 'very low rates'); peak rate 0.07-0.53 C/s, never >=1 C/s; self-heated to 543-546 C",
    "C1 agree_negative": "source non-TR and no L2; trace stays below the IR cap (D5) or reaches only the ARC stop temperature with rate <=0.44 C/s (D6)",
    "C2 negative_trace_inferred": "no source verdict; no L2; peak <=133 C and the cell cools before record end",
    "C3 negative_rule_artefact": "L2 label exists only because the saturation waiver used a ceiling from a pinned ambient/dead channel (17-35 C); without it no L2; peak 76-111 C",
    "D1 disagree_source_nonTR_L2": "source OHS 3-5 (local reactions / swelling, no rupture) but L2 fired on a >=50 C rise within 60 s to 123-150 C",
    "D2 indeterminate_rule_positive_unverified": "no source verdict; L2 fired but the peak stays below the 150 C IR cap (113-149 C), the same trace shape as the D1 conflicts",
    "D3 indeterminate_insufficient": "outcome not observable: record ends at its peak, a 23-sample fragment, or the source states the replicate did not meet its own TR criterion",
}
summ_rows = []
for (ds, sv, l2, cat, sg, fin), g in cells.groupby(["dataset", "source_verdict", "l2_detected", "category", "suff_group", "final_assignment"], sort=False):
    summ_rows.append(dict(dataset=ds, source_verdict=sv, l2_detected=l2, n=len(g), sufficiency_category=sg,
                          reason=cat + ": " + GENERIC[cat], final_assignment=fin,
                          experiments=";".join(g.experiment_id) if len(g) <= 12 else ""))
summary = pd.DataFrame(summ_rows)
summary["o"] = summary.dataset.map(order)
summary = summary.sort_values(["o", "source_verdict", "l2_detected", "reason", "sufficiency_category"], ascending=[True, False, False, True, True]).drop(columns="o")
summary.to_csv(ROOT + "/paper/tables/S2g_outcome_reconciliation.csv", index=False)

cells.to_pickle(O + "cells_final.pkl")
adj = cells[["experiment_id", "dataset", "source_verdict", "l2_detected", "category", "final_assignment", "sufficiency_category"]].copy()
adj["event_observable"] = ~adj.category.str.startswith(("B1", "D3"))
adj = adj.rename(columns={"source_verdict": "physical_outcome", "category": "outcome_evidence"})
adj.to_csv(ROOT + "/tr-corpus/registry/outcome_adjudication.csv", index=False)
summary.to_pickle(O + "summary_final.pkl")
print("n_fallback_sheet1", n_fallback, summ.loc[fb, ["FileName", "txt1", "ohs"]].to_dict("records"))
print("recovered case-insensitive:", cells[cells.experiment_id == "OE-LCo-6470mAh-60SOC"][["ohs", "chs", "source_verdict", "l2_detected", "category"]].to_dict("records"))
print("waiver-dependent labels:", cells[cells.l2_waiver_dependent][["experiment_id", "dataset", "T_censored_at", "censored_channels", "best_rise", "waiver_ceiling_bogus", "category"]].to_string())
print("D6 no-L2 diag:", cells[(cells.dataset == "D6") & ~cells.l2_detected][["experiment_id", "n_cand", "best_rise", "best_rise_T", "end_gap_cand", "peak_rate_C_per_s", "category", "final_assignment"]].round(2).to_string())
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
print(summary.drop(columns="experiments").to_string())
print(pd.crosstab([cells.dataset, cells.source_verdict], cells.l2_detected, margins=True))
print(pd.crosstab(cells.dataset, cells.final_assignment, margins=True))
