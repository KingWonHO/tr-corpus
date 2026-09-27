import os, pandas as pd, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O = os.path.join(ROOT, "tr-corpus", "registry", "adjudication_cache") + os.sep
cells = pd.read_pickle(O + "cells_final.pkl")
summary = pd.read_pickle(O + "summary_final.pkl")

def md(df, floatfmt="%.2f"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else (("%d" % v) if float(v).is_integer() and abs(v) < 1e6 and c not in ("frac_rows_native",) else floatfmt % v))
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

d5 = cells[cells.dataset == "D5"]; d6 = cells[cells.dataset == "D6"]
L = []
A = L.append
A("# 26 Outcome reconciliation: source verdict vs. computed L2 (D1-D6)\n")
A("Date: 2026-09-27 (label v1.1; first run 2026-09-24 on v1.0). Inputs: `tr-corpus/registry/experiments.csv`, the harmonized 1 s parquet records, `Dataset/9/.../main.xlsx`, the raw KIT ARC files under `Dataset/12`, Lin et al. 2023 (J. Energy Storage 61:106798, local PDF), Ohneseit et al. 2023 (Batteries 9:237, KIT repository PDF), the three Zenodo record pages and the Schoeberl/Ohneseit 2025 J. Power Sources abstract. Outputs: `paper/tables/S2g_outcome_reconciliation.csv` (dataset x category counts) and `paper/tables/S2g_outcome_reconciliation_cells.csv` (one row per experiment, 343 rows). Scripts: `trbench/adjudication/{diag,reconcile,report}.py`.\n")
A("## 1. What is source-stated and what is inferred\n")
A("| Dataset | Source verdict field | Status |\n|---|---|---|")
A("| D1-D4 (25) | publications: every experiment reported as reaching thermal runaway | source-stated (as given in the manuscript's dataset descriptions; not re-audited here) |")
A("| D5 (252) | `main.xlsx` sheet `Summary`, column `Observed Score` (EUCAR-style observed hazard severity, OHS 1-7) and `Calculated Score` (CHS 5-100) | OHS present for 118 files (117 exact filename matches + `OE-LCo-6470mAh-60SOC`, recovered by case-insensitive match, OHS 6). 134 experiments have no score (all `SNL_*`, `ChevyVolt*`, `Nissan*Leaf*`, `LCO_4Ah/4000mAh*_MAX`, `LFP_15Ah*_MAX`, `NMC_10000mAh*_MAX`, and 6 `OE-*` files). The paper never states a binary TR threshold on the OHS scale; the mapping used here (OHS 6-7 = TR: 'rupture of pouch, gas release' / '... smoke, gas release, fire'; OHS 1-5 = non-TR: 'no effect ...', 'moderate effect, extended joule heating, local reactions (no / limited spread)', 'physical effect (pouch swelling), extended joule heating, local reactions') is ours, supported by the paper's text ('the burned-out LCO and NMC cells ... went through severe thermal runaway. At lower SOCs, the cells only showed the indentation marks'). CHS = 100 whenever max T > 160 C, so CHS cannot separate the censored 150 C cases. |")
A("| D6 21700 (55) | Ohneseit et al. 2023 text; no per-cell table in the paper, the Zenodo README or the data files | source-stated at condition level: 'at SOC 0 and SOC 30 for LFP cells and at SOC 0 for the other cell chemistries, no thermal runaway occurred until 350 C'; 'at SOC 60, LFP cells went into thermal runaway but with very low rates ... one of two LFP cells ... met the criterion used to define thermal runaway ..., whereas the other cell tested did not'; start-of-TR temperatures are reported for every other chemistry x SOC (Fig. 7b, Table 19). Which physical file is 'the other cell' at LFP SOC 60 is inferred from the ARC rate column (M1 reaches 20.5 C/min, M2 only 4.1 C/min; the paper says one of the two exceeded 10 C/min). The authors' own TR criterion is 'more than 1 K between two logged steps' of a log written every 1 K or 0.05 min. |")
A("| D6 18650 + 4680 (11) | Zenodo records 14956641 / 14956635 and the 2025 J. Power Sources abstract | not stated per cell. Inferred from the raw files: ARC rate column maximum, presence of >1 K steps between consecutive fast-logged points (dt <= 6 s; the authors' criterion applied to their exotherm log), and the peak temperature. |")
A("\nTemperature-trace quantities (peak, peak rate, largest 60 s rise, which L2 guard failed) are computed by replaying `trbench.common.onset_L2` on `T_surface_max` of each shipped record; the replay reproduces every registry `t_onset_L2` (343/343). For the 21 D6 records without L2 the replay is repeated on the full raw-derived 1 s grid, because the shipped record keeps only the last 3 h and for `21700__LFP_SOC80_M3` the fast event lies before that window.\n")

A("## 2. Cross-tabulation: source verdict x L2 detected\n")
ct = pd.crosstab([cells.dataset, cells.source_verdict], cells.l2_detected).reset_index()
ct.columns = ["dataset", "source_verdict", "L2 not detected", "L2 detected"]
A(md(ct))
A("\nD5 by observed hazard level (OHS) x L2:\n")
o = d5.copy(); o["OHS"] = o.ohs.fillna(-1).astype(int).astype(str).replace("-1", "none")
ct2 = pd.crosstab(o.OHS, o.l2_detected).reset_index(); ct2.columns = ["OHS", "L2 not detected", "L2 detected"]
ct2["OHS"] = pd.Categorical(ct2.OHS, ["1", "2", "3", "4", "5", "6", "7", "none"]); ct2 = ct2.sort_values("OHS")
A(md(ct2))
A("\nD6 by chemistry x SOC x L2 (21700 unless stated):\n")
d6c = d6.copy(); d6c["form"] = d6c.experiment_id.str.split("__").str[0]
ct3 = pd.crosstab([d6c.form, d6c.chemistry, d6c.soc_pct.astype(int)], d6c.l2_detected).reset_index(); ct3.columns = ["format", "chemistry", "SOC %", "L2 not detected", "L2 detected"]
A(md(ct3))

A("\n## 3. Categories and final assignment\n")
A("Category codes (column `category` in the cells table):\n")
A("* A1 agree_positive: source TR and L2 detected.\n* A2 positive_trace_inferred: no source verdict; L2 detected and the trace reaches the 150 C IR cap or >150 C (D5), or the ARC rate reaches >=200 C/min with >1 K log steps (D6).\n* B1 rule_negative_truncated: source TR; a >=1 C/s crossing above 60 C exists but the +50 C/60 s confirmation is cut by the end of the ARC log (5-12 s after the crossing) or by a logging gap.\n* B2 rule_negative_slow: source TR; the peak rate never reaches 1 C/s (slow LFP runaway tracked to 543-546 C).\n* C1 agree_negative: source non-TR and no L2.\n* C2 negative_trace_inferred: no source verdict, no L2, peak <=133 C and the cell cools before the record ends.\n* C3 negative_rule_artefact: the L2 label exists only because the saturation waiver in `onset_L2` used a ceiling taken from a pinned ambient / dead thermocouple (17-35 C); without the waiver there is no L2; peak 76-111 C.\n* D1 disagree_source_nonTR_L2: source OHS 3-5 but L2 fired.\n* D2 indeterminate_rule_positive_unverified: no source verdict; L2 fired but the peak stays below the IR cap (113-149 C), the same trace shape as D1.\n* D3 indeterminate_insufficient: record ends at its temperature maximum, a 23-sample fragment, or the source states that this replicate did not meet its own TR criterion.\n")
s = summary.drop(columns=["experiments"]).copy()
A(md(s))
A("\nFinal assignment per dataset:\n")
fa = pd.crosstab(cells.dataset, cells.final_assignment, margins=True).reset_index()
A(md(fa))
A("\nPositives by rule (L2) versus positives after reconciliation: D5 83 -> %d (21 OHS 6-7 + 40 unscored cells at the IR cap; minus 7 source conflicts, 9 unverified sub-cap positives and 6 waiver artefacts), D6 45 -> %d (44 L2 positives with sufficient evidence + 7 rule-negatives reassigned from the source and the ARC rate column; 1 fragment moved to indeterminate). D1-D4 unchanged (25/25)." % (
    (d5.final_assignment == "positive").sum(), (d6.final_assignment == "positive").sum()))

A("\n## 4. Disagreements and rule-negatives, experiment by experiment\n")
A("### 4.1 D5: source non-TR (OHS 3-5) but L2 detected (7 cells, all Soteria-programme control NMC811 pouches at 0-50 % SOC)\n")
x = d5[d5.category.str.startswith("D1")][["experiment_id", "ohs", "chs", "soc_pct", "max_surface_T", "T_last", "best_rise", "best_rise_T", "peak_rate_C_per_s", "t_onset_L2", "end_gap_after_L2_s", "T_censored_at"]].copy()
x.columns = ["experiment", "OHS", "CHS", "SOC %", "peak T (C)", "T at end (C)", "rise within 60 s (C)", "from T (C)", "peak rate (C/s)", "t L2 (s)", "record after L2 (s)", "censored at"]
A(md(x))
A("\nAll seven satisfy the rule as written (>=1 C/s for 2 s above 60 C, then >=50 C within 60 s): the cell heats from about 60-80 C to 123-150 C within a minute after the short and then cools back to 30-75 C by the end of the record. The authors score them 3-5 ('extended joule heating, local reactions (no / limited spread)' or 'pouch swelling'), i.e. no venting, rupture or fire, and the paper states that six Control NMC-811 cells reached EUCAR 5-6 and the other eight 2-3. Two of the seven peak exactly at the 150.24 C IR-camera cap, so their true peak is unknown. Final assignment: indeterminate. Note the mirror image does not occur: no OHS 6-7 cell lacks L2 (the three OHS 7 cells that once lacked it are covered by the saturation waiver).\n")
A("### 4.2 D5: no source verdict, L2 detected, peak below the IR cap (9 cells)\n")
x = d5[d5.category.str.startswith("D2")][["experiment_id", "max_surface_T", "T_last", "best_rise", "best_rise_T", "peak_rate_C_per_s", "t_onset_L2", "end_gap_after_L2_s"]].copy()
x.columns = ["experiment", "peak T (C)", "T at end (C)", "rise within 60 s (C)", "from T (C)", "peak rate (C/s)", "t L2 (s)", "record after L2 (s)"]
A(md(x))
A("\nSame trace shape as 4.1 (60-90 C to 113-149 C within a minute, then cooling). Because the source is silent, these cannot be assigned either way: indeterminate.\n")
A("### 4.3 D5: L2 labels that depend on the saturation waiver (9 cells)\n")
x = d5[d5.l2_waiver_dependent][["experiment_id", "ohs", "max_surface_T", "T_censored_at", "censored_channels", "best_rise", "category", "final_assignment"]].copy()
x.columns = ["experiment", "OHS", "peak T (C)", "waiver ceiling (C)", "censored channels", "rise within 60 s (C)", "category", "final"]
A(md(x))
A("\n`onset_L2` waives the +50 C confirmation when the trace reaches a censoring ceiling inside the 60 s window; `standardize()` lets the `T_surface_max` aggregate inherit `max()` of every censored `T_*` channel. For the three OHS 7 cells the ceiling is the 150.24 C IR cap and the waiver does what it was written for. For the six `SNL_*` cells the ceiling is 16.7-34.9 C, i.e. a pinned ambient or dead thermocouple, so the waiver disables the confirmation entirely; without it none of the six has an L2 (largest 60 s rise 16-34 C, peaks 76-111 C, cooling to ambient by record end). These six labels are artefacts and are assigned negative. The same inheritance affects D6 through `T_heater` (see 4.5).\n")
A("### 4.4 D5: records without L2\n")
n = d5[~d5.l2_detected]
A("* %d records; peak surface temperature never reaches 150 C (maximum %.0f C, `%s`, OHS 5); %d of them show a >=50 C rise within 60 s (fast local Joule heating) that the guards reject.\n" % (len(n), n.max_surface_T.max(), n.loc[n.max_surface_T.idxmax(), "experiment_id"], (n.max_rise_60s >= 50).sum()))
why = n.reason.str.extract(r"no L2 \((.*?)(?:; largest|\))")[0].str.replace(r"\(largest.*", "", regex=True).str.strip()
why = why.str.replace(r"peak rate [0-9.]+ C/s < 1 C/s", "peak rate < 1 C/s", regex=True).str.replace(r"\(largest rise .*", "", regex=True).str.replace(r" \(largest.*", "", regex=True)
why = why.str.replace(r"above-60 C crossing but \+50 C/60 s confirmation failed.*", "above-60 C crossing, +50 C/60 s confirmation failed", regex=True)
A("Why the rule did not fire (by source verdict):\n")
w = pd.crosstab(why, n.source_verdict).reset_index(); w.columns = ["failing condition"] + list(w.columns[1:])
A(md(w))
A("\n* Sufficiency notes: %d records log surface temperature in fewer than half of the 1 s rows (`ChevyVolt*`, `Nissan*Leaf*`, `OE-*`; native 5-10 s logging), %d end fewer than 300 s after the 25 mV short (peaks 25-117 C, all cooling or flat), and one unscored record (`LFP_15Ah_40SOC_cell1_MAX`) ends within a minute of its 115 C maximum while still at that temperature; it is the only D5 negative assigned indeterminate. `Soteria-Control-50SOC` (OHS 4) also ends at its 105 C maximum but carries a source verdict and stays negative.\n" % (
    (n.frac_T_rows_finite < 0.5).sum(), (n.post_isc_s < 300).sum()))

A("### 4.5 D6: the eight records without L2 outside the source-stated negative conditions\n")
x = d6[~d6.l2_detected & (d6.source_verdict != "non-TR")][["experiment_id", "source_verdict", "raw_Tmax", "arc_rate_max_C_per_min", "kit_fast_step", "kit_first_step_T", "peak_rate_C_per_s", "best_rise", "best_rise_T", "end_gap_cand", "category", "final_assignment"]].copy()
x.columns = ["experiment", "source", "raw peak T (C)", "ARC rate max (C/min)", ">1 K fast log step", "at T (C)", "peak rate on 1 s grid (C/s)", "rise within 60 s of crossing (C)", "crossing T (C)", "log end after crossing (s)", "category", "final"]
A(md(x))
A("\n* `NCA_HEII_SOC60_M1`, `NCA_HP_SOC80_M1`, `NMC_SOC60_M1`, `LIB_LFP_4680_cell1`: the runaway is in the file (ARC rate 1,172-16,642 C/min, >1 K steps between consecutive logged points, the authors' criterion) but the exotherm log ends 5-12 s after the >=1 C/s crossing, so the logged rise inside the 60 s confirmation window is 24-48 C (`NCA_HP_SOC80_M1` misses by 2.3 C). Assigned positive (B1).\n* `LFP_SOC80_M3`: a 180-225 C/min event at 281-315 C (>1 K steps) followed by a logging gap whose next point is 394 C; the rise inside 60 s of the crossing is 44.4 C. Assigned positive (B1).\n* `LFP_SOC60_M1`, `LFP_SOC80_M1`: self-heating tracked continuously to 546 / 543 C with a peak rate of 0.34 / 0.52 C/s (ARC rate 20.5 / 31.6 C/min) and no >1 K fast step; the rule cannot fire on a runaway this slow. The source calls LFP SOC 60 TR 'with very low rates' and reports TR-start temperatures for LFP SOC 80 while noting the replicate variability at that SOC. Assigned positive (B2).\n* `LFP_SOC60_M2`: 4.1 C/min at most, reached 546 C; the source states that one of the two SOC 60 replicates did not meet its criterion and adds that 'with a different definition of thermal runaway, a different conclusion is possible'. Assigned indeterminate.\n")
A("### 4.6 D6: source-stated negatives (13)\n")
x = d6[d6.source_verdict == "non-TR"][["experiment_id", "raw_Tmax", "arc_rate_max_C_per_min", "peak_rate_C_per_s", "record_duration_s"]].copy()
x.columns = ["experiment", "raw peak T (C)", "ARC rate max (C/min)", "peak rate on 1 s grid (C/s)", "raw record length (s)"]
A(md(x))
A("\nAll SOC 0 cells stop at 349-351 C (the authors' 350 C end criterion); the three LFP SOC 30 cells were heated on to 449-450 C after their exotherm stopped at 250-275 C. None has a >=1 C/s crossing anywhere on the full grid (peak 0.01-0.44 C/s). Six of them log an ARC rate above 10 C/min at some point (`LFP_SOC30_M1` 12.4, `NCA_HEI_SOC0_M1/M2` 13.4/13.7, `NCA_HEII_SOC0_M2` 21.9, `NMC_SOC0_M1/M2` 26.4/19.7 C/min) without any >1 K fast step, consistent with the paper's remarks that one LFP SOC 30 cell 'exceeded 10 C/min shortly' and that at SOC 0 'the rates reduced at several points of the experiment'. No disagreement: no source-stated negative carries an L2.\n")
A("### 4.7 D6: positives and their sufficiency\n")
p = d6[d6.l2_detected]
A("* %d L2 positives; in every one the raw exotherm log ends %.0f-%.0f s after L2 (median %.0f s) with %d-%d native samples after L2. The +50 C confirmation therefore rests on the terminal jump of the log, which is 54-461 C.\n* Two positives depend on the saturation waiver with a ceiling inherited from `T_heater` (15.4 W and 0.9 W): `18650__LIB_LFP_18650_cell1` (rise within 60 s 40.6 C; ARC rate 369 C/min, >1 K steps, peak 305 C: kept positive on the trace) and `4680__LIB_LFP_4680_cell3` (a 23-sample, 22 s fragment from 176 to 198 C ending 7 s after the label; rise 6.9 C: indeterminate).\n* The 18650 sodium-ion cells (2) and the remaining 18650 / 4680 cells have no per-cell source statement; they are positive on the trace (ARC rate 369-26,135 C/min, >1 K steps present in all).\n" % (
    len(p), p.end_gap_after_L2_s.min(), p.end_gap_after_L2_s.max(), p.end_gap_after_L2_s.median(), int(p.n_native_after_L2.min()), int(p.n_native_after_L2.max())))

A("## 5. D6 native sampling\n")
A("Native samples are the rows of the raw `.txt` / `.EXO` files (time in minutes, one point per 1 K step or per 0.05 min while the ARC is in exotherm mode; the file starts only once self-heating exceeds 0.02 C/min). Two measures of how much of the shipped 1 s record is measured: (i) `frac_rows_native` = native samples inside the shipped window / 1 s rows; (ii) `1 - T_surface_mid_interp` from the parquet flag. The flag counts a grid row as measured whenever it lies within a quarter of the record's median native interval of a real sample, so it is generous by construction (0.05-0.99) and should not be read as an observation fraction; measure (i) is 0.2-1.9 %.\n")
A("Overall (66 records): median native interval %.1f s (range %.1f-%.1f s across records), 90th percentile %.0f s (range %.0f-%.0f s); shortest interval 0.12-0.18 s in runaway records and 1.4-188 s in the negatives. Within +/-300 s of L2 (45 records): %d-%d native samples, median interval %.2f s (record range %.2f-%.2f s), 90th percentile %.1f s (range %.1f-%.1f s).\n" % (
    d6.dt_median_s.median(), d6.dt_median_s.min(), d6.dt_median_s.max(), d6.dt_p90_s.median(), d6.dt_p90_s.min(), d6.dt_p90_s.max(),
    int(p.n_native_pm300.min()), int(p.n_native_pm300.max()), p.dt_median_pm300_s.median(), p.dt_median_pm300_s.min(), p.dt_median_pm300_s.max(), p.dt_p90_pm300_s.median(), p.dt_p90_pm300_s.min(), p.dt_p90_pm300_s.max()))
x = d6[["experiment_id", "l2_detected", "n_native", "record_duration_s", "n_native_in_window", "frac_rows_native", "frac_rows_not_flagged_interp", "dt_median_s", "dt_p90_s", "n_native_pm300", "dt_median_pm300_s", "dt_p90_pm300_s", "end_gap_after_L2_s", "n_native_after_L2"]].copy()
x["frac_rows_native"] = (100 * x.frac_rows_native).round(2)
x.columns = ["experiment", "L2", "native n", "raw length (s)", "native in shipped window", "% rows native", "1 - interp flag", "dt median (s)", "dt p90 (s)", "n +/-300 s of L2", "dt median +/-300 s", "dt p90 +/-300 s", "log end after L2 (s)", "native after L2"]
A(md(x))

A("\n## 6. What could not be determined\n")
A("* D5: no per-test TR statement exists; the OHS -> TR mapping (6-7 = TR) is an interpretation of the level descriptions. 134 of 252 files have no OHS at all (not in `main.xlsx`, or listed without a score), and the Data in Brief paper (PMC11639431) adds no per-test verdict. The Sheet1 tab of `main.xlsx` carries the same levels as `Summary` and adds nothing.\n* D5: for the 16 L2 positives whose peak stays below the IR cap (7 with OHS 3-5, 9 unscored) the record cannot show whether the reaction was self-sustaining; the source says it was not for the 7 it scored.\n* D6: the 2023 paper states outcomes per condition, not per file; the assignment of 'the replicate that did not meet the criterion' at LFP SOC 60 to `M2` is inferred from the ARC rate column. The 18650 and 4680 records (11 cells, including the two sodium-ion cells) have no source outcome statement anywhere that was reachable (Zenodo READMEs, 2025 JPS abstract); their verdicts are trace-inferred.\n* D6: the exotherm log stops at the runaway in every positive, so the maximum temperature and the post-onset behaviour are not observed in the file (the authors say the same: 'the reaction occurs faster than the ARC can track; therefore, the recording can miss the maximum temperature').\n* Two implementation points surfaced by the check and left unchanged here: (a) `standardize()` passes `max()` of all censored `T_*` channels, including `T_heater` and pinned ambient thermocouples, as the saturation ceiling of `T_surface_max`, which disables the +50 C guard for 8 labels (6 D5 artefacts, 2 D6); (b) for D6 the shipped 3 h window can exclude an earlier fast event (`LFP_SOC80_M3`), so rule diagnostics on D6 negatives must use the raw file.\n")
open(ROOT + "/reports/26_outcome_reconciliation.md", "w").write("\n".join(L) + "\n")
print("written", len("\n".join(L)))
