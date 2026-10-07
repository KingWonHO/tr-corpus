"""Emit the distributable corpus documentation from what was actually built.

Everything here is generated from registry/experiments.csv, so the README can
never drift from the data.  CC BY requires that modifications be stated, which
is what preprocessing_log.md is for.
"""
from __future__ import annotations

import os
import re
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import schema as S     # noqa: E402
import make_windows as W   # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "tr-corpus")

SOURCES = {
    "ds01_bak": ("Golubkov, A. Report Package - Thermal Runaway of Li-Ion cell "
                 "BAK N21700CG-50 at 60% SOC. Zenodo (2026). "
                 "doi:10.5281/zenodo.18849418 (Virtual Vehicle Research GmbH, "
                 "BATCAVE laboratory)", "CC BY 4.0"),
    "ds02_overcharge": ("Shen, J. overcharge-induced Thermal Runaway data in "
                        "Prismatic Lithium Iron Phosphate Battery. Mendeley "
                        "Data V1 (2025). doi:10.17632/8zjttd77my.1", "CC BY 4.0"),
    "ds03_warwick": ("Gulsoy, B., Briggs, C., Ngo, Q., Faraji Niri, M. & Marco, J. "
                     "Dataset of internal temperature and gas pressure in "
                     "cylindrical lithium-ion cells during thermal runaway. Data "
                     "in Brief 63, 112190 (2025). doi:10.1016/j.dib.2025.112190 | "
                     "data: doi:10.17632/rgfhdhcd9k.2 | method: Gulsoy et al., "
                     "J. Power Sources 617, 235147 (2024)", "CC BY 4.0"),
    "ds04_osf": ("Kwak, E., Jeong, J., Kim, J. H., Shin, Y., Park, M. & Oh, K.-Y. "
                 "Multi-modal thermal runaway dataset of fresh and aged "
                 "lithium-ion battery cells and modules. Scientific Data 13, "
                 "1084 (2026). doi:10.1038/s41597-026-07857-1 | data: "
                 "doi:10.17605/OSF.IO/C2HNQ", "CC0 1.0"),
    "ds09_mech": ("Lin, L.S. et al. Mechanically induced thermal runaway severity "
                  "analysis for Li-ion batteries. Journal of Energy Storage 61, "
                  "106798 (2023). doi:10.1016/j.est.2023.106798 (ORNL / Sandia)",
                  "CC BY 4.0"),
    "ds12_arc": ("Ohneseit, S. et al. Exothermal data from thermal safety "
                 "assessment by means of Accelerating Rate Calorimetry (ARC). "
                 "Zenodo, three records: 18650 doi:10.5281/zenodo.14956641; "
                 "21700 doi:10.5281/zenodo.7707929; 4680 "
                 "doi:10.5281/zenodo.14956635 (KIT)", "CC BY 4.0"),
    "ds15_sim": ("Kriston, A., Pfrang, A., Podias, A. & Ibtissam, A. Analysis of "
                 "the effect of thermal runaway initiation conditions on the "
                 "severity of thermal runaway - numerical simulation and machine "
                 "learning study. Mendeley Data V2 (2020). "
                 "doi:10.17632/9cykcc3svn.2 (European Commission JRC)",
                 "CC BY 4.0"),
}

LICENSE = """Creative Commons Attribution 4.0 International (CC BY 4.0)

This harmonized corpus combines seven independently published datasets.  Six are
released under CC BY 4.0 and one (#4, OSF) under CC0 1.0.  A mixed corpus is
governed by its most restrictive component, so the corpus as a whole is
distributed under CC BY 4.0.

CC BY 4.0 requires attribution AND a statement of changes.  ATTRIBUTION.md gives
the per-dataset citations; preprocessing_log.md states every change made.

#4 is CC0 and carries no legal attribution requirement.  It is cited anyway, as
academic norms require.

Note: CC0 and CC BY cover copyright only.  Patents, trademarks, and any
third-party rights held by cell manufacturers are outside their scope.  This is
not a problem for academic use but needs separate review before commercial use.
"""


def _readme(exp):
    ds = pd.read_csv(os.path.join(OUT, "registry", "datasets.csv"))
    sp = exp.groupby(["dataset_id", "split"]).size().unstack(fill_value=0)
    for c in ("train", "val", "test", "test_zeroshot", "pretrain"):
        if c not in sp.columns:
            sp[c] = 0

    lines = [
        "# TR corpus -- harmonized thermal-runaway early-detection benchmark",
        "",
        "Seven public thermal-runaway datasets aligned onto one schema, one time",
        "axis convention and one set of label definitions, so that early-warning",
        "lead time can be compared across sensor channels rather than across",
        "papers that each defined onset differently.",
        "",
        "Built by `trbench/build_corpus.py`.  Every number below is generated",
        "from `registry/experiments.csv`; nothing here is hand-maintained.",
        "",
        "## Contents",
        "",
        "```",
        "measured/{dataset_id}/{experiment_id}.parquet    six measured datasets",
        "simulated/ds15_sim/*.parquet                     simulated, PRETRAINING ONLY",
        "registry/datasets.csv                            Table 1",
        "registry/label_matrix.csv                        Table 2, label & panel expressibility",
        "registry/channel_availability.csv                per-channel experiment counts",
        "registry/experiments.csv                         one row per experiment, all metadata",
        "registry/preprocessing_notes.txt                 per-experiment processing notes",
        "splits/split_assignment.csv                      train / val / test / zero-shot",
        "splits/loeo_folds.json                           leave-one-experiment-out folds",
        "scalers/baseline_stats.json                      pre-trigger normalization stats",
        "windows/{split}.npz                              model-ready windows",
        "preprocessing_log.md                             what was changed and why",
        "ATTRIBUTION.md                                   original sources and citations",
        "```",
        "",
        "## Datasets",
        "",
        "| # | id | role | trigger | experiments | duration (s) | channels | L1 | L2 | L3 |",
        "|---|----|------|---------|-------------|--------------|----------|----|----|----|",
    ]
    for _, r in ds.iterrows():
        lines.append("| %d | %s | %s | %s | %d | %d | %d | %d | %d | %d |" % (
            r["no"], r["dataset_id"], r["role"], r["trigger"],
            r["n_experiments"], r["total_duration_s"], r["n_channels"],
            r["L1_computable"], r["L2_computable"], r["L3_computable"]))

    n_core = int((exp["role"] == "core").sum())
    lines += [
        "",
        "L1/L2/L3 columns count the experiments for which that onset definition",
        "is computable at all -- a zero is a data gap, not a modelling choice.",
        "",
        "## Splits",
        "",
        "The split unit is the experiment.  Windows from one experiment never",
        "appear on both sides of a split; the effective sample size is the",
        "experiment count, not the window count.",
        "",
        "| dataset | train | val | test | zero-shot | pretrain |",
        "|---|---|---|---|---|---|",
    ]
    for d in sp.index:
        lines.append("| %s | %d | %d | %d | %d | %d |" % (
            d, sp.loc[d, "train"], sp.loc[d, "val"], sp.loc[d, "test"],
            sp.loc[d, "test_zeroshot"], sp.loc[d, "pretrain"]))

    lines += [
        "",
        "`ds12_arc` is held out entirely as the zero-shot test (temperature only,",
        "sparse sampling -- the realistic worst case).  `ds15_sim` is simulated and",
        "is never evaluated.",
        "",
        "## Schema",
        "",
        "Each parquet file is one experiment on a %g Hz grid:" % S.BASE_HZ,
        "",
        "- `time_s`, `t_rel_trigger`, `t_rel_onset` -- both time anchors are stored",
        "- %d physical channels in fixed units (degC, kPa, V, A, ppm, N, mm)" % len(S.CHANNELS),
        "- `mask_<channel>` -- 1 where the channel was actually measured.",
        "  Missing modalities are NOT zero-filled: a physical 0 ppm and an",
        "  unmeasured channel are different things, and the mask is what tells",
        "  them apart.",
        "- `is_interpolated` -- 1 where a value came from upsampling",
        "- `qc_flag` -- bit 0 thermocouple detachment, bit 1 sensor saturation,",
        "  bit 2 beyond t_onset + %g s, bit 3 interpolated" % S.ALPHA_S,
        "",
        "Channels never merged, because their response times differ by tens of",
        "seconds: `T_surface_*` / `T_internal_core` / `T_vent`, and",
        "`P_internal` / `P_chamber`.",
        "",
        "`T_heater`, `F_penetrator` and `disp_penetrator` are ABUSE ACTUATION, not",
        "vehicle sensors.  They locate the trigger on the time axis and are",
        "excluded from every sensor panel and from the model features.",
        "",
        "## Label definitions",
        "",
        "| id | definition | needs |",
        "|---|---|---|",
        "| L1 | venting from pressure -- steepest DROP after the peak for an in-cell "
        "transducer, steepest RISE for a chamber transducer | any pressure |",
        "| L2 | dT/dt >= %g degC/s held %g s, above %g degC, confirmed by a further %g degC rise within %g s | any T |"
        % (S.L2_PRIMARY, S.L2_SUSTAIN_S, S.L2_T_FLOOR, S.L2_RISE_MIN, S.L2_HORIZON_S),
        "| L3 | cell voltage below 80%% of its pre-trigger level for 3 s | V_cell |",
        "| ISC | open-circuit voltage down %g mV from baseline -- the internal-short "
        "criterion from #9 own indentation protocol, and the earliest electrical "
        "precursor in the corpus | V_cell |" % (S.ISC_DROP_V * 1000),
        "",
        "All three are computed for every experiment where the channels exist, and",
        "L2 is swept at %s degC/s.  `t_onset_L2_unguarded` stores the bare rate"
        % ", ".join(str(x) for x in S.L2_THRESHOLDS),
        "crossing without the physical guards, so the sensitivity of the label",
        "definition itself can be measured.",
        "",
        "## Windows",
        "",
        "`windows/{split}.npz` holds `X` (N, %d, %d), `mask` (same shape),"
        % (W.WINDOW_S, len(W.FEATURES)),
        "`y_ttl` (seconds from window end to onset), `y_pre` (1 if the window ends",
        "within %g s before onset), `y_tr`, `experiment`, `features`." % W.LEAD_HORIZON_S,
        "",
        "Sensor panels M1..M6 are applied by masking columns at training time, not",
        "by rebuilding the windows -- otherwise the panels would differ in more",
        "than their channel set.  `make_windows.panel_mask(features, 'M3')` gives",
        "the selector.",
        "",
        "Normalization uses statistics fitted on the pre-trigger baseline only.",
        "A z-score over the whole sequence would pull the runaway excursion into",
        "the statistics and leak the future into the normalizer.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python trbench/build_corpus.py     # adapters -> tr-corpus/",
        "python trbench/make_splits.py      # experiment-level splits + LOEO",
        "python trbench/make_windows.py     # windows/{split}.npz",
        "```",
        "",
        "## License",
        "",
        "CC BY 4.0.  See LICENSE, ATTRIBUTION.md and preprocessing_log.md.",
        "%d core experiments were built in total." % n_core,
    ]
    return "\n".join(lines) + "\n"


def _attribution(exp):
    lines = ["# Attribution", "",
             "This corpus is derived from seven independently published datasets.",
             "Each retains its original license; cite the original work when",
             "using any part of it.", ""]
    for ds_id, info in S.DATASETS.items():
        sub = exp[exp["dataset_id"] == ds_id]
        if not len(sub):
            continue
        title, lic = SOURCES[ds_id]
        dois = re.findall(r"doi:(\S+?)(?=[;,)\s]|$)", title)
        lines += [
            "## #%d %s" % (info["no"], ds_id),
            "",
            "- Citation: %s" % title,
            "- DOI: %s" % ("; ".join(dict.fromkeys(dois)) if dois else "not assigned"),
            "- License: %s" % lic,
            "- Experiments used here: %d" % len(sub),
            "- Role in this corpus: %s" % info["role"],
            "",
        ]
    return "\n".join(lines)


def _log(exp):
    notes_path = os.path.join(OUT, "registry", "preprocessing_notes.txt")
    notes = open(notes_path, encoding="utf-8").read() if os.path.exists(notes_path) else ""
    n_lines = len([x for x in notes.splitlines() if x.strip()])

    lines = [
        "# Preprocessing log",
        "",
        "CC BY 4.0 requires that changes to the source data be stated.  This is",
        "that statement.  Nothing here was applied silently: every per-experiment",
        "note is in `registry/preprocessing_notes.txt` (%d entries)." % n_lines,
        "",
        "## Applied to every dataset",
        "",
        "1. **Channel renaming** to the standard taxonomy in `trbench/schema.py`,",
        "   with unit conversion to degC / kPa / V / A / ppm / N / mm.  Physically",
        "   distinct quantities are never merged.",
        "2. **Resampling to %g Hz.**  Channels faster than the grid are reduced by" % S.BASE_HZ,
        "   bin mean, which anti-aliases; #3's 10 kHz pressure would otherwise fold",
        "   the vent transient back into the band.  Channels slower than the grid",
        "   are linearly interpolated and every filled sample is marked in",
        "   `is_interpolated`.",
        "3. **Two time anchors recorded**, `t_trigger` (abuse applied) and",
        "   `t_onset` (runaway).  Where the trigger is not instrumented, t_trigger",
        "   is the record start and the experiment says so in its notes.",
        "4. **Three onset labels computed** (L1 pressure, L2 temperature rate,",
        "   L3 voltage collapse) wherever the required channel exists, plus an",
        "   L2 threshold sweep and an unguarded L2 variant.",
        "5. **Missing modalities left missing.**  No zero-filling; a per-channel",
        "   `mask_*` column carries availability.",
        "6. **QC flags**: thermocouple detachment (a >150 degC collapse to ambient",
        "   that then holds -- the affected channel is masked from that point on),",
        "   sensor saturation, and everything beyond `t_onset + %g s`." % S.ALPHA_S,
        "7. **Normalization statistics fitted on the pre-trigger baseline only**",
        "   and stored separately; the parquet files hold physical units.",
        "",
        "## Dataset-specific decisions",
        "",
        "### #1 BAK",
        "- Read from the 100 Hz MATLAB v7.3 file, not the ~0.6 Hz export workbook.",
        "- Thermocouples are mapped by the file's own semantic group",
        "  (`cell_can`, `heater`, `vent_gas`, `gas`), because the TC numbering is",
        "  per-cell: cell A uses TC1..TC4 and cell H uses TC29..TC32.  A fixed",
        "  TC-name table silently drops seven of the eight experiments.",
        "- Heater current/voltage are used only to find t_trigger and then",
        "  discarded.  They are NOT mapped to `I`, which means cell current;",
        "  #1 has no cell-current channel, so its panel M2 is voltage-only.",
        "- `gas_total_mol` is derived from cassette pressure, so it is not an",
        "  independent gas sensor.  Speciated gas (H2/CO/CO2/HF/...) exists only as",
        "  a single post-test GC measurement and is carried as metadata.  Panels",
        "  M4/M5 are therefore partly degenerate for #1.",
        "",
        "### #2 Overcharge LFP",
        "- The published 9,795 records are TIMESTEPS.  There are FIVE",
        "  experiments, one per overcharge current (16/24/32/40/48 A).",
        "- The Mendeley record gives a five-stage current protocol",
        "  (0.5C -> 0.75C -> 1C -> 1.25C -> 1.5C).  16/0.5 = 24/0.75 = 32/1 =",
        "  40/1.25 = 48/1.5 = 32, so **the cell is 32 Ah** and each file is one",
        "  C-rate stage.",
        "- It also says surface/gas temperatures at five locations -- two on the",
        "  cell and three on the flue gas, independently confirming that",
        "  yqs/yqz/yqx are gas probes, not cell surface.",
        "- **COD is read as CO**, settled from the data because the record has no",
        "  companion paper and its own gas list (CO/CO2/HF) does not match the",
        "  columns present (H2, COD, HF, CH4; no CO2 at all).  Three grounds:",
        "  the COD baseline is 16-28 ppm across all five files, where atmospheric",
        "  CO2 sits near 400-450 ppm, so a CO2 sensor could not idle there while",
        "  an electrochemical CO cell idles near zero with a small offset; COD",
        "  peaks at 5,200-7,500 ppm some 40-90 s BEFORE the cell temperature peak",
        "  and in step with H2, which is vent-gas behaviour; and the record does",
        "  name CO, with no other column able to carry it.  The baseline is the",
        "  discriminating argument.  Recorded because it rests on inference, not",
        "  on a published column definition.",
        "- The index column is used as the 1 Hz time axis; 24A.csv's timestamps",
        "  have minute resolution only.",
        "- The COD probe is reported as `gas_CO`; CH4 converted from vol%% to ppm.",
        "- No pressure channel, so panels M3/M5/M6 are not expressible.",
        "",
        "### #3 Warwick",
        "",
        "Paper: Gulsoy et al., Data in Brief 63:112190 (2025).  It confirms three",
        "assumptions this adapter had been making and corrects a fourth:",
        "",
        "- `IntPre` really is in **bar** (logged as a voltage and converted by a",
        "  calibration transfer function).",
        "- The trigger is a 50x50 mm heating pad at a constant **40 W**, applied",
        "  from the start of the record until runaway.  So `t_trigger = 0` is the",
        "  real trigger here, not the fallback it is elsewhere.",
        "- Cells are at 100% SOC (CC-CV to 4.25 V at 1C, taper to C/20).",
        "- **The cell is a Sony VTC6A: NCA cathode with a graphite/SiOx anode,",
        "  4 Ah, 21700 - not the NMC previously recorded here.**",
        "",
        "- Recovered from an opaque MATLAB-table MCOS blob by reading the",
        "  `__function_workspace__` subsystem directly.",
        "- Rates: the paper states 1 kHz on the voltage side and 10 Hz on the",
        "  temperature side.  The files carry 10 kHz on the voltage side, and",
        "  10 Hz temperature for tests 1-2 but 100 Hz for test 3.  Resampling is",
        "  driven by each channel measured interval, so this does not affect the",
        "  output, but a reader comparing paper to files will hit it.",
        "- Test 3 has no vent thermocouples.",
        "- `P_internal` is gauge pressure; the negative excursions after rupture",
        "  are a real sensor artefact and are kept and flagged, not clipped.",
        "- This is the ONLY dataset in the corpus with a genuine in-cell pressure",
        "  transducer and an embedded internal thermocouple, so panels M3 (in-cell",
        "  pressure) and M6 rest entirely on its three experiments.",
        "",
        "### #4 OSF",
        "",
        "The dataset ships with its paper (Kwak et al., Scientific Data 13:1084,",
        "2026, doi:10.1038/s41597-026-07857-1), which was used as ground truth.",
        "Four file-level problems are resolved from it:",
        "",
        "- **C1/C2 temperature and pressure records are cross-paired.**  The paper",
        "  states that temperature and pressure come from one synchronized DAQ",
        "  (GL860) at 10 Hz, so the two records of a test must span the same time.",
        "  Published durations are C1_T 2,637 s, C2_T 5,373 s, C1_P 5,373 s,",
        "  C2_P 2,599 s: C1_P matches C2_T to the second and C2_P matches C1_T to",
        "  within 38 s, while the published pairing is off by ~2,740 s.  Correctly",
        "  paired B2 differs by 80 s, so sub-minute offsets are normal here and a",
        "  46-minute one is not.  **Re-paired.**  The pressure peaks then fall",
        "  10-12 s from their temperature peaks instead of 35-37 s.  Which of the",
        "  two file names is wrong cannot be determined, and does not matter:",
        "  C1 and C2 are two repeats of one condition (2170 cell, 86% SOH).",
        "- **`D1_Pressure.xlsx` contains force, not pressure.**  Fig. 9(b) is",
        "  captioned \"Temperature and force measurements for the aged cell (D1)\",",
        "  and the Data Record says pouch folders carry Temperature and Force.  The",
        "  values confirm it: the file starts at 310 N like the other pouch",
        "  preloads (A2 274 N, M1 477 N) and peaks at 929 N, whereas the real",
        "  pressure files start near 0.  Mapped to `F_expansion`.",
        "- **D1 is also truncated at t = 7,730 s.**  Beyond that its value column",
        "  stops being a measurement and becomes a copy of the row counter",
        "  (offset +63) for 1,450 rows, then 63 rows of the literal string `--`.",
        "- **The aged module M2 is 76% SOH, not 86%.**  The folder is named",
        "  \"Aged\" like the 86% cells, but Technical Validation says \"1 NMC pouch",
        "  module at 100% SOH and 1 NMC pouch module at 76% SOH\".",
        "",
        "Two further points:",
        "",
        "- Module temperature files: the `ms` column is a cycling sub-second field",
        "  (0,200,...,800, then restarts), not elapsed time.  Reading it literally",
        "  compresses a 4,400 s test into 0.8 s.  The axis is rebuilt from the row",
        "  index at the file's own step.  Module thermocouple positions are given",
        "  only by cell grouping, so the 15 channels feed `T_surface_max` /",
        "  `T_surface_mean` only.",
        "- A1 has no force record and B1 no pressure record because, per Technical",
        "  Validation, two channels failed and were withheld; B1's transducer is",
        "  named explicitly.  Those absences are intentional, not missing files.",
        "",
        "**Pressure unit.**  The Data Record says kPa and the SOH section quotes",
        "\"peak pressures of 8.68 and 8.00 kPa\", which are exactly the raw column",
        "maxima.  Read literally that cannot be right: the transducer has a 3 MPa",
        "range and a stated uncertainty of +/-0.06 MPa = 60 kPa, so an 8.68 kPa",
        "peak would sit seven times below the sensor's own uncertainty; the same",
        "paper reports a fresh-cell \"peak pressure of 900 kPa\" where the raw",
        "maximum is 9.73; and ~0.33 mol of vent gas (the measured figure for a",
        "comparable 21700 in #1) released into the 2.829 L canister gives several",
        "hundred kPa.  The column is read as **bar** and converted to kPa, which",
        "makes all four statements agree.  Flagged here because it is a deliberate",
        "departure from the published unit.",
        "",
        "### #9 Mechanical indentation",
        "- 253 workbooks, 14 distinct header layouts.  Columns are parsed",
        "  positionally: each time-like column opens a block and the value columns",
        "  after it belong to it, which is how one sheet carries a load frame and",
        "  thermocouples at different rates.",
        "- `Penetrator Force (N)` / `Cell Voltage (V)` are the calibrated versions",
        "  of `Load (lb)` / `Voltage (V)`; the calibrated pair wins, unless it is",
        "  present but empty (Sorteria-*), in which case the raw pair is used.",
        "- Force sign convention is not consistent across the collection.  Series",
        "  whose compression is negative are flipped so compression is positive.",
        "- `F_penetrator` and `disp_penetrator` are abuse actuation, excluded from",
        "  every sensor panel.",
        "- **About half of these tests do not reach thermal runaway.**  main.xlsx",
        "  scores each 1..7 (1 = no effect, 7 = rupture and fire); scores 1-2 are",
        "  genuine negative outcomes under abuse.  This matters for the",
        "  false-alarm question: the assumption that public TR data contains no",
        "  negatives at all is not true for the mechanical trigger.",
        "",
        "From the paper (Lin et al., J. Energy Storage 61:106798, 2023):",
        "",
        "- **The infrared camera range was set to Tmax = 150 degC**, and the",
        "  maximum temperature after runaway was not tracked closely.  Twenty-five",
        "  experiments sit pinned at exactly 150.24 degC: their peak temperature is",
        "  CENSORED, not measured.  Until this was handled the L2 confirmation rule",
        "  could not fire on a censored trace, and three cells the authors scored",
        "  EUCAR 7 (rupture, smoke, fire) carried no onset label at all.  Saturation",
        "  now counts as confirmation, which raised L2 coverage from 74 to 83",
        "  experiments and made the label agree with the authors own scores on",
        "  **all 20** of the severity 6-7 cells.",
        "- Protocol: single-side indentation at 0.05 in/min (1.27 mm/min), DAQ at",
        "  10 Hz or faster, internal short declared at a 25 mV drop in open-circuit",
        "  voltage, indenter then held for over 10 minutes.  That 25 mV criterion",
        "  is used corpus-wide as the `t_isc` label.",
        "- The paper records load as confirmation of loading/failure conditions and",
        "  excludes it from its own severity calculation, which is independent",
        "  support for treating force and displacement as trigger-side here.",
        "- `severity_score` is the EUCAR/OHS level (1-7); `calculated_severity` is",
        "  the paper CHS (5-100) from peak temperature, peak dT/dt and a",
        "  voltage-drop score, weighted by capacity and SOC.",
        "",
        "### #12 KIT ARC",
        "- The .EXO files name four columns but write six numeric fields per row.",
        "  Reading with a header makes pandas promote the extra leading fields",
        "  into an index and the time axis comes out non-monotonic and negative.",
        "  Parsed positionally instead.",
        "- Time is in minutes and event-driven.  On a 1 Hz grid these records are",
        "  >99%% interpolated, so records longer than %g s are cropped to their" % S.MAX_DURATION_S,
        "  last %g s.  Every crop is recorded in `crop_start_s` and in the notes." % S.CROP_PRE_S,
        "- Temperature only: expressible in panel M1 alone.  Held out as the",
        "  zero-shot test set; never used for training.",
        "",
        "### #15 Simulated",
        "",
        "Source paper: Kriston, Podias, Adanouj & Pfrang, J. Electrochem. Soc.",
        "167, 090555 (2020).  It defines the columns, and two of them are not what",
        "their names suggest:",
        "",
        "- **`V` is the model OVERPOTENTIAL** (V_OP, the unknown in the",
        "  equivalent-circuit Kirchhoff solution), not a cell terminal voltage.",
        "- **`I` is a CURRENT RATE normalised by cell capacity**, units 1/s, not",
        "  amperes.",
        "",
        "  Both are therefore dropped rather than mapped onto `V_cell` and `I`;",
        "  keeping them would put incompatible scales into columns that hold volts",
        "  and amps everywhere else, and would corrupt globally normalised",
        "  pretraining.  Before this was caught, `V` was producing 8 spurious L3",
        "  voltage-collapse labels.",
        "",
        "- `Temp` is exported in degC even though the model works in kelvin.  The",
        "  per-severity means of `Max_Temp` are 130/257/367/566/936, against the",
        "  paper's cluster maxima of ~127/253/363/670/928 degC, which both",
        "  confirms the unit and maps `Severity` 1-5 onto the paper's five",
        "  clusters: no TR/no ISC, no TR/ISC (soft short), mild TR, severe ISC TR,",
        "  severe TR.  The workbook mixes units: `IniT` is in kelvin (300-700).",
        "- **The `TR` and `Severe TR` columns hold strings** (\"TR\"/\"No TR\",",
        "  \"Severe TR\"/\"Mild TR\"), and `bool(\"No TR\")` is True.  Read as booleans",
        "  they marked all 781 events as runaways, including ones whose peak",
        "  temperature never leaves ambient.  Compared as strings now.",
        "- Extracted from a Deflate64 archive (Python's zipfile cannot read it) and",
        "  reduced to the columns that map onto the measured schema.",
        "- Kept in `simulated/`, physically separate from `measured/`, and assigned",
        "  the `pretrain` split so it cannot reach an evaluation set by accident.",
        "",
        "## Known limitations",
        "",
        "- Trigger coverage is uneven: 252 of the measured experiments are",
        "  mechanical indentation and 8 are external heating.  Any comparison",
        "  across triggers must be weighted or restricted, not pooled.",
        "- Panel M4/M5 depends almost entirely on #2, and M6 entirely on #3.",
        "  Panel comparisons are only valid within a dataset that expresses both",
        "  panels being compared.",
        "- Pressure is not one quantity across the corpus.  Only #3 measures",
        "  inside the cell (`P_internal`); #1 (`P_chamber` reactor, `P_chamber_2`",
        "  cassette) and #4 (`P_chamber` canister) measure the vessel around it.",
        "  Venting has the OPPOSITE SIGN in the two: an in-cell transducer",
        "  collapses at rupture, a chamber transducer jumps as gas arrives, so the",
        "  L1 rule is chosen per channel and the columns are never merged.  #1 is",
        "  absolute, #3 and #4 gauge.  Any cross-dataset pressure comparison has to",
        "  account for all three differences.",
        "- Panel M6 (internal temperature) rests entirely on #3 three",
        "  experiments, and true in-cell pressure likewise exists only in #3.",
        "- `t_isc` is defined by #9 indentation protocol, where a 25 mV step is an",
        "  unambiguous short.  On the externally heated datasets the same threshold",
        "  can also be crossed by ordinary thermal drift of the open-circuit",
        "  voltage, so treat `t_isc` as sensitive but less specific there.",
    ]
    return "\n".join(lines) + "\n"


def main():
    exp = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))
    for name, text in (("README.md", _readme(exp)),
                       ("LICENSE", LICENSE),
                       ("ATTRIBUTION.md", _attribution(exp)),
                       ("preprocessing_log.md", _log(exp))):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", name)

    sim_readme = (
        "# SIMULATED DATA -- NOT MEASUREMENTS\n\n"
        "Everything in this directory comes from a numerical thermal-runaway\n"
        "model, not from a physical cell.  It is provided for pretraining only\n"
        "and carries the `pretrain` split.\n\n"
        "Do not mix it with `measured/`, and do not report any evaluation\n"
        "number computed on it as a measurement result.\n")
    p = os.path.join(OUT, "simulated")
    if os.path.isdir(p):
        with open(os.path.join(p, "README_SIMULATED.md"), "w", encoding="utf-8") as f:
            f.write(sim_readme)
        print("wrote simulated/README_SIMULATED.md")


if __name__ == "__main__":
    main()
