"""#9 Mechanically induced TR (indentation) -- 253 experiment workbooks.

Lin, Li, Fishman, Torres-Castro, Preger, De Angelis, Lamb, Zhu, Allu & Wang,
"Mechanically induced thermal runaway severity analysis for Li-ion batteries",
Journal of Energy Storage 61, 106798 (2023), doi:10.1016/j.est.2023.106798.
ORNL / Sandia National Laboratories.

The protocol: single-side indentation to force an internal short, indenter
driven at 0.05 in/min (1.27 mm/min), DAQ at 10 Hz or faster, short declared
when open-circuit voltage drops by 25 mV, indenter then held in place for over
10 minutes.  Load is recorded only "as confirmation of loading/failure
conditions", not as a severity input -- which is exactly why force and
displacement belong on the trigger side here and not in a sensor panel.

The one fact that changes the labels: "The infrared camera temperature range
was set at Tmax = 150 degC.  The maximum temperature after thermal runaway was
not tracked closely."  Twenty-five of the 252 experiments sit pinned at exactly
150.24 degC -- their peak temperature is CENSORED, not measured.  Before this
was handled, the L2 confirmation rule could not fire on a censored trace and
three cells the authors scored EUCAR 7 (rupture, smoke, fire) carried no onset
label at all.  common.onset_L2 now accepts saturation as confirmation.

main.xlsx "Observed Score" is the EUCAR hazard level (the paper's OHS, 1-7) and
"Calculated Score" is the paper's CHS, a 5-100 scale combining peak temperature,
peak dT/dt and a voltage-drop score, weighted by capacity and SOC.

Two things here change the plan's assumptions, both worth carrying into the
paper:

1.  It is 253 experiments, not the handful implied by "temperature-only, thin".
    That single-handedly clears the >=40-experiment gate in plan section 4.
2.  Roughly half of them do NOT go into thermal runaway.  main.xlsx scores each
    test 1..7 (1 = no effect, 7 = rupture + fire), and scores 1-2 are genuine
    NEGATIVE events under abuse.  Plan E9 assumes the public corpus has no
    negatives at all; for the mechanical trigger that is not true, and these
    tests give a real, if narrow, false-alarm denominator.

Fourteen different header layouts.  Columns are parsed positionally: each
time-like column opens a block and the value columns after it belong to that
block, which is how the same sheet can carry a load frame at one rate and
thermocouples at another.  `Penetrator Force (N)` / `Cell Voltage (V)` are the
calibrated versions of `Load (lb)` / `Voltage (V)`; when both are present the
calibrated pair wins.

Penetrator force and displacement are the ABUSE ACTUATION, not sensors a vehicle
would have.  They are kept (they locate t_trigger exactly) but live in
schema.TRIGGER_SIDE and are excluded from every M1..M6 panel.
"""
import glob
import os
import re
import warnings
import numpy as np
import pandas as pd

from common import RawExperiment

warnings.filterwarnings("ignore", module="openpyxl")

DS = "ds09_mech"
ROOT = "Dataset/9/Mechanically Induced Thermal Runaway for Li-ion Batteries"
EXCEL = os.path.join(ROOT, "excel")
MAIN = os.path.join(ROOT, "main.xlsx")

LBF_TO_N = 4.4482216


def _classify(name):
    """-> ('time', None) | ('val', channel, scale) | None"""
    n = str(name).strip().lower()
    if n.startswith("unnamed") or n in ("", "nan"):
        return None
    if "time" in n:
        return ("time", None, 1.0)
    if n in ("column1",):
        return ("time", None, 1.0)
    if "penetrator force" in n:
        return ("val", "F_penetrator", 1.0)
    if n.startswith("load") or n in ("column2",):
        return ("val", "F_penetrator_lb", LBF_TO_N)
    if "displacement" in n:
        return ("val", "disp_penetrator", 1.0)
    if "cell voltage" in n or n.startswith("vcell"):
        return ("val", "V_cell", 1.0)
    if n.startswith("voltage") or n in ("column3",):
        return ("val", "V_cell_raw", 1.0)
    if "tambient" in n or "ambient" in n:
        return ("val", "T_ambient", 1.0)
    if "near positive" in n:
        return ("val", "T_surface_pos", 1.0)
    if "near negative" in n:
        return ("val", "T_surface_neg", 1.0)
    if re.match(r"^tc\d", n) or "punch" in n or "bottom" in n:
        return ("val", "T_surface_x_" + re.sub(r"\W+", "", n)[:14], 1.0)
    if any(k in n for k in ("temp", "[c]", "(c)", "max ")) or n == "c":
        return ("val", "T_surface_x_" + re.sub(r"\W+", "", n)[:14], 1.0)
    return None


def _blocks(df):
    """Split a sheet into (time_array, {channel: values}) blocks."""
    out, cur_t, cur = [], None, {}
    for c in df.columns:
        k = _classify(c)
        if k is None:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if k[0] == "time":
            if cur_t is not None and cur:
                out.append((cur_t, cur))
            cur_t, cur = v, {}
        elif cur_t is not None:
            ch, scale = k[1], k[2]
            if ch not in cur:
                cur[ch] = v * scale
    if cur_t is not None and cur:
        out.append((cur_t, cur))
    return out


def _pick_sheet(xl):
    best, best_n = None, -1
    for s in xl.sheet_names:
        df = xl.parse(s)
        n = sum(1 for c in df.columns if _classify(c))
        if n > best_n:
            best, best_n = df, n
    return best


def _summary(root):
    p = os.path.join(root, MAIN)
    if not os.path.exists(p):
        return {}
    df = pd.read_excel(p, sheet_name="Summary")
    df["score"] = pd.to_numeric(df["Observed Score.1"], errors="coerce")
    out = {}
    for _, r in df.iterrows():
        fn = str(r["FileName"]).strip()
        if fn and fn != "nan":
            chs = pd.to_numeric(r.get("Calculated Score"), errors="coerce")
            out[fn] = dict(
                severity_score=None if pd.isna(r["score"]) else int(r["score"]),
                calculated_severity=None if pd.isna(chs) else float(chs),
                severity_text=(None if pd.isna(r["Observed Score"])
                               else str(r["Observed Score"])),
                capacity_mah=(None if pd.isna(r.get("Capacity"))
                              else float(r["Capacity"])),
                soc_pct=None if pd.isna(r.get("SOC")) else float(r["SOC"]),
            )
    return out


CHEM_PAT = [("LFP", "LFP"), ("NMC", "NMC"), ("NCA", "NCA"), ("LCO", "LCO"),
            ("LMO-LNO", "LMO-LNO"), ("NMC-LMO", "NMC-LMO")]

# Filenames that name a manufacturer or programme instead of a chemistry.
# Soteria comes from the paper's Table 1: both the Control and the MFCC
# (metallized film current collector) cells are LiNiMnCoO2 (811), 5200 mAh.
# The 500/1500/2000 mAh unprefixed files are the commercial LCO pouches of that
# same table.  ChevyVolt and NissanLeaf are commercial EV cells that the paper
# does not tabulate, so their chemistry stays unknown and only the pack is named.
PROGRAMME = [
    ("soteria", ("NMC811", "Soteria")), ("sorteria", ("NMC811", "Soteria")),
    ("soetria", ("NMC811", "Soteria")),
    ("chevyvolt", (None, "Chevrolet Volt")),
    ("nissanleaf", (None, "Nissan Leaf")), ("nisanleaf", (None, "Nissan Leaf")),
    ("nissan_leaf", (None, "Nissan Leaf")),
]
LCO_MAH = re.compile(r"^(500|1500|2000)mAh", re.I)


def _from_name(stem):
    m = re.search(r"(\d+)\s*SOC", stem, re.I)
    soc = float(m.group(1)) if m else None
    low = stem.lower().replace("-", "").replace("_", "")
    chem = next((c for pat, c in CHEM_PAT if pat.lower() in stem.lower()), None)
    source = None
    if chem is None:
        for pat, (c, src) in PROGRAMME:
            if pat.replace("_", "") in low:
                chem, source = c, src
                break
    if chem is None and LCO_MAH.match(stem):
        chem, source = "LCO", "commercial LCO pouch (paper Table 1)"
    return chem or "unknown", soc, source


def load(root):
    summ = _summary(root)
    for f in sorted(glob.glob(os.path.join(root, EXCEL, "*.xlsx"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            df = _pick_sheet(pd.ExcelFile(f))
        except Exception as e:
            print("  [ds09] skip %s: %s" % (stem, e))
            continue
        if df is None:
            continue

        ser, notes = {}, []
        for t, block in _blocks(df):
            for ch, v in block.items():
                n = min(len(t), len(v))
                ser.setdefault(ch, (t[:n], v[:n]))

        # A "calibrated" column is sometimes present but entirely empty
        # (Sorteria-*: Penetrator Force (N) is all NaN while Load (lb) is real),
        # so drop empty channels before deciding which version wins.
        ser = {k: v for k, v in ser.items() if np.isfinite(v[1]).sum() >= 5}

        if "F_penetrator" in ser:
            ser.pop("F_penetrator_lb", None)
        elif "F_penetrator_lb" in ser:
            ser["F_penetrator"] = ser.pop("F_penetrator_lb")
        if "V_cell" in ser:
            ser.pop("V_cell_raw", None)
        elif "V_cell_raw" in ser:
            ser["V_cell"] = ser.pop("V_cell_raw")

        if not ser:
            print("  [ds09] no recognised channel in %s" % stem)
            continue

        # Sign convention is not consistent across the collection: some load
        # frames report compression negative (1500mAh*: 2 N -> -380 N).  Flip so
        # compression is positive everywhere, otherwise contact detection and
        # any cross-file comparison of force are meaningless.
        if "F_penetrator" in ser:
            t, v = ser["F_penetrator"]
            if abs(np.nanmin(v)) > abs(np.nanmax(v)):
                ser["F_penetrator"] = (t, -v)
                notes.append("F_penetrator sign flipped so compression is "
                             "positive (raw file reports compression negative)")

        # t_trigger = penetrator contact: first departure from the pre-contact
        # baseline by more than both 10% of the stroke and 8 x its noise level
        t_trig = None
        for key in ("disp_penetrator", "F_penetrator"):
            if key not in ser:
                continue
            t, v = ser[key]
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            head = v[ok][:max(10, ok.sum() // 50)]
            base = float(np.nanmedian(head))
            mad = float(np.nanmedian(np.abs(head - base))) or 1e-9
            span = float(np.nanmax(v[ok]) - base)
            if not np.isfinite(span) or span <= 0:
                continue
            thr = max(0.10 * span, 8.0 * mad)
            hit = np.flatnonzero(ok & (v - base > thr))
            if len(hit):
                t_trig = float(t[hit[0]])
                notes.append("t_trigger from %s departing baseline by %.3g"
                             % (key, thr))
                break

        chem, soc, source = _from_name(stem)
        s = summ.get(stem, {})
        meta = dict(
            cell_type=stem, chemistry=chem, cell_source=source,
            form_factor="pouch/cylindrical",
            capacity_ah=(s.get("capacity_mah") / 1000.0
                         if s.get("capacity_mah") else None),
            soc_pct=s.get("soc_pct", soc), soh="fresh",
            trigger="mechanical_indentation", scale="cell",
            severity_score=s.get("severity_score"),      # EUCAR / OHS, 1-7
            severity_text=s.get("severity_text"),
            calculated_severity=s.get("calculated_severity"),   # CHS, 5-100
            indenter_speed_mm_per_min=1.27,
            isc_criterion_mv=25.0,
            ir_camera_ceiling_c=150.0,
            source_doi="10.1016/j.est.2023.106798",
        )
        notes.append("F_penetrator/disp_penetrator are abuse actuation "
                     "(indenter at 1.27 mm/min), excluded from panels M1-M6")
        notes.append("much of this dataset's temperature comes from an infrared "
                     "camera capped at 150 degC; where the trace is pinned there "
                     "the peak is censored, not measured")
        if meta["severity_score"] is not None and meta["severity_score"] <= 2:
            notes.append("severity score %d: no thermal runaway -- usable as a "
                         "negative sample" % meta["severity_score"])
        yield RawExperiment(DS, stem, ser, meta, t_trig, notes)
