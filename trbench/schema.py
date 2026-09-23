"""Standard channel taxonomy and dataset registry for the TR benchmark corpus.

Follows TR_benchmark_research_plan_v2.md 1-1 (channel taxonomy).  Physically
distinct quantities are never merged: surface / internal / vent temperatures and
internal / chamber pressures stay in separate columns.

Units are fixed here and converted inside each adapter:
    temperature  degC
    pressure     kPa (absolute unless the column name says gauge)
    voltage      V
    current      A
    gas          ppm  (gas_total_mol is the one exception: mol)
    force        N
    displacement mm
"""

# --- physical channels ------------------------------------------------------
TEMPERATURE = [
    "T_surface_neg",    # near negative terminal
    "T_surface_pos",    # near positive terminal
    "T_surface_mid",    # can mid-height
    "T_surface_max",    # max over every surface TC present in the raw file
    "T_surface_mean",   # mean over every surface TC present in the raw file
    "T_internal_core",  # in-cell / core thermocouple
    "T_vent",           # vent gas, nearest probe
    "T_vent_2",         # vent gas, second probe
    "T_ambient",
    "T_heater",         # abuse heater surface -- trigger side, not a BMS sensor
]
# Venting shows up with OPPOSITE SIGN in these two, so they must never be
# merged: an in-cell transducer collapses when the cell ruptures, while a
# chamber transducer jumps as the released gas arrives.  Only #3 has a genuine
# in-cell sensor; #1 and #4 measure the vessel around the cell.
PRESSURE = ["P_internal", "P_chamber", "P_chamber_2"]
ELECTRICAL = ["V_cell", "V_module", "I"]
GAS = ["gas_H2", "gas_CO", "gas_CO2", "gas_HF", "gas_CH4", "gas_total_mol"]
MECHANICAL = ["F_expansion"]
# Trigger-side actuation.  Present so the abuse can be located on the time axis;
# excluded from every sensor panel M1..M6 because using them leaks the trigger.
TRIGGER_SIDE = ["T_heater", "F_penetrator", "disp_penetrator"]

CHANNELS = TEMPERATURE + PRESSURE + ELECTRICAL + GAS + MECHANICAL + [
    "F_penetrator", "disp_penetrator",
]

TIME_COLS = ["time_s", "t_rel_trigger", "t_rel_onset"]
QC_COLS = ["is_interpolated", "qc_flag"]


def interp_col(ch):
    """Per-channel interpolation flag: 1 where this channel's value was filled.

    The row-level `is_interpolated` is an OR across channels and is far too
    coarse for a causal rebuild -- see the experiment log, section 15-C.
    """
    return ch + "_interp"

MASK_PREFIX = "mask_"
def mask_col(ch: str) -> str:
    return MASK_PREFIX + ch

ALL_COLUMNS = TIME_COLS + CHANNELS + [mask_col(c) for c in CHANNELS] + QC_COLS

# --- qc_flag bit field ------------------------------------------------------
QC_TC_DETACH   = 1 << 0   # thermocouple fell off (hot -> ambient step)
QC_SATURATED   = 1 << 1   # channel clipped at its range limit
QC_POST_ONSET  = 1 << 2   # beyond t_onset + ALPHA_S, kept but excluded from analysis
QC_INTERP      = 1 << 3   # value came from upsampling, not measurement

# --- sensor panels (plan section E3) ---------------------------------------
SENSOR_PANELS = {
    "M1": ["T_surface_neg", "T_surface_pos", "T_surface_mid",
           "T_surface_max", "T_surface_mean", "T_ambient"],
    "M2": None,   # filled below: M1 + V/I
    "M3": None,   # M2 + pressure
    "M4": None,   # M2 + gas
    "M5": None,   # M2 + P + gas
    "M6": None,   # M5 + T_internal_core
}
SENSOR_PANELS["M2"] = SENSOR_PANELS["M1"] + ["V_cell", "V_module", "I"]
SENSOR_PANELS["M3"] = SENSOR_PANELS["M2"] + PRESSURE
SENSOR_PANELS["M4"] = SENSOR_PANELS["M2"] + GAS
SENSOR_PANELS["M5"] = SENSOR_PANELS["M2"] + PRESSURE + GAS
SENSOR_PANELS["M6"] = SENSOR_PANELS["M5"] + ["T_internal_core"]
# The plan's M6 (M5 + internal temperature) is expressible by ZERO experiments:
# #3 carries the internal thermocouple and in-cell pressure but no gas.  M6i is
# the node #3 can actually express, and is what the "internal sensor upper
# bound" argument has to rest on.
SENSOR_PANELS["M6i"] = SENSOR_PANELS["M3"] + ["T_internal_core"]

# ---- panel expressibility -------------------------------------------------
# A panel is expressible by an experiment only when EVERY required group has at
# least one channel present.  Checking only the incremental channel (the earlier
# bug) reported M5/M6 as available for datasets that cannot express them.
GROUP_T_SURF = ["T_surface_max", "T_surface_mean", "T_surface_mid",
                "T_surface_neg", "T_surface_pos"]
GROUP_V = ["V_cell", "V_module"]
GROUP_P = ["P_internal", "P_chamber", "P_chamber_2"]
GROUP_GAS = ["gas_H2", "gas_CO", "gas_CO2", "gas_HF", "gas_CH4", "gas_total_mol"]
GROUP_TINT = ["T_internal_core"]

# Cell current is present in #2 alone, so M2 is voltage-driven in practice and
# `I` is optional rather than required -- requiring it would collapse M2 from
# 267 experiments to 5.
PANEL_REQUIRES = {
    "M1": [GROUP_T_SURF],
    "M2": [GROUP_T_SURF, GROUP_V],
    "M3": [GROUP_T_SURF, GROUP_V, GROUP_P],
    "M4": [GROUP_T_SURF, GROUP_V, GROUP_GAS],
    "M5": [GROUP_T_SURF, GROUP_V, GROUP_P, GROUP_GAS],
    "M6": [GROUP_T_SURF, GROUP_V, GROUP_P, GROUP_GAS, GROUP_TINT],
    "M6i": [GROUP_T_SURF, GROUP_V, GROUP_P, GROUP_TINT],
}

# How much a panel comparison can carry.  Measured, not assumed: see
# registry/panel_power.csv.
#   primary     enough experiments across enough datasets for a population claim
#   case_study  report inside the one or two datasets that express it, with n
#   empty       no experiment expresses it; the count is itself the result
PANEL_ROLE = {
    "M1": "primary", "M2": "primary",
    "M3": "case_study", "M4": "case_study", "M5": "case_study",
    "M6i": "case_study", "M6": "empty",
}


def panel_expressible(channels):
    """-> set of panel ids an experiment with `channels` can express."""
    have = set(channels)
    return {p for p, groups in PANEL_REQUIRES.items()
            if all(any(c in have for c in g) for g in groups)}

# Which pressure channel a dataset carries decides how venting is detected.
# See common.onset_L1.
PRESSURE_SITE = {"P_internal": "in_cell", "P_chamber": "chamber",
                 "P_chamber_2": "chamber"}

# --- dataset registry -------------------------------------------------------
# role: core | aux_holdout | pretrain_only
DATASETS = {
    "ds01_bak": dict(
        no=1, name="BAK N21700CG-50 (60% SOC, one-sided heating)",
        license="CC BY 4.0", role="core", scale="cell",
        chemistry="NMC", form_factor="21700", capacity_ah=5.0, soc_pct=60,
        trigger="external_heating", native_hz=100.0, redistributable=True,
    ),
    "ds02_overcharge": dict(
        no=2, name="Overcharge-induced TR, prismatic LFP",
        license="CC BY 4.0", role="core", scale="cell",
        chemistry="LFP", form_factor="prismatic", capacity_ah=32.0, soc_pct=100,
        trigger="overcharge", native_hz=1.0, redistributable=True,
    ),
    "ds03_warwick": dict(
        no=3, name="Warwick internal temperature & pressure (Gulsoy et al.)",
        license="CC BY 4.0", role="core", scale="cell",
        chemistry="NCA", form_factor="21700", capacity_ah=4.0, soc_pct=100,
        trigger="external_heating", native_hz=10000.0, redistributable=True,
    ),
    "ds04_osf": dict(
        no=4, name="OSF multi-modal TR (cell & module, fresh/aged)",
        license="CC0 1.0", role="core", scale="cell+module",
        chemistry="NMC", form_factor="21700+pouch", capacity_ah=None, soc_pct=100,
        trigger="external_heating", native_hz=5.0, redistributable=True,
    ),
    "ds09_mech": dict(
        no=9, name="Mechanically induced TR (indentation)",
        license="CC BY 4.0", role="core", scale="cell",
        chemistry="mixed", form_factor="mixed", capacity_ah=None, soc_pct=None,
        trigger="mechanical_indentation", native_hz=10.0, redistributable=True,
    ),
    "ds12_arc": dict(
        no=12, name="KIT ARC 18650 / 21700 / 4680 (Ohneseit et al.)",
        license="CC BY 4.0", role="aux_holdout", scale="cell",
        chemistry="mixed", form_factor="18650+21700+4680", capacity_ah=None,
        soc_pct=None, trigger="arc_hws", native_hz=None, redistributable=True,
    ),
    "ds15_sim": dict(
        no=15, name="780 simulated TR events, JRC (Kriston et al.)",
        license="CC BY 4.0", role="pretrain_only", scale="cell",
        chemistry="NMC111", form_factor="simulated", capacity_ah=None, soc_pct=None,
        trigger="simulated_heating", native_hz=5.0, redistributable=True,
    ),
}

# --- preprocessing constants -----------------------------------------------
BASE_HZ = 1.0            # plan 1-4: common grid
HIGH_HZ = 10.0           # plan 1-4: pressure verification grid
ALPHA_S = 60.0           # plan 1-6: analysis truncated at t_onset + ALPHA_S
L2_THRESHOLDS = (0.5, 1.0, 2.0)   # degC/s, plan E1 sweep
L2_PRIMARY = 1.0
L2_SUSTAIN_S = 2.0
# Physical guards on L2, so the heater turn-on ramp and thermocouple glitches
# do not register as runaway.  See common.onset_L2 for the cases that motivated
# each value; t_onset_L2_unguarded is stored alongside for the E1 sensitivity.
L2_T_FLOOR = 60.0        # degC, crossing must occur above this
L2_RISE_MIN = 50.0       # degC, further rise required to confirm
L2_HORIZON_S = 60.0      # s, window in which that rise must happen

# Records longer than MAX_DURATION_S are cropped to the last CROP_PRE_S seconds.
# Only #12 (ARC, 6-50 h runs sampled ~200 times) trips this.
MAX_DURATION_S = 14400.0
CROP_PRE_S = 10800.0

# Internal-short criterion from the ORNL/Sandia indentation protocol (#9):
# open-circuit voltage down 25 mV from baseline.
ISC_DROP_V = 0.025
