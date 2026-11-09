# channel taxonomy, sensor panels and L2 event constants of the corpus

TEMPERATURE = ["T_surface_neg", "T_surface_pos", "T_surface_mid", "T_surface_max", "T_surface_mean",
               "T_internal_core", "T_vent", "T_vent_2", "T_ambient", "T_heater"]
PRESSURE = ["P_internal", "P_chamber", "P_chamber_2"]
ELECTRICAL = ["V_cell", "V_module", "I"]
GAS = ["gas_H2", "gas_CO", "gas_CO2", "gas_HF", "gas_CH4", "gas_total_mol"]
MECHANICAL = ["F_expansion"]
TRIGGER_SIDE = ["T_heater", "F_penetrator", "disp_penetrator"]
CHANNELS = TEMPERATURE + PRESSURE + ELECTRICAL + GAS + MECHANICAL + ["F_penetrator", "disp_penetrator"]

SENSOR_PANELS = {"M1": ["T_surface_neg", "T_surface_pos", "T_surface_mid", "T_surface_max", "T_surface_mean", "T_ambient"]}
SENSOR_PANELS["M2"] = SENSOR_PANELS["M1"] + ["V_cell", "V_module", "I"]
SENSOR_PANELS["M3"] = SENSOR_PANELS["M2"] + PRESSURE
SENSOR_PANELS["M4"] = SENSOR_PANELS["M2"] + GAS
SENSOR_PANELS["M5"] = SENSOR_PANELS["M2"] + PRESSURE + GAS
SENSOR_PANELS["M6"] = SENSOR_PANELS["M5"] + ["T_internal_core"]
SENSOR_PANELS["M6i"] = SENSOR_PANELS["M3"] + ["T_internal_core"]

GROUP_T_SURF = ["T_surface_max", "T_surface_mean", "T_surface_mid", "T_surface_neg", "T_surface_pos"]
GROUP_V = ["V_cell", "V_module"]
GROUP_P = ["P_internal", "P_chamber", "P_chamber_2"]
GROUP_GAS = ["gas_H2", "gas_CO", "gas_CO2", "gas_HF", "gas_CH4", "gas_total_mol"]
GROUP_TINT = ["T_internal_core"]
PANEL_REQUIRES = {
    "M1": [GROUP_T_SURF],
    "M2": [GROUP_T_SURF, GROUP_V],
    "M3": [GROUP_T_SURF, GROUP_V, GROUP_P],
    "M4": [GROUP_T_SURF, GROUP_V, GROUP_GAS],
    "M5": [GROUP_T_SURF, GROUP_V, GROUP_P, GROUP_GAS],
    "M6": [GROUP_T_SURF, GROUP_V, GROUP_P, GROUP_GAS, GROUP_TINT],
    "M6i": [GROUP_T_SURF, GROUP_V, GROUP_P, GROUP_TINT],
}

# L2 thermal-rate onset (degC/s, s, degC)
L2_PRIMARY = 1.0
L2_SUSTAIN_S = 2.0
L2_T_FLOOR = 60.0
L2_RISE_MIN = 50.0
L2_HORIZON_S = 60.0

DATASET_LABEL = {"ds01_bak": "D1", "ds02_overcharge": "D2", "ds03_warwick": "D3",
                 "ds04_osf": "D4", "ds09_mech": "D5", "ds12_arc": "D6"}


# panels whose every required channel group is present
def panel_expressible(channels):
    have = set(channels)
    return {p for p, groups in PANEL_REQUIRES.items() if all(any(c in have for c in g) for g in groups)}
