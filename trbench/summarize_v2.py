"""Reproducible uncertainty summaries for evaluator-V2 event-level CSV files.

The input rows are already out-of-fold predictions.  This module never fits a
model.  Its intervals therefore describe variation across held experiments,
conditional on the fitted models, split, seeds, and calibrated thresholds that
produced the CSV files.

Example
-------
uv run python trbench/summarize_v2.py \
    --inputs tr-corpus/results/v2_e3a_raw.csv \
    --out-dir tr-corpus/results/v2_uncertainty \
    --registry tr-corpus/registry/experiments.csv \
    --stratify dataset trigger
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


KEY = ["model", "panel", "held", "label"]
REQUIRED = KEY + ["detected", "lead"]
NUMERIC = [
    "tau",
    "alarm_time",
    "t_trigger",
    "onset",
    "lead",
    "far_check",
    "risk_max",
]
BOOLEAN = ["detected", "pre_trigger"]
DEFAULT_L2 = "L2@1.0"
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 20260914

CAVEATS = [
    (
        "All intervals are conditional on the fitted out-of-fold models, "
        "split, model seeds, and calibrated thresholds represented in the "
        "input CSV; they do not include model-fitting or split uncertainty."
    ),
    (
        "Overlapping leave-one-experiment-out fits and repeated model/panel "
        "evaluations are not independent experiments. Bootstrap resampling "
        "uses the matched held experiment as the sampling unit."
    ),
    (
        "The lead-time contrast is conditional on detection by both panels. "
        "It is not an unconditional performance contrast."
    ),
    (
        "A non-significant McNemar result or an interval containing zero does "
        "not establish equivalence; no equivalence margin is imposed here."
    ),
]


def _parse_bool(value: object) -> object:
    """Parse a CSV boolean while preserving missing values."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _first_nonmissing(series: pd.Series) -> object:
    values = series[series.notna()]
    return values.iloc[0] if len(values) else pd.NA


def _display_key(names: Sequence[str], values: object) -> str:
    if not isinstance(values, tuple):
        values = (values,)
    return ", ".join(f"{name}={value!r}" for name, value in zip(names, values))


def load_v2(inputs: Sequence[str | Path]) -> tuple[pd.DataFrame, dict]:
    """Load, validate, and deduplicate one or more V2 CSV files.

    Duplicate ``(model, panel, held, label)`` rows are allowed only when every
    non-missing value agrees.  When a seed column exists, it is part of that
    key and all downstream groups. Conflicting native builds or reruns fail
    instead of being counted as extra experiments.
    """
    if not inputs:
        raise ValueError("at least one --inputs path is required")

    frames: list[pd.DataFrame] = []
    seed_presence: list[bool] = []
    file_records: list[dict] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        frame = pd.read_csv(path)
        seed_presence.append("seed" in frame.columns)
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        frame["_source_input"] = str(path.resolve())
        frames.append(frame)
        file_records.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": int(len(frame)),
            }
        )

    if any(seed_presence) and not all(seed_presence):
        raise ValueError(
            "cannot mix V2 inputs with and without a seed column; their fitted "
            "model replicates cannot be matched safely"
        )
    data = pd.concat(frames, ignore_index=True, sort=False)
    input_rows = len(data)
    key_columns = KEY + (["seed"] if all(seed_presence) else [])

    for column in KEY:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"key column {column!r} contains a missing value")
        data[column] = data[column].astype(str)

    if "seed" in data:
        seed_values = pd.to_numeric(data["seed"], errors="coerce")
        invalid_seed = (
            seed_values.isna()
            | ~np.isfinite(seed_values)
            | ~np.isclose(seed_values, np.round(seed_values))
        )
        if invalid_seed.any():
            raise ValueError("column 'seed' must contain an integer on every row")
        data["seed"] = seed_values.astype(np.int64)

    for column in BOOLEAN:
        if column in data:
            try:
                data[column] = data[column].map(_parse_bool).astype("boolean")
            except ValueError as exc:
                raise ValueError(f"column {column!r}: {exc}") from exc
    if data["detected"].isna().any():
        raise ValueError("column 'detected' contains a missing value")

    for column in NUMERIC:
        if column in data:
            original = data[column]
            converted = pd.to_numeric(original, errors="coerce")
            invalid = original.notna() & converted.isna()
            if invalid.any():
                sample = original[invalid].iloc[0]
                raise ValueError(f"column {column!r} contains non-numeric value {sample!r}")
            data[column] = converted

    detected_without_lead = data["detected"].fillna(False) & ~np.isfinite(data["lead"])
    if detected_without_lead.any():
        key = tuple(data.loc[detected_without_lead, key_columns].iloc[0])
        raise ValueError(
            f"detected row has no finite lead: {_display_key(key_columns, key)}"
        )

    payload_columns = [
        c for c in data.columns if c not in key_columns + ["_source_input"]
    ]
    strict_columns = set(REQUIRED + NUMERIC + BOOLEAN) & set(payload_columns)
    rows: list[pd.Series] = []
    duplicate_rows = 0
    duplicate_keys = 0
    for key, group in data.groupby(key_columns, sort=False, dropna=False):
        if len(group) > 1:
            duplicate_keys += 1
            duplicate_rows += len(group) - 1
            conflicts: list[str] = []
            for column in payload_columns:
                values = group[column]
                if column in strict_columns:
                    differs = values.nunique(dropna=False) > 1
                else:
                    differs = values.dropna().nunique(dropna=True) > 1
                if differs:
                    conflicts.append(column)
            if conflicts:
                raise ValueError(
                    "conflicting duplicate for "
                    f"{_display_key(key_columns, key)} in columns {conflicts}"
                )

        row = group.iloc[0].copy()
        for column in payload_columns:
            if pd.isna(row[column]):
                row[column] = _first_nonmissing(group[column])
        rows.append(row)

    clean = pd.DataFrame(rows).drop(columns="_source_input").reset_index(drop=True)
    clean["detected"] = clean["detected"].astype(bool)
    if "pre_trigger" in clean:
        clean["pre_trigger"] = clean["pre_trigger"].astype("boolean")

    accounting = {
        "input_files": file_records,
        "input_rows": int(input_rows),
        "deduplicated_rows": int(len(clean)),
        "identical_duplicate_keys": int(duplicate_keys),
        "identical_duplicate_rows_removed": int(duplicate_rows),
        "unique_held_experiments": int(clean["held"].nunique()),
        "model_panel_held_evaluations": int(
            clean[
                ["model", "panel", *(["seed"] if "seed" in clean else []), "held"]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "model_panel_held_label_rows": int(len(clean)),
        "seed_column_present": bool("seed" in clean),
        "model_seeds": sorted(map(int, clean["seed"].unique())) if "seed" in clean else [],
    }
    return clean, accounting


def attach_registry(
    data: pd.DataFrame,
    registry_path: str | Path,
    strata: Sequence[str],
) -> pd.DataFrame:
    """Attach requested dataset/trigger strata from the experiment registry."""
    unsupported = sorted(set(strata) - {"dataset", "trigger"})
    if unsupported:
        raise ValueError(f"unsupported strata {unsupported}; use dataset and/or trigger")
    if not strata:
        return data.copy()

    registry = pd.read_csv(registry_path)
    needed = ["dataset_id", "experiment_id"]
    if "trigger" in strata:
        needed.append("trigger")
    missing = [column for column in needed if column not in registry]
    if missing:
        raise ValueError(f"registry is missing columns {missing}")

    reg = registry[needed].copy()
    reg["held"] = reg["dataset_id"].astype(str) + "/" + reg["experiment_id"].astype(str)
    for column in ["dataset_id", "trigger"]:
        if column not in reg:
            continue
        conflicts = reg.groupby("held", dropna=False)[column].nunique(dropna=True)
        if (conflicts > 1).any():
            held = conflicts[conflicts > 1].index[0]
            raise ValueError(f"registry has conflicting {column!r} values for {held}")
    reg = reg.drop_duplicates("held")
    rename = {"dataset_id": "dataset"}
    keep = ["held"] + [rename.get(s, s) for s in strata]
    reg = reg.rename(columns=rename)[keep]

    merged = data.merge(reg, on="held", how="left", validate="many_to_one")
    missing_held = merged.loc[merged[list(strata)].isna().any(axis=1), "held"].unique()
    if len(missing_held):
        sample = ", ".join(map(str, missing_held[:5]))
        raise ValueError(f"registry metadata missing for held experiments: {sample}")
    return merged


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def exact_mcnemar_p(rescue: int, lost: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    if rescue < 0 or lost < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = rescue + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(rescue, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _derived_seed(seed: int, *parts: object) -> int:
    text = "\x1f".join([str(seed), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def bootstrap_interval(
    values: Iterable[float],
    statistic: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile interval from resampling independent held experiments."""
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not len(values):
        return math.nan, math.nan, math.nan

    fn = np.mean if statistic == "mean" else np.median
    point = float(fn(values))
    if len(values) < 2:
        return point, math.nan, math.nan

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    # Chunking avoids allocating n_bootstrap x n_experiments for large inputs.
    chunk = max(1, min(n_bootstrap, 1_000_000 // len(values)))
    for start in range(0, n_bootstrap, chunk):
        stop = min(start + chunk, n_bootstrap)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        sampled = values[index]
        estimates[start:stop] = fn(sampled, axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha], method="linear")
    return point, float(low), float(high)


def _scoped_groups(
    data: pd.DataFrame,
    keys: Sequence[str],
    strata: Sequence[str],
):
    """Yield overall groups and, when requested, registry strata."""
    for values, group in data.groupby(list(keys), sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        identity = dict(zip(keys, values))
        yield "overall", identity, {}, group
    if strata:
        group_keys = list(keys) + list(strata)
        for values, group in data.groupby(group_keys, sort=True, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            identity = dict(zip(keys, values[: len(keys)]))
            stratum = dict(zip(strata, values[len(keys) :]))
            yield "stratified", identity, stratum, group


def summarize_l2_detection(
    data: pd.DataFrame,
    l2_label: str = DEFAULT_L2,
    confidence: float = 0.95,
    strata: Sequence[str] = (),
) -> pd.DataFrame:
    l2 = data[data["label"] == l2_label]
    if l2.empty:
        raise ValueError(f"no rows found for L2 label {l2_label!r}")
    rows: list[dict] = []
    group_keys = ["model", "panel", *(["seed"] if "seed" in data else [])]
    for scope, identity, stratum, group in _scoped_groups(
        l2, group_keys, strata
    ):
        detected = int(group["detected"].sum())
        total = len(group)
        low, high = wilson_interval(detected, total, confidence)
        rows.append(
            {
                "scope": scope,
                **identity,
                **{s: stratum.get(s, pd.NA) for s in strata},
                "label": l2_label,
                "n_unique_cells": int(group["held"].nunique()),
                "n_oof_evaluations": int(total),
                "n_detected": detected,
                "detection_rate": detected / total,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def summarize_panel_pairs(
    data: pd.DataFrame,
    l2_label: str = DEFAULT_L2,
    baseline_panel: str = "M1",
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
    strata: Sequence[str] = (),
) -> pd.DataFrame:
    """Matched L2 comparison of M1 with every other panel."""
    l2 = data[data["label"] == l2_label].copy()
    if baseline_panel not in set(l2["panel"]):
        raise ValueError(
            f"baseline panel {baseline_panel!r} has no rows for label {l2_label!r}"
        )
    panels = sorted(set(l2["panel"]) - {baseline_panel})
    rows: list[dict] = []
    for panel in panels:
        pair_source = l2[l2["panel"].isin([baseline_panel, panel])]
        group_keys = ["model", *(["seed"] if "seed" in data else [])]
        for scope, identity, stratum, group in _scoped_groups(
            pair_source, group_keys, strata
        ):
            base = group[group["panel"] == baseline_panel][
                ["held", "detected", "lead"]
            ].rename(columns={"detected": "base_detected", "lead": "base_lead"})
            other = group[group["panel"] == panel][
                ["held", "detected", "lead"]
            ].rename(columns={"detected": "panel_detected", "lead": "panel_lead"})
            paired = base.merge(other, on="held", how="inner", validate="one_to_one")
            paired = paired.sort_values("held", kind="stable").reset_index(drop=True)
            if paired.empty:
                continue

            rescue_mask = paired["panel_detected"] & ~paired["base_detected"]
            lost_mask = paired["base_detected"] & ~paired["panel_detected"]
            both_mask = paired["base_detected"] & paired["panel_detected"]
            rescue, lost = int(rescue_mask.sum()), int(lost_mask.sum())
            detection_delta = (
                paired["panel_detected"].astype(float)
                - paired["base_detected"].astype(float)
            ).to_numpy()
            group_id = [
                identity["model"],
                identity.get("seed", "no-seed"),
                panel,
                scope,
                *(stratum.get(s) for s in strata),
            ]
            diff, diff_low, diff_high = bootstrap_interval(
                detection_delta,
                "mean",
                n_bootstrap,
                confidence,
                _derived_seed(seed, "detection", *group_id),
            )

            both = paired.loc[both_mask].copy()
            lead_delta = (both["panel_lead"] - both["base_lead"]).to_numpy(float)
            lead_delta = lead_delta[np.isfinite(lead_delta)]
            lead, lead_low, lead_high = bootstrap_interval(
                lead_delta,
                "median",
                n_bootstrap,
                confidence,
                _derived_seed(seed, "lead", *group_id),
            )
            rows.append(
                {
                    "scope": scope,
                    **identity,
                    **{s: stratum.get(s, pd.NA) for s in strata},
                    "baseline_panel": baseline_panel,
                    "panel": panel,
                    "label": l2_label,
                    "n_unique_cells": int(paired["held"].nunique()),
                    "n_paired_oof_evaluations": int(2 * len(paired)),
                    "baseline_detected": int(paired["base_detected"].sum()),
                    "panel_detected": int(paired["panel_detected"].sum()),
                    "rescue": rescue,
                    "lost": lost,
                    "mcnemar_exact_p": exact_mcnemar_p(rescue, lost),
                    "detection_rate_difference": diff,
                    "detection_diff_ci_low": diff_low,
                    "detection_diff_ci_high": diff_high,
                    "n_detected_by_both": int(both_mask.sum()),
                    "n_both_with_finite_lead": int(len(lead_delta)),
                    "conditional_lead_delta_median_s": lead,
                    "conditional_lead_delta_ci_low_s": lead_low,
                    "conditional_lead_delta_ci_high_s": lead_high,
                    "bootstrap_unit": "held experiment",
                    "n_bootstrap": n_bootstrap,
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(rows)


def summarize_label_sensitivity(
    data: pd.DataFrame,
    l2_label: str = DEFAULT_L2,
    strata: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare L2 with each label on matched model-panel experiment cohorts."""
    labels = sorted(set(data["label"]) - {l2_label})
    pair_rows: list[dict] = []
    for label in labels:
        source = data[data["label"].isin([l2_label, label])]
        group_keys = ["model", "panel", *(["seed"] if "seed" in data else [])]
        for scope, identity, stratum, group in _scoped_groups(
            source, group_keys, strata
        ):
            base = group[group["label"] == l2_label][["held", "detected"]].rename(
                columns={"detected": "l2_detected"}
            )
            other = group[group["label"] == label][["held", "detected"]].rename(
                columns={"detected": "label_detected"}
            )
            paired = base.merge(other, on="held", how="inner", validate="one_to_one")
            if paired.empty:
                continue
            label_only = paired["label_detected"] & ~paired["l2_detected"]
            l2_only = paired["l2_detected"] & ~paired["label_detected"]
            flips = label_only | l2_only
            pair_rows.append(
                {
                    "scope": scope,
                    **identity,
                    **{s: stratum.get(s, pd.NA) for s in strata},
                    "reference_label": l2_label,
                    "comparison_label": label,
                    "n_unique_cells": int(paired["held"].nunique()),
                    "n_label_evaluations": int(2 * len(paired)),
                    "n_flip": int(flips.sum()),
                    "flip_rate": float(flips.mean()),
                    "comparison_only_detected": int(label_only.sum()),
                    "l2_only_detected": int(l2_only.sum()),
                }
            )

    all_rows: list[dict] = []
    required_labels = [l2_label, *labels]
    group_keys = ["model", "panel", *(["seed"] if "seed" in data else [])]
    for scope, identity, stratum, group in _scoped_groups(
        data, group_keys, strata
    ):
        pivot = group.pivot(index="held", columns="label", values="detected")
        if not set(required_labels).issubset(pivot.columns):
            complete = pivot.iloc[0:0]
        else:
            complete = pivot[required_labels].dropna()
        if len(complete):
            any_flip = complete.nunique(axis=1) > 1
            n_flip = int(any_flip.sum())
            flip_rate = float(any_flip.mean())
        else:
            n_flip = 0
            flip_rate = math.nan
        all_rows.append(
            {
                "scope": scope,
                **identity,
                **{s: stratum.get(s, pd.NA) for s in strata},
                "labels": "|".join(required_labels),
                "n_labels": len(required_labels),
                "n_unique_cells": int(len(complete)),
                "n_label_evaluations": int(len(complete) * len(required_labels)),
                "n_cells_with_any_detection_flip": n_flip,
                "any_flip_rate": flip_rate,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(all_rows)


def summarize_common_cohort_accounting(
    data: pd.DataFrame,
    l2_label: str = DEFAULT_L2,
    strata: Sequence[str] = (),
) -> pd.DataFrame:
    """Count biological cells separately from repeated model evaluations."""
    labels = [l2_label, *sorted(set(data["label"]) - {l2_label})]
    index = ["model", "panel", *(["seed"] if "seed" in data else []), "held"]
    index += list(strata)
    pivot = data.pivot(index=index, columns="label", values="detected")
    if not set(labels).issubset(pivot.columns):
        complete = pivot.iloc[0:0]
    else:
        complete = pivot[labels].dropna()
    records = complete.reset_index()

    def make_row(scope: str, group: pd.DataFrame, stratum: dict) -> dict:
        values = group[labels] if len(group) else group.reindex(columns=labels)
        flips = values.nunique(axis=1) > 1 if len(group) else pd.Series(dtype=bool)
        n_evaluations = len(group)
        return {
            "scope": scope,
            **{s: stratum.get(s, pd.NA) for s in strata},
            "labels": "|".join(labels),
            "n_labels": len(labels),
            "n_unique_cells": int(group["held"].nunique()) if len(group) else 0,
            "n_model_panel_seed_cell_evaluations": int(n_evaluations),
            "n_label_evaluations": int(n_evaluations * len(labels)),
            "n_repeated_evaluations_with_any_flip": int(flips.sum()),
            "repeated_evaluation_flip_rate": (
                float(flips.mean()) if n_evaluations else math.nan
            ),
        }

    rows = [make_row("overall", records, {})]
    if strata and len(records):
        for values, group in records.groupby(list(strata), sort=True, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            stratum = dict(zip(strata, values))
            rows.append(make_row("stratified", group, stratum))
    return pd.DataFrame(rows)


def analyze(
    inputs: Sequence[str | Path],
    registry: str | Path | None = None,
    strata: Sequence[str] = (),
    l2_label: str = DEFAULT_L2,
    baseline_panel: str = "M1",
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return all summary tables and a serializable analysis manifest."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if strata and registry is None:
        raise ValueError("--registry is required when --stratify is used")
    data, accounting = load_v2(inputs)
    if strata:
        data = attach_registry(data, registry, strata)

    l2 = summarize_l2_detection(data, l2_label, confidence, strata)
    panels = summarize_panel_pairs(
        data,
        l2_label,
        baseline_panel,
        n_bootstrap,
        confidence,
        seed,
        strata,
    )
    label_pairs, all_labels = summarize_label_sensitivity(data, l2_label, strata)
    accounting_table = summarize_common_cohort_accounting(data, l2_label, strata)
    tables = {
        "l2_detection": l2,
        "panel_pairs": panels,
        "label_sensitivity": label_pairs,
        "all_label_cohort": all_labels,
        "cohort_accounting": accounting_table,
    }
    manifest = {
        "analysis": "Evaluator V2 held-experiment uncertainty summary",
        "l2_label": l2_label,
        "baseline_panel": baseline_panel,
        "confidence": confidence,
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_seed": seed,
        "strata": list(strata),
        "accounting": accounting,
        "caveats": CAVEATS,
        "outputs": {name: f"{name}.csv" for name in tables},
    }
    return tables, manifest


def _write_readme(path: Path, manifest: dict) -> None:
    accounting = manifest["accounting"]
    lines = [
        "# Evaluator V2 uncertainty summary",
        "",
        f"Input rows: {accounting['input_rows']}",
        f"Deduplicated rows: {accounting['deduplicated_rows']}",
        f"Unique held experiments: {accounting['unique_held_experiments']}",
        (
            "Repeated model-panel-held evaluations: "
            f"{accounting['model_panel_held_evaluations']}"
        ),
        "",
        "## Interpretation limits",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["caveats"])
    lines.extend(
        [
            "",
            "Positive conditional lead differences mean that the comparison "
            "panel has more lead than M1 among experiments detected by both.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis(out_dir: str | Path, tables: dict[str, pd.DataFrame], manifest: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_readme(out / "README.md", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize evaluator-V2 CSVs with held-experiment uncertainty; "
            "this command does not fit models."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="V2 CSV path(s)")
    parser.add_argument("--out-dir", required=True, help="separate output directory")
    parser.add_argument("--registry", help="experiments.csv used for optional strata")
    parser.add_argument(
        "--stratify",
        nargs="*",
        default=[],
        choices=["dataset", "trigger"],
        help="also report registry-derived dataset and/or trigger strata",
    )
    parser.add_argument("--l2-label", default=DEFAULT_L2)
    parser.add_argument("--baseline-panel", default="M1")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables, manifest = analyze(
        inputs=args.inputs,
        registry=args.registry,
        strata=args.stratify,
        l2_label=args.l2_label,
        baseline_panel=args.baseline_panel,
        n_bootstrap=args.bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    write_analysis(args.out_dir, tables, manifest)
    print(
        f"{manifest['accounting']['deduplicated_rows']} rows -> "
        f"{Path(args.out_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
