"""Re-score saved evaluator-V2 alarms after event-reference relabelling.

Only ``onset``, ``detected`` and ``lead`` may change.  Model scores, fitted
thresholds, alarm times and false-alarm estimates are copied verbatim, so this
path never fits a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {
    "L2@1.0": "t_onset_L2_1.0",
    "L2@0.5": "t_onset_L2_0.5",
    "L2@2.0": "t_onset_L2_2.0",
    "L1 venting": "t_onset_L1",
    "L3 voltage": "t_onset_L3",
    "ISC 25mV": "t_isc",
}
IMMUTABLE = ["model", "panel", "held", "seed", "tau", "alarm_time",
             "t_trigger", "pre_trigger", "far_check", "risk_max"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescore(frame: pd.DataFrame, registry: pd.DataFrame):
    reg = registry.copy()
    reg["held"] = reg.dataset_id.astype(str) + "/" + reg.experiment_id.astype(str)
    lookup = reg.set_index("held")
    out_rows, audit = [], []

    group_cols = ["model", "panel", "held"]
    if "seed" in frame.columns:
        group_cols.append("seed")
    for key, group in frame.groupby(group_cols, sort=False, dropna=False):
        held = str(group.held.iloc[0])
        if held not in lookup.index:
            out_rows.extend(group.to_dict("records"))
            continue
        reg_row = lookup.loc[held]
        if isinstance(reg_row, pd.DataFrame):
            raise ValueError("duplicate registry key: " + held)
        by_label = {str(r.label): r.copy() for _, r in group.iterrows()}
        template = group.iloc[0].copy()
        for label, column in LABELS.items():
            onset = reg_row.get(column, np.nan)
            old = by_label.get(label)
            if not np.isfinite(onset):
                if old is not None:
                    audit.append({"held": held, "label": label,
                                  "action": "drop", "old_onset": old.onset,
                                  "new_onset": np.nan,
                                  "old_detected": bool(old.detected),
                                  "new_detected": np.nan})
                continue

            row = old.copy() if old is not None else template.copy()
            alarm = row.alarm_time
            pre = bool(row.pre_trigger)
            detected = bool(np.isfinite(alarm) and float(alarm) < float(onset)
                            and not pre)
            lead = float(onset) - float(alarm) if detected else np.nan
            old_onset = old.onset if old is not None else np.nan
            old_detected = bool(old.detected) if old is not None else np.nan
            changed = (old is None or not np.isclose(float(old_onset), float(onset),
                                                      rtol=0.0, atol=1e-9) or
                       bool(old_detected) != detected)
            row["label"] = label
            row["onset"] = float(onset)
            row["detected"] = detected
            row["lead"] = lead
            out_rows.append(row.to_dict())
            if changed:
                audit.append({"held": held, "label": label,
                              "action": "add" if old is None else "update",
                              "old_onset": old_onset,
                              "new_onset": float(onset),
                              "old_detected": old_detected,
                              "new_detected": detected})

    out = pd.DataFrame(out_rows, columns=frame.columns)
    order = {name: i for i, name in enumerate(LABELS)}
    out["_label_order"] = out.label.map(order)
    out = out.sort_values(group_cols + ["_label_order"], kind="stable").drop(
        columns="_label_order").reset_index(drop=True)

    # Threshold, alarm and all saved fit/false-alarm quantities are invariant
    # within each original group and are never recomputed here.
    left = frame.groupby(group_cols, dropna=False)[IMMUTABLE].first().reset_index(drop=True)
    right = out.groupby(group_cols, dropna=False)[IMMUTABLE].first().reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False,
                                  check_exact=True)
    return out, pd.DataFrame(audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("use a fresh output directory: %s" % args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry)
    audits, outputs = [], []
    for path in args.inputs:
        before = pd.read_csv(path)
        after, audit = rescore(before, registry)
        target = args.out_dir / path.name
        after.to_csv(target, index=False)
        audit.insert(0, "input", path.name)
        audits.append(audit)
        outputs.append({"input": str(path), "input_sha256": _sha256(path),
                        "output": target.name, "output_sha256": _sha256(target),
                        "rows_before": len(before), "rows_after": len(after)})
    # An input that is already re-scored against this registry produces no audit
    # rows at all, and the concatenation then has no columns to count flips on.
    audit = pd.concat(audits, ignore_index=True)
    for col in ("input", "held", "label", "action", "old_onset", "new_onset",
                "old_detected", "new_detected"):
        if col not in audit.columns:
            audit[col] = pd.Series(dtype="object")
    audit.to_csv(args.out_dir / "rescore_changes.csv", index=False)
    manifest = {
        "operation": "event-reference-only re-score; no model fitting",
        "registry": str(args.registry),
        "registry_sha256": _sha256(args.registry),
        "inputs": outputs,
        "changed_or_removed_rows": len(audit),
        "status_flips": int((audit.old_detected.fillna(False).astype(bool) !=
                             audit.new_detected.fillna(False).astype(bool)).sum()),
        "invariants": ["model", "tau", "alarm_time", "far_check", "risk_max"],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("%d input rows -> %d output rows; %d changed/removed, %d status flips" %
          (sum(x["rows_before"] for x in outputs),
           sum(x["rows_after"] for x in outputs), len(audit),
           manifest["status_flips"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
