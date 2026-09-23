"""Recompute non-primary event references from the standardized corpus.

This is the bounded update path for changes to ``onset_L1``, ``onset_L3`` or
``onset_isc``.  It does not rebuild windows, fit a model or touch L2.  The
registry update and its change log are written atomically enough for a local
analysis run (temporary file followed by ``replace``).

Usage::

    python trbench/relabel_event_references.py --datasets ds03_warwick --write
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tr-corpus"
REGISTRY = CORPUS / "registry" / "experiments.csv"
AUDIT = CORPUS / "registry" / "event_reference_relabel_20260915.csv"


def _same(old, new) -> bool:
    return ((pd.isna(old) and new is None) or
            (pd.notna(old) and new is not None and
             np.isclose(float(old), float(new), rtol=0.0, atol=1e-9)))


def recompute(registry: pd.DataFrame, datasets: set[str] | None = None):
    updated = registry.copy()
    changes = []
    for ix, row in registry.iterrows():
        if row.get("role") == "pretrain_only":
            continue
        if datasets and row["dataset_id"] not in datasets:
            continue

        channels = set(str(row.get("channels_present", "")).split(","))
        pch = row.get("L1_source")
        columns = ["time_s"]
        if pd.notna(pch):
            columns.append(str(pch))
        if "V_cell" in channels:
            columns.append("V_cell")
        if len(columns) == 1:
            continue

        frame = pd.read_parquet(CORPUS / row["file"],
                                columns=list(dict.fromkeys(columns)))
        t = frame["time_s"].to_numpy(float)
        trigger = float(row.get("t_trigger", 0.0) or 0.0)
        values = {}
        if pd.notna(pch):
            site = "in_cell" if pch == "P_internal" else "chamber"
            values["t_onset_L1"] = C.onset_L1(
                t, frame[str(pch)].to_numpy(float), trigger, site)
        if "V_cell" in channels:
            voltage = frame["V_cell"].to_numpy(float)
            values["t_onset_L3"] = C.onset_L3(t, voltage, trigger)
            values["t_isc"] = C.onset_isc(t, voltage, trigger)

        for label, new in values.items():
            old = row[label]
            if _same(old, new):
                continue
            updated.at[ix, label] = np.nan if new is None else float(new)
            changes.append({
                "dataset_id": row["dataset_id"],
                "experiment_id": row["experiment_id"],
                "label": label,
                "old_s": old,
                "new_s": np.nan if new is None else float(new),
                "change_s": (np.nan if pd.isna(old) or new is None else
                             float(new) - float(old)),
            })
    return updated, pd.DataFrame(changes)


def _write_csv_atomic(frame: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _patch_registry_atomic(path: Path, changes: pd.DataFrame):
    """Change selected CSV cells without reformatting unrelated float fields."""
    updates = {(r.dataset_id, r.experiment_id, r.label): r.new_s
               for r in changes.itertuples(index=False)}
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header = next(csv.reader([raw_lines[0]]))
    pos = {name: i for i, name in enumerate(header)}
    output = [raw_lines[0]]
    for raw in raw_lines[1:]:
        row = next(csv.reader([raw]))
        key = (row[pos["dataset_id"]], row[pos["experiment_id"]])
        touched = False
        for label in ("t_onset_L1", "t_onset_L3", "t_isc"):
            update = updates.get((*key, label), "__missing__")
            if update == "__missing__":
                continue
            row[pos[label]] = "" if pd.isna(update) else str(float(update))
            touched = True
        if not touched:
            output.append(raw)
            continue
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerow(row)
        output.append(stream.getvalue())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(output), encoding="utf-8", newline="")
    tmp.replace(path)


def _update_counts(registry: pd.DataFrame):
    datasets_path = CORPUS / "registry" / "datasets.csv"
    datasets = pd.read_csv(datasets_path)
    for ix, row in datasets.iterrows():
        sub = registry[registry.dataset_id == row.dataset_id]
        datasets.at[ix, "L1_computable"] = int(sub.t_onset_L1.notna().sum())
        datasets.at[ix, "L3_computable"] = int(sub.t_onset_L3.notna().sum())
    _write_csv_atomic(datasets, datasets_path)

    matrix_path = CORPUS / "registry" / "label_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    for ix, row in matrix.iterrows():
        sub = registry[registry.dataset_id == row.dataset_id]
        n = len(sub)
        matrix.at[ix, "L1"] = "%d/%d" % (sub.t_onset_L1.notna().sum(), n)
        matrix.at[ix, "L3"] = "%d/%d" % (sub.t_onset_L3.notna().sum(), n)
    _write_csv_atomic(matrix, matrix_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="dataset ids to patch from stored standardized traces")
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    before = pd.read_csv(args.registry)
    datasets = set(args.datasets)
    after, changes = recompute(before, datasets)
    if not changes.empty:
        print(changes.to_string(index=False))
    print("%d event-reference values changed" % len(changes))

    if args.write:
        _patch_registry_atomic(args.registry, changes)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        _write_csv_atomic(changes, args.audit)
        if args.registry.resolve() == REGISTRY.resolve():
            _update_counts(after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
