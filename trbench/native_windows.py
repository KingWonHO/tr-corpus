"""Build a causal, full-record window view directly from the raw adapters.

The standardized parquet corpus is useful for inspection, but it is not a
valid source for a strict prospective experiment: its original 1 Hz resampler
uses centred bins or linear interpolation, and long ARC records were cropped
with knowledge of the onset.  This module deliberately bypasses those files.

At grid time ``g`` a channel is formed from information available by ``g``:

* if the trailing interval ``(g - 1, g]`` contains native observations, use
  their mean (anti-aliasing without the centred-bin half-second look-ahead);
* otherwise carry the last native observation at or before ``g``;
* store the time since that last observation as ``age__<channel>``.

Surface maximum and mean are recomputed from these causal member channels.
Windows end every ten seconds from the start of the complete raw record.  No
onset-dependent dense sampling, crop, truncation, interpolation, or
retrospective QC mask is applied.  Labels and split assignments are copied
from the immutable experiment registry, whose times use the same earliest
native observation as zero.  When a current raw acquisition has a longer tail
than that frozen registry, ``label_observed`` is false on the extra windows;
the signal is retained without inventing additional negative follow-up.

The output schema is compatible with :func:`run_e3.load_windows`.  Large
arrays are staged as memory maps, so even the uncropped ARC split does not need
to be materialised in RAM before it is compressed into an NPZ.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "tr-corpus"
sys.path[:0] = [str(HERE)]

import schema as S  # noqa: E402
import make_windows as MW  # noqa: E402


WINDOW_S = 60
GRID_S = 1.0
STRIDE_S = 10
LEAD_HORIZON_S = MW.LEAD_HORIZON_S
AGE_PREFIX = "age__"
MEASURED_DATASETS = tuple(
    ds for ds, info in S.DATASETS.items() if info.get("role") != "pretrain_only"
)
ADAPTERS = {ds: "adapters.%s" % ds for ds in MEASURED_DATASETS}
FEATURES = tuple(MW.FEATURES)
OUTPUT_FEATURES = FEATURES + tuple(AGE_PREFIX + ch for ch in FEATURES)


def age_channel(channel: str) -> str:
    """Return the feature name carrying ``channel`` observation age."""
    return AGE_PREFIX + channel


def _clean_series(t, v):
    """Finite, time-sorted observations with the first duplicate retained."""
    t = np.asarray(t, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    n = min(len(t), len(v))
    t, v = t[:n], v[:n]
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    if not len(t):
        return t, v
    order = np.argsort(t, kind="mergesort")
    t, v = t[order], v[order]
    unique = np.r_[True, np.diff(t) > 0]
    return t[unique], v[unique]


def _safe_experiment_id(value):
    """Apply the registry's filename/id normalization exactly."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def _series_hash(series):
    """Digest the exact finite adapter output consumed by this builder."""
    h = hashlib.sha256()
    for channel in sorted(series):
        h.update(channel.encode("utf-8"))
        h.update(b"\0")
        for array in series[channel]:
            data = np.ascontiguousarray(array, dtype="<f8")
            h.update(str(data.shape).encode("ascii"))
            h.update(b":")
            h.update(memoryview(data).cast("B"))
    return h.hexdigest()


def causal_channel(t, v, grid, bin_s: float = GRID_S):
    """Causally place one native channel on an increasing time grid.

    Returns ``(values, available, age_s)``.  For each grid time ``g``, values
    in the right-closed trailing bin ``(g-bin_s, g]`` are averaged.  When that
    bin is empty, the most recent native value at or before ``g`` is held.
    ``age_s`` always refers to the most recent native observation, including
    when the output value is a mean of several observations.

    The rule depends only on the prefix ending at ``g``.  In particular it
    does not classify a whole record as high- or low-rate using a future-aware
    median sampling interval.
    """
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("bin_s must be finite and positive")
    grid = np.asarray(grid, dtype=np.float64).ravel()
    if len(grid) and (not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0)):
        raise ValueError("grid must be finite and strictly increasing")
    t, v = _clean_series(t, v)
    out = np.full(len(grid), np.nan, dtype=np.float64)
    age = np.full(len(grid), np.nan, dtype=np.float64)
    available = np.zeros(len(grid), dtype=bool)
    if not len(t) or not len(grid):
        return out, available, age

    # side='right' makes the interval exactly (g-bin_s, g]: a sample on the
    # left boundary belongs to the preceding bin and one on g is available.
    right = np.searchsorted(t, grid, side="right")
    left = np.searchsorted(t, grid - bin_s, side="right")
    available = right > 0

    prefix = np.r_[0.0, np.cumsum(v, dtype=np.float64)]
    count = right - left
    has_bin = count > 0
    out[has_bin] = ((prefix[right[has_bin]] - prefix[left[has_bin]]) /
                    count[has_bin])
    held = available & ~has_bin
    out[held] = v[right[held] - 1]
    age[available] = grid[available] - t[right[available] - 1]
    # Tiny negative ages can only be floating-point subtraction at equality.
    age[available] = np.maximum(age[available], 0.0)
    return out, available, age


def _surface_members(channels):
    named = ("T_surface_neg", "T_surface_pos", "T_surface_mid")
    extra = sorted(ch for ch in channels if ch.startswith("T_surface_x"))
    return [ch for ch in named if ch in channels] + extra


def causal_surface_aggregates(values, masks, ages):
    """Add causal ``T_surface_max``/``mean`` and their conservative ages.

    The aggregate age is the oldest member age used at that grid point.  This
    tells a model when any component of a mean or maximum may be stale and does
    not expose clocks from channels outside the aggregate itself.
    """
    members = _surface_members(values)
    if not members:
        return
    vv = np.vstack([np.asarray(values[ch], dtype=np.float64) for ch in members])
    mm = np.vstack([np.asarray(masks[ch], dtype=bool) for ch in members])
    aa = np.vstack([np.asarray(ages[ch], dtype=np.float64) for ch in members])
    present = mm.any(axis=0)
    use = np.where(mm, vv, np.nan)
    use_age = np.where(mm, aa, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx = np.nanmax(use, axis=0)
        mean = np.nanmean(use, axis=0)
        agg_age = np.nanmax(use_age, axis=0)
    for name, data in (("T_surface_max", mx), ("T_surface_mean", mean)):
        values[name] = data
        masks[name] = present.copy()
        ages[name] = agg_age.copy()


def causal_record(raw, grid_s: float = GRID_S):
    """Return one raw adapter record on the causal 1 Hz representation.

    ``time_origin_s`` is the earliest finite observation among channels with
    at least two observations, matching ``common.standardize`` and therefore
    the registry label axis.
    """
    clean = {}
    for ch, pair in raw.series.items():
        t, v = _clean_series(*pair)
        if len(t) >= 1:
            clean[ch] = (t, v)
    # A singleton is a valid past observation, but cannot establish a record
    # clock on its own.  Keeping it in `clean` makes a prefix containing only
    # the first observation invariant to later samples; using only `axis` for
    # t0/t1 preserves the registry's common.standardize convention.
    axis = {ch: pair for ch, pair in clean.items() if len(pair[0]) >= 2}
    if not axis:
        raise ValueError("%s/%s: no usable native channel" %
                         (raw.dataset_id, raw.experiment_id))
    t0 = min(t[0] for t, _ in axis.values())
    t1 = max(t[-1] for t, _ in axis.values())
    duration = float(t1 - t0)
    # This is the same endpoint convention as common.standardize.
    grid = np.arange(0.0, duration + grid_s / 2.0, grid_s, dtype=np.float64)

    values, masks, ages = {}, {}, {}
    for ch, (t, v) in clean.items():
        out, mask, age = causal_channel(t - t0, v, grid, grid_s)
        values[ch], masks[ch], ages[ch] = out, mask, age
    # Always replace adapter-supplied aggregate names with aggregates over the
    # causal member traces.  Extra surface probes participate without becoming
    # output features of their own.
    values.pop("T_surface_max", None)
    values.pop("T_surface_mean", None)
    masks.pop("T_surface_max", None)
    masks.pop("T_surface_mean", None)
    ages.pop("T_surface_max", None)
    ages.pop("T_surface_mean", None)
    causal_surface_aggregates(values, masks, ages)

    f = np.zeros((len(grid), len(OUTPUT_FEATURES)), dtype=np.float32)
    m = np.zeros((len(grid), len(OUTPUT_FEATURES)), dtype=np.uint8)
    for j, ch in enumerate(FEATURES):
        if ch not in values:
            continue
        ok = np.asarray(masks[ch], dtype=bool) & np.isfinite(values[ch])
        f[:, j] = np.where(ok, values[ch], 0.0).astype(np.float32)
        m[:, j] = ok.astype(np.uint8)
        a = len(FEATURES) + j
        age_ok = ok & np.isfinite(ages[ch])
        f[:, a] = np.where(age_ok, ages[ch], 0.0).astype(np.float32)
        m[:, a] = age_ok.astype(np.uint8)
    return (grid, f, m, float(t0), tuple(sorted(clean)),
            _series_hash(clean))


def _window_starts(n: int, window_s: int = WINDOW_S,
                   stride_s: int = STRIDE_S):
    last = n - window_s
    if last < 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, last + 1, stride_s, dtype=np.int64)


def _is_finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _grid_count_from_registry(row, grid_s=GRID_S):
    """Capacity estimate from the uncropped duration saved in the registry."""
    duration = float(row["record_duration_s"])
    return len(np.arange(0.0, duration + grid_s / 2.0, grid_s))


def _registry_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _file_hash(path: Path):
    return _registry_hash(path)


def _git_head():
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True)
        return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read_warwick_table_compat(path):
    """Read ds03's MATLAB table under SciPy's current MatlabOpaque layout.

    SciPy 1.17 exposes the MCOS FileWrapper payload as ``MatlabOpaque.arr``;
    the adapter's older supported layout exposed the identical 14-cell payload
    under ``_ObjectMetadata``.  Only the container access differs.  Channel
    names, test ids and arrays are still interpreted by the canonical adapter.
    """
    import scipy.io as sio
    from scipy.io.matlab._mio5 import MatFile5Reader

    workspace = sio.loadmat(path, variable_names=["__function_workspace__"])
    payload = workspace["__function_workspace__"].tobytes()
    reader = MatFile5Reader(io.BytesIO(payload[8:]), byte_order="<",
                            struct_as_record=True)
    reader.initialize_read()
    header, _ = reader.read_var_header()
    array = reader.read_var_array(header, process=False)
    mcos = array["MCOS"][0, 0]
    if "arr" not in (getattr(mcos.dtype, "names", None) or ()):
        raise ValueError("unsupported Warwick MCOS container")
    cells = mcos["arr"].ravel()[0].ravel()
    names = [str(x[0]) for x in cells[9].ravel()]
    test_ids = [str(x[0]) for x in cells[2].ravel()]
    cols = cells[4].ravel()
    return names, test_ids, cols


def _adapter_module(dataset):
    """Import an adapter, adding only a SciPy-container compatibility shim."""
    module = importlib.import_module(ADAPTERS[dataset])
    if dataset == "ds03_warwick":
        original = module._read_table

        def read_table(path):
            try:
                return original(path)
            except ValueError as exc:
                if "_ObjectMetadata" not in str(exc):
                    raise
                return _read_warwick_table_compat(path)

        module._read_table = read_table
    return module


def _open_memmap(path, dtype, shape, fill=None):
    a = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        a[...] = fill
    return a


def _allocate_split(stage: Path, split: str, capacity: int,
                    window_s: int, key_chars: int, trigger_chars: int):
    """Allocate an on-disk output buffer for one registry split."""
    spec = {
        "X": (np.float32, (capacity, window_s, len(OUTPUT_FEATURES)), 0.0),
        "mask": (np.uint8, (capacity, window_s, len(OUTPUT_FEATURES)), 0),
        "y_time": (np.float32, (capacity,), np.nan),
        "y_event": (np.uint8, (capacity,), 0),
        "t_end": (np.float64, (capacity,), np.nan),
        "trigger": ("<U%d" % max(1, trigger_chars), (capacity,), ""),
        "y_ttl": (np.float32, (capacity,), np.nan),
        "y_pre": (np.uint8, (capacity,), 0),
        "y_tr": (np.uint8, (capacity,), 0),
        "label_observed": (np.uint8, (capacity,), 0),
        "experiment": ("<U%d" % max(1, key_chars), (capacity,), ""),
    }
    arrays = {}
    paths = {}
    for name, (dtype, shape, fill) in spec.items():
        path = stage / ("%s_%s.npy" % (split, name))
        arrays[name] = _open_memmap(path, dtype, shape, fill)
        paths[name] = path
    return {"arrays": arrays, "paths": paths, "cursor": 0,
            "meta": [], "records": []}


def _write_record(buf, row, raw, grid, f, m, t0, native_channels,
                  series_sha256, window_s, stride_s, chunk_windows):
    starts = _window_starts(len(grid), window_s, stride_s)
    n = len(starts)
    lo = buf["cursor"]
    hi = lo + n
    if hi > len(buf["arrays"]["X"]):
        raise RuntimeError("staging capacity underestimated for %s/%s" %
                           (raw.dataset_id, raw.experiment_id))

    key = "%s/%s" % (raw.dataset_id, _safe_experiment_id(raw.experiment_id))
    registry_duration = float(row["record_duration_s"])
    label_end = min(registry_duration, float(grid[-1]))
    registry_trigger = (float(row["t_trigger"])
                        if _is_finite(row["t_trigger"]) else np.nan)
    raw_trigger = (max(0.0, float(raw.t_trigger) - t0)
                   if raw.t_trigger is not None else 0.0)
    trigger_delta = (raw_trigger - registry_trigger
                     if np.isfinite(registry_trigger) else np.nan)
    if np.isfinite(trigger_delta) and abs(trigger_delta) > GRID_S + 1e-7:
        raise ValueError(
            "%s: raw trigger axis differs from registry by %.6f s; registry "
            "labels cannot safely be reused" % (key, trigger_delta))

    onset = float(row["t_onset_L2"]) if _is_finite(row["t_onset_L2"]) else None
    if onset is not None and onset > label_end + GRID_S + 1e-7:
        raise ValueError(
            "%s: registry onset %.6f lies beyond label-observed end %.6f" %
            (key, onset, label_end))

    aa = buf["arrays"]
    offsets = np.arange(window_s, dtype=np.int64)
    for p in range(0, n, chunk_windows):
        q = min(n, p + chunk_windows)
        take = starts[p:q, None] + offsets[None, :]
        aa["X"][lo + p:lo + q] = f[take]
        aa["mask"][lo + p:lo + q] = m[take]

    end_t = grid[starts + window_s - 1]
    observed = end_t <= label_end + 1e-7
    if onset is None:
        # The registry only establishes absence of L2 through label_end.  A
        # longer electrical/mechanical tail remains in X but is right-unknown,
        # rather than silently extending the verified negative duration.
        aa["y_time"][lo:hi] = (label_end - end_t).astype(np.float32)
        aa["y_event"][lo:hi] = 0
        aa["y_ttl"][lo:hi] = np.nan
        aa["y_pre"][lo:hi] = 0
        event = 0
    else:
        ttl = onset - end_t
        aa["y_time"][lo:hi] = ttl.astype(np.float32)
        aa["y_event"][lo:hi] = 1
        aa["y_ttl"][lo:hi] = ttl.astype(np.float32)
        aa["y_pre"][lo:hi] = ((ttl > 0) &
                               (ttl <= LEAD_HORIZON_S)).astype(np.uint8)
        event = 1
    aa["t_end"][lo:hi] = end_t
    aa["trigger"][lo:hi] = str(row["trigger"])
    aa["y_tr"][lo:hi] = event
    aa["label_observed"][lo:hi] = observed.astype(np.uint8)
    aa["experiment"][lo:hi] = key
    buf["cursor"] = hi

    meta = dict(
        key=key, dataset_id=raw.dataset_id,
        experiment_id=_safe_experiment_id(raw.experiment_id), n_windows=n,
        t_onset=onset, t_trigger=row["t_trigger"],
        record_end_s=float(grid[-1]), trigger=row["trigger"],
        split=row["split"], native_time_origin_s=t0,
        label_end_s=label_end,
        registry_crop_start_s=row.get("crop_start_s", np.nan),
        registry_standardized_duration_s=row.get("duration_s", np.nan),
        registry_record_duration_s=registry_duration,
        native_duration_delta_s=float(grid[-1]) - registry_duration,
        n_native_channels=len(native_channels))
    buf["meta"].append(meta)
    record = dict(
        key=key, split=str(row["split"]), raw_time_origin_s=t0,
        native_grid_end_s=float(grid[-1]),
        registry_record_duration_s=registry_duration,
        native_duration_delta_s=float(grid[-1]) - registry_duration,
        label_observed_end_s=label_end,
        registry_crop_start_s=(None if not _is_finite(row.get("crop_start_s"))
                               else float(row["crop_start_s"])),
        registry_t_trigger_s=(None if not np.isfinite(registry_trigger)
                              else registry_trigger),
        recomputed_raw_t_trigger_s=raw_trigger,
        trigger_axis_delta_s=(None if not np.isfinite(trigger_delta)
                              else float(trigger_delta)),
        n_windows=n, n_label_observed_windows=int(observed.sum()),
        native_channels=list(native_channels),
        adapter_series_sha256=series_sha256)
    buf["records"].append(record)


def _save_split(out_dir: Path, stage: Path, split: str, buf,
                window_s: int, grid_s: float):
    n = buf["cursor"]
    if not n:
        raise ValueError("no windows generated for split %s" % split)
    for a in buf["arrays"].values():
        a.flush()
    target = out_dir / (split + ".npz")
    temporary = out_dir / (split + ".npz.tmp")
    arrays = {name: a[:n] for name, a in buf["arrays"].items()}
    with temporary.open("wb") as fh:
        np.savez_compressed(
            fh, **arrays, features=np.asarray(OUTPUT_FEATURES),
            window_s=np.int64(window_s), hz=np.float64(1.0 / grid_s),
            norm=np.asarray("raw_native_causal"))
    os.replace(temporary, target)
    pd.DataFrame(buf["meta"]).to_csv(
        out_dir / (split + "_experiments.csv"), index=False)
    size_mb = target.stat().st_size / 1e6

    # Release mappings before deleting their backing files on Windows.  Relying
    # on garbage collection alone leaves the loop's final memmap referenced and
    # makes its staging file undeletable on that platform.
    arrays.clear()
    maps = list(buf["arrays"].values())
    buf["arrays"].clear()
    for mapping in maps:
        mmap = getattr(mapping, "_mmap", None)
        if mmap is not None:
            mmap.close()
    del maps
    gc.collect()
    for path in buf["paths"].values():
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return {"file": target.name, "n_windows": n,
            "n_experiments": len(buf["meta"]), "size_mb": size_mb}


def _close_buffers(buffers):
    """Flush and close any staging maps that remain after an interrupted run."""
    for buf in buffers.values():
        for mapping in list(buf.get("arrays", {}).values()):
            try:
                mapping.flush()
            except (AttributeError, ValueError):
                pass
            mmap = getattr(mapping, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except (OSError, ValueError):
                    pass
        buf.get("arrays", {}).clear()
    gc.collect()


def _clean_stage(stage: Path, out_dir: Path):
    """Delete only this builder's flat, verified staging directory."""
    if not stage.exists():
        return
    if stage.is_symlink() or stage.parent.resolve() != out_dir.resolve() or \
            stage.name != ".native_staging":
        raise RuntimeError("refusing to clean unverified staging path %s" % stage)
    entries = list(stage.iterdir())
    bad = [p for p in entries if p.is_symlink() or not p.is_file() or
           p.suffix != ".npy"]
    if bad:
        raise RuntimeError("refusing to clean unexpected staging entries: %s" % bad)
    for path in entries:
        path.unlink()
    stage.rmdir()


def _guard_outputs(out_dir: Path, splits, force: bool):
    names = ["manifest.json"]
    for split in splits:
        names.extend([split + ".npz", split + "_experiments.csv"])
    existing = [out_dir / name for name in names if (out_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to replace native outputs without force: %s" %
            ", ".join(str(p) for p in existing))


def build(out_dir=OUT / "windows" / "W60_native", datasets=None,
          splits=("train", "val", "test", "test_zeroshot"),
          window_s=WINDOW_S, grid_s=GRID_S, stride_s=STRIDE_S,
          chunk_windows=2048, force=False):
    """Build the native causal window corpus and return its manifest."""
    if abs(grid_s - 1.0) > 1e-12:
        raise ValueError("current labels/window duration require a 1 Hz grid")
    if window_s <= 0 or stride_s <= 0 or chunk_windows <= 0:
        raise ValueError("window, stride and chunk sizes must be positive")
    datasets = tuple(datasets or MEASURED_DATASETS)
    unknown = set(datasets) - set(MEASURED_DATASETS)
    if unknown:
        raise ValueError("not a measured adapter: %s" % sorted(unknown))
    splits = tuple(dict.fromkeys(splits))
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _guard_outputs(out_dir, splits, force)
    stage = out_dir / ".native_staging"
    if stage.exists():
        _clean_stage(stage, out_dir)
    stage.mkdir()

    registry_path = OUT / "registry" / "experiments.csv"
    reg = pd.read_csv(
        registry_path, dtype={"dataset_id": str, "experiment_id": str})
    reg = reg[reg["dataset_id"].isin(datasets) & reg["split"].isin(splits)].copy()
    if not len(reg):
        raise ValueError("no registry rows for requested datasets/splits")
    reg["key"] = reg["dataset_id"] + "/" + reg["experiment_id"]
    if reg["key"].duplicated().any():
        raise ValueError("duplicate experiment keys in registry")
    by_key = reg.set_index("key", drop=False)

    key_chars = int(reg["key"].str.len().max())
    trigger_chars = int(reg["trigger"].astype(str).str.len().max())
    buffers = {}
    for split in splits:
        rr = reg[reg["split"] == split]
        if not len(rr):
            continue
        expected = sum(len(_window_starts(_grid_count_from_registry(row),
                                          window_s, stride_s))
                       for _, row in rr.iterrows())
        # Three ds09 workbooks currently contain a longer V/F acquisition tail
        # than the registry build exposed (174 additional 10 s endpoints in
        # total).  Keep room for that audited difference while retaining a hard
        # bound: a larger, unreviewed source change must fail rather than resize
        # and copy a multi-gigabyte map implicitly.
        capacity = expected + max(1024, len(rr) * 4)
        buffers[split] = _allocate_split(
            stage, split, capacity, window_s, key_chars, trigger_chars)

    seen = set()
    try:
        for dataset in datasets:
            module = _adapter_module(dataset)
            wanted = set(reg.loc[reg["dataset_id"] == dataset, "key"])
            if not wanted:
                continue
            print("== %s (%d registry records)" % (dataset, len(wanted)))
            for raw in module.load(str(ROOT)):
                safe_id = _safe_experiment_id(raw.experiment_id)
                key = "%s/%s" % (raw.dataset_id, safe_id)
                if key not in wanted:
                    continue
                if key in seen:
                    raise ValueError("adapter yielded duplicate %s" % key)
                row = by_key.loc[key]
                (grid, f, m, t0, native_channels,
                 series_sha256) = causal_record(raw, grid_s)
                _write_record(
                    buffers[str(row["split"])], row, raw, grid, f, m, t0,
                    native_channels, series_sha256, window_s, stride_s,
                    chunk_windows)
                seen.add(key)
                print("  %-52s %7d s  %6d windows" %
                      (key, round(float(grid[-1])),
                       len(_window_starts(len(grid), window_s, stride_s))))
                del grid, f, m
            missing = wanted - seen
            if missing:
                raise ValueError("adapter did not yield registry records: %s" %
                                 sorted(missing))

        missing = set(reg["key"]) - seen
        if missing:
            raise ValueError("missing registry records: %s" % sorted(missing))

        outputs = {}
        records = []
        for split, buf in buffers.items():
            outputs[split] = _save_split(
                out_dir, stage, split, buf, window_s, grid_s)
            records.extend(buf["records"])
            print("%-14s %7d windows / %3d exp  %.1f MB" %
                  (split, outputs[split]["n_windows"],
                   outputs[split]["n_experiments"],
                   outputs[split]["size_mb"]))

        manifest = {
            "format": "trbench-native-causal-v1",
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head": _git_head(),
            "registry": str(registry_path.relative_to(ROOT)).replace("\\", "/"),
            "registry_sha256": _registry_hash(registry_path),
            "builder_source_sha256": _file_hash(Path(__file__)),
            "adapter_source_sha256": {
                dataset: _file_hash(Path(_adapter_module(dataset).__file__))
                for dataset in datasets
            },
            "datasets": list(datasets), "splits": list(buffers),
            "window_s": window_s, "grid_s": grid_s,
            "window_endpoint_stride_s": stride_s,
            "features": list(OUTPUT_FEATURES),
            "changes_from_standardized_corpus": [
                "rebuilt directly from the six measured raw-data adapters",
                "each 1 s value uses only native observations in (t-1,t]",
                "empty trailing bins hold the last native observation at or before t",
                "per-channel observation age is stored as age__<channel>",
                "surface maximum and mean are recomputed from causal member values",
                "complete raw record retained; no onset-dependent crop or post-onset truncation",
                "uniform 10 s window endpoint cadence independent of labels",
                "retrospective detachment/saturation masks are not applied",
                "split, trigger and onset labels copied from the experiment registry",
                "label_observed excludes raw tails beyond registry-verified follow-up",
                "adapter source files and finite adapted series are SHA-256 fingerprinted"
            ],
            "outputs": outputs, "experiments": sorted(records, key=lambda x: x["key"]),
        }
        with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        return manifest
    finally:
        # Keep no anonymous scratch directory after success or failure.  All
        # durable outputs are separately named NPZ/CSV/JSON files.
        _close_buffers(buffers)
        if stage.exists():
            _clean_stage(stage, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT / "windows" / "W60_native"))
    ap.add_argument("--datasets", default=",".join(MEASURED_DATASETS),
                    help="comma-separated measured adapter ids")
    ap.add_argument("--splits", default="train,val,test,test_zeroshot")
    ap.add_argument("--window", type=int, default=WINDOW_S)
    ap.add_argument("--stride", type=int, default=STRIDE_S)
    ap.add_argument("--chunk-windows", type=int, default=2048)
    ap.add_argument("--force", action="store_true",
                    help="replace existing outputs for the selected splits")
    args = ap.parse_args()
    build(out_dir=args.out,
          datasets=tuple(x for x in args.datasets.split(",") if x),
          splits=tuple(x for x in args.splits.split(",") if x),
          window_s=args.window, stride_s=args.stride,
          chunk_windows=args.chunk_windows, force=args.force)


if __name__ == "__main__":
    main()
