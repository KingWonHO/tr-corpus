# TR corpus: stored evaluation results

Evaluation outputs for *Evaluating thermal runaway early warning in lithium-ion
batteries: A framework for lead time, false alarms and validation planning*.

- Results and analysis code: [GitHub release](https://github.com/KingWonHO/tr-corpus/releases/tag/v1.2.0).
- Harmonized input data, labels, adjudicated outcomes and experiment roles:
  [Zenodo, DOI 10.5281/zenodo.23181869](https://doi.org/10.5281/zenodo.23181869).
- Evaluation and build code: [KingWonHO/tr-corpus](https://github.com/KingWonHO/tr-corpus).

The archive contains the stored experiment-level alarms, calibrated thresholds,
available per-window risks, table sources and analysis scripts. It includes all
five seeds of the 120 s-horizon sequence-model runs, the longer-training internal
experiments, the additional sensitivity analyses and the physical-timeline
outputs used in the manuscript. Original source datasets and manuscript files
are not bundled.

## Use the archive

Extract `tr-corpus-results-v1.2.0.zip` and use its top-level directory as the
project root. It includes a snapshot of `trbench/`, `tests/`, `pyproject.toml`
and `uv.lock`, as well as `paper/` analysis scripts. Unpack the Zenodo data
archives into the corresponding `tr-corpus/` subdirectories. For example, the
registry must be at `tr-corpus/registry/experiments.csv` and windows at
`tr-corpus/windows/W60_native/`.

```bash
uv sync --extra paper
uv run python paper/review_tables.py
uv run python paper/review_internal_training_length.py
uv run python paper/h120_profile.py
uv run python paper/timeline_figure.py
```

These commands summarize stored results or draw the physical timeline; they do
not fit new models. The timeline also reads a harmonized measured record from
the Zenodo deposit. External-case reprocessing requires the separately cited
source releases.

## Files supporting the manuscript

Directory names are stable identifiers. Use this map, rather than selecting
the newest file by modification time. Figure and table filenames retain their
analysis identifiers even when manuscript numbering changes.

| Path, relative to the archive root | Contents |
|---|---|
| `tr-corpus/results/validation_20260925_all7_rescored/` | Internal leave-one-experiment-out alarms, re-scored against the reference events. |
| `tr-corpus/results/validation_20260925_e3a_trees/` | Internal tree/rule runs, including per-fold risk traces for matched-FAR analysis and the physical timeline. |
| `tr-corpus/results/validation_20260925_e3a_seq/` | Internal sequence-model alarm records. |
| `tr-corpus/results/validation_20260925_e3a_h120/` | 120 s training horizon: tree/rule results, seeds 0-4. |
| `tr-corpus/results/validation_20260925_e3a_h120_seq/` | 120 s training horizon: four sequence models, seeds 0-4, split between `e3a_seq_h120_s0.csv` and `e3a_seq_h120_s1to4.csv`. |
| `tr-corpus/results/validation_20260925_native/`, `validation_20260925_common_surface/`, `validation_20260925_seq/` | Held-out dataset design: calibration, record-level results and per-window predictions for the input representations. |
| `tr-corpus/results/validation_20260925_shortcut_controls/` | Availability-only and observation-age-only controls. |
| `tr-corpus/results/validation_20260925_split_sensitivity/` | Alternative negative-record allocation and its predictions. |
| `tr-corpus/results/validation_20260925_e3b/`, `summary_20260925_e3b/` | Pressure and gas sensor comparisons. |
| `tr-corpus/results/summary_20260925_all7/`, `summary_20260925_cluster_bootstrap/` | Reference-event, sensor-panel and cluster-bootstrap summaries. |
| `tr-corpus/results/claims_20260925/` | Required-lead support, success profiles, false-alarm evidence and claim assessment. |
| `tr-corpus/results/review_20261006/epochs_internal/` | GRU and Mamba internal-design results at 12 epochs, seeds 0-4. |
| `tr-corpus/results/review_20261006/convergence/` | Held-out sequence-model runs at 6, 12 and 24 epochs, stored predictions and training-loss records. |
| `tr-corpus/results/review_20261006/cadence/` | Evaluation at 1 s window-end intervals. |
| `tr-corpus/results/review_20261006/stored/`, `disputed/`, `disputed_heldout/` | Source-confirmed outcome checks and sensitivity to disputed outcome assignments. |
| `tr-corpus/results/review_20261006/fixed_rules/`, `fixed_rules_calibrated/` | Fixed and calibrated temperature-level baselines. |
| `tr-corpus/results/review_20261006/d8_layout/` | Vehicle-input layout control. |
| `tr-corpus/results/external_20260925_d7_gardner/`, `external_20260925_d8_field/` | External-case summaries, alarms and exposure accounting. |
| `paper/tables/*.csv` | Stored table sources, including `S8h_horizon120_profile*`, `S3c_internal_training_length*` and the worked assessment. |
| `paper/tables/F8_timeline_events.csv`, `paper/figures/F8_timeline_a.pdf` through `F8_timeline_d.pdf` | Event/first-alarm times and the four physical-timeline panels. `F8` is the historical file identifier; these panels are Figure 2 in the current manuscript. |
| `paper/timeline_figure.py`, `paper/review_*.py`, `paper/h120_profile.py`, other included analysis scripts | Generation and summary code for the archived outputs. |

Other result directories retain development and intermediate runs for traceability.
The paths above identify the principal analyses of the manuscript. Do not pool
development runs with the reported evaluation populations.

## Re-scoring and inference

The window-index arrays in stored predictions refer to
`tr-corpus/windows/W60_native/{train,val,test,test_zeroshot}.npz`, concatenated
in that order, unless the run manifest specifies another build. The window
fields `experiment`, `t_end` and `y_tr`, together with the registry event times,
connect risks to records and outcomes. Run manifests record model settings and
input choices. `trbench/survival.py`, `trbench/run_validation.py` and
`trbench/claims.py` implement threshold calibration, first alarms and assessment.

Stored first-alarm records support re-scoring against alternative reference
events and required leads. Threshold changes require per-window risks. These
are included where they were saved (including held-out-design predictions and
internal tree traces); the internal sequence-model alarm tables do not contain
complete per-window risks. New threshold analyses of those models require
inference. Stored outputs are the reference for exact reported results because
fresh GPU fits, particularly the convolution-Transformer, may differ.

## Integrity and licences

`tr-corpus-results-v1.2.0.file_manifest.csv` lists the relative path, size and
SHA-256 digest of every archive member. `tr-corpus-results-v1.2.0.SHA256SUMS`
contains the same per-file digests; run `sha256sum -c` inside the extracted
top-level directory. `tr-corpus-results-v1.2.0.archive.sha256` checks the ZIP
and the two manifest assets from their download directory.

Stored evaluation outputs and timeline figures: CC BY 4.0. Code: MIT (see
`LICENSE-CODE.md`). Source data retain their own licences and attribution
requirements, as documented in the Zenodo deposit.
