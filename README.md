# tr-corpus

Code for *Evaluating lithium-ion battery thermal-runaway early warning across event definitions, data roles and
test conditions*. The harmonized input data are on Zenodo ([10.5281/zenodo.23181869](https://doi.org/10.5281/zenodo.23181869));
this repository holds only code. Earlier analyses (internal LOEO validation, external cases D7 and D8) are in
release [v1.2.0](https://github.com/KingWonHO/tr-corpus/tree/v1.2.0).

## Folder structure

```
main.py            entry point: one --exp name per step
core/              shared library: windows, survival targets and thresholds, models, fitting, perturbations
preprocess/        Zenodo corpus -> cohort, folds, role memberships, task inputs (README inside)
experiments/       one class per experiment
configs/           fixed inputs from earlier analyses (outcome reconciliation, native sampling, model shortlist, protocols)
data/              downloaded inputs, not tracked (README inside)
results/           outputs, not tracked
```

## Setup

```bash
uv sync                      # Python 3.11, torch 2.11 (CUDA 12.8), XGBoost 3.2, LightGBM 4.7, pinned in uv.lock
```

Place the Zenodo corpus in `data/tr-corpus` and, for the native-rate rule, the KIT ARC files in `data/raw/ds12_arc`
(see `data/README.md`).

## Usage

```bash
uv run python main.py --exp preprocess
uv run python main.py --exp common_cohort
uv run python main.py --exp task_comparison
uv run python main.py --exp robustness          # any of the re-scoring experiments, after task_comparison
uv run python main.py --exp <name> --help       # options of one experiment
```

`preprocess` runs first, then `common_cohort` and `task_comparison`. The other experiments only read the stored
task states and decisions in `results/task_comparison`, and can run in any order. Outputs go to `results/<name>`.

## Experiments

| `--exp` | Analysis | Main outputs (article item) |
|---|---|---|
| `preprocess` | cohort, adjudicated outcomes, design and record folds, four-role memberships, task plans and inputs | `results/preprocess/{cohort,roles,tasks}` |
| `common_cohort` | four-role audit on 77 positives and 173 negatives: XGBoost, GRU, Mamba and two rules under design vs record splits, calibration evidence, 60/300 s horizons, post-trigger fitting; fixed-alarm re-scoring against L3 (`--stage fit,analyze,check`) | `summary.csv` (split, evidence and control tables), `reference_flips.csv` (Fig. 5), `fit_diagnostics.csv`, `per_record.parquet` |
| `task_comparison` | task-specific comparisons (heating/overcharge, indentation, Li-ion ARC) with fitting additions: LightGBM, XGBoost, LSTM, GRU, grid rules and the native-rate rule, 5 folds x 5 seeds (`--stage fit,native_rate,analyze,audit,replay,interpret`) | `primary_summary.csv` (task table, Fig. 4, Table S5), `summary_seed.csv` (Fig. 6), `evidence_strata.csv`, `opportunity_constraints*.csv`, `paired_model_rule.csv`, `per_record.parquet` |
| `alarm_timing` | first-decision alarms, after-crossing successes, false-alarm timing and ARC negative state | `alarm_timing_summary.csv`, `false_alarm_*.csv`, `arc_negative_state.csv` (Fig. 7) |
| `explanation` | feature attribution of the stored task models | `explain_importance.csv` |
| `robustness` | sensor noise, bias, sample-and-hold and delay at stored and recalibrated thresholds, incl. the native-rate rule | `robustness_summary.csv`, `robustness_native_summary.csv`, `latency_summary.csv` |
| `cadence` | stored states re-scored at a 1 s decision cadence | `r3_cadence_1s.csv` |
| `supplements` | requirement bounds, chemistry/format strata, opportunity windows, compute per decision | `requirement_bounds.csv`, `strata_chemistry_format.csv`, `opportunity_windows.csv`, `compute_per_decision.csv` |
| `review` | revision analyses: alternative events (GB 38031-style), combined perturbations, severity, calibration expectation, design-unit bounds, initial temperature, budget area, seeds, model size, bootstrap (`--jobs`) | `r3_*.csv`, `r3_event_variants.json` |
| `refits` | targeted refits: ARC noise augmentation, fresh-sample summaries, no window cap, D1 replay and source balance, D5 window-mean removal (`--variants`) | `retrain_r3_*_seed.csv`, `retrain_r3_d1_replay_check.csv` |

Fits are deterministic (fixed seeds, deterministic cuDNN, CPU trees with 4 threads). Refits of sampled settings
reproduced the article's stored states bit for bit with the pinned environment on an RTX 5080. CPU timings in
`compute_per_decision.csv` depend on the machine.

## Licence

MIT (`LICENSE`). The corpus on Zenodo is CC BY 4.0; the original source datasets keep their own licences.
