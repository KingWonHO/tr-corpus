# Validation-selected retraining protocol, 2026-10-30

This protocol is frozen before the new model outputs are examined. It is a retrospective reanalysis of previously inspected records, not prospective independent validation.

## Population and roles

Use the frozen W60_native causal inputs and the 2026-10-29 outer five-fold assignments. The source paired cohort has 267 records: 77 adjudicated positives, 173 adjudicated negatives and 17 indeterminate records. All outer test records are predicted; primary metrics use 77/173, with source-supported 37/90 and trace-supported strata reported separately. The seven source-nonTR/L2-discordant records receive a separate negative sensitivity analysis.

Primary model fitting excludes indeterminate records. This is an explicit protocol change, not evidence that the remaining labels are physically verified. Outer calibration and test roles remain as before. For the design split, a subset of the remaining training designs is reserved for checkpoint/configuration validation. Groups are selected using only outcome counts, source-support counts and a deterministic hash. Both fit and validation must contain positive and negative targets. No calibration or test prediction is consulted during model selection.

For the record-split comparison, fit-positive and fit-negative record counts exactly match the corresponding design-split fit counts. The remaining development records validate models; calibration and test records retain the frozen record-wise assignment. This controls fit-record counts, not observation duration, window counts, design composition or validation-population differences. All these sizes will be reported.

## Inputs, targets and fixed comparisons

T is the existing temperature-only panel, including available ambient temperature and availability masks. TV adds only V_cell; neither current nor observation-age channels are included. W=60 s and decision spacing=10 s are held fixed. h=300 s is primary for required leads 60/120 s; h=60 s is a target-mismatch sensitivity, not a model optimized for 120 s warning. The pre-trigger first-alarm failure rule remains unchanged.

Arms: design/h300, design/h60, record-count-matched/h300, source-supported-fit-and-validation/design/h300, and pre-trigger-training-exclusion/design/h300 (TV; XGBoost/Mamba). Source-only fitting reuses the same role assignments and restricts fit and validation; calibration/test remain all adjudicated records for the main comparison. All/source-supported calibration is crossed afterward using the same predictions; test records never choose a threshold. Both all-record and source-supported test metrics are retained.

Models: XGBoost, GRU, Mamba, temperature and temperature-rate rules. All learned final fits use seeds 0–4 and all five outer folds. The extra pre-trigger control uses XGBoost/Mamba only. No architecture is newly proposed.

## Model selection

Sequence models keep the existing 64-dimensional architectures. Maximum 100 epochs, batch size 1024, AdamW, gradient clipping 1.0, three-epoch fixed warm-up, then validation-plateau LR reduction (factor .5, patience 3). Early stopping requires 10 non-improving epochs after at least 10 epochs, min_delta=1e-4. The globally lowest validation-loss checkpoint is restored even when improvements were smaller than min_delta. Validation loss is BCE macro-averaged over records within each target class, then averaged equally over the two target classes. Unknown/censored targets are excluded. Scaling uses fitting data only.

Three fixed candidate configurations are selected on seed 0 validation loss separately for each arm/fold/panel/model: lr=.001 with ratio class weight, lr=.0003 with ratio weight, lr=.001 with sqrt-ratio weight (all weight_decay=.0001). The chosen configuration is frozen for seeds 1–4; each seed independently uses the same validation set for early stopping. Candidate count and losses are saved. This small search does not establish globally optimal hyperparameters.

XGBoost uses CPU hist for reproducible scoring, at most 2000 boosting rounds, early_stopping_rounds=50, validation logloss with the same class/record weights, and candidates (depth, learning rate, class weight) = (5,.08,ratio), (3,.03,ratio), (3,.03,sqrt-ratio). Other settings match the existing implementation. Trees receive the same nine-per-channel window summaries as before.

The epoch-6 checkpoint of each selected sequence trajectory is preserved and scored with its own calibration threshold. This is a controlled within-trajectory checkpoint comparison under the NEW scheduler/fit population, not an exact reproduction of the old six-epoch OneCycleLR experiment. Historical six-epoch results remain archived separately. No test outcome selects the retained checkpoint.

## Evaluation and preservation

Calibration uses negative-record maximum risk and inclusive >= thresholds; budgets 0, 2.5, 5, 10, 15, 20 and 30 percent are swept, primary=10 percent. Report realised FAR, its denominator and monitoring exposure, alongside unconditional timely detection, observable-opportunity denominators, misses, pre-trigger alarms and conditional lead. Rules use the same calibration/test records; no fixed 110.4 C threshold is imported from a different experiment.

Re-score fixed alarms against alternative events and stratify by D1–D3 vs D5 and adjudication evidence. Evaluate evidence-restricted calibration with separate denominators. Preserve all seeds, candidate histories, states, scaling, per-window risks, split membership, operating points, source hashes and replay checks. Seed variation and conditional design-bootstrap uncertainty are separate. New outputs do not overwrite earlier runs, paper figures or manuscript sources until analyses and checks pass.

Reframe the manuscript around the unified selected-model protocol and the conclusions supported by these outputs. Old auxiliary D6/D7/D8, physical-event and sampling evidence remains explicitly identified where retained; old detector performance is not relabelled as new retraining. Do not imply these results repair independent physical labels, missing heating/overcharge negatives, or real-vehicle validation. No remote publication or submission is part of this run.
