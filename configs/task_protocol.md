# Task-specific decomposition and controlled pooling, 2026-10-30

This is a retrospective follow-up on already inspected data. All new results are stored under `tr-corpus/results/task_decomposition_20261030`; earlier runs and manuscripts are unchanged. Existing P5/P7/P10/P17 evidence is inventoried before fitting. A source-classification probe is not a warning model and re-scoring is not refitting.

## Model shortlist

Use the existing P7 inner-validation rows for `p7_nested_L60_h300`, budget 0.10. Rank learned families by mean(tune timely@60 - tune FAR), then timely, FAR and name. This transparent screening utility weights the two event-level rates equally; it is not a universal engineering cost. The four selected families are LightGBM, XGBoost, LSTM and GRU. All four are retained in every feasible new task, without selecting on new test outcomes. Historical validation records are not a new independent model-ranking sample. Temperature-level and maximum one-second rise within the causal 60-second window rules accompany each distinct calibration/test split.

## Data and tasks

Inventory all 343 records: provenance, trigger, scale, chemistry/design, sensors, sampling, endpoint/outcome support, positive/negative/indeterminate and usable windows. D4 contains seven cell and two module tests. Module records are never added to cell training; they are separate transfer cases. D1-D4 lack within-source non-TR trials. Their negative calibration/evaluation uses explicitly identified D5 donor records, and is reported as cross-abuse FAR, never thermal FAR. D6 has 43 L2-timed positives and 13 negatives; nine physical positives without L2 and one indeterminate remain separately inventoried and predicted, not relabelled negative.

Use a shared *maximum available surface-temperature* channel, preserving each source's sensor semantics. It denotes a single probe in D6. No source ID, actuator force/displacement, current, age, or other channel-presence signature is a predictor. The archived causal 60-second window and 10-second decision grid remain fixed. Native observation age is retained only for observability audits. This common-input diagnostic is distinct from a full multimodal comparison.

- D1/D2/D3/D4-cell within-source: three record-held-out folds, a separate validation record subset, remaining source positives for fitting. This evaluates same-source anticipation, not unseen-design generalization. h=300 s; leads 0/10/30/60/120 s reported, 60 s highlighted.
- D5: five design-held-out folds, with fit/validation/calibration/test design separation. h=120 s, 30 s highlighted; 0/10/30/60/120 also reported. Add D1, D2, D3, D4-cell, all heating cells (D1/D3/D4-cell), or all thermal cells (D1-D4-cell) to fitting only. D5 validation/calibration/test are fixed across additions. h=300 s repeats own-only and all-thermal additions as a horizon sensitivity.
- Thermal source-held-out: each of D1, D2, D3 and D4-cell is the target source; another complete donor source is validation and remaining thermal cells fit. Compare thermal-only, +D5 negatives, +D5 positives, +both using the same D5 donor split and fixed thermal validation. h=300 s. Report target-specific results and cross-abuse negative FAR. The two D4 module records are separate transfer cases in these models.
- D6 ARC: five design-held-out folds, h=600 s, leads 60/120/300 s highlighted. Fit within ARC, +thermal cells, and +eligible D5 records, with ARC validation/calibration/test held fixed. All physical positives without L2 receive predicted alarms but no fabricated lead time. Observation support and negative duration accompany results.

## Training and selection

Sequence models: one 64-unit recurrent layer, fixed AdamW learning rate 0.001, sqrt(negative/positive window ratio) loss weight, maximum 100 epochs, minimum ten epochs, patience ten, delta 1e-4; restore the global minimum validation-loss checkpoint. Validation loss equally weights classes and records within each class. Scaling is fit-only. Trees use up to 2,000 rounds, 50-round early stopping and record/class-balanced validation loss; XGBoost depth3/lr0.03, LightGBM depth3/leaves7/lr0.05. Both trees use 0.8 row/feature subsampling; LightGBM samples rows every round. Five seeds 0-4; no new hyperparameter search. Fixed training choices make this a data-composition comparison, not an architecture optimization study.

For long ARC and other records, deterministically cap FIT and VALIDATION at 256 windows per record per target class, spread across the available indices. Full calibration/test trajectories are retained. The same cap applies to every task and addition. No post-event or unknown-censoring window trains the model. No indeterminate record enters fit/validation/calibration. Learning before trigger is permitted; it is a causal observed-history input, and this choice is explicitly fixed.

## Alarms and reporting

Primary task policy starts monitoring at the known experimental trigger once a full causal input window exists. This is a test-protocol start time, not an inferred deployment trigger or a predictor. Both calibration and test use this policy. A secondary archival policy takes the first crossing over the whole record and counts pre-trigger alarms separately. These policies are not conflated with old results.

For each policy, calibrate negative-record maxima at budgets 0/5/10/20%; primary 10%. Ties are not split. No test label or prediction selects thresholds. Report actual FAR and calibration sample size; full monitoring and fixed first-300-second burdens are separate. Minimum lead is a task probe, not an asserted action or regulatory deadline.

Report per-record and per-source timing, misses, late alarms, useful lead conditional on detection, all-positive and executable-opportunity denominators, actual fresh-observation support, outcome evidence, and full/300-second negative exposure. Aggregate seeds within physical records before paired design bootstrap; seed counts do not multiply records. Source-held-out results with one nominal design are descriptive. Pooling effects are paired on unchanged target records; added training size, label prevalence and normalization changes are part of the intervention, not isolated physical causation. Keep all settings, including unfavorable/null results.

## Integrity

Freeze membership, inputs, protocol and code hashes before training. Save states/scalers, validation histories, selected epochs, fit/validation indices and complete validation/calibration/test risks. Replay predictions after state reload; independently re-check split disjointness, threshold budgets, first-alarm accounting and aggregate arithmetic. Record infeasible tasks explicitly instead of weakening their separation or fabricating labels.
