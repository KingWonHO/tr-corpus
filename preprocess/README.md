# Preprocessing outputs

`python main.py --exp preprocess [--steps cohort,roles,tasks]` writes three folders under `results/preprocess/`.
Steps run in this order; `roles` and `tasks` read `cohort`.

## cohort

| file | content |
|---|---|
| `cohort.csv` | one row per record of the six physical sources: registry fields, adjudicated outcome, source verdict, nominal cell design, `paired` (expressible by the M1 and M2 panels), `scored_positive` / `scored_negative`, window-build times |
| `split_assignment.csv` | the 267 paired records with `group_fold` (design-wise outer fold) and `record_fold` (record-wise comparison fold) |
| `split_counts.csv` | records, designs and outcomes per mode, fold and role (test, next-fold calibration, train) |
| `input_manifest.json` | SHA-256 of the registry, adjudication, reconciliation table and window build; window count and features |

## roles

| file | content |
|---|---|
| `roles_group_f{0-4}.csv` | design split: fit, validation (designs chosen from outcome counts only), calibration, test and excluded records |
| `roles_record_f{0-4}.csv` | record split with fit-positive and fit-negative counts matched to the design split |
| `role_counts.csv` | records, designs and outcomes per mode, fold and role |
| `input_manifest.json` | SHA-256 of the protocol, cohort, split assignment and window build; checked by `common_cohort --stage check` |
| `protocol.md` | the common-cohort protocol (`configs/common_cohort_protocol.md`) |

## tasks

| file | content |
|---|---|
| `cohort.csv` | the 343 records with `eligible` (timed positive or negative) |
| `inventory.csv` | records, designs and outcomes per dataset, scale, trigger and outcome |
| `model_selection.csv`, `model_selection.json` | the four learned families shortlisted from the earlier inner-validation rows |
| `memberships/<task>__<arm>__f<fold>.csv` | roles of every record in each of the 88 task plans; the two sodium-ion ARC cells are transfer cases or excluded |
| `plans.csv` | task, arm, fold, horizon, highlighted lead and membership file of each plan |
| `role_counts.csv` | records, outcomes and designs per plan and role |
| `inputs.npz` | per-window `T_surface_max`, its mask and observation age, targets, end times and record keys, including the zero-shot ARC windows |
| `manifest.json` | SHA-256 of the protocol, prior shortlist rows, inputs, cohort, plans and memberships |
| `arc_chemistry_amendment.json` | the sodium-ion rule for the ARC memberships and the resulting Li-ion counts |
| `data_characteristics.csv`, `data_characteristics_summary.csv` | per-record observation support (duration, temperature range, freshness, lead opportunities) and its summary |
| `data_audit.json` | result of the construction checks |
| `protocol.md` | the task-decomposition protocol (`configs/task_protocol.md`) |
