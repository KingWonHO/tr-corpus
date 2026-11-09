# Data

Nothing in this folder is tracked by git. Download the inputs and place them as below.

```
data/
├── tr-corpus/                  # harmonized corpus, Zenodo 10.5281/zenodo.23181869 (CC BY 4.0)
│   ├── registry/               # experiments.csv (one row per record), outcome_adjudication.csv, ...
│   ├── splits/                 # split_assignment.csv
│   ├── measured/               # harmonized 1 s records, <dataset>/<record>.parquet
│   └── windows/W60_native/     # causal 60 s windows: train/val/test/test_zeroshot .npz + *_experiments.csv
└── raw/
    └── ds12_arc/               # original KIT ARC files: 18650/, 21700/, 4680/ (*.txt, *.EXO)
```

- **tr-corpus**: unpack the Zenodo archive so that `data/tr-corpus/registry/experiments.csv` exists.
- **raw/ds12_arc**: only the native-interval rate rule needs it (`task_comparison`, `robustness`, `cadence`,
  `review`). Download the three KIT ARC records and keep their folder names:
  18650 (10.5281/zenodo.14956641), 21700 (10.5281/zenodo.7707929), 4680 (10.5281/zenodo.14956635).

Other locations can be passed with `--data` and `--raw`.

## Preprocessing

```bash
uv run python main.py --exp preprocess
```

writes the derived inputs of every experiment into `results/preprocess/`:

| Folder | Content |
|---|---|
| `cohort/` | analysis cohort (record outcome, evidence, nominal design) and the design- and record-level outer folds |
| `roles/` | fit / validation / calibration / test memberships of the common-cohort audit, per fold |
| `tasks/` | task cohort, compact surface-temperature inputs (`inputs.npz`), memberships and plans of the task comparisons |

File-by-file descriptions are in `preprocess/README.md`.
