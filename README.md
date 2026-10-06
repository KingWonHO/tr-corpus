# tr-corpus

Code for *Conditions for comparable thermal-runaway early warning in lithium-ion batteries: a public corpus and evaluation framework* (submitted to eTransportation, 2026). The paper describes the method; this repository lets you rebuild the corpus and rerun the experiments.

The harmonized corpus is on Zenodo: [10.5281/zenodo.23181869](https://doi.org/10.5281/zenodo.23181869) (version 1.1.0, CC BY 4.0; version 1.0.0: [10.5281/zenodo.22908328](https://doi.org/10.5281/zenodo.22908328)). From version 1.1.0 the Zenodo deposit holds **data only**; all code, including the corpus builder, is here, and the stored outputs of the reported runs are attached to this repository's releases.

## Setup

```bash
uv sync            # Python 3.11, pinned by uv.lock
```

## Get the data

Either download the harmonized corpus from Zenodo and place its `tr-corpus/` folder next to `trbench/`, or rebuild it from the raw sources. The raw datasets are not redistributed; download each from its DOI into `Dataset/<id>/`:

| `Dataset/<id>` | Source | DOI |
|---|---|---|
| `1` | Golubkov, BAK N21700CG-50 at 60 % SOC (Virtual Vehicle) | 10.5281/zenodo.18849418 |
| `2` | Shen, overcharge of prismatic LFP | 10.17632/8zjttd77my.1 |
| `3` | Gulsoy et al., internal temperature and gas pressure (WMG) | 10.17632/rgfhdhcd9k.1 |
| `4` | Kwak et al., multi-modal TR of fresh and aged cells | 10.17605/OSF.IO/C2HNQ |
| `9` | Lin et al., mechanically induced TR (ORNL/Sandia), Mendeley Data V2 | 10.17632/sn2kv34r4h.2 |
| `12` | Ohneseit et al., ARC data (KIT), three records | 10.5281/zenodo.14956641, 7707929, 14956635 |
| `15` | Kriston et al., numerical simulation (pretraining only; not used by any reported analysis and not in the Zenodo v1.1.0 deposit) | 10.17632/9cykcc3svn.2 |
| `16_gardner` | Gardner et al., trace-H2 mitigation (external case D7) | 10.6084/m9.figshare.28677122.v2 |
| `BMS/Tsinghua`, `BMS/{DTI,GIS,QAS}` | Zhang et al.; Cao et al., EV field data (external case D8) | 10.6084/m9.figshare.23659323; 10.5281/zenodo.10656500 |

```bash
uv run python trbench/build_corpus.py      # adapters -> harmonized 1 s records + registry
uv run python trbench/make_splits.py       # frozen experiment roles
uv run python trbench/make_windows.py      # W60_native window builds
uv run python trbench/validate_corpus.py   # schema, time-axis and label checks
```

`trbench/claims.py` pins `registry/experiments.csv` and `splits/split_assignment.csv` by SHA-256, so a rebuild that matches those hashes reproduces the paper's inputs exactly.

## Run the experiments

```bash
uv run python -m pytest -q tests
uv run python trbench/run_v2.py                    # internal leave-one-experiment-out design
uv run python trbench/rescore_fixed_alarm.py       # re-score the issued alarms under every reference event
uv run python trbench/summarize_v2.py
uv run python trbench/run_validation.py            # held-dataset design, seven models, seeds 0-4
uv run python trbench/cluster_bootstrap.py         # cell-model bootstrap
uv run python trbench/run_split_sensitivity.py     # alternative negative allocation
uv run python trbench/claims.py all                # required lead and false-alarm evidence
uv run python trbench/external_d7.py events; uv run python trbench/external_d7.py alarms
uv run python trbench/external_d8.py index; uv run python trbench/external_d8.py accounting; uv run python trbench/external_d8.py score
```

Outputs go to `tr-corpus/results/`.

### Sensitivity analyses for the external review (October 2026)

```bash
uv run python trbench/native_windows.py --out tr-corpus/windows/W60_native_s1 --splits train,val,test --stride 1
uv run python trbench/run_review_20261006.py cadence --models rule,xgboost,lightgbm --out tr-corpus/results/review_20261006/cadence/e3a_trees_cadence_s0to4.csv
uv run python trbench/run_review_20261006.py disputed --models rule,xgboost,lightgbm --out tr-corpus/results/review_20261006/disputed/e3a_trees_disputed_s0to4.csv
uv run python trbench/run_review_20261006.py disputed-heldout --models rule,xgboost,lightgbm,gru,mamba,itransformer,convtransformer --out tr-corpus/results/review_20261006/disputed_heldout
uv run python trbench/run_convergence_20261006.py --out tr-corpus/results/review_20261006/convergence
uv run python trbench/run_epochs_internal_20261006.py --epochs 12 -- --arm e3a --models gru --windows W60_native --without-age --seeds 0,1,2,3,4 --out tr-corpus/results/review_20261006/epochs_internal/e3a_gru_ep12_s0to4.csv
uv run python trbench/run_fixed_rules_20261006.py --out tr-corpus/results/review_20261006/fixed_rules            # add --calibrated for the 10 % budget
uv run python trbench/run_d8_layout_control.py
```

`cadence` scores the held cell on windows ending every 1 s with models fitted exactly as the stored internal run; `disputed` moves the two source-stated non-runaway test records that satisfy L2 into the check negatives; `run_convergence` refits the sequence probes for 6, 12 and 24 epochs with per-epoch loss logging; `run_fixed_rules` scores fixed and calibrated surface-temperature thresholds under the same evaluation; `run_d8_layout_control` holds the D8 target tensor fixed while the source representation changes. These are exploratory analyses run after the evaluation design was fixed.

The convolution–Transformer does not refit bit-identically on GPU (cuDNN convolution kernels); all other models reproduce their stored thresholds to 1e-9.

## Licence

MIT (`LICENSE`). The corpus and its six source datasets keep their own licences (CC BY 4.0 / CC0).
