# tr-corpus

Code for *Conditions for comparable thermal-runaway early warning in lithium-ion batteries: a public corpus and evaluation framework* (submitted to eTransportation, 2026). The paper describes the method; this repository lets you rebuild the corpus and rerun the experiments.

The harmonized corpus itself is on Zenodo: [10.5281/zenodo.22908328](https://doi.org/10.5281/zenodo.22908328) (CC BY 4.0).

## Setup

```bash
uv sync            # Python 3.11, pinned by uv.lock
```

## Get the data

Either download the harmonized corpus from Zenodo and unpack it as `tr-corpus/` next to `trbench/`, or rebuild it from the raw sources. The raw datasets are not redistributed; download each from its DOI into `Dataset/<id>/`:

| `Dataset/<id>` | Source | DOI |
|---|---|---|
| `1` | Golubkov, BAK N21700CG-50 at 60 % SOC (Virtual Vehicle) | 10.5281/zenodo.18849418 |
| `2` | Shen, overcharge of prismatic LFP | 10.17632/8zjttd77my.1 |
| `3` | Gulsoy et al., internal temperature and gas pressure (WMG) | 10.17632/rgfhdhcd9k.1 |
| `4` | Kwak et al., multi-modal TR of fresh and aged cells | 10.17605/OSF.IO/C2HNQ |
| `9` | Lin et al., mechanically induced TR (ORNL/Sandia) | 10.17632/sn2kv34r4h.1 |
| `12` | Ohneseit et al., ARC data (KIT), three records | 10.5281/zenodo.14956641, 7707929, 14956635 |
| `15` | Kriston et al., numerical simulation (pretraining only) | 10.17632/9cykcc3svn.2 |
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

Outputs go to `tr-corpus/results/`. The convolution–Transformer does not refit bit-identically on GPU (cuDNN convolution kernels); all other models reproduce their stored thresholds to 1e-9.

## Licence

MIT (`LICENSE`). The corpus and its six source datasets keep their own licences (CC BY 4.0 / CC0).
