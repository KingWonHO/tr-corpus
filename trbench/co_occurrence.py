"""Channel-group co-occurrence over the measured registry.

Channel availability alone understates the gap: a corpus can carry every
channel somewhere and still support no comparison, because a comparison needs
BOTH channels in the SAME experiment.  This module counts, for every pair of
channel groups, how many measured experiments observe both.

It is the single source for the two registry files and for the manuscript's
figure 1.  (build_corpus.py once computed the same thing inline and rewrote
the files whenever it ran -- including a `--only ds01_bak` rebuild that left
the registry with ds01's 8 experiments instead of all 343.  Recompute from
the frozen registry with:

    uv run python trbench/co_occurrence.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "tr-corpus")
sys.path[:0] = [HERE]
import schema as S  # noqa: E402

GROUPS = {
    "T_surface": S.GROUP_T_SURF, "T_internal": S.GROUP_TINT,
    "T_vent": ["T_vent", "T_vent_2"],
    "P_internal": ["P_internal"], "P_chamber": ["P_chamber", "P_chamber_2"],
    "V": S.GROUP_V, "I": ["I"],
    "gas_speciated": ["gas_H2", "gas_CO", "gas_CO2", "gas_HF", "gas_CH4"],
    "gas_derived": ["gas_total_mol"], "F_expansion": ["F_expansion"],
}


def compute(reg):
    """(matrix, pairs) over measured experiments (role != pretrain_only)."""
    meas = reg[reg["role"] != "pretrain_only"]
    sets = [set(str(c).split(",")) for c in meas["channels_present"].fillna("")]
    dsid = list(meas["dataset_id"])
    names = list(GROUPS)
    have = {g: [any(c in s for c in GROUPS[g]) for s in sets] for g in names}

    matrix = pd.DataFrame(
        [[sum(x and y for x, y in zip(have[a], have[b])) for b in names] for a in names],
        index=pd.Index(names, name="group"), columns=names)

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            hit = [k for k in range(len(sets)) if have[a][k] and have[b][k]]
            pairs.append(dict(group_a=a, group_b=b, n_experiments=len(hit),
                              datasets=",".join(sorted({dsid[k] for k in hit}))))
    pairs = pd.DataFrame(pairs).sort_values(
        ["n_experiments", "group_a", "group_b"]).reset_index(drop=True)
    return matrix, pairs


def main():
    reg = pd.read_csv(os.path.join(OUT, "registry", "experiments.csv"))
    matrix, pairs = compute(reg)
    matrix.to_csv(os.path.join(OUT, "registry", "co_occurrence.csv"))
    pairs.to_csv(os.path.join(OUT, "registry", "co_occurrence_pairs.csv"), index=False)
    n_meas = int((reg["role"] != "pretrain_only").sum())
    print("%d measured experiments; %d pairs, %d never co-observed"
          % (n_meas, len(pairs), int((pairs.n_experiments == 0).sum())))


if __name__ == "__main__":
    main()
