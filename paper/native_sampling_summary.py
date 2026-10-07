"""Summarize cached native observations against the manuscript's active registry.

Only records with a current L2 event contribute event-relative statistics.
The per-record native-sampling measurements are independent of model training.
Require unchanged timestamps for retained events: a shifted/new event requires
a new raw-sample audit, not a silent reuse of event-relative measurements.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    cells=pd.read_csv(ROOT/'paper/tables/S2h_native_sampling_cells.csv')
    reg=pd.read_csv(ROOT/'tr-corpus/registry/experiments.csv',low_memory=False)
    data=cells.merge(reg[['dataset_id','experiment_id','t_onset_L2']],on=['dataset_id','experiment_id'],validate='one_to_one')
    assert len(data)==343
    active=data[data.t_onset_L2.notna()].copy()
    assert np.allclose(active.grid_L2_s,active.t_onset_L2,equal_nan=False), 'Native audit needs recomputing at changed event times'
    # A stricter ceiling cannot produce a new native event where the more
    # permissive cached calculation already found none.
    low=active[active.ceiling_used.lt(110)]
    assert low.native1s_L2_s.isna().all(), 'Recompute native timing with eligible surface ceilings'
    rows=[]
    for ds,g in data.groupby('dataset',sort=True):
        a=g[g.t_onset_L2.notna()]
        # Cached absolute timestamps were CSV-rounded to six significant
        # digits; use the separately stored high-resolution time difference.
        delta=a.native1s_L2_minus_grid_s
        row=dict(dataset=ds,n=len(g),n_L2=len(a))
        for c in ('native_dt_median_s','native_dt_p90_s','native_dt_max_s'):
            row[c]=g[c].median()
        for c in ('near_L2_dt_median_s','near_L2_dt_p90_s'):
            row[c]=a[c].median()
        row.update(native1s_L2_minus_grid_L2_median_s=delta.median(),
                   native1s_L2_minus_grid_L2_p90_s=delta.abs().quantile(.9),
                   native1s_L2_minus_grid_L2_max_s=delta.abs().max(),
                   n1s_diff_gt_5s=int((delta.abs()>5).sum()),
                   n1s_native_L2_missing=int(delta.isna().sum()),
                   n_stale_jump_crossings=int(g.causal_n_stale_jump_crossings.sum()),
                   n_stale_jump_at_grid_L2=int(a.stale_jump_at_grid_L2.sum()))
        rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/'paper/tables/S2h_native_sampling.csv',index=False,float_format='%.9g')
    # Inspectable cohort manifest, explicitly tied to the active registry.
    active[['dataset','dataset_id','experiment_id','t_onset_L2','native1s_L2_minus_grid_s','L2_bracket_gap_s']].to_csv(ROOT/'paper/tables/S2h_event_cohort.csv',index=False)
    arc=active[active.dataset=='D6']
    info=dict(records=len(data),L2_events=len(active),native_missing=int(active.native1s_L2_s.isna().sum()),
              shifted_over_5s=int((active.native1s_L2_minus_grid_s.abs()>5).sum()),
              D6_events=len(arc),D6_gaps_over_2s=int((arc.L2_bracket_gap_s>2).sum()),
              D6_gap_median_s=float(arc.L2_bracket_gap_s.median()),D6_gap_max_s=float(arc.L2_bracket_gap_s.max()))
    (ROOT/'paper/tables/S2h_sampling_summary.json').write_text(json.dumps(info,indent=2),'utf-8')
    print(out.to_string(index=False));print(json.dumps(info))

if __name__=='__main__':main()
