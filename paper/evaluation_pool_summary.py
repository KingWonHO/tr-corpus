"""Build the manuscript evaluation-pool counts from the active registry."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    r=pd.read_csv(ROOT/'tr-corpus/registry/experiments.csv',low_memory=False)
    ds=['ds01_bak','ds02_overcharge','ds03_warwick','ds04_osf','ds09_mech','ds12_arc']
    r=r[r.dataset_id.isin(ds)]
    a=pd.read_csv(ROOT/'tr-corpus/registry/outcome_adjudication.csv')
    r=r.merge(a[['experiment_id','final_assignment']],on='experiment_id',validate='one_to_one')
    l2=r.t_onset_L2.notna();pos=r.final_assignment.eq('positive');neg=r.final_assignment.eq('negative')
    src=r.dataset_id.ne('ds12_arc');external=~src
    paired=r.dataset_id.isin(['ds01_bak','ds02_overcharge','ds03_warwick','ds09_mech'])
    rows=[('Source L2 events',int((src&l2).sum())),('Eligible source positives',int((src&l2&pos).sum())),
          ('Internal paired cohort',int((paired&l2&pos).sum())),('Calibration negatives',int((src&~l2&neg&r.split.eq('val')).sum())),
          ('Check negatives',int((src&~l2&neg&r.split.eq('test')).sum())),('D6 positives',int((external&l2&pos).sum())),
          ('D6 negatives',int((external&~l2&neg).sum())),('D6 outside scored pools',int((external&~((l2&pos)|(~l2&neg))).sum()))]
    out=pd.DataFrame(rows,columns=['pool','records']);out.to_csv(ROOT/'paper/tables/S2i_evaluation_pools.csv',index=False)
    print(out.to_string(index=False))
if __name__=='__main__':main()
