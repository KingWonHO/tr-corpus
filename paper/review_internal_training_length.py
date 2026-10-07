"""Recompute S3c summaries from stored six- and twelve-epoch internal fits; no model fitting."""
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
r=root/"tr-corpus/results"
# Generate the supplementary rows from the saved experiment-level results.
old=pd.concat([pd.read_csv(f) for f in (r/'validation_20260925_all7_rescored').glob('e3a_*s0to4.csv')])
sp=pd.read_csv(r/'claims_20260925/lead_support_cells.csv'); span=sp[sp.event_set=='L2@1.0'].drop_duplicates('held').set_index('held').span_s
groups={}; audit=[]
for model in ['gru','mamba']:
 for ep in [6,12]:
  d=old[old.model==model].copy() if ep==6 else pd.read_csv(r/f'review_20261006/epochs_internal/e3a_{model}_ep12_s0to4.csv')
  keys=['model','panel','seed','held','label']; assert not d.duplicated(keys).any()
  base=old[old.model==model]; assert set(map(tuple,d[keys].values))==set(map(tuple,base[keys].values))
  groups[(model,ep)]=d
  for (panel,seed),z in d[d.label=='L2@1.0'].groupby(['panel','seed']):
   assert len(z)==77
   row=dict(model=model,epochs=ep,panel=panel,seed=int(seed),n=77,detected=int(z.detected.sum()),far_median=z.far_check.median(),far_max=z.far_check.max())
   for ell in [60,120]:
    q=z[z.held.map(span)>=ell];row[f'observable_{ell}']=len(q);row[f'met_{ell}']=int((q.detected & (q.lead>=ell)).sum())
   for lab,key in [('L2@0.5','rate05'),('L2@2.0','rate2'),('L3 voltage','L3'),('ISC 25mV','ISC')]:
    q=z.merge(d[(d.panel==panel)&(d.seed==seed)&(d.label==lab)],on='held',suffixes=('_a','_b'))
    row['flips_'+key]=int((q.detected_a!=q.detected_b).sum());row['n_'+key]=len(q)
   audit.append(row)
a=pd.DataFrame(audit)
def ran(v,fmt='d'):
 lo,hi=min(v),max(v)
 def f(x):return format(int(x),'d') if fmt=='d' else format(x,fmt)
 return f(lo) if lo==hi else f(lo)+'--'+f(hi)
metrics=[('M1 detected /77','M1 검출 /77','detected','M1','d'),('M2 detected /77','M2 검출 /77','detected','M2','d'),('M1 check FAR, fold median','M1 점검 FAR, 폴드 중앙값','far_median','M1','.2f'),('M2 check FAR, fold median','M2 점검 FAR, 폴드 중앙값','far_median','M2','.2f'),('M1 check FAR, all folds','M1 점검 FAR, 전체 폴드 범위','far_check','M1','.2f'),('M2 check FAR, all folds','M2 점검 FAR, 전체 폴드 범위','far_check','M2','.2f'),(r'M2$-$M1 detection rate',r'M2$-$M1 검출률','delta',None,'+.3f'),(r'Rate reversals, 0.5 /77',r'상승률 기준 판정 변화, 0.5 /77','flips_rate05',None,'d'),(r'Rate reversals, 2 /77',r'상승률 기준 판정 변화, 2 /77','flips_rate2',None,'d'),('L3 reversals /76','L3 판정 변화 /76','flips_L3',None,'d'),('25 mV reversals /77','25 mV 판정 변화 /77','flips_ISC',None,'d'),('60 s successes /53','60 s 선행경보 성공 /53','met_60',None,'d'),('120 s successes /29','120 s 선행경보 성공 /29','met_120',None,'d')]
rows=[]
for en,ko,key,panel,fmt in metrics:
 vals=[]
 for model,ep in [('gru',6),('gru',12),('mamba',6),('mamba',12)]:
  z=a[(a.model==model)&(a.epochs==ep)];d=groups[(model,ep)]
  if key=='delta':
   v=z.pivot(index='seed',columns='panel',values='detected');v=(v.M2-v.M1)/77
  elif key=='far_check':v=d[(d.label=='L2@1.0')&(d.panel==panel)].far_check
  else:v=z[z.panel==panel][key] if panel else z[key]
  vals.append(ran(v,fmt))
 rows.append((en,ko,vals))

a.to_csv(root/"paper/tables/S3c_internal_training_length_by_seed.csv",index=False)
pd.DataFrame([dict(metric=x[0],GRU6=x[2][0],GRU12=x[2][1],Mamba6=x[2][2],Mamba12=x[2][3]) for x in rows]).to_csv(root/"paper/tables/S3c_internal_training_length.csv",index=False)
