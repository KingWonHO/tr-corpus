"""Package stored evaluation outputs and a reproducible analysis-code snapshot.

Run from a workspace containing the results: python tools/make_results_deposit.py
Use --check to inspect the selection; existing release assets are never replaced.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = '1.2.0'
TREES = {'tr-corpus/results': {'.csv', '.json', '.npy', '.npz'},
         'paper/tables': {'.csv'}, 'trbench': {'.py'}, 'tests': {'.py'}}
ANALYSIS = ['make_figures.py', 'review_controls.py', 'h120_profile.py',
            'matched_far.py', 'case_claims.py', 'case_d7.py', 'case_lin2025.py',
            'd4_label_check.py', 'd8_evidence.py', 'd8_tables.py',
            'evaluation_pool_summary.py', 'mechanism_tables.py',
            'native_sampling_summary.py', 'review_cadence.py', 'review_e2_summary.py',
            'review_internal_training_length.py', 'review_sensitivity.py',
            'review_tables.py', 'timeline_figure.py']
FILES = ['RESULTS_README.md', 'LICENSE-CODE.md', 'pyproject.toml', 'uv.lock',
         'tools/make_results_deposit.py'] + ['paper/' + n for n in ANALYSIS]
FILES += ['paper/figures/F8_timeline_' + p + ext for p in 'abcd' for ext in ['.pdf', '.png']]

def collect(root=ROOT):
    out=[]
    for tree, suffixes in TREES.items():
        base=root/tree
        if not base.is_dir(): raise FileNotFoundError(base)
        for dp,dns,fns in os.walk(base):
            dns[:]=sorted(d for d in dns if d not in {'__pycache__','.pytest_cache','.git'})
            out.extend(Path(dp)/fn for fn in fns if Path(fn).suffix in suffixes and fn != 'test_check_tables.py')
    for f in FILES:
        p=root/f
        if not p.is_file(): raise FileNotFoundError(p)
        out.append(p)
    return sorted(set(out))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--version',default=VERSION)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--output-dir',type=Path)
    a=ap.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+',a.version): ap.error('version must be X.Y.Z')
    root=a.root.resolve(); name='tr-corpus-results-v'+a.version
    dest=a.output_dir or root/'dist'; files=collect(root)
    print(f'{len(files)} files selected',flush=True)
    if a.check:
        print(f'{sum(p.stat().st_size for p in files):,} bytes'); return
    dest.mkdir(parents=True,exist_ok=True)
    names=[name+'.zip',name+'.file_manifest.csv',name+'.SHA256SUMS',name+'.archive.sha256']
    for fn in names:
        if (dest/fn).exists(): raise FileExistsError(dest/fn)
    rows=[]
    with zipfile.ZipFile(dest/names[0],'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for i,p in enumerate(files,1):
            rel=p.relative_to(root).as_posix(); payload=p.read_bytes()
            z.writestr(name+'/'+rel,payload)
            rows.append((rel,len(payload),hashlib.sha256(payload).hexdigest()))
            if i%500==0: print(f'Archived {i}/{len(files)}',flush=True)
    with (dest/names[1]).open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['path','bytes','sha256']); w.writerows(rows)
    (dest/names[2]).write_text(''.join(f'{h}  {rel}\n' for rel,n,h in rows),encoding='utf-8')
    # Read every archived member back and verify the manifest against its actual bytes.
    with zipfile.ZipFile(dest/names[0]) as z:
        assert len(z.namelist())==len(rows)==len(set(z.namelist()))
        for rel,n,h in rows:
            data=z.read(name+'/'+rel)
            assert len(data)==n and hashlib.sha256(data).hexdigest()==h,rel
    checks=[]
    for fn in names[:3]:
        h=hashlib.file_digest((dest/fn).open('rb'),'sha256').hexdigest()
        checks.append(f'{h}  {fn}\n')
    (dest/names[3]).write_text(''.join(checks),encoding='utf-8')
    print(f'Verified all {len(rows)} archive members; ZIP {(dest/names[0]).stat().st_size:,} bytes',flush=True)

if __name__=='__main__': main()
