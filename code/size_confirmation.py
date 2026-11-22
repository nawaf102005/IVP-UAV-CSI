"""Fresh, fixed follow-up of two historically reported network size and duration slices."""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from dense_alignment_study import final_record, record
from fast_backend import library
from mobility_cost_study import make_case, run_policy
from model import Config
from persistence_campaign import BASE, ci, jsave, write_csv

CELLS=((6,50,171000000),(8,100,172000000))
FRACTIONS=(0.,.025)
FRAMES=512
METHODS=('Predictive-NR','Predictive-IVP','Predictive-Margin')
FAMILY=24

def run_frame(job):
    n,ms,seed=job;scale=ms/20.
    cfg=replace(Config(),n=n,frame_s=ms/1000,ul_bits=30000*scale,dl_bits=1800*scale)
    rho=float(np.exp(-20*cfg.frame_s/10));sc,mu,cov=make_case(cfg,seed,rho);results=[]
    for method in METHODS:
        for f in ([0.] if method==METHODS[0] else FRACTIONS):
            cost=f*cfg.frame_s;p=run_policy(cfg,sc,mu,cov,method,cost,enforce_elapsed=False)
            for kind,row in [('cert',p),('front',final_record(cfg,sc,mu,cov,p,cost))]:
                results.append({**record(row,f,kind),'rule':method,'n':n,'duration_ms':ms})
    return results

def evaluate(out):
    groups=defaultdict(dict);count=0
    for n,ms,start in CELLS:
        with gzip.open(out/f'N{n}_T{ms}.jsonl.gz','rt') as stream:
            for line in stream:
                a=json.loads(line);key=(a['n'],a['duration_ms'],a['rule'],a['kind'],a['alpha'])
                assert a['seed'] not in groups[key] and a['probes']==len(a['observations'])
                assert start<=a['seed']<start+FRAMES and a['n']==n and a['duration_ms']==ms
                groups[key][a['seed']]=a;count+=1
    summary=[]
    for n,ms,start in CELLS:
        for trm in ('cert','front'):
            for f in FRACTIONS:
                for l,r in [(METHODS[1],METHODS[0]),(METHODS[2],METHODS[0]),(METHODS[1],METHODS[2])]:
                    def data(m):return groups[n,ms,m,trm,0. if m==METHODS[0] else f]
                    a,b=data(l),data(r);assert set(a)==set(b)==set(range(start,start+FRAMES))
                    seeds=sorted(a);x=np.array([a[s]['completed_kbit'] for s in seeds]);y=np.array([b[s]['completed_kbit'] for s in seeds])
                    summary.append(dict(n=n,T_ms=ms,terminal=trm,alpha=f,cost_ms=f*ms,left=l,right=r,frames=FRAMES,
                        **ci(x-y,FAMILY),left_kbit=x.mean(),right_kbit=y.mean(),left_outages=sum(v['outage'] for v in a.values()),
                        right_outages=sum(v['outage'] for v in b.values()),left_reports=sum(v['probes'] for v in a.values()),
                        right_reports=sum(v['probes'] for v in b.values())))
    assert count==2*512*10 and len(summary)==FAMILY
    write_csv(out/'size_contrasts.csv',summary)
    jsave(out/'audit.json',dict(frame_method_terminal_records=count,paired_contrasts=FAMILY,independent_frames_per_setting=FRAMES,disjoint_seed_sets=True))

def run(out,workers):
    out.mkdir(parents=True,exist_ok=False)
    jsave(out/'protocol.json',dict(cells=CELLS,frames=FRAMES,alphas=FRACTIONS,charge='incremental per acquisition round; no-report avoids charge; terminal computation excluded',
        seed='Distinct fresh seed blocks per size, paired across three methods, two budgets, two terminals within the size.',
        origin='Reviewed size/duration settings from earlier 128-frame study; selected before this follow-up sample. Separate study from persistence and planning resolution.',
        settings='Fixed 0.75-ms report and 20 m/s, dcor=10m. Payload proportional to duration relative to the 20-ms default. Full original certificate-based one-step forecasts, up to 4 reports. Both terminals score identical acquired histories. Frontier uses separate 4096-draw original final gate.',
        uncertainty='24 simultaneous two-sided Bonferroni paired-t 95% intervals over two cells x two terminals x two per-round costs x three pairwise methods. Radio outages descriptive. No timing, reuse, or dense boundary estimation in this study.',
        methods=METHODS,software={name:hashlib.sha256((BASE/'code'/name).read_bytes()).hexdigest() for name in ['size_confirmation.py','dense_alignment_study.py','mobility_cost_study.py','aligned_frontier.py','fast_kernel.cpp']}))
    library()
    with ProcessPoolExecutor(workers) as pool:
        for n,ms,start in CELLS:
            jobs=[(n,ms,s) for s in range(start,start+FRAMES)]
            with gzip.open(out/f'N{n}_T{ms}.jsonl.gz','wt',compresslevel=4) as stream:
                for idx,rr in enumerate(pool.map(run_frame,jobs,chunksize=1),1):
                    for r in rr:stream.write(json.dumps(r,separators=(',',':'))+'\n')
                    if idx%64==0:print(n,ms,idx,'/',FRAMES,flush=True)
    evaluate(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    evaluate(a.out) if a.audit_only else run(a.out,a.workers)
