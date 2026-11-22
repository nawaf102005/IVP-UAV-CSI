"""Paired per-round acquisition budget curves with a common finite terminal frontier."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t

from aligned_frontier import FINAL_OFFSET
from certificate_work_audit import counts, reset
from dense_alignment_study import final_record, record
from fast_backend import library
from mobility_cost_study import make_case, run_policy
from model import Config
from persistence_campaign import pin

ROOT=Path(__file__).resolve().parents[1]
CELLS={6:(50,171000000,181000000),8:(100,172000000,182000000)}
RULES={'C-IVP':'Predictive-IVP','Sensitivity':'Predictive-Margin','No-report':'Predictive-NR'}
CONTRASTS=(('C-IVP','No-report'),('Sensitivity','No-report'),('C-IVP','Sensitivity'))
GRID=np.arange(11)/400.
NFRAME=512
FAMILY=2*len(GRID)*len(CONTRASTS)
BOOT=5000
TIME_N=128

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def gzip_rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:yield json.loads(line)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def settings(n):
    ms=CELLS[n][0]
    return replace(Config(),n=n,frame_s=ms/1000,ul_bits=30000*ms/20,dl_bits=1800*ms/20)
def case(cfg,seed):
    return make_case(cfg,seed,float(np.exp(-20*cfg.frame_s/10)))

def freeze(out):
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'protocol.json',dict(
        design='S18 budget refinement and independent confirmation. Endpoints of the original S18 curves were known. The intermediate budgets, new seeds, analyses and timing procedure were fixed locally before running them. Original and fresh outputs remain separate.',
        cells={n:dict(T_ms=v[0],original_start=v[1],confirmation_start=v[2]) for n,v in CELLS.items()},
        frames=NFRAME,grid_fractions=GRID.tolist(),complete_reoptimization=True,seed_matching='Same geometry, payload, report errors, execution realization and 4096-draw final validation seed per frame across methods and costs. The two sizes and two seed blocks are disjoint.',
        roles='S18 rows and curves descriptive; original 24-contrast endpoint family retained. Primary intervals from extra 512 independent frames per size, simultaneous two-sided Bonferroni paired t over two sizes x eleven costs x three contrasts (66). No pooling. All costs are incremental per acquisition round; common final frontier excluded. No-report incurs no acquisition charge.',
        mean_crossing='First crossing of the piecewise-linear mean gain starting at zero cost. Linear interpolation only inside the first positive-to-nonpositive bracket. All sampled sign changes retained; no monotonicity assumed. If the curve starts nonpositive, record no positive initial region. If always positive, crossing is right censored.',
        uncertainty='5000 paired frame-bootstrap resamples using the same frame indices at all costs in that cell. Percentile interval for grid-linear first crossing, pointwise per cell/rule. For bootstrap quantiles only, no initial positive region maps to 0 and rights censored map to maximum tested cost; counts and flags retained. No claim of a continuous threshold between tested costs.',
        supported='Largest initial contiguous tested grid prefix with strictly positive simultaneous lower limits starting at zero. No supported prefix if zero-cost lower limit is nonpositive. Never label an interpolated crossing as a supported limit.',
        work='Count all C++ feasible(action,mode,z) calls in the certificate report/stop planner, including stop rounds; exclude the final frontier. Also record scalar forecasts, directional solves and preparations. One feasibility call can have different arithmetic cost at another size. The no-report acquisition workload is zero.',
        timing='After parallel runs: pinned logical CPU, BLAS/OMP single-threaded, original uninstrumented run_policy wall_s. Two randomized repeats per first 128 confirmation identities. At each size evaluate C-IVP at the estimated first crossing, at the largest positive supported budget (when defined), and at a fixed 0.5% budget. Two separate development identities warm up the code. Clock includes certificate-valued acquisition and updates through stopping, excludes common frontier, truth-based scoring and compilation. A separate counted replay of identical decisions yields tests/round. Aggregate wall across repeats divided by aggregate rounds. Speedup = measured acquisition ms/round divided by target ms/round; equivalent throughput = mean calls/round divided by target ms/round. Neither is a deadline guarantee.',
        bootstrap_replicates=BOOT,bootstrap_seed=2026112100,confirmation_family=FAMILY,timing_frames=TIME_N,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        code_sha256={name:sha(ROOT/'code'/name) for name in ['frontier_break_even.py','certificate_work_audit.py','aligned_frontier.py','mobility_cost_study.py','fast_kernel.cpp','dense_alignment_study.py']},
        original_files_sha256={f'N{n}_T{ms}.jsonl.gz':sha(ROOT/'size_confirmation_results'/f'N{n}_T{ms}.jsonl.gz') for n,(ms,_,_) in CELLS.items()}))

def simulate_frame(job):
    n,seed=job;cfg=settings(n);sc,mu,cov=case(cfg,seed);output=[]
    for rule,method in RULES.items():
        for i in ([0] if rule=='No-report' else range(len(GRID))):
            cost=GRID[i]*cfg.frame_s
            if rule!='No-report':reset()
            policy=run_policy(cfg,sc,mu,cov,method,reserve_s=cost,enforce_elapsed=False)
            work=counts() if rule!='No-report' else {}
            terminal=final_record(cfg,sc,mu,cov,policy,cost)
            r=record(terminal,float(GRID[i]),'front')
            r.update(rule=rule,index=i,rounds=0 if rule=='No-report' else policy['probes']+1,
                     scalar_forecasts=len(policy['history'])*(len(sc['candidates']) if rule=='C-IVP' else 1),
                     certificate_service=policy['completed_kbit'],work=work)
            output.append(r)
    return output

def simulate(out,workers):
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['confirmation_family']==FAMILY and not any(out.glob('frames*.gz'))
    library()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n,(ms,old,new) in CELLS.items():
            for block,start in [('S18',old),('confirmation',new)]:
                target=out/f'frames_{block}_{n}.jsonl.gz'
                with gzip.open(target,'wt',compresslevel=3) as fp:
                    for j,rr in enumerate(pool.map(simulate_frame,[(n,s) for s in range(start,start+NFRAME)]),1):
                        for r in rr:fp.write(json.dumps(r,separators=(',',':'))+'\n')
                        if j%64==0:print('Grid',n,block,j,'/',NFRAME,flush=True)

def grid_matrix(groups,rule,field):
    return np.asarray([[a[field] for a in groups[rule,0 if rule=='No-report' else i]] for i in range(len(GRID))],float).T

def first_cross(y,x):
    if y[0]<=0:return None,'no_initial_gain',None
    where=np.flatnonzero(y[1:]<=0)
    if not len(where):return None,'right_censored',None
    j=int(where[0]);lo,hi=y[j:j+2]
    return float(x[j]+(x[j+1]-x[j])*lo/(lo-hi)),'interior',[float(x[j]),float(x[j+1])]

def analyze(out):
    curves=[];crossings=[];workrows=[];critical=t.isf(.025/FAMILY,NFRAME-1)
    for n,(ms,old,new) in CELLS.items():
        for bidx,block in enumerate(('S18','confirmation')):
            groups=defaultdict(list)
            for r in gzip_rows(out/f'frames_{block}_{n}.jsonl.gz'):groups[r['rule'],r['index']].append(r)
            for rr in groups.values():rr.sort(key=lambda r:r['seed']);assert len(rr)==NFRAME
            assert len(groups)==2*len(GRID)+1
            expected=list(range((old,new)[bidx],(old,new)[bidx]+NFRAME))
            assert all([r['seed'] for r in rr]==expected for rr in groups.values())
            z={m:grid_matrix(groups,m,'completed_kbit') for m in RULES}
            x=ms*GRID;rng=np.random.default_rng(2026112100+100*n+bidx)
            indices=rng.integers(NFRAME,size=(BOOT,NFRAME),dtype=np.int32)
            for left,right in CONTRASTS:
                d=z[left]-z[right];mean=d.mean(axis=0);half=critical*d.std(axis=0,ddof=1)/np.sqrt(NFRAME)
                lower=mean-half;upper=mean+half
                for i,alpha in enumerate(GRID):
                    l=groups[left,i];r=groups[right,0 if right=='No-report' else i]
                    curves.append(dict(block=block,n=n,T_ms=ms,index=i,alpha=alpha,cost_ms=x[i],left=left,right=right,
                                       mean=mean[i],lo=lower[i] if bidx else None,hi=upper[i] if bidx else None,
                                       left_service=z[left][:,i].mean(),right_service=z[right][:,i].mean(),
                                       left_outages=sum(a['outage'] for a in l),right_outages=sum(a['outage'] for a in r),
                                       left_reports=sum(a['probes'] for a in l),right_reports=sum(a['probes'] for a in r)))
                if right!='No-report':continue
                value,status,bracket=first_cross(mean,x);support=None;bootlo=boothi=None;freq={}
                if bidx:
                    for i in range(len(GRID)):
                        if lower[i]<=0:break
                        support=float(x[i])
                    stats=[];freq=defaultdict(int)
                    for chunk in np.array_split(indices,50):
                        for bootmean in d[chunk].mean(axis=1):
                            val,flag,_=first_cross(bootmean,x);freq[flag]+=1
                            stats.append(0. if flag=='no_initial_gain' else x[-1] if flag=='right_censored' else val)
                    bootlo,boothi=np.percentile(stats,[2.5,97.5]).tolist()
                crossings.append(dict(block=block,n=n,T_ms=ms,rule=left,mean_cross_ms=value,status=status,bracket_ms=bracket,
                                      grid_sign_changes=int(np.count_nonzero((mean[1:]>0)!=(mean[:-1]>0))),
                                      supported_ms=support,bootstrap_lo=bootlo,bootstrap_hi=boothi,bootstrap_status=dict(freq)))
            for (rule,i),rr in groups.items():
                if rule=='No-report':continue
                rounds=sum(a['rounds'] for a in rr);workrows.append(dict(n=n,block=block,rule=rule,index=i,cost_ms=x[i],frames=NFRAME,
                    total_rounds=rounds,forecasts_per_round=sum(a['scalar_forecasts'] for a in rr)/rounds,
                    **{k+'_per_round':sum(a['work'][k] for a in rr)/rounds for k in rr[0]['work']}))
    write_csv(out/'curves.csv',curves);write_csv(out/'work.csv',workrows);write_json(out/'crossings.json',crossings)
    print('Paired summaries complete',flush=True)

def timing(out):
    assert not (out/'timing_records.csv').exists()
    cpu=pin();library();cross=json.loads((out/'crossings.json').read_text());jobs={}
    for n,(ms,_,_) in CELLS.items():
        result=next(r for r in cross if r['block']=='confirmation' and r['rule']=='C-IVP' and r['n']==n)
        jobs[n]={'fixed_half_percent':ms*.005}
        if result['mean_cross_ms']:jobs[n]['mean_crossing']=result['mean_cross_ms']
        if result['supported_ms'] and result['supported_ms']>0:jobs[n]['supported_grid']=result['supported_ms']
    raw=[];summary=[]
    for n,(ms,_,start) in CELLS.items():
        cfg=settings(n);distinct=sorted(set(jobs[n].values()))
        for seed in (start-991,start-990):
            sc,mu,cov=case(cfg,seed)
            for budget in distinct:run_policy(cfg,sc,mu,cov,RULES['C-IVP'],budget/1000,False)
        for seed in range(start,start+TIME_N):
            sc,mu,cov=case(cfg,seed);order=np.random.default_rng([3821,seed]);stable={}
            for rep in range(2):
                for budget in order.permutation(distinct):
                    p=run_policy(cfg,sc,mu,cov,RULES['C-IVP'],budget/1000,False)
                    signature=(p['observations'],p['selected'],p['mode'],p['probes'])
                    if budget in stable:assert signature==stable[budget]
                    stable[budget]=signature
                    raw.append(dict(n=n,seed=seed,budget_ms=float(budget),rep=rep,rounds=p['probes']+1,wall_ms=1000*p['wall_s']))
        # Separate pass keeps native work counters out of timed decisions.
        by_case={(r['seed'],r['budget_ms']):[] for r in raw if r['n']==n}
        for rr in raw:
            if rr['n']==n:by_case[rr['seed'],rr['budget_ms']].append(rr)
        for seed in range(start,start+TIME_N):
            sc,mu,cov=case(cfg,seed)
            for budget in distinct:
                reset();p=run_policy(cfg,sc,mu,cov,RULES['C-IVP'],budget/1000,False);work=counts()
                for rr in by_case[seed,budget]:
                    assert rr['rounds']==p['probes']+1;rr.update(work)
        for kind,budget in jobs[n].items():
            rs=[r for r in raw if r['n']==n and r['budget_ms']==budget];rounds=sum(r['rounds'] for r in rs)
            wall=sum(r['wall_ms'] for r in rs)/rounds;calls=sum(r['certificate_tests'] for r in rs)/rounds
            summary.append(dict(n=n,T_ms=ms,target=kind,budget_ms=budget,frames=TIME_N,repeats=2,
                      total_rounds_per_pass=rounds//2,acquisition_ms_per_round=wall,certificate_calls_per_round=calls,
                      required_million_calls_s=.001*calls/budget,mean_speedup_to_target=wall/budget))
        print('Matched acquisition timing complete',n,flush=True)
    write_csv(out/'timing_records.csv',raw);write_csv(out/'timing_summary.csv',summary)
    with open('/proc/cpuinfo') as f:cpu_name=next((line.split(':',1)[1].strip() for line in f if line.startswith('model name')),None)
    write_json(out/'timing_host.json',dict(pinned_cpu=cpu,cpu_model=cpu_name,platform=platform.platform(),
        thread_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},settings=jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['freeze','simulate','analyze','timing'],required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.stage=='simulate':simulate(a.out,a.workers)
    else:{'freeze':freeze,'analyze':analyze,'timing':timing}[a.stage](a.out)
