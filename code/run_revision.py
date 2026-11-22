"""Frozen revision experiments; disjoint development and evaluation seeds."""
import argparse,json,os,platform,sys,time
from dataclasses import replace,asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from model import Config,scenario
from policies import run_policy
from revised_policies import run_fast
from fast_backend import build

CORE=['C-IVP','A-stop','D-stop','Sensitivity-3pt','GH-KG','No-report','OMA','Nominal','Perfect-CSI']
SMALL=['C-IVP','A-stop','D-stop','Sensitivity-3pt','No-report','OMA']

def both_catalogue(sc,n):
    mu=sc['mean'].reshape(2,n,n);ix=[]
    for d in range(2):
        x=mu[d].copy();np.fill_diagonal(x,-np.inf)
        ix.extend((d*n*n+np.arange(n)*(n+1)).tolist())
        ix.extend((d*n*n+np.arange(n)*n+x.argmax(axis=1)).tolist())
    sc['candidates']=np.unique(ix)

def make_case(label,cfg,seed):
    sc=scenario(cfg,seed);truth_cov=sc['covariance'].copy()
    if label=='sigma_mismatch':
        truth_cov*=25/9
        sc['true']=sc['mean']+(sc['true']-sc['mean'])*5/3
    elif label=='reciprocity_mismatch':
        true_sc=scenario(replace(cfg,reciprocity=.2),seed)
        sc['true']=true_sc['true'];truth_cov=true_sc['covariance']
    elif label=='structure_mismatch':
        n=cfg.n;K=np.eye(n*n)*(cfg.shadow_db*np.log(10)/10)**2
        p=np.arange(n*n).reshape(n,n).T.ravel()
        truth_cov=np.block([[K,.9*K[:,p]],[.9*K[p,:],K[np.ix_(p,p)]]])
        sc['true']=sc['mean']+np.linalg.cholesky(truth_cov)@np.random.default_rng(seed+876).normal(size=2*n*n)
    elif label=='mean_mismatch':
        n=cfg.n;ix=np.r_[np.arange(n)*(n+1),n*n+np.arange(n)*(n+1)]
        sc['true'][ix]-=3*np.log(10)/10
    elif label=='blockage':
        n=cfg.n;blocked=np.random.default_rng(seed+77).random(n)<.1
        x=sc['true'].reshape(2,n,n).copy()
        for i in np.flatnonzero(blocked):x[:,i,i]-=8*np.log(10)/10
        sc['execution_true']=x.ravel()
    if label in ['balanced','DL-heavy','UL-both']:
        both_catalogue(sc,cfg.n)
    sc['truth_covariance']=truth_cov
    return sc

def specification(suite):
    base=Config();settings=[]
    def add(name,cfg,count,start,methods):settings.append((name,cfg,count,start,methods))
    if suite=='smoke':
        add('smoke',base,4,990000,CORE[:4]);return settings
    add('main',base,1024,110000,CORE)
    fixed=[]
    for k in [1,2]:
        for method in ['C-IVP','A-fixed','D-fixed','Random-fixed']:
            fixed.append((f'{method}-{k}',method,dict(fixed_count=k)))
    add('fixed-count',base,1024,110000,fixed)
    for delta in [.01,.025,.05,.10,.20,.40]:
        add(f'delta-{delta:g}',replace(base,outage=delta),512,120000,SMALL)
        add(f'cal-delta-{delta:g}',replace(base,outage=delta),256,125000,['No-report'])
    for cap in [0,1,2,4,8]:add(f'cap-{cap}',replace(base,max_probes=cap),512,130000,['C-IVP','A-stop'])
    for label in ['sigma_mismatch','reciprocity_mismatch','structure_mismatch','mean_mismatch']:
        add(label,base,512,140000,SMALL+[('Oracle-cov','C-IVP',dict(oracle_cov=True))])
    add('blockage',base,1024,150000,SMALL+[('Guarded-IVP','C-IVP',dict(guard_db=8.)),('Guarded-no-report','No-report',dict(guard_db=8.))])
    add('UL-both',base,512,155000,SMALL)
    add('balanced',replace(base,ul_fraction=.5,ul_bits=18000.,dl_bits=18000.,dl_power_w=.2,reciprocity=.2),512,155000,SMALL)
    add('DL-heavy',replace(base,ul_fraction=.2,ul_bits=1800.,dl_bits=30000.,ul_power_w=1.,dl_power_w=.2,reciprocity=.2),512,155000,SMALL)
    for n,count,start in [(6,256,160000),(8,128,170000)]:
        meth=['C-IVP','A-stop','D-stop','Sensitivity-3pt','No-report']
        meth += [(f'C-IVP-C{k}','C-IVP',dict(catalogue_limit=k)) for k in [16,32,64] if k<2**n-1]
        add(f'N-{n}',replace(base,n=n),count,start,meth)
    add('two-step',replace(base,max_probes=2),128,180000,['C-IVP','Two-step','GH-KG','A-stop','Sensitivity-3pt'])
    add('reference',base,128,110000,['C-IVP','IVP-reference'])
    return settings

def one_job(job):
    label,cfg,seed,methods=job;sc=make_case(label,cfg,seed);records=[]
    for spec in methods:
        if isinstance(spec,str):name=method=spec;kwargs={}
        else:name,method,kwargs=spec;kwargs=dict(kwargs)
        source=sc
        if kwargs.pop('oracle_cov',False):source={**sc,'covariance':sc['truth_covariance']}
        if method=='IVP-reference':out=run_policy(cfg,source,'IVP')
        else:out=run_fast(cfg,source,method,**kwargs)
        out.update(scenario=label,method=name,outage_budget=cfg.outage)
        records.append(out)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['smoke','paper'],default='smoke')
    p.add_argument('--output',default='../revision_results');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);build()
    settings=specification(a.suite);jobs=[(name,cfg,s,methods) for name,cfg,count,start,methods in settings for s in range(start,start+count)]
    meta=dict(suite=a.suite,python=sys.version,platform=platform.platform(),workers=a.workers,
              grid_size=25,grid_range=[-6,6],boundary_refinements=8,
              development_seeds='92001--92500 numerical/development; excluded from evaluation',
              snapshots=len(jobs),settings=[dict(name=name,config=asdict(cfg),count=count,seed_start=start,methods=methods) for name,cfg,count,start,methods in settings],
              timing_note='Loaded batch times are diagnostic only. Dedicated sequential benchmark supplies online timing results.')
    (out/'revision_protocol.json').write_text(json.dumps(meta,indent=2));t=time.perf_counter()
    with (out/'revision_records.jsonl').open('w') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,records in enumerate(pool.map(one_job,jobs),1):
            for r in records:f.write(json.dumps(r)+'\n')
            if i%256==0 or i==len(jobs):f.flush();print(i,len(jobs),round(time.perf_counter()-t,2),flush=True)
    meta['elapsed_s']=time.perf_counter()-t;(out/'revision_protocol.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
