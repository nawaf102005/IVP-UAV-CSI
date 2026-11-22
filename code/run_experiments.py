"""Run the frozen synthetic experiment suite; never invent unavailable results.

Usage: python run_experiments.py --suite paper --output ../results
       python run_experiments.py --suite smoke --output ../smoke_results
"""
import argparse, json, os, platform, sys, time
from pathlib import Path
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy
from model import Config,scenario
from policies import run_policy,METHODS


def jobs_for_suite(suite):
    base=Config()
    if suite=='smoke':
        return [('smoke',base,s,METHODS) for s in range(90000,90004)]
    settings=[('main',base,256,10000,METHODS+['A-opt-fixed','IVP-diagonal'])]
    for spacing in [120.,160.,240.]:
        settings.append((f'spacing_{int(spacing)}',replace(base,spacing_m=spacing),96,20000,METHODS))
    for overhead in [.00025,.0015]:
        settings.append((f'overhead_{overhead*1000:g}ms',replace(base,probe_s=overhead),96,30000,METHODS))
    for reciprocity in [.2,.6]:
        settings.append((f'reciprocity_{reciprocity:g}',replace(base,reciprocity=reciprocity),96,40000,METHODS))
    settings += [('six_uav',replace(base,n=6),64,50000,METHODS),
                 ('shadow_5dB',replace(base,shadow_db=5.),96,60000,METHODS),
                 ('low_load',replace(base,ul_bits=13000.),96,70000,METHODS),
                 ('post_report_blockage',base,128,80000,METHODS)]
    return [(label,cfg,seed,methods) for label,cfg,count,start,methods in settings for seed in range(start,start+count)]


def one_job(job):
    label,cfg,seed,methods=job
    sc=scenario(cfg,seed)
    if label=='post_report_blockage':
        rng=np.random.default_rng(seed+77)
        extra=np.zeros((2,cfg.n,cfg.n))
        blocked=rng.random(cfg.n)<.1
        for i in np.flatnonzero(blocked):extra[:,i,i]=-8*np.log(10)/10
        sc['execution_true']=sc['true']+extra.ravel()
    records=[]
    for method in methods:
        out=run_policy(cfg,sc,method)
        out['scenario']=label
        records.append(out)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['paper','smoke'],default='smoke')
    p.add_argument('--output',default='../results');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    jobs=jobs_for_suite(args.suite)
    metadata=dict(suite=args.suite,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),machine=platform.machine(),workers=args.workers,
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  scenarios={label:asdict(cfg) for label,cfg,seed,methods in jobs},
                  jobs=len(jobs),note='Independent synthetic snapshots. No external channel measurements. CPU time excluded from air-interface frame feasibility.')
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2))
    before=time.time()
    with (out/'records.jsonl').open('w') as f,ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,records in enumerate(pool.map(one_job,jobs),1):
            for rec in records:f.write(json.dumps(rec)+'\n')
            f.flush()
            if i%32==0 or i==len(jobs):print(f'{i}/{len(jobs)} snapshots, elapsed {time.time()-before:.1f} s',flush=True)
    metadata['elapsed_s']=time.time()-before
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
