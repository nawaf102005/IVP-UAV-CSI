"""Rebuild all registered comparisons from frame records and check archived planner parity."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import sem, t
from mobility_cost_study import make_case
from frontier_tradeoff_campaign import run_one
from persistence_campaign import BASE, CFG

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def records(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)

def verify_interval(values,entry,family):
    values=np.array(values)
    mean=values.mean()
    half=sem(values)*t.isf(.025/family,len(values)-1)
    np.testing.assert_allclose([mean,mean-half,mean+half],[float(entry[k]) for k in ('mean','lo','hi')],rtol=1e-10,atol=1e-10)

def audit(out):
    p=json.loads((out/'protocol.json').read_text())
    for file,digest in p['software'].items():
        assert hashlib.sha256((BASE/'code'/file).read_bytes()).hexdigest()==digest,file
    totals={};nr=0
    for fp in sorted((out/'trajectory_records').glob('*.jsonl.gz')):
        seed=int(fp.name.split('.')[0]);sums=defaultdict(lambda:np.zeros(6));visited=set()
        for v,frame,k,c,rule,idx,y,res,cs,co,cm,ca,fs,fo,fm,fa in records(fp):
            key=(v,k,c,rule,frame);assert key not in visited;visited.add(key)
            report=int(idx>=0);due=bool(k and frame%k==0)
            assert (report==0 or due) and (y is not None)==bool(report)
            if rule=='Fixed-IVP':assert report==int(due)
            assert abs(res-(1+report)*c*due)<1e-9
            assert ca in ('reuse','OMA','empty') and fa in ('reuse','OMA','empty')
            assert (cm>0)==(ca!='empty') and (fm>0)==(fa!='empty')
            assert 0<=cs and 0<=fs and co in (0,1) and fo in (0,1)
            sums[v,k,c,rule]+=[report,cs,co,fs,fo,1]
        assert len(visited)==3*40*25
        nr+=len(visited)
        for (v,k,c,rule),value in sums.items():
            assert value[-1]==40
            totals[seed,v,k,c,rule]=value
    sets=sorted({k[0] for k in totals})
    for r in rows(out/'trajectory_means.csv'):
        value=totals[int(r['seed']),int(r['speed']),int(r['K']),float(r['cost_ms']),r['rule']]
        np.testing.assert_allclose(value[:5],[float(r['reports']),40*float(r['cert_kbit']),float(r['cert_outages']),40*float(r['front_kbit']),float(r['front_outages'])])
    reuse=rows(out/'reuse_contrasts.csv')
    for r in reuse:
        k=int(r['K']);v=int(r['speed']);c=float(r['cost_ms']);col=1 if r['terminal']=='cert' else 3
        def fetch(s,m):return totals[s,v,0 if m=='No-report' else k,0 if m=='No-report' else c,m][col]/40
        verify_interval([fetch(s,r['left'])-fetch(s,r['right']) for s in sets],r,192)
    host=defaultdict(list);hf=0
    for seed,tick,k,ix,y,wall,res,rad,net,outage,late,active in records(out/'host_frames.jsonl.gz'):
        assert int(wall>res)==late and np.isclose(net,0. if late else rad)
        assert (ix==-1)==(y is None)
        assert ix<0 or (k and tick%k==0)
        host[seed,k].append(net);hf+=1
    hs=sorted({s for s,_ in host});assert all(len(x)==40 for x in host.values())
    for r in rows(out/'host_contrasts.csv'):
        verify_interval([np.mean(host[s,int(r['K'])])-np.mean(host[s,0]) for s in hs],r,2)
    bycell=defaultdict(dict);profile_frames=0
    for r in records(out/'profile_frames.jsonl.gz'):
        b=r['budget_ms'];late=bool(b and r['wall_ms']>b)
        assert r['deadline_miss']==late and r['active_overrun']==(r['active'] and late)
        assert r['net_kbit']==(0. if late else r['radio_kbit'])
        assert r['reports']==len(r['observations'])<=4
        key=r['draws'],r['nodes'],b
        assert r['seed'] not in bycell[key]
        bycell[key][r['seed']]=r;profile_frames+=1
    for r in rows(out/'profile_summary.csv'):
        l,j,b=int(r['draws']),int(r['nodes']),float(r['budget_ms'])
        d=bycell[l,j,b];base=bycell[0,0,b]
        assert len(d)==len(base)==128
        verify_interval([d[s]['net_kbit']-base[s]['net_kbit'] for s in sorted(d)],r,18)
        np.testing.assert_allclose(float(r['p95_wall_ms']),np.percentile([a['wall_ms'] for a in d.values()],95))
    old=BASE/'dense_alignment_results';oldprotocol=json.loads((old/'protocol.json').read_text());first=oldprotocol['seed_start'];checked=0
    for r in records(old/'records.jsonl.gz'):
        if r['method']!='Frontier-aware-IVP' or r['alpha']!=0 or not first<=r['seed']<first+12:continue
        sc,m,P=make_case(CFG,r['seed'],oldprotocol['rho']);z=run_one(sc,m,P,(4096,49,0.))
        for a,b in [('observations','observations'),('history','history'),('selected','selected'),('mode','mode'),
                    ('validation_bad','validation_bad'),('probes','reports'),('outage','radio_outage'),('completed_kbit','net_kbit')]:
            assert r[a]==z[b],(r['seed'],a)
        checked+=1
    assert (nr,hf,profile_frames,len(sets),len(hs),len(reuse),checked)==(384000,7680,2560,128,64,192,12)
    summary=dict(reuse_frame_policy_records=nr,host_frame_policy_records=hf,profile_frame_policy_records=profile_frames,
                 independently_reconstructed_reuse_intervals=192,independently_reconstructed_host_intervals=2,
                 independently_reconstructed_profile_intervals=18,archived_full_frontier_histories_exactly_reproduced=checked,
                 protocol_source_hashes_match=True)
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args().out)
