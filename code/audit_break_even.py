"""Independently rescore the released frontier, endpoint matches, paired bands and work targets."""
import argparse
import csv
import gzip
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
import numpy as np
from scipy.stats import sem,t
from admission import execute
from aligned_frontier import acceptance_count
from mobility_cost_study import make_case
from model import Config

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def records(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)
def verify(out):
    root=Path(__file__).resolve().parents[1];pr=json.loads((out/'protocol.json').read_text())
    for file,digest in pr['code_sha256'].items():assert hashlib.sha256((root/'code'/file).read_bytes()).hexdigest()==digest
    store={};tot=matches=0;rounds=defaultdict(int);refwork=defaultdict(lambda:np.zeros(2))
    names={'Predictive-NR':'No-report','Predictive-IVP':'C-IVP','Predictive-Margin':'Sensitivity'}
    grid=np.array(pr['grid_fractions'])
    for ns,cell in pr['cells'].items():
        n=int(ns);ms=cell['T_ms'];cfg=replace(Config(),n=n,frame_s=ms/1000,ul_bits=30000*ms/20,dl_bits=1800*ms/20)
        refpath=root/'size_confirmation_results'/f'N{n}_T{ms}.jsonl.gz';assert hashlib.sha256(refpath.read_bytes()).hexdigest()==pr['original_files_sha256'][refpath.name]
        archived={(z['seed'],names[z['rule']],round(z['alpha']*400)):z for z in records(refpath) if z['kind']=='front'}
        for block in ('S18','confirmation'):
            lower=cell['original_start' if block=='S18' else 'confirmation_start'];last=None;seen=set()
            for r in records(out/f'frames_{block}_{n}.jsonl.gz'):
                seed,rule,i=r['seed'],r['rule'],r['index'];m=r['probes'];k=(seed,rule,i)
                assert lower<=seed<lower+512 and k not in seen;seen.add(k)
                if last!=seed:sc=make_case(cfg,seed,np.exp(-20*cfg.frame_s/10))[0];last=seed
                assert m==len(r['observations']) and r['rounds']==(0 if rule=='No-report' else m+1)
                c=cfg.frame_s*grid[i];assert abs(r['reserved_s']-(m+1)*c)<1e-12
                a=dict(group=np.array(r['selected'],int),mode=r['mode'],power=np.asarray(r['power']),value=r['admitted_kbit'])
                if r['mode']=='OMA':a['airtime']=np.asarray(r['airtime'])
                score=execute(replace(cfg,frame_s=cfg.frame_s-(m+1)*c),sc,a,m)
                assert abs(score['completed_kbit']-r['completed_kbit'])<1e-8 and score['outage']==r['outage']
                if r['active']:assert 0<=r['validation_bad']<=acceptance_count(n)
                if block=='S18' and k in archived:
                    old=archived[k]
                    for field in ['observations','history','selected','mode','power','probes','validation_bad','completed_kbit','outage']:
                        assert r[field]==old[field],(n,seed,i,field)
                    matches+=1
                if rule!='No-report':
                    assert r['scalar_forecasts']==(len(sc['candidates']) if rule=='C-IVP' else 1)*len(r['history'])
                    assert r['work']['certificate_tests']>0
                    refwork[block,n,rule,i]+=[r['work']['certificate_tests'],r['rounds']]
                store[block,n,seed,rule,i]=r['completed_kbit'];tot+=1
            assert len(seen)==512*(2*len(grid)+1)
    cs=read_csv(out/'curves.csv');chk={}
    for r in cs:
        b,n,i,le,ri=r['block'],int(r['n']),int(r['index']),r['left'],r['right']
        start=pr['cells'][str(n)]['original_start' if b=='S18' else 'confirmation_start']
        vals=np.array([store[b,n,s,le,i]-store[b,n,s,ri,0 if ri=='No-report' else i] for s in range(start,start+512)])
        mean=vals.mean();assert abs(mean-float(r['mean']))<1e-10
        low=None
        if b=='confirmation':
            half=t.ppf(1-.025/66,511)*sem(vals);low=mean-half
            np.testing.assert_allclose([float(r['lo']),float(r['hi'])],[low,mean+half],atol=1e-10)
        elif r['lo'] or r['hi']:raise AssertionError('archive labelled as confirmatory')
        chk[b,n,le,ri,i]=(mean,low)
    boundaries=json.loads((out/'crossings.json').read_text())
    for q in boundaries:
        b,n,l=q['block'],q['n'],q['rule'];x=q['T_ms']*grid
        rows=[chk[b,n,l,'No-report',j] for j in range(len(grid))]
        s=[r[0] for r in rows];j=next((i for i,m in enumerate(s) if m<=0),len(s))
        if 0<j<len(s):
            val=(x[j]*s[j-1]-x[j-1]*s[j])/(s[j-1]-s[j]);assert abs(val-q['mean_cross_ms'])<1e-10
        else:assert q['mean_cross_ms'] is None
        if b=='confirmation':
            fail=next((i for i,r in enumerate(rows) if r[1]<=0),len(rows));assert q['supported_ms']==(None if not fail else x[fail-1])
            assert sum(q['bootstrap_status'].values())==pr['bootstrap_replicates']
    for x in read_csv(out/'work.csv'):
        w,rs=refwork[x['block'],int(x['n']),x['rule'],int(x['index'])]
        assert rs==int(x['total_rounds']) and abs(w/rs-float(x['certificate_tests_per_round']))<1e-8
    raw=read_csv(out/'timing_records.csv');lookup=defaultdict(list)
    for r in raw:lookup[int(r['n']),float(r['budget_ms'])].append(r)
    for r in read_csv(out/'timing_summary.csv'):
        v=float(r['budget_ms']);rows=lookup[int(r['n']),v];assert len(rows)==256
        repeats=defaultdict(list)
        for z in rows:repeats[int(z['seed'])].append(z)
        assert len(repeats)==128
        for pair in repeats.values():
            assert sorted(int(z['rep']) for z in pair)==[0,1] and pair[0]['rounds']==pair[1]['rounds'] and pair[0]['certificate_tests']==pair[1]['certificate_tests']
        nr=sum(int(z['rounds']) for z in rows);wall=sum(float(z['wall_ms']) for z in rows)/nr;calls=sum(int(z['certificate_tests']) for z in rows)/nr
        np.testing.assert_allclose([wall,calls,calls/(1000*v),wall/v],[float(r[f]) for f in ['acquisition_ms_per_round','certificate_calls_per_round','required_million_calls_s','mean_speedup_to_target']],rtol=1e-10)
    assert tot==47104 and matches==5120 and len(cs)==132
    result=dict(scored_frontier_outcomes=tot,identical_original_S18_frontier_outcomes=matches,heldout_paired_bands_checked=66,
        archived_descriptive_contrasts_checked=66,estimated_boundary_records_checked=len(boundaries),
        hardware_timing_records_checked=len(raw),frozen_source_hashes_match=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);verify(p.parse_args().out)
