"""Independent execution replay, paired-input check and interval reconstruction for larger clusters."""
import argparse
import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import sem, t
from admission import execute
from aligned_frontier import acceptance_count
from mobility_cost_study import make_case
from model import Config
from persistence_campaign import BASE, read_csv, jsave

def check(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for name,sha in protocol['software'].items():
        assert hashlib.sha256((BASE/'code'/name).read_bytes()).hexdigest()==sha
    groups={};rc=0
    for n,ms,first in protocol['cells']:
        cfg=replace(Config(),n=n,frame_s=ms*.001,ul_bits=30000*ms/20.,dl_bits=1800*ms/20.)
        rho=np.exp(-20*cfg.frame_s/10)
        with gzip.open(out/f'N{n}_T{ms}.jsonl.gz','rt') as f:
            last=None
            for line in f:
                r=json.loads(line);s=r['seed'];k=(n,ms,s,r['alpha'],r['rule'],r['kind'])
                assert k not in groups and first<=s<first+protocol['frames']
                if s!=last:sc=make_case(cfg,s,rho)[0];last=s
                m=r['probes'];assert m==len(r['observations'])<=4
                reserve=cfg.frame_s*r['alpha']*(m+1)
                assert abs(reserve-r['reserved_s'])<1e-12
                a=dict(group=np.asarray(r['selected'],int),mode=r['mode'],power=np.asarray(r['power']),value=r['admitted_kbit'])
                if 'airtime' in r:a['airtime']=np.asarray(r['airtime'])
                z=execute(replace(cfg,frame_s=cfg.frame_s-reserve),sc,a,m)
                assert abs(z['completed_kbit']-r['completed_kbit'])<1e-8 and z['outage']==r['outage']
                if r['kind']=='front':
                    other=groups[n,ms,s,r['alpha'],r['rule'],'cert']
                    assert r['observations']==other['observations'] and r['history']==other['history']
                    if r['selected']:assert 0<=r['validation_bad']<=acceptance_count(n,4096)
                groups[k]=r;rc+=1
    rr=read_csv(out/'size_contrasts.csv')
    for r in rr:
        n=int(r['n']);ms=int(r['T_ms']);alpha=float(r['alpha']);trm=r['terminal']
        first=next(c[2] for c in protocol['cells'] if c[:2]==[n,ms]);diff=[]
        for seed in range(first,first+protocol['frames']):
            def fetch(rule):return groups[n,ms,seed,0. if rule=='Predictive-NR' else alpha,rule,trm]['completed_kbit']
            diff.append(fetch(r['left'])-fetch(r['right']))
        v=np.mean(diff);h=sem(diff)*t.isf(.025/24,len(diff)-1)
        np.testing.assert_allclose([v,v-h,v+h],[float(r[k]) for k in ('mean','lo','hi')],rtol=1e-10,atol=1e-10)
    assert (rc,len(rr))==(10240,24)
    jsave(out/'independent_audit.json',dict(replayed_held_out_execution_outcomes=rc,matched_history_terminal_pairs=rc//2,
                                            independently_reconstructed_intervals=len(rr),protocol_source_hashes_match=True))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    check(p.parse_args().out)
