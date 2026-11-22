"""Replay all R5 terminal allocations, observation draws and deadline penalties."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,scenario,posterior
from fast_backend import schedule
from admission import execute

def audit(root,results=None):
    out=results or root/'round5_results';counts={};errors=[];cache={}
    for filename,expected in [('scale',4608),('latency',5376),('mismatch',22528)]:
        records=[json.loads(s) for s in (out/(filename+'_records.jsonl')).read_text().splitlines()];counts[filename]=len(records);assert len(records)==expected
        for r in records:
            try:
                n=r['n'];T=r.get('duration_ms',20);cfg=replace(Config(),n=n,frame_s=T/1000,ul_bits=30000*T/20,dl_bits=1800*T/20)
                key=(filename,n,T,r['seed'],r.get('dimension'),r.get('true_parameter'),r['method'].startswith('Matched'))
                if key not in cache:
                    sc=scenario(cfg,r['seed'])
                    if filename=='mismatch':
                        true=scenario(replace(cfg,**{r['dimension']:r['true_parameter']}),r['seed']);sc['true']=true['true']
                        if r['method'].startswith('Matched'):sc['covariance']=true['covariance']
                    cache[key]=sc
                sc=cache[key];mu=sc['mean'].copy();cov=sc['covariance'].copy();used=np.zeros(len(mu),int);rv=(cfg.measurement_db*np.log(10)/10)**2
                for o in r['observations']:
                    k=o['index'];y=sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]];used[k]+=1
                    assert abs(y-o['observation'])<1e-10;mu,cov=posterior(mu,cov,k,y,rv)
                count=len(r['observations']);assert count==r['probes'];reserve=r['reserve_s'];act=schedule(cfg,mu,cov,sc['payload'],count,reserve_s=reserve)
                assert r['selected']==act['group'].tolist() and r['mode']==act['mode']
                np.testing.assert_allclose(r['power'],act['power'],rtol=1e-8,atol=1e-12)
                if 'airtime' in act:np.testing.assert_allclose(r['airtime'],act['airtime'],rtol=1e-8,atol=1e-12)
                e=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-(count+1)*reserve)),sc,act,count)
                assert abs(e['completed_kbit']-r['radio_completed_kbit'])<1e-7
                over=reserve>0 and r['wall_s']>(count+1)*reserve;assert int(over)==r['deadline_overrun']
                if over and e['active']:e.update(completed_kbit=0,completed_tasks=0,outage=1)
                for k in ['completed_kbit','completed_tasks','admitted_kbit','outage','active']:assert abs(e[k]-r[k])<1e-7
            except (AssertionError,ValueError) as e:errors.append(dict(file=filename,seed=r['seed'],method=r['method'],error=str(e)))
        cache.clear()
    result=dict(decisions=sum(counts.values()),counts=counts,errors=errors,scope='All final allocations, observations and outcomes including deadline penalties; full forecast recomputation belongs to runner.')
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if errors:raise RuntimeError('Replay mismatch')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results',type=Path);a=p.parse_args();audit(Path(a.root),a.results)
