"""Replay platform-cost data, allocations, observations and frontier validation."""
import argparse,gzip,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,posterior
from mobility_cost_study import make_case
from fast_backend import schedule,library
from admission import execute
from power_frontier import reference

def audit_file(job):
    path,cfg_dict,rho,front=job;cfg=Config(**cfg_dict);library();count=0;keys=set()
    with gzip.open(path,'rt') as f:
        for line in f:
            r=json.loads(line);key=(r['seed'],r['method'],r['alpha']);assert key not in keys;keys.add(key)
            assert r['enforce_elapsed'] is False
            sc,mu,cov=make_case(cfg,r['seed'],rho);D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(D,int)
            for obs in r['observations']:
                k=obs['index'];assert k in sc['candidates'];expected=sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]];used[k]+=1;assert abs(expected-obs['observation'])<1e-11;mu,cov=posterior(mu,cov,k,obs['observation'],rv)
            assert len(r['observations'])==r['probes']
            c=0. if r['method'].startswith('Predictive-NR') else r['alpha']*cfg.frame_s;reserved=(r['probes']+1)*c
            assert abs(c-r['reserve_s'])<1e-12 and abs(reserved-r['reserved_s'])<1e-12
            if front:
                remaining=cfg.frame_s-r['probes']*cfg.probe_s-reserved
                if remaining<=0:action=dict(group=np.array([],int),power=np.zeros((2,0)),mode='empty',value=0.);trials=[]
                else:action,trials=reference(replace(cfg,frame_s=remaining),{**sc,'mean':mu[D:],'covariance':cov[D:,D:]})
                assert trials==r['trials']
                if trials and action['mode']!='empty':assert trials[-1]['validation_upper']<=cfg.outage
            else:action=schedule(cfg,mu[D:],cov[D:,D:],sc['payload'],r['probes'],reserve_s=c)
            assert action['group'].tolist()==r['selected'] and action['mode']==r['mode'];np.testing.assert_allclose(action['power'],r['power'],rtol=1e-10,atol=1e-12)
            if 'airtime' in action:np.testing.assert_allclose(action['airtime'],r['airtime'],rtol=1e-10,atol=1e-12)
            result=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,action,r['probes'])
            for k in ['completed_kbit','admitted_kbit','completed_tasks','admitted_tasks','outage','active']:assert abs(result[k]-r[k])<1e-9,(path,key,k)
            count+=1
    return Path(path).name,count

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.results);protocol=json.loads((out/'protocol.json').read_text());jobs=[]
    for c in protocol['cells']:jobs.append((str(out/(c['id']+'.jsonl.gz')),c['config'],c['rho'],c['kind']=='frontier'))
    la=json.loads((out/'lookahead_summary.json').read_text());jobs.append((str(out/'lookahead_records.jsonl.gz'),la['config'],la['rho'],False));total=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(audit_file,j) for j in jobs]):name,count=future.result();total+=count;print('replayed',name,count,flush=True)
    expected=protocol['test_frames']*(sum(1+2*len(c['alphas']) for c in protocol['cells'])+9);assert total==expected
    report=dict(records_replayed=total,cells=len(protocol['cells']),mismatches=0,elapsed_host_time_used=False,observations_recomputed=True,terminal_allocations_recomputed=True,frontier_candidate_validation_recomputed=True,full_forecasts_recomputed=False)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
