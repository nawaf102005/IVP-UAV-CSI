"""Independent terminal replay of fresh dense and frontier-alignment records."""
import argparse,gzip,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,posterior
from mobility_cost_study import make_case
from fast_backend import schedule
from admission import execute
from power_frontier import reference
from aligned_frontier import run_aligned

def replay(job):
    cfgdict,rho,rs=job;cfg=Config(**cfgdict);total=0;forecasts=0
    for r in rs:
        sc,mu,cov=make_case(cfg,r['seed'],rho);D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(D,int)
        for o in r['observations']:
            k=o['index'];assert k in sc['candidates'];y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1;assert abs(y-o['observation'])<1e-11;mu,cov=posterior(mu,cov,k,y,rv)
        m=r['probes'];assert m==len(r['observations']);c=r['reserve_s'];reserved=(m+1)*c;assert abs(reserved-r['reserved_s'])<1e-12 and r['enforce_elapsed'] is False
        if r['kind']=='dense':a=schedule(cfg,mu[D:],cov[D:,D:],sc['payload'],m,reserve_s=c)
        else:
            remaining=cfg.frame_s-m*cfg.probe_s-reserved
            a,trials=reference(replace(cfg,frame_s=remaining),{**sc,'mean':mu[D:],'covariance':cov[D:,D:]})
            if a['mode']!='empty':assert trials[-1]['validation_upper']<=cfg.outage and trials[-1]['validation_bad']==r['validation_bad']
        assert a['group'].tolist()==r['selected'] and a['mode']==r['mode'];np.testing.assert_allclose(a['power'],r['power'],rtol=1e-8,atol=1e-12)
        if 'airtime' in a:np.testing.assert_allclose(a['airtime'],r['airtime'],rtol=1e-8,atol=1e-12)
        result=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,a,m)
        for k in ['completed_kbit','admitted_kbit','completed_tasks','admitted_tasks','outage','active']:assert abs(result[k]-r[k])<1e-9,(r['seed'],r['method'],k)
        if r['method']=='Frontier-aware-IVP' and r['alpha']==0. and r['seed']<97000016:
            sc,jm,jc=make_case(cfg,r['seed'],rho);full=run_aligned(cfg,sc,jm,jc,c)
            for k in ['observations','history','selected','mode','completed_kbit']:assert full[k]==r[k],k
            forecasts+=1
        total+=1
    return total,forecasts

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.results);protocol=json.loads((out/'protocol.json').read_text())
    with gzip.open(out/'records.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    keys=[(r['kind'],r['seed'],r['alpha'],r['method']) for r in rows];assert len(set(keys))==len(rows)
    jobs=[(protocol['config'],protocol['rho'],rows[i::a.workers]) for i in range(a.workers)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:parts=list(pool.map(replay,jobs))
    total=sum(x[0] for x in parts);forecast=sum(x[1] for x in parts);expected=protocol['frames']*(2+2*len(protocol['dense_alphas'])+2*len(protocol['frontier_alphas']));assert total==expected
    r=dict(records_replayed=total,mismatches=0,full_frontier_policy_replays=forecast,terminal_frontier_checked_against_independent_python=True,observations_and_execution_recomputed=True,scope='All terminal allocations and outcomes replayed; 16 complete zero-cost frontier-aware policy forecasts additionally recomputed. Not all internal forecasts are rerun.')
    (out/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()
