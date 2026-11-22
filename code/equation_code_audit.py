"""Cross-language checks of timing, posterior and terminal certificates.

This is an implementation audit, not a formal verification of all arithmetic
or validation of the channel model. No performance-study record is changed.
"""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,posterior
from mobility_cost_study import make_case
from admission import best_schedule
from fast_backend import schedule,parameters
from aligned_frontier import terminal
from power_frontier import reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    states=0;checks=0;worst=0.;frontier=0
    for n in [1,2,4,6]:
        cfg=replace(Config(),n=n)
        for i in range(16):
            seed=100000000+n*1000+i;sc,jm,jc=make_case(cfg,seed,.75);D=len(sc['mean']);mu=jm.copy();cov=jc.copy();rv=(cfg.measurement_db*np.log(10)/10)**2;states+=1
            for m in range(3):
                for c in [0.,.0002]:
                    remaining=cfg.frame_s-m*cfg.probe_s-(m+1)*c;assert abs(parameters(cfg,m,reserve_s=c)[0]-remaining)<1e-15
                    fast=schedule(cfg,mu[D:],cov[D:,D:],sc['payload'],m,reserve_s=c);ref=best_schedule(replace(cfg,frame_s=cfg.frame_s-(m+1)*c),mu[D:],cov[D:,D:],sc['payload'],m)
                    gap=abs(fast['value']-ref['value']);worst=max(worst,gap);assert gap<1e-8,(n,seed,m,c,gap);checks+=1
                    if n==4 and i<4:
                        f=terminal(cfg,sc,mu[D:],cov[D:,D:],m,c);r,trials=reference(replace(cfg,frame_s=remaining),{**sc,'mean':mu[D:],'covariance':cov[D:,D:]});assert f['mode']==r['mode'] and f['group'].tolist()==r['group'].tolist();np.testing.assert_allclose(f['power'],r['power'],rtol=1e-8,atol=1e-12)
                        if 'airtime' in f:np.testing.assert_allclose(f['airtime'],r['airtime'],rtol=1e-8,atol=1e-12)
                        frontier+=1
                k=int(sc['candidates'][m%len(sc['candidates'])]);y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,m]);mu,cov=posterior(mu,cov,k,y,rv)
    frozen=json.loads((Path(__file__).resolve().parents[1]/'dense_alignment_results/protocol.json').read_text())['code_sha256'];source=Path(__file__).parent
    hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in frozen};assert hashes==frozen
    r=dict(states=states,certificate_checkpoints=checks,frontier_checkpoints=frontier,max_certificate_value_difference=worst,all_checks_passed=True,frozen_alignment_code_hashes_match=True,scope='Independent Python spectral/eigensolve scheduler versus C++ positive-linear-solve scheduler across sizes, posterior histories and charged durations. Frontier outputs additionally checked against the Python reference. Not formal floating-point or physical-model verification.')
    (out/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':main()
