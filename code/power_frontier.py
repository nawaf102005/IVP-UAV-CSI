"""Independently validated finite-power-family chance-feasibility reference."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,scenario
from admission import coefficients,minimum_power,subsets,orthogonal_data,orthogonal_airtime,execute
from chance_benchmark import draw,failures,cp_upper

def reference(cfg,sc):
    knobs=[.01,.025,.05,.1,.2,.4,.8]
    groups=sorted(subsets(cfg.n),key=lambda g:-sc['payload'][:,g].sum())
    family=len(groups)*2*len(knobs);alpha=.05/family
    validation=draw(sc['mean'],sc['covariance'],4096,sc['seed']+4200000,cfg.n)
    trials=[]
    for group in groups:
        for knob in knobs:
            local=replace(cfg,outage=knob)
            coeff=coefficients(local,sc['mean'],sc['covariance'],sc['payload'],0,group)
            power=minimum_power(local,coeff)
            actions=[]
            if power is not None:
                # Uniform per-direction scaling improves every physical SINR.
                cap=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
                power=power*np.min(cap/power,axis=1)[:,None]
                actions.append(dict(group=group,power=power,mode='reuse',value=float(sc['payload'][:,group].sum()/1000)))
            data=orthogonal_data(local,sc['mean'],sc['covariance'],sc['payload'],0,group)
            airtime=orthogonal_airtime(data,0)
            if np.all(airtime.sum(axis=1)<=data['duration']):
                airtime=airtime*(data['duration']/airtime.sum(axis=1))[:,None]
                actions.append(dict(group=group,power=np.broadcast_to(np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None],airtime.shape).copy(),airtime=airtime,mode='OMA',value=float(sc['payload'][:,group].sum()/1000)))
            for action in actions:
                bad=int(failures(cfg,sc['payload'],validation,action).sum());upper=cp_upper(bad,4096,alpha)
                trials.append(dict(group=list(map(int,group)),knob=knob,mode=action['mode'],validation_bad=bad,validation_upper=upper))
                if upper<=cfg.outage:return action,trials
    return dict(group=np.array([],int),power=np.zeros((2,0)),mode='empty',value=0.),trials

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='../revision_results');a=p.parse_args();out=Path(a.output);cfg=Config();t=time.perf_counter()
    with (out/'frontier_records.jsonl').open('w') as f:
        for seed in range(190000,190128):
            sc=scenario(cfg,seed);before=time.perf_counter();action,trials=reference(cfg,sc);elapsed=time.perf_counter()-before
            r=execute(cfg,sc,action,0);g=draw(sc['mean'],sc['covariance'],16384,seed+3000000,cfg.n);bad=int(failures(cfg,sc['payload'],g,action).sum())
            r.update(method='Validated-frontier',seed=seed,selected=list(map(int,action['group'])),mode=action['mode'],power=action['power'].tolist(),probes=0,
                     wall_s=elapsed,conditional_risk=bad/16384,conditional_bad=bad,conditional_draws=16384,trials=trials)
            if 'airtime' in action:r['airtime']=action['airtime'].tolist()
            f.write(json.dumps(r)+'\n');f.flush()
    (out/'frontier_protocol.json').write_text(json.dumps(dict(count=128,seed_start=190000,candidate_quantile_budgets=[.01,.025,.05,.1,.2,.4,.8],validation=4096,
        evaluation=16384,family_size=210,confidence=.95,elapsed_s=time.perf_counter()-t,
        limitation='Finite candidate family with simultaneous binomial validation; not an exact chance-constrained optimum.'),indent=2))

if __name__=='__main__':main()
