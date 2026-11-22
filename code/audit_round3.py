"""Replay R3 decisions, check bound invariants and simultaneous endpoint events.

Observed timing is retained, never regenerated as a correctness condition.
The checks are numerical; they are not formal interval-arithmetic certification.
"""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from model import Config,scenario,posterior
from admission import execute
from fast_backend import schedule,value
from bounded_admission import bounded_schedule
from predictive_acquisition import endpoint_fit,endpoint_schedule
from finite_blocklength import run_fbl
from ray_evaluation import fit,case
from conformal_report_filter import safe_schedule
from round2_experiments import read_records,regularized_covariance,score_quantile,full_scores,dump
from vector_bounds import BoxEnvelope,vector_fantasy

def audit(root,out):
    errors=[];counts={};events=0;active_events=0;vector_points=0
    data=np.load(root/'dynamic_ray_data/canyon_low.npz');meta=json.loads((root/'dynamic_ray_data/canyon_low.json').read_text())
    fitted=endpoint_fit(data,meta);initial=fit(data,meta,'coherent');c0=regularized_covariance(initial['cov'],1.);q0=score_quantile(initial,meta,c0)
    expected={'branch':480,'bounded_policy':192,'temporal':4608,'endpoint':1280,'catalogue':512,'finite_blocklength':1024}
    for kind,num in expected.items():
        records=read_records(out/(kind+'_records.jsonl'));counts[kind]=len(records)
        if len(records)!=num:errors.append(dict(kind=kind,error='count',expected=num,actual=len(records)))
        for r in records:
            cfg=replace(Config(),n=r.get('n',4),max_probes=2) if kind in ['branch','bounded_policy','catalogue'] else Config()
            if kind=='finite_blocklength':
                cfg=replace(Config(),outage=.04,max_probes=2);s=scenario(cfg,r['seed'])
                rr=run_fbl(cfg,s,r['blocklength'],r['method']=='FBL-IVP')
                ok=all(rr[k]==r[k] for k in ['selected','mode','probes','outage','target_outage','blocks'])
                ok &= all(np.allclose(rr[k],r[k],rtol=1e-8,atol=1e-10) for k in ['power','thresholds','completed_kbit','admitted_kbit','packet_error_probabilities'])
                if not ok:errors.append(dict(kind=kind,seed=r['seed'],method=r['method']))
                continue
            if kind=='endpoint':
                cfg=replace(Config(),ul_fraction=.5);s=case(cfg,data,meta,initial,r['ray_index'],'coherent',False,True,r['seed'])
                s['execution_true']=fitted['xe'][meta['train']+meta['calibration']+r['ray_index']]
            else:s=scenario(cfg,r['seed'])
            mu=s['mean'].copy();cov=s['covariance'].copy();D=len(mu);rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(D,int);ok=True
            if kind=='temporal':
                rho=r['rho'];eps=np.linalg.cholesky(cov+1e-14*np.eye(D))@np.random.default_rng(r['seed']+15000000).normal(size=D)
                s['execution_true']=mu+rho*(s['true']-mu)+np.sqrt(1-rho*rho)*eps
                if r['method']!='Frozen-IVP':mu=np.r_[mu,mu];cov=np.block([[cov,rho*cov],[rho*cov,cov]])
            elif kind=='endpoint' and r['method']=='Predictive-IVP':mu=fitted['joint_mean'].copy();cov=fitted['joint_cov'].copy()
            obs=r.get('observations',[])
            for o in obs:
                k=o['index'];y=s['true'][k]+np.sqrt(rv)*s['report_noise'][k,used[k]];used[k]+=1
                ok &= abs(y-o['observation'])<1e-10;mu,cov=posterior(mu,cov,k,o['observation'],rv)
            m=len(obs);ok &= m==r['probes'];event=False
            if kind=='branch':
                act,info=bounded_schedule(cfg,mu,cov,s['payload'],0,r['node_budget'])
                ok &= all(abs(info[k]-r[k])<1e-7 for k in ['lower','upper','gap','ratio'])
                if cfg.n<=8:
                    exact=schedule(cfg,mu,cov,s['payload'],0)['value'];ok &= r['lower']<=exact+1e-8<=r['upper']+2e-8
            elif kind=='bounded_policy':
                act,info=bounded_schedule(cfg,mu,cov,s['payload'],m,256)
                cat=schedule(cfg,mu,cov,s['payload'],m,masks=np.array(r['catalogue'],np.int32))
                if cat['value']>act['value']:act=cat
                ok &= abs(info['upper']-r['terminal_upper'])<1e-7 and abs(max(0,info['upper']-act['value'])-r['terminal_gap'])<1e-7
            elif kind=='endpoint' and r['method'] in ['Frozen-Safe','Endpoint-Safe','Endpoint-NR']:
                act=safe_schedule(cfg,s,c0,q0,obs) if r['method']=='Frozen-Safe' else endpoint_schedule(cfg,s,fitted,obs)
                if r['method'].startswith('Endpoint'):
                    x=s['execution_true'];score=max(full_scores(x[None,:],fitted['mean'],fitted['cov'])[0],np.max(abs(x-s['true']-fitted['drift_mean'])/fitted['drift_sd']))
                    radius=norm.isf(.01/(2*cfg.max_probes))*np.sqrt(rv)
                    event=score<=fitted['q'] and all(abs(o['observation']-s['true'][o['index']])<=radius for o in obs)
            else:
                if len(mu)>D:mu=mu[D:];cov=cov[D:,D:]
                act=schedule(cfg,mu,cov,s['payload'],m)
            ok &= list(map(int,act['group']))==r['selected'] and act['mode']==r['mode']
            ok &= bool(np.allclose(act['power'],r['power'],rtol=1e-8,atol=1e-12))
            if 'airtime' in act:ok &= bool(np.allclose(act['airtime'],r['airtime'],rtol=1e-8,atol=1e-12))
            phy=execute(cfg,s,act,m)
            for k in ['completed_kbit','admitted_kbit','outage','active']:ok &= abs(phy[k]-r[k])<1e-7
            if event:events+=1;active_events+=phy['active'];ok &= phy['outage']==0
            if not ok:errors.append(dict(kind=kind,seed=r['seed'],method=r.get('method'),n=cfg.n))
        print('replayed',kind,len(records),flush=True)
    vals=read_records(out/'catalogue_value_records.jsonl')
    if len(vals)!=128:errors.append(dict(kind='catalogue_values',error='count'))
    for r in vals:
        cfg=replace(Config(),max_probes=2);s=scenario(cfg,r['seed']);q=np.array([value(cfg,s['mean'],s['covariance'],s['payload'],0,k) for k in range(32)])
        if not np.allclose(q,r['values'],atol=1e-8,rtol=1e-9):errors.append(dict(kind='catalogue_values',seed=r['seed']))
    vectors=read_records(out/'vector_bound_records.jsonl')
    if len(vectors)!=32:errors.append(dict(kind='vector',error='count'))
    for r in vectors:
        cfg=replace(Config(),max_probes=2);s=scenario(cfg,r['seed']);b,c=vector_fantasy(s['covariance'],r['indices'],(cfg.measurement_db*np.log(10)/10)**2)
        env=BoxEnvelope(cfg,s['mean'],c,b,s['payload'],2);leaves=r['leaves'];ok=True
        masses=np.array([np.prod(ndtr(x['hi'])-ndtr(x['lo'])) for x in leaves])
        ok &= np.allclose(masses,[x['mass'] for x in leaves],rtol=1e-10,atol=1e-14)
        ok &= abs(masses.sum()+r['tail_mass']-1)<1e-10
        final=r['checkpoints'][-1]
        ok &= abs(sum(x['mass']*x['lower'] for x in leaves)-final['lower'])<1e-7
        ok &= abs(sum(x['mass']*x['upper'] for x in leaves)+r['tail_mass']*env.total-final['upper'])<1e-7
        for a,d in zip(r['checkpoints'],r['checkpoints'][1:]):ok &= a['lower']<=d['lower']+1e-7 and a['upper']+1e-7>=d['upper']
        rng=np.random.default_rng(r['seed']+100000)
        for j in rng.choice(len(leaves),min(16,len(leaves)),replace=False):
            leaf=leaves[j];l,u=env.bounds(np.array(leaf['lo']),np.array(leaf['hi']))
            ok &= abs(l-leaf['lower'])<1e-8 and abs(u-leaf['upper'])<1e-8
            z=rng.uniform(leaf['lo'],leaf['hi']);v=schedule(cfg,s['mean']+b@z,c,s['payload'],2)['value'];vector_points+=1
            ok &= l<=v+1e-8<=u+2e-8
        if not ok:errors.append(dict(kind='vector',seed=r['seed']))
    result=dict(decisions_checked=sum(counts.values()),counts=counts,candidate_values_replayed=len(vals)*32,vector_states=len(vectors),vector_point_checks=vector_points,
        endpoint_score_noise_events=events,active_endpoint_score_noise_events=active_events,errors=errors)
    dump(out/'audit_round3.json',result);print(json.dumps(result,indent=2))
    if errors:raise RuntimeError('R3 verification failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results');a=p.parse_args()
    root=Path(a.root);audit(root,Path(a.results) if a.results else root/'round3_results')
