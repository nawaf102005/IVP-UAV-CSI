"""Frozen-protocol experiments for the third review. See round3_results/protocol.json."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.stats import beta
from model import Config,scenario,posterior
from fast_backend import schedule,value,library
from admission import execute
from revised_policies import run_fast
from bounded_admission import bounded_schedule,run_bounded
from vector_bounds import value_bounds
from predictive_acquisition import run_predictive,endpoint_fit,endpoint_schedule
from round2_experiments import dump,write_record,batch_value,regularized_covariance,score_quantile,full_scores
from ray_evaluation import fit,case
from conformal_report_filter import safe_schedule

def branch_study(root,out):
    library()
    with (out/'branch_records.jsonl').open('w') as f,(out/'bounded_policy_records.jsonl').open('w') as pf:
        for n in [4,8,12,16,24]:
            cfg=replace(Config(),n=n,max_probes=2)
            for seed in range(310000,310032):
                s=scenario(cfg,seed)
                exact=schedule(cfg,s['mean'],s['covariance'],s['payload'],0)['value'] if n<=8 else None
                for budget in [64,256,1024]:
                    a,info=bounded_schedule(cfg,s['mean'],s['covariance'],s['payload'],0,budget)
                    r=execute(cfg,s,a,0);r.update(info);r.update(seed=seed,n=n,exact=exact,probes=0,selected=a['group'].tolist(),power=a['power'].tolist(),mode=a['mode'])
                    if 'airtime' in a:r['airtime']=a['airtime'].tolist()
                    write_record(f,r)
                if n in [8,16,24]:
                    for reporting in [False,True]:write_record(pf,run_bounded(cfg,s,reporting,node_budget=256,catalogue_size=64))
            f.flush();pf.flush();print('branch',n,flush=True)

def vector_study(root,out):
    cfg=replace(Config(),max_probes=2);library()
    with (out/'vector_bound_records.jsonl').open('w') as f:
        for seed in range(320000,320032):
            s=scenario(cfg,seed);ids=s['candidates'][:2]
            r=value_bounds(cfg,s['mean'],s['covariance'],s['payload'],0,ids,keep_leaves=True)
            r.update(seed=seed,gh17=batch_value(cfg,s,ids,17),gh33=batch_value(cfg,s,ids,33))
            write_record(f,r)
            if seed%8==7:f.flush();print('vector',seed-320000+1,flush=True)

def temporal_study(root,out):
    cfg=Config();library()
    with (out/'temporal_records.jsonl').open('w') as f:
        for rho in [.95,.75,.4]:
            for seed in range(330000,330512):
                s=scenario(cfg,seed);mu=s['mean'];c=s['covariance'];D=len(mu)
                rng=np.random.default_rng(seed+15000000)
                eps=np.linalg.cholesky(c+1e-14*np.eye(D))@rng.normal(size=D)
                s['execution_true']=mu+rho*(s['true']-mu)+np.sqrt(1-rho*rho)*eps
                jm=np.r_[mu,mu];jc=np.block([[c,rho*c],[rho*c,c]])
                for method in ['Frozen-IVP','Predictive-IVP','Predictive-NR']:
                    r=run_fast(cfg,s,'C-IVP') if method=='Frozen-IVP' else run_predictive(cfg,s,jm,jc,method)
                    r.update(method=method,rho=rho);write_record(f,r)
            f.flush();print('temporal',rho,flush=True)

def endpoint_study(root,out):
    name='canyon_low';data=np.load(root/'dynamic_ray_data'/f'{name}.npz');meta=json.loads((root/'dynamic_ray_data'/f'{name}.json').read_text())
    fitted=endpoint_fit(data,meta);initial=fit(data,meta,'coherent');c0=regularized_covariance(initial['cov'],1.);q0=score_quantile(initial,meta,c0)
    cfg=replace(Config(),ul_fraction=.5);start=meta['train']+meta['calibration']
    dump(out/'endpoint_protocol.json',dict(scene=name,seed=meta['seed'],train=meta['train'],calibration=meta['calibration'],test=meta['test'],
        q_endpoint=fitted['q'],q_frozen=q0,endpoint_calibration_scores=fitted['calibration_scores'].tolist(),floor_sd_db=1.,
        score='max of endpoint coordinate/contrast score and all standardized initial-to-endpoint drift residuals',
        guarantee='Marginal endpoint admission for exchangeable initial/endpoint pairs and known report noise. No trajectory or packet-decoder guarantee.'))
    with (out/'endpoint_records.jsonl').open('w') as f:
        for index in range(meta['test']):
            seed=360000+index;s=case(cfg,data,meta,initial,index,'coherent',False,True,seed)
            s['execution_true']=fitted['xe'][start+index]
            frozen=run_fast(cfg,s,'C-IVP');frozen.update(method='Frozen-IVP',ray_index=index);write_record(f,frozen)
            pred=run_predictive(cfg,s,fitted['joint_mean'],fitted['joint_cov']);pred.update(ray_index=index);write_record(f,pred)
            for method in ['Frozen-Safe','Endpoint-Safe','Endpoint-NR']:
                obs=frozen['observations'] if method=='Frozen-Safe' else pred['observations'] if method=='Endpoint-Safe' else []
                a=safe_schedule(cfg,s,c0,q0,obs) if method=='Frozen-Safe' else endpoint_schedule(cfg,s,fitted,obs)
                r=execute(cfg,s,a,len(obs));r.update(method=method,ray_index=index,seed=seed,probes=len(obs),observations=obs,
                    selected=a['group'].tolist(),power=a['power'].tolist(),mode=a['mode'])
                if 'airtime' in a:r['airtime']=a['airtime'].tolist()
                write_record(f,r)
            if index%64==63:f.flush();print('endpoint',index+1,flush=True)

def report_catalogue(mu,n,both):
    ids=[]
    for d in range(2 if both else 1):
        for i in range(n):
            row=mu[d*n*n+i*n:d*n*n+(i+1)*n].copy();row[i]=-np.inf
            ids.extend([d*n*n+i*(n+1),d*n*n+i*n+int(np.argmax(row))])
    return np.array(sorted(set(ids)),int)

def catalogue_study(root,out):
    cfg=replace(Config(),max_probes=2);library()
    with (out/'catalogue_records.jsonl').open('w') as f,(out/'catalogue_value_records.jsonl').open('w') as vf:
        for seed in range(340000,340128):
            s=scenario(cfg,seed);rv=(cfg.measurement_db*np.log(10)/10)**2
            qs=np.array([value(cfg,s['mean'],s['covariance'],s['payload'],0,k) for k in range(len(s['mean']))])
            cats={'UL-fixed':report_catalogue(s['mean'],cfg.n,False),'Bidirectional-fixed':report_catalogue(s['mean'],cfg.n,True),'All-coordinates':np.arange(len(s['mean']))}
            write_record(vf,dict(seed=seed,full_best=float(qs.max()),default_best=float(qs[cats['UL-fixed']].max()),bidirectional_best=float(qs[cats['Bidirectional-fixed']].max()),
                                values=qs.tolist(),catalogues={k:v.tolist() for k,v in cats.items()}))
            for method in ['UL-fixed','Bidirectional-fixed','Bidirectional-updated','All-coordinates']:
                mu=s['mean'].copy();cov=s['covariance'].copy();obs=[];used=np.zeros(len(mu),int)
                for m in range(2):
                    ids=report_catalogue(mu,cfg.n,True) if method=='Bidirectional-updated' else cats[method]
                    forecast=qs[ids] if m==0 else np.array([value(cfg,mu,cov,s['payload'],m,int(k)) for k in ids])
                    k=int(ids[np.argmax(forecast)]);y=float(s['true'][k]+np.sqrt(rv)*s['report_noise'][k,used[k]]);used[k]+=1
                    mu,cov=posterior(mu,cov,k,y,rv);obs.append(dict(index=k,observation=y))
                a=schedule(cfg,mu,cov,s['payload'],2);r=execute(cfg,s,a,2)
                r.update(method=method,seed=seed,probes=2,observations=obs,selected=a['group'].tolist(),power=a['power'].tolist(),mode=a['mode'])
                if 'airtime' in a:r['airtime']=a['airtime'].tolist()
                write_record(f,r)
            if seed%32==31:f.flush();vf.flush();print('catalogue',seed-340000+1,flush=True)
    n=64;alpha=.04;confidence=.95;ranks=[k for k in range(1,n+1) if beta.sf(1-alpha,k,n+1-k)>=confidence]
    dump(out/'beta_calibration.json',dict(n=n,miscoverage=alpha,confidence=confidence,finite_rank_available=bool(ranks),
        maximum_rank_confidence=1-(1-alpha)**n,minimum_n_for_finite_maximum=int(np.ceil(np.log(1-confidence)/np.log(1-alpha)))))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output');p.add_argument('--mode',choices=['branch','vector','temporal','endpoint','catalogue'],required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output) if a.output else root/'round3_results';out.mkdir(parents=True,exist_ok=True)
    globals()[a.mode+'_study'](root,out)

if __name__=='__main__':main()
