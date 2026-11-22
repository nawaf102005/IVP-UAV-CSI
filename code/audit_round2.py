"""Replay every additional stored decision and check conditional score events."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.stats import norm
from model import Config,scenario,posterior
from admission import execute
from fast_backend import schedule
from ray_evaluation import fit,case
from conformal_report_filter import safe_schedule,conformal_quantile
from round2_experiments import read_records,fresh_case_data,regularized_covariance,score_quantile,full_scores,dump

def audit(root,out):
    errors=[];counts={};events=0;active_events=0;cache={}
    for fresh in [False,True]:
        for name in (['munich'] if fresh else ['canyon_low','canyon_high','munich']):
            if fresh:data,meta=fresh_case_data(root,name)
            else:data=np.load(root/'ray_data'/f'{name}.npz');meta=json.loads((root/'ray_data'/f'{name}.json').read_text())
            for kind,mobile in [('incoherent',False),('coherent',True)]:
                prior=fit(data,meta,kind);reg=regularized_covariance(prior['cov'],1.)
                cache[fresh,name+('-motion' if mobile else '-static')]=(data,meta,prior,reg,kind,mobile)
    for filename in ['vector_records.jsonl','calibration_records.jsonl','fresh_calibration_records.jsonl','kg_timing_records.jsonl','reservation_records.jsonl']:
        records=read_records(out/filename);counts[filename]=len(records)
        expected={'vector_records.jsonl':512,'calibration_records.jsonl':3072,'fresh_calibration_records.jsonl':2560,'kg_timing_records.jsonl':512,'reservation_records.jsonl':1536}
        if len(records)!=expected[filename]:errors.append(dict(file=filename,expected=expected[filename],observed=len(records)))
        for r in records:
            cfg=replace(Config(),max_probes=2) if filename.startswith('vector') else Config();is_ray='calibration' in filename
            if is_ray:
                cfg=replace(Config(),ul_fraction=.5,ul_power_w=.2,dl_power_w=1.)
                data,meta,prior,reg,kind,mobile=cache[r.get('fresh',False),r['scenario']]
                sc=case(cfg,data,meta,prior,r['ray_index'],kind,mobile,True,r['seed'])
            else:sc=scenario(cfg,r['seed'])
            mu=sc['mean'].copy();cov=sc['covariance'].copy();used=np.zeros(len(mu),int);rv=(cfg.measurement_db*np.log(10)/10)**2
            ok=True
            for obs in r['observations']:
                k=obs['index'];expected=sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]];used[k]+=1
                ok &= abs(obs['observation']-expected)<1e-10
                mu,cov=posterior(mu,cov,k,obs['observation'],rv)
            m=len(r['observations']);ok &= m==r['probes']
            if is_ray and r['method']!='C-IVP':
                c=prior['cov'] if r['method'].startswith('Safe') else reg
                q=conformal_quantile(prior,meta) if r['method'].startswith('Safe') else score_quantile(prior,meta,reg)
                act=safe_schedule(cfg,sc,c,q,r['observations']);ok &= abs(r['q']-q)<1e-10
                radius=norm.isf(.01/(2*cfg.max_probes))*np.sqrt(rv)
                event=full_scores(sc['true'][None,:],prior['mean'],c)[0]<=q and all(abs(o['observation']-sc['true'][o['index']])<=radius for o in r['observations'])
            else:
                act=schedule(cfg,mu,cov,sc['payload'],m,reserve_s=r.get('reserve_s',0.));event=False
            ok &= r['selected']==list(map(int,act['group'])) and r['mode']==act['mode']
            ok &= bool(np.allclose(np.array(r['power']),act['power'],rtol=1e-8,atol=1e-12))
            if 'airtime' in act:ok &= bool(np.allclose(r['airtime'],act['airtime'],rtol=1e-8,atol=1e-12))
            reserve=r.get('reserve_s',0.);ecfg=replace(cfg,frame_s=max(1e-12,cfg.frame_s-(m+1)*reserve))
            physical=execute(ecfg,sc,act,m)
            if event and not mobile:
                events+=1;active_events+=int(physical['active']);ok &= physical['outage']==0
            overrun=reserve>0 and r['wall_s']>(m+1)*reserve
            if overrun and physical['active']:
                physical.update(completed_kbit=0.,completed_tasks=0,outage=1)
            for key in ['completed_kbit','completed_tasks','outage','active','admitted_kbit']:
                ok &= abs(physical[key]-r[key])<1e-7
            if not ok:errors.append(dict(file=filename,seed=r['seed'],method=r['method']))
    result=dict(counts=counts,decisions_checked=sum(counts.values()),errors=errors,
                static_score_noise_events_checked=events,active_static_score_noise_events=active_events)
    dump(out/'audit_round2.json',result);print(json.dumps(result,indent=2))
    if errors:raise RuntimeError('Round-two replay mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results');a=p.parse_args()
    root=Path(a.root);audit(root,Path(a.results) if a.results else root/'round2_results')
