"""Physical two-pilot observations; Gaussian log-power conditioning is approximate.

Fixed flat channels, two orthogonal complex waveforms, LS estimates, explicit
prior-averaged delta variance. No exact physical-posterior guarantee is claimed.
The conditional-CDF rule uses outer quadrature, so is not a vector lower bound.
"""
import argparse,json,time,platform,sys
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from scipy import stats
from model import Config,scenario,noise_power
from admission import execute
from fast_backend import library,schedule,affine_value
from revised_policies import gaussian_rule
from vector_bounds import vector_fantasy,value_bounds

def pilot_matrix(length):
    if length<2:raise ValueError('Two orthogonal columns require at least two symbols')
    t=np.arange(length)
    return np.stack([np.ones(length),np.exp(2j*np.pi*t/length)],axis=1)

def matched_filter(channel,noise,power,length):
    S=pilot_matrix(length)
    received=np.sqrt(power)*(S@np.asarray(channel))+noise
    return S.conj().T@received/(np.sqrt(power)*length)

def batches(cfg,sc):
    ul=sc['mean'][:cfg.n**2].reshape(cfg.n,cfg.n).copy();np.fill_diagonal(ul,-np.inf)
    return [(i*cfg.n+i,i*cfg.n+int(np.argmax(ul[i]))) for i in range(cfg.n)]

def noise_covariance(cfg,sc,ids,length):
    ids=np.asarray(ids,int);sigma_e2=noise_power(cfg)/(length*cfg.ul_power_w)
    return np.diag(2*sigma_e2*np.exp(-sc['mean'][ids]+np.diag(sc['covariance'])[ids]/2))

def physical_observations(cfg,sc,length):
    rng=np.random.default_rng(sc['seed']+700000+length*1000)
    phases=rng.uniform(-np.pi,np.pi,len(sc['true']));hh=np.exp(sc['true']/2+1j*phases);ans=[]
    for ids in batches(cfg,sc):
        eps=np.sqrt(noise_power(cfg)/2)*(rng.normal(size=length)+1j*rng.normal(size=length))
        hat=matched_filter(hh[list(ids)],eps,cfg.ul_power_w,length)
        y=np.log(np.maximum(np.abs(hat)**2,np.finfo(float).tiny));R=noise_covariance(cfg,sc,ids,length)
        snr=length*cfg.ul_power_w*np.exp(sc['true'][list(ids)])/noise_power(cfg)
        ans.append(dict(indices=list(ids),y=y.tolist(),snr=snr.tolist(),
             standardized_error=((y-sc['true'][list(ids)])/np.sqrt(np.diag(R))).tolist(),
             ls_error=(abs(hat-hh[list(ids)])**2/(noise_power(cfg)/(length*cfg.ul_power_w))).tolist()))
    return ans

def batch_value(cfg,sc,ids,length,method,order=17):
    R=noise_covariance(cfg,sc,ids,length);b,post=vector_fantasy(sc['covariance'],ids,R);z,w=gaussian_rule(order)
    if method=='Conditional-CDF':
        return float(sum(ww*affine_value(cfg,sc['mean']+b[:,0]*zz,post,b[:,1],sc['payload'],1) for zz,ww in zip(z,w)))
    if method=='Product-GH':
        return float(sum(wi*wj*schedule(cfg,sc['mean']+b[:,0]*zi+b[:,1]*zj,post,sc['payload'],1)['value'] for zi,wi in zip(z,w) for zj,wj in zip(z,w)))
    raise ValueError(method)

def update(sc,ids,y,R):
    ids=np.array(ids,int);cov=sc['covariance'];K=np.linalg.solve(cov[np.ix_(ids,ids)]+R,cov[ids,:]).T
    mean=sc['mean']+K@(np.asarray(y)-sc['mean'][ids]);post=cov-K@cov[ids,:]
    return mean,(post+post.T)/2

def run_policy(cfg,sc,length,method,observations):
    t=time.perf_counter();tc=time.process_time();act=schedule(cfg,sc['mean'],sc['covariance'],sc['payload'],0)
    vv=[];chosen=None;count=0;obs=[]
    if method!='No-report':
        vv=[batch_value(cfg,sc,ids,length,method) for ids in batches(cfg,sc)];chosen=int(np.argmax(vv))
        if vv[chosen]>act['value']+cfg.gain_tol_kbit:
            o=observations[chosen];ids=o['indices'];obs=o['y'];R=noise_covariance(cfg,sc,ids,length)
            mean,cov=update(sc,ids,obs,R);act=schedule(cfg,mean,cov,sc['payload'],1);count=1
    wall=time.perf_counter()-t;cpu=time.process_time()-tc;r=execute(cfg,sc,act,count)
    r.update(seed=sc['seed'],length=length,method=method,probes=count,chosen=chosen,observation=obs,forecasts=vv,
        mode=act['mode'],selected=act['group'].tolist(),power=act['power'].tolist(),wall_s=wall,cpu_s=cpu,probe_s=cfg.probe_s)
    if 'airtime' in act:r['airtime']=act['airtime'].tolist()
    return r

def paired(a,b):
    x=np.asarray(a)-np.asarray(b);m=float(x.mean());h=float(stats.t.ppf(.975,len(x)-1)*stats.sem(x))
    return dict(mean=m,ci=[m-h,m+h],n=len(x))

def summarize(out):
    rr=[json.loads(s) for s in (out/'pilot_records.jsonl').read_text().splitlines()]
    oo=[json.loads(s) for s in (out/'pilot_observations.jsonl').read_text().splitlines()];summary={}
    for L in sorted({r['length'] for r in rr}):
        aa={m:sorted([r for r in rr if r['length']==L and r['method']==m],key=lambda r:r['seed']) for m in ['Conditional-CDF','Product-GH','No-report']}
        dd={m:dict(completed_kbit=float(np.mean([r['completed_kbit'] for r in x])),outages=sum(r['outage'] for r in x),
                   reports=float(np.mean([r['probes'] for r in x])),wall_ms=1000*float(np.mean([r['wall_s'] for r in x])),
                   cpu_ms=1000*float(np.mean([r['cpu_s'] for r in x]))) for m,x in aa.items()}
        for m in ['Product-GH','No-report']:dd['cdf_minus_'+m]=paired([r['completed_kbit'] for r in aa['Conditional-CDF']],[r['completed_kbit'] for r in aa[m]])
        dd['same_acquisition']=sum((a['probes'],a['chosen'] if a['probes'] else None)==(b['probes'],b['chosen'] if b['probes'] else None) for a,b in zip(aa['Conditional-CDF'],aa['Product-GH']))
        errors=np.array([o['standardized_error'] for r in oo if r['length']==L for o in r['batches']]).ravel()
        snr=np.array([o['snr'] for r in oo if r['length']==L for o in r['batches']]).ravel()
        ls=np.array([o['ls_error'] for r in oo if r['length']==L for o in r['batches']]).ravel()
        dd['observation_diagnostic']=dict(n=len(errors),mean=float(errors.mean()),variance=float(errors.var(ddof=1)),outside_95=float(np.mean(abs(errors)>stats.norm.ppf(.975))),
             snr_db_quantiles=(10*np.log10(np.quantile(snr,[0,.05,.5,.95,1]))).tolist(),ls_normalized_mse=float(ls.mean()))
        summary[str(L)]=dd
    (out/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='round4_results');p.add_argument('--frames',type=int,default=256);p.add_argument('--audit-states',type=int,default=16);p.add_argument('--seed',type=int,default=980000)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'pilot_records.jsonl').exists():raise FileExistsError('Choose a fresh output directory')
    proto=dict(seed=a.seed,frames_per_length=a.frames,lengths=[16,64],power_w=.2,feedback_s=.0015,max_batches=1,methods=['Conditional-CDF','Product-GH','No-report'],gh_order=17,audit_order=33,audit_states_per_length=a.audit_states,
        delta_covariance='diag(2 N0B/(L P) exp(-mu+diag(Sigma)/2))',channel='Original Gaussian log gains with independent uniform phases; channel constant over sounding and data.',
        accounting='One batch costs L/B+1.5ms; radio-only service; computation is measured separately.',bounds='Gaussian-surrogate box bounds only; outer quadrature does not make Conditional-CDF a lower bound.',
        config=asdict(Config()),python=sys.version,platform=platform.platform())
    (out/'pilot_protocol.json').write_text(json.dumps(proto,indent=2)+'\n');library();gaussian_rule(17);gaussian_rule(33)
    for L in proto['lengths']:
        cfg=replace(Config(),probe_s=.0015+L/1e6,max_probes=1);sc=scenario(cfg,a.seed-1);obs=physical_observations(cfg,sc,L)
        for m in proto['methods']:run_policy(cfg,sc,L,m,obs)
    with (out/'pilot_records.jsonl').open('w') as f,(out/'pilot_observations.jsonl').open('w') as g,(out/'pilot_value_audit.jsonl').open('w') as h:
        for L in proto['lengths']:
            cfg=replace(Config(),probe_s=.0015+L/1e6,max_probes=1)
            for i in range(a.frames):
                sc=scenario(cfg,a.seed+i);obs=physical_observations(cfg,sc,L);g.write(json.dumps(dict(seed=sc['seed'],length=L,batches=obs))+'\n');records={}
                for m in np.random.default_rng(sc['seed']+L).permutation(proto['methods']):
                    r=run_policy(cfg,sc,L,str(m),obs);records[m]=r;f.write(json.dumps(r,allow_nan=False)+'\n')
                if i<a.audit_states:
                    j=records['Conditional-CDF']['chosen'];ids=batches(cfg,sc)[j]
                    bd=value_bounds(cfg,sc['mean'],sc['covariance'],sc['payload'],0,ids,budgets=(256,1024),noise_cov=noise_covariance(cfg,sc,ids,L),next_count=1)
                    h.write(json.dumps(dict(seed=sc['seed'],length=L,ids=list(ids),cdf17=records['Conditional-CDF']['forecasts'][j],gh17=records['Product-GH']['forecasts'][j],
                       cdf33=batch_value(cfg,sc,ids,L,'Conditional-CDF',33),gh33=batch_value(cfg,sc,ids,L,'Product-GH',33),bounds=bd),allow_nan=False)+'\n')
                if (i+1)%32==0:f.flush();g.flush();h.flush();print('pilots',L,i+1,a.frames,flush=True)
    print(json.dumps(summarize(out),indent=2))

if __name__=='__main__':main()
