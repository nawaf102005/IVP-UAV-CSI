"""Split-conformal prior bounds intersected with simultaneous report bounds.

Unlike a Gaussian posterior inflation, this terminal filter retains a marginal
static-frame guarantee for arbitrary adaptive report selection. The query policy
is still Gaussian C-IVP; its expected-utility theorem is not transferred to this
different terminal filter. Moving-channel experiments are explicit stress tests.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from model import Config,noise_power,thresholds
from dataclasses import replace
from admission import subsets,minimum_power,execute
from ray_evaluation import fit,case

def conformal_quantile(prior,meta,prior_risk=.04):
    start=meta['train'];nc=meta['calibration'];x=prior['x'][start:start+nc]-prior['mean']
    marginal=np.max(np.abs(x)/np.sqrt(np.diag(prior['cov'])),axis=1)
    score=np.maximum(marginal,prior['cal_scores'])
    rank=int(np.ceil((nc+1)*(1-prior_risk)))
    return float(np.sort(score)[rank-1]) if rank<=nc else np.inf

def safe_schedule(cfg,sc,cov,q,observations,reciprocal=True):
    n=cfg.n;mu=sc['mean'];std=np.sqrt(np.diag(cov));lower=mu-q*std;upper=mu+q*std
    # Known report-error law; union bound covers all M adaptive queries.
    radius=norm.isf(.01/(2*max(cfg.max_probes,1)))*cfg.measurement_db*np.log(10)/10
    for obs in observations:
        index=obs['index'];indices=[index]
        if reciprocal:
            d,ij=divmod(index,n*n);i,j=divmod(ij,n)
            indices.append((1-d)*n*n+j*n+i)
        for k in indices:
            lower[k]=max(lower[k],obs['observation']-radius)
            upper[k]=min(upper[k],obs['observation']+radius)
    empty=dict(group=np.array([],int),power=np.zeros((2,0)),mode='empty',value=0.)
    if np.any(lower>upper):return empty
    theta=thresholds(cfg,sc['payload'],len(observations));duration=(cfg.frame_s-len(observations)*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
    for group in sorted(subsets(n),key=lambda g:-sc['payload'][:,g].sum()):
        s=len(group);lf=[];lu=[]
        for d in range(2):
            ix=d*n*n+group[:,None]*n+group[None,:];di=d*n*n+group*(n+1)
            variance=np.diag(cov)[ix]+np.diag(cov)[di][:,None]-2*cov[ix,di[:,None]]
            prior_bound=mu[ix]-mu[di][:,None]+q*np.sqrt(np.maximum(variance,0))
            observed_bound=upper[ix]-lower[di][:,None]
            logf=np.log(theta[d,group])[:,None]+np.minimum(prior_bound,observed_bound);np.fill_diagonal(logf,-np.inf)
            lf.append(logf);lu.append(np.log(theta[d,group]*noise_power(cfg))-lower[di])
        lf=np.array(lf);lu=np.array(lu);power=minimum_power(cfg,(lf,np.zeros_like(lf),lu,np.zeros_like(lu)))
        value=float(sc['payload'][:,group].sum()/1000)
        if power is not None:return dict(group=group,power=power,mode='reuse',value=value)
        di=np.array([group*(n+1),n*n+group*(n+1)]);cap=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
        rate=cfg.bandwidth_hz*np.logaddexp(0.,np.log(cap/noise_power(cfg))+lower[di])/np.log(2)
        with np.errstate(divide='ignore',over='ignore'):
            airtime=sc['payload'][:,group]/rate
        if np.all(airtime.sum(axis=1)<=duration):
            return dict(group=group,power=np.broadcast_to(cap,airtime.shape).copy(),airtime=airtime,mode='OMA',value=value)
    return empty

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='../ray_data');p.add_argument('--output',default='../revision_results');a=p.parse_args();root=Path(a.data);out=Path(a.output)
    records=[json.loads(x) for x in (out/'ray_records.jsonl').read_text().splitlines()]
    indexed={(r['scenario'],r['ray_index']):r for r in records if r['method']=='C-IVP'}
    cfg=replace(Config(),ul_fraction=.5,ul_power_w=.2,dl_power_w=1.);meta_out=[]
    with (out/'conformal_records.jsonl').open('w') as f:
        for scene_index,name in enumerate(['canyon_low','canyon_high','munich']):
            data=np.load(root/f'{name}.npz');meta=json.loads((root/f'{name}.json').read_text())
            for kind,mobile in [('incoherent',False),('coherent',True)]:
                prior=fit(data,meta,kind);q=conformal_quantile(prior,meta);label=name+('-motion' if mobile else '-static')
                meta_out.append(dict(scenario=label,q=q,prior_risk=.04,report_risk=.01))
                for index in range(meta['test']):
                    seed=210000+scene_index*1000+index;sc=case(cfg,data,meta,prior,index,kind,mobile,False,seed)
                    for method in ['Safe-NR','Safe-IVP']:
                        observations=indexed[label,index]['observations'] if method=='Safe-IVP' else []
                        t=time.perf_counter();act=safe_schedule(cfg,sc,prior['cov'],q,observations);elapsed=time.perf_counter()-t
                        r=execute(cfg,sc,act,len(observations));r.update(method=method,scenario=label,seed=seed,ray_index=index,probes=len(observations),
                            observations=observations,mode=act['mode'],selected=list(map(int,act['group'])),power=act['power'].tolist(),
                            wall_s=elapsed+(indexed[label,index]['wall_s'] if method=='Safe-IVP' else 0.),demand_kbit=float(sc['payload'].sum()/1000))
                        if 'airtime' in act:r['airtime']=act['airtime'].tolist()
                        f.write(json.dumps(r)+'\n')
                f.flush()
    (out/'conformal_protocol.json').write_text(json.dumps(dict(settings=meta_out,guarantee='Marginal over exchangeable static calibration/test frames and independent report noise; no Gaussian posterior assumption; reciprocity is exact in these ray scenes.',
        limitation='Motion breaks static-frame exchangeability with execution; nondegradation of Gaussian certified utility is not claimed for the filtered action.'),indent=2))

if __name__=='__main__':main()
