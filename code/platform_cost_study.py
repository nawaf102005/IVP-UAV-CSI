"""Platform-independent acquisition-cost sweeps and allocator ablation.

c is a counterfactual acquisition-decision budget per round. Elapsed host time
NEVER changes these outcomes. No-report avoids this additional planning cost;
common terminal-optimizer time is excluded. Reporting policies are compared
at equal hypothetical c, not equal measured implementation latency.
"""
import argparse,gzip,json,time,platform,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from scipy import stats
from model import Config,posterior
from mobility_cost_study import make_case,run_policy
from fast_backend import library,schedule,affine_value
from power_frontier import reference
from admission import execute
from revised_policies import gaussian_rule

ALPHAS=[0.,.001,.0025,.005,.01,.02,.04,.08,.16,.4]
SPEEDS=[0.,5.,10.,20.,30.,40.]
FRACTIONS=[0.,.01,.025,.05,.1]
SEED_START=93000000

def shadow_correlation(speed,age_s,correlation_length_m):
    if speed<0 or age_s<0 or correlation_length_m<=0:raise ValueError('Invalid physical correlation parameters')
    return float(np.exp(-speed*age_s/correlation_length_m))

def configurations():
    cells=[]
    for i,v in enumerate(SPEEDS):
        for j,f in enumerate(FRACTIONS):
            cfg=replace(Config(),probe_s=f*Config().frame_s)
            cells.append(dict(id=f'map_{i}_{j}',kind='map',speed=v,dcor=10.,rho=shadow_correlation(v,cfg.frame_s,10.),report_fraction=f,config=asdict(cfg),alphas=ALPHAS))
    for n in [4,6,8]:
        for ms in [20,50,100]:
            scale=ms/20;cfg=replace(Config(),n=n,frame_s=ms/1000,ul_bits=30000*scale,dl_bits=1800*scale)
            cells.append(dict(id=f'slice_N{n}_T{ms}',kind='size_duration',speed=20.,dcor=10.,rho=shadow_correlation(20.,cfg.frame_s,10.),report_fraction=cfg.probe_s/cfg.frame_s,config=asdict(cfg),alphas=ALPHAS))
    cfg=Config()
    cells.append(dict(id='frontier_N4_T20',kind='frontier',speed=20.,dcor=10.,rho=shadow_correlation(20.,cfg.frame_s,10.),report_fraction=cfg.probe_s/cfg.frame_s,config=asdict(cfg),alphas=ALPHAS[:6]))
    return cells

def frontier_terminal(cfg,sc,jm,jc,policy,cost_s):
    mean=jm.copy();cov=jc.copy();D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2
    for obs in policy['observations']:mean,cov=posterior(mean,cov,obs['index'],obs['observation'],rv)
    m=policy['probes'];reserved=(m+1)*cost_s;remaining=cfg.frame_s-m*cfg.probe_s-reserved
    if remaining<=0:action=dict(group=np.array([],int),power=np.zeros((2,0)),mode='empty',value=0.);trials=[]
    else:action,trials=reference(replace(cfg,frame_s=remaining),{**sc,'mean':mean[D:],'covariance':cov[D:,D:]})
    r=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,action,m)
    r.update(method=policy['method']+'-F',seed=policy['seed'],probes=m,observations=policy['observations'],history=policy['history'],reserve_s=cost_s,reserved_s=reserved,enforce_elapsed=False,selected=action['group'].tolist(),mode=action['mode'],power=action['power'].tolist(),trials=trials)
    if 'airtime' in action:r['airtime']=action['airtime'].tolist()
    return r

def compact(r,alpha):
    keep=['method','seed','completed_kbit','admitted_kbit','completed_tasks','admitted_tasks','outage','active','probes','observations','history','selected','power','mode','airtime','reserve_s','reserved_s','enforce_elapsed','trials']
    return {**{k:r[k] for k in keep if k in r},'alpha':alpha}

def interval(d,family):
    d=np.asarray(d,float);mean=float(d.mean());h=float(stats.t.ppf(1-.05/(2*family),len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
    return dict(mean=mean,low=mean-h,high=mean+h)

def run_cell(job):
    out,cell,frames,family=job;out=Path(out);library();cfg=Config(**cell['config']);front=cell['kind']=='frontier';records=[];baseline=[];byalpha=[]
    cached=[]
    for seed in range(SEED_START,SEED_START+frames):
        sc,jm,jc=make_case(cfg,seed,cell['rho']);cached.append((sc,jm,jc));r=run_policy(cfg,sc,jm,jc,'Predictive-NR',enforce_elapsed=False)
        if front:r=frontier_terminal(cfg,sc,jm,jc,r,0.)
        r=compact(r,0.);baseline.append(r);records.append(r)
    for alpha in cell['alphas']:
        policies={m:[] for m in ['Predictive-IVP','Predictive-Margin']}
        for sc,jm,jc in cached:
            for method in policies:
                r=run_policy(cfg,sc,jm,jc,method,reserve_s=alpha*cfg.frame_s,enforce_elapsed=False)
                if front:r=frontier_terminal(cfg,sc,jm,jc,r,alpha*cfg.frame_s)
                r=compact(r,alpha);policies[method].append(r);records.append(r)
        result=dict(alpha=alpha,methods={})
        for label,rs in [('NR',baseline)]+list(policies.items()):result['methods'][label]=dict(completed=float(np.mean([r['completed_kbit'] for r in rs])),outages=sum(r['outage'] for r in rs),mean_reports=float(np.mean([r['probes'] for r in rs])))
        for label,aa,bb in [('IVP-NR',policies['Predictive-IVP'],baseline),('Margin-NR',policies['Predictive-Margin'],baseline),('IVP-Margin',policies['Predictive-IVP'],policies['Predictive-Margin'])]:
            result[label]=interval([a['completed_kbit']-b['completed_kbit'] for a,b in zip(aa,bb)],family)
        byalpha.append(result)
    path=out/(cell['id']+'.jsonl.gz')
    with gzip.open(path,'wt') as f:
        for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    summary=dict(cell=cell,frames=frames,records=len(records),costs=byalpha)
    (out/(cell['id']+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    return cell['id'],len(records)

def lookahead_policy(cfg,sc,jm,jc,cost_s):
    """Two-step look-ahead adaptation; correlated noisy reports, 7-node first stage.

    Same reward and terminal allocator as IVP. This is an adaptation of the
    look-ahead strategy in Chang--Liu, not reproduction of their independent
    multichannel MAC model or its optimality result.
    """
    library();mean=jm.copy();cov=jc.copy();D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(D,int);obs=[];history=[];z,w=gaussian_rule(7)
    while True:
        m=len(obs);current=schedule(cfg,mean[D:],cov[D:,D:],sc['payload'],m,reserve_s=cost_s)
        if m==2 or current['value']>=sc['payload'].sum()/1000-1e-9:break
        estimates=[]
        for k in sc['candidates']:
            b=cov[:,k]/np.sqrt(cov[k,k]+rv);cs=cov-np.outer(b,b)
            if m==1:
                estimates.append(affine_value(cfg,mean[D:],cs[D:,D:],b[D:],sc['payload'],m+1,reserve_s=cost_s));continue
            q=0.
            for zz,ww in zip(z,w):
                mm=mean+b*zz;base=schedule(cfg,mm[D:],cs[D:,D:],sc['payload'],1,reserve_s=cost_s)['value']
                if base<sc['payload'].sum()/1000-1e-9:
                    follow=[]
                    for j in sc['candidates']:
                        bb=cs[D:,j]/np.sqrt(cs[j,j]+rv);cc=cs[D:,D:]-np.outer(bb,bb)
                        follow.append(affine_value(cfg,mm[D:],cc,bb,sc['payload'],2,reserve_s=cost_s))
                    base=max(base,max(follow))
                q+=ww*base
            estimates.append(q)
        k=int(sc['candidates'][np.argmax(estimates)]);q=float(max(estimates));history.append(dict(index=k,current=current['value'],forecast=q))
        if q<=current['value']+cfg.gain_tol_kbit:break
        y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1;mean,cov=posterior(mean,cov,k,y,rv);obs.append(dict(index=k,observation=y))
    m=len(obs);reserved=(m+1)*cost_s;r=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,current,m)
    r.update(method='Two-step-LA',seed=int(sc['seed']),probes=m,observations=obs,history=history,reserve_s=cost_s,reserved_s=reserved,enforce_elapsed=False,selected=current['group'].tolist(),mode=current['mode'],power=current['power'].tolist())
    if 'airtime' in current:r['airtime']=current['airtime'].tolist()
    return r

def lookahead_study(out,frames,family):
    cfg=replace(Config(),max_probes=2);rho=shadow_correlation(20.,cfg.frame_s,10.);rows=[];summary=[]
    for alpha in [0.,.005,.02]:
        by={m:[] for m in ['Predictive-NR','Predictive-IVP','Two-step-LA']}
        for seed in range(SEED_START,SEED_START+frames):
            sc,jm,jc=make_case(cfg,seed,rho)
            for method in by:
                r=lookahead_policy(cfg,sc,jm,jc,alpha*cfg.frame_s) if method=='Two-step-LA' else run_policy(cfg,sc,jm,jc,method,reserve_s=0. if method=='Predictive-NR' else alpha*cfg.frame_s,enforce_elapsed=False)
                r=compact(r,alpha);by[method].append(r);rows.append(r)
        s=dict(alpha=alpha,methods={m:dict(completed=float(np.mean([r['completed_kbit'] for r in rs])),outages=sum(r['outage'] for r in rs),mean_reports=float(np.mean([r['probes'] for r in rs]))) for m,rs in by.items()})
        for label,a,b in [('LA-IVP','Two-step-LA','Predictive-IVP'),('LA-NR','Two-step-LA','Predictive-NR')]:s[label]=interval([x['completed_kbit']-y['completed_kbit'] for x,y in zip(by[a],by[b])],family)
        summary.append(s)
    with gzip.open(out/'lookahead_records.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    (out/'lookahead_summary.json').write_text(json.dumps(dict(config=asdict(cfg),rho=rho,frames=frames,costs=summary),indent=2)+'\n');return len(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=128);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Use a new directory; do not overwrite a partially completed protocol')
    cells=configurations();family=3*sum(len(c['alphas']) for c in cells)+6
    protocol=dict(study='Platform-independent acquisition-cost tolerance',cells=cells,test_frames=a.frames,test_seed=SEED_START,cost_definition='Additional acquisition-decision budget c per decision round, paid m+1 times; no-report avoids the additional evaluator. Common terminal-optimization time is excluded. IVP and margin are evaluated at equal counterfactual c; elapsed host time never affects outcomes.',scale='T=20/50/100 ms with payload proportional to T, fixed tau=0.75 ms in size/duration slices; N=4/6/8. Physical correlation exp(-v*T/dcor), v=20 m/s and illustrative 1/e dcor=10 m.',frontier='N4 T20 v20 tau0.75ms, same seeds and first six cost levels; all methods use same independently validated finite terminal frontier. Report selection remains certificate-based: terminal-allocator ablation, not frontier-optimal acquisition.',lookahead='Chang--Liu-style two-step adaptation on separate M=2 comparison with matched utility, posterior and terminal allocator; first-stage GH7 and second-stage CDF25. Not an exact replication of the original independent-channel system.',statistics=dict(simultaneous_contrasts=family,method='Bonferroni 95% paired t intervals; pointwise paired bootstrap boundary intervals reported separately',boundary='First crossing of piecewise-linear empirical mean gain if positive at c=0; all sign changes and positive regions retained. No continuous monotonicity assumption or universal threshold guarantee.'),workers=a.workers,python=sys.version,platform=platform.platform())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');library()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_cell,(str(out),cell,a.frames,family)):cell['id'] for cell in cells}
        for f in as_completed(futures):print('completed',*f.result(),flush=True)
    print('lookahead decisions',lookahead_study(out,a.frames,family),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(cells=len(cells),frames=a.frames,simultaneous_family=family,complete=True),indent=2)+'\n')
if __name__=='__main__':main()
