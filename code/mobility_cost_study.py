"""Paired large-scale aging/report-cost map with optional charged computation.

Known jointly Gaussian initial/execution log gains; no fast-fading or speed
calibration is implied by rho. A fixed N=4 UL-heavy system isolates two axes.
CDF, sensitivity and no-report use the SAME terminal certificate allocator.
Settings are fixed in protocol.json before the new run. Development and test
seeds are separate. This extends the model study, not the field evidence.
"""
import argparse,json,time,sys,platform
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from scipy import stats
from model import Config,scenario,posterior
from admission import coefficients,execute
from fast_backend import schedule,affine_value,library

METHODS=['Predictive-IVP','Predictive-Margin','Predictive-NR']
RHOS=[.4,.6,.75,.9,.95,1.]
FRACTIONS=[0.,.01,.025,.05,.1]

def target_margin_scores(cfg,mean,cov,payload,count,candidates,cross_cov,observation_cov):
    lf,_,lu,_=coefficients(cfg,mean,cov,payload,count,np.arange(cfg.n))
    f=np.exp(lf);u=np.exp(lu)/np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
    row=f.sum(axis=-1)+u;d,i=np.unravel_index(np.argmax(row),row.shape)
    gradient=np.zeros(2*cfg.n*cfg.n)
    gradient[d*cfg.n*cfg.n+i*cfg.n+np.arange(cfg.n)]=f[d,i]
    gradient[d*cfg.n*cfg.n+i*(cfg.n+1)]=-row[d,i]
    rv=(cfg.measurement_db*np.log(10)/10)**2
    return (gradient@cross_cov[:,candidates])**2/(np.diag(observation_cov)[candidates]+rv)

def make_case(cfg,seed,rho):
    sc=scenario(cfg,seed);c=sc['covariance'];mu=sc['mean'];D=len(mu)
    eps=np.linalg.cholesky(c+np.eye(D)*1e-14)@np.random.default_rng(seed+15000000).normal(size=D)
    sc['execution_true']=mu+rho*(sc['true']-mu)+np.sqrt(max(0,1-rho*rho))*eps
    joint_mean=np.r_[mu,mu];joint_cov=np.block([[c,rho*c],[rho*c,c]])
    return sc,joint_mean,joint_cov

def run_policy(cfg,sc,joint_mean,joint_cov,method,reserve_s=0.,enforce_elapsed=True):
    library();started=time.perf_counter();mean=joint_mean.copy();cov=joint_cov.copy();D=len(sc['mean']);used=np.zeros(D,int);obs=[];history=[]
    rv=(cfg.measurement_db*np.log(10)/10)**2
    while True:
        m=len(obs);current=schedule(cfg,mean[D:],cov[D:,D:],sc['payload'],m,reserve_s=reserve_s)
        if method=='Predictive-NR' or m==cfg.max_probes or current['value']>=sc['payload'].sum()/1000-1e-9:break
        if cfg.frame_s-m*cfg.probe_s-(m+1)*reserve_s<=0:break
        candidates=sc['candidates']
        if method=='Predictive-Margin':
            timing_cfg=replace(cfg,frame_s=max(1e-12,cfg.frame_s-(m+1)*reserve_s))
            scores=target_margin_scores(timing_cfg,mean[D:],cov[D:,D:],sc['payload'],m,candidates,cov[D:,:D],cov[:D,:D]);candidates=candidates[[int(np.argmax(scores))]]
        forecasts=[]
        for k in candidates:
            b=cov[D:,k]/np.sqrt(cov[k,k]+rv);post=cov[D:,D:]-np.outer(b,b)
            forecasts.append(affine_value(cfg,mean[D:],post,b,sc['payload'],m+1,reserve_s=reserve_s))
        k=int(candidates[np.argmax(forecasts)]);q=float(max(forecasts));history.append(dict(current=current['value'],forecast=q,index=k))
        if q<=current['value']+cfg.gain_tol_kbit:break
        y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1;mean,cov=posterior(mean,cov,k,y,rv);obs.append(dict(index=k,observation=y))
    wall=time.perf_counter()-started;m=len(obs);reserved=(m+1)*reserve_s;eval_cfg=replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved));result=execute(eval_cfg,sc,current,m)
    radio_completed=result['completed_kbit'];radio_outage=result['outage'];overrun=enforce_elapsed and reserve_s>0 and wall>reserved
    if overrun and result['active']:result.update(completed_kbit=0.,completed_tasks=0,outage=1)
    result.update(method=method,seed=int(sc['seed']),probes=m,wall_s=wall,reserve_s=reserve_s,reserved_s=reserved,deadline_overrun=int(overrun),radio_completed_kbit=radio_completed,radio_outage=radio_outage,observations=obs,history=history,selected=current['group'].tolist(),power=current['power'].tolist(),mode=current['mode'],config=asdict(cfg),enforce_elapsed=enforce_elapsed)
    if 'airtime' in current:result['airtime']=current['airtime'].tolist()
    return result

def paired_interval(d,family):
    d=np.asarray(d,float);mean=float(d.mean());half=float(stats.t.ppf(1-.05/(2*family),len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
    return dict(mean=mean,low=mean-half,high=mean+half)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--frames',type=int,default=128);parser.add_argument('--development',type=int,default=128);args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'records.jsonl').exists():raise FileExistsError('Choose fresh output')
    cfg=Config();protocol=dict(study='Post-review joint aging and report-cost diagnostic',config=asdict(cfg),rhos=RHOS,report_fractions=FRACTIONS,methods=METHODS,test_frames=args.frames,test_seed=92000000,development_frames=args.development,development_seed=91000000,development_condition=dict(rho=.75,report_fraction=cfg.probe_s/cfg.frame_s),reservations='Twice development p95 of per-round average wall time, rounded upward to 0.05 ms; one reservation per method, fixed over rho and tau/T.',timing='One warmed thread, serial randomized method order; generation and execution checks excluded from online wall time.',statistics='Bonferroni simultaneous 95% paired t intervals across both IVP contrasts at all 30 cells and both cost modes (120 contrasts). Wilson risk intervals are pointwise; no reliability certification from finite counts.',scope='Known Gaussian initial/execution log gains at fixed N=4, UL-heavy load. rho is not converted to speed or a Doppler coherence time. Same sufficient-certificate allocator, not the stronger validated frontier. Endpoint execution, not time-integrated packets.',python=sys.version,platform=platform.platform())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');library()
    for seed in range(90999997,91000000):
        sc,jm,jc=make_case(cfg,seed,.75)
        for m in METHODS:run_policy(cfg,sc,jm,jc,m)
    development=[];costs={m:[] for m in METHODS}
    for seed in range(91000000,91000000+args.development):
        sc,jm,jc=make_case(cfg,seed,.75)
        for method in np.random.default_rng(seed).permutation(METHODS):
            r=run_policy(cfg,sc,jm,jc,method);development.append(r);costs[method].append(r['wall_s']/(r['probes']+1))
    reserves={m:float(np.ceil(2*np.quantile(v,.95)/.00005)*.00005) for m,v in costs.items()}
    (out/'reservations.json').write_text(json.dumps(dict(reserves=reserves,development=development),indent=2)+'\n');print('reservations',reserves,flush=True)
    summaries=[];family=2*len(RHOS)*len(FRACTIONS)*2
    with (out/'records.jsonl').open('w') as stream:
        for rho in RHOS:
            for fraction in FRACTIONS:
                cfg=replace(Config(),probe_s=fraction*Config().frame_s)
                for charged in [False,True]:
                    records=[]
                    for seed in range(92000000,92000000+args.frames):
                        sc,jm,jc=make_case(cfg,seed,rho)
                        for method in np.random.default_rng(seed+int(rho*1000)+int(fraction*100000)+int(charged)*7).permutation(METHODS):
                            r=run_policy(cfg,sc,jm,jc,method,reserves[method] if charged else 0.);r.update(rho=rho,report_fraction=fraction,charged=charged);stream.write(json.dumps(r)+'\n');records.append(r)
                    by_method={m:[r for r in records if r['method']==m] for m in METHODS}
                    for m in METHODS:by_method[m].sort(key=lambda r:r['seed'])
                    q=dict(rho=rho,report_fraction=fraction,charged=charged,frames=args.frames,methods={})
                    for method,rs in by_method.items():
                        successes=sum(r['outage'] for r in rs);z=stats.norm.ppf(.975);ph=successes/len(rs);den=1+z*z/len(rs);center=(ph+z*z/(2*len(rs)))/den;half=z*np.sqrt(ph*(1-ph)/len(rs)+z*z/(4*len(rs)**2))/den
                        q['methods'][method]=dict(completed=float(np.mean([r['completed_kbit'] for r in rs])),outages=successes,outage_wilson=[float(center-half),float(center+half)],mean_reports=float(np.mean([r['probes'] for r in rs])),mean_ms=float(1000*np.mean([r['wall_s'] for r in rs])),overruns=sum(r['deadline_overrun'] for r in rs))
                    for comparator in ['Predictive-Margin','Predictive-NR']:
                        q['IVP-minus-'+comparator]=paired_interval([a['completed_kbit']-b['completed_kbit'] for a,b in zip(by_method['Predictive-IVP'],by_method[comparator])],family)
                    summaries.append(q);stream.flush();print('map',rho,fraction,'charged',charged,flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()
