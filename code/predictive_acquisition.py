"""Value acquisition about an execution state correlated with observations.

Joint Gaussian conditioning gives affine target means and fixed covariance.
Ray experiments also use a separate pair-calibrated endpoint filter; its
marginal guarantee does not depend on a Gaussian posterior or cover a trajectory.
"""
import time
import numpy as np
from scipy.stats import norm
from model import posterior,noise_power,thresholds
from fast_backend import schedule,affine_value,library
from admission import execute,subsets,minimum_power
from ray_evaluation import log_vectors
from round2_experiments import full_scores,regularized_covariance

def run_predictive(cfg,sc,joint_mean,joint_cov,method='Predictive-IVP'):
    library();started=time.perf_counter();mean=joint_mean.copy();cov=joint_cov.copy();D=len(sc['mean'])
    used=np.zeros(D,int);obs=[];rv=(cfg.measurement_db*np.log(10)/10)**2;history=[]
    while True:
        m=len(obs);current=schedule(cfg,mean[D:],cov[D:,D:],sc['payload'],m)
        if method.endswith('NR') or m==cfg.max_probes or current['value']>=sc['payload'].sum()/1000-1e-9:break
        forecasts=[]
        for k in sc['candidates']:
            direction=cov[D:,k]/np.sqrt(cov[k,k]+rv)
            new_cov=cov[D:,D:]-np.outer(direction,direction)
            forecasts.append(affine_value(cfg,mean[D:],new_cov,direction,sc['payload'],m+1))
        k=int(sc['candidates'][np.argmax(forecasts)]);q=max(forecasts)
        history.append(dict(current=current['value'],forecast=q,index=k))
        if q<=current['value']+cfg.gain_tol_kbit:break
        y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1
        mean,cov=posterior(mean,cov,k,y,rv);obs.append(dict(index=k,observation=y))
    r=execute(cfg,sc,current,len(obs));r.update(method=method,seed=sc['seed'],probes=len(obs),observations=obs,
        selected=current['group'].tolist(),power=current['power'].tolist(),mode=current['mode'],history=history,wall_s=time.perf_counter()-started)
    if 'airtime' in current:r['airtime']=current['airtime'].tolist()
    return r

def endpoint_fit(data,meta,noise_db=1.):
    x0=log_vectors(data['coherent']);xe=log_vectors(data['coherent_future']);t=meta['train'];c=meta['calibration'];D=x0.shape[1]
    joint=np.c_[x0,xe];mean=joint[:t].mean(axis=0);raw=np.cov(joint[:t],rowvar=False)
    cov=.85*raw+.15*np.diag(np.diag(raw))+np.eye(2*D)*1e-6
    ecov=regularized_covariance(cov[D:,D:],noise_db)
    drift=xe-x0;dm=drift[:t].mean(axis=0);ds=np.maximum(drift[:t].std(axis=0,ddof=1),noise_db*np.log(10)/10)
    scores=np.maximum(full_scores(xe[t:t+c],mean[D:],ecov),np.max(np.abs(drift[t:t+c]-dm)/ds,axis=1))
    rank=int(np.ceil((c+1)*.96));q=float(np.sort(scores)[rank-1]) if rank<=c else np.inf
    return dict(joint_mean=mean,joint_cov=cov,mean=mean[D:],cov=ecov,drift_mean=dm,drift_sd=ds,q=q,
                calibration_scores=scores,x0=x0,xe=xe)

def endpoint_schedule(cfg,sc,fitted,observations):
    n=cfg.n;mu=fitted['mean'];cov=fitted['cov'];q=fitted['q'];std=np.sqrt(np.diag(cov))
    lower=mu-q*std;upper=mu+q*std
    nu=norm.isf(.01/(2*max(cfg.max_probes,1)))*cfg.measurement_db*np.log(10)/10
    for obs in observations:
        k=obs['index'];d,ij=divmod(k,n*n);i,j=divmod(ij,n)
        for ix in [k,(1-d)*n*n+j*n+i]: # exact reciprocity of these ray arrays
            center=obs['observation']+fitted['drift_mean'][ix]
            radius=nu+q*fitted['drift_sd'][ix]
            lower[ix]=max(lower[ix],center-radius);upper[ix]=min(upper[ix],center+radius)
    empty=dict(group=np.array([],int),power=np.zeros((2,0)),mode='empty',value=0.)
    if np.any(lower>upper):return empty
    theta=thresholds(cfg,sc['payload'],len(observations));duration=(cfg.frame_s-len(observations)*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
    for group in sorted(subsets(n),key=lambda g:-sc['payload'][:,g].sum()):
        lf=[];lu=[]
        for d in range(2):
            ix=d*n*n+group[:,None]*n+group[None,:];di=d*n*n+group*(n+1)
            variance=np.diag(cov)[ix]+np.diag(cov)[di][:,None]-2*cov[ix,di[:,None]]
            cap=mu[ix]-mu[di][:,None]+q*np.sqrt(np.maximum(variance,0))
            logf=np.log(theta[d,group])[:,None]+np.minimum(cap,upper[ix]-lower[di][:,None]);np.fill_diagonal(logf,-np.inf)
            lf.append(logf);lu.append(np.log(theta[d,group]*noise_power(cfg))-lower[di])
        lf=np.array(lf);lu=np.array(lu);power=minimum_power(cfg,(lf,np.zeros_like(lf),lu,np.zeros_like(lu)))
        value=float(sc['payload'][:,group].sum()/1000)
        if power is not None:return dict(group=group,power=power,mode='reuse',value=value)
        di=np.array([group*(n+1),n*n+group*(n+1)]);cap=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
        rate=cfg.bandwidth_hz*np.logaddexp(0.,np.log(cap/noise_power(cfg))+lower[di])/np.log(2)
        times=sc['payload'][:,group]/rate
        if np.all(times.sum(axis=1)<=duration):return dict(group=group,power=np.broadcast_to(cap,times.shape).copy(),airtime=times,mode='OMA',value=value)
    return empty
