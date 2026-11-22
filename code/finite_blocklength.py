"""Reuse-only finite-blocklength NORMAL-APPROXIMATION diagnostic.

No exact code-achievability claim. Time is split into fixed-length complex-AWGN
blocks; deterministic SINR targets preserve scalar innovation intervals.
"""
import json,ctypes,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from model import Config,scenario,posterior,fantasy,noise_power
from fast_backend import schedule,library,all_masks,quantiles,parameters,ptr,iptr
from innovation import expected_envelope
from round2_experiments import write_record

def coding_targets(cfg,payload,m,blocklength):
    durations=(cfg.frame_s-m*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
    if blocklength<=0:raise ValueError('Block length must be positive')
    blocks=np.floor(cfg.bandwidth_hz*durations/blocklength).astype(int)
    if np.any(blocks<1):raise ValueError('Each direction must have time for at least one complete codeword')
    rate=payload/(blocks[:,None]*blocklength);eps=.01/(2*cfg.n*blocks)
    theta=np.zeros_like(payload)
    for d in range(2):
        for i in range(cfg.n):
            def residual(g):return np.log2(1+g)-norm.isf(eps[d])*np.sqrt(1-(1+g)**-2)/(np.log(2)*np.sqrt(blocklength))-rate[d,i]
            hi=2.**(rate[d,i]+2)
            while residual(hi)<0:hi*=2
            theta[d,i]=brentq(residual,0.,hi,xtol=1e-12)
    proxy=cfg.bandwidth_hz*durations[:,None]*np.log2(1+theta)
    return theta,blocks,rate,proxy

def reuse_schedule(cfg,mu,cov,payload,proxy,m):
    masks=all_masks(cfg.n);weights=payload.sum(axis=0)/1000
    order=sorted(masks,key=lambda mask:-sum(weights[i] for i in range(cfg.n) if mask&(1<<i)))
    for mask in order:
        a=schedule(cfg,mu,cov,proxy,m,masks=np.array([mask],np.int32))
        if a['mode']=='reuse':a['value']=float(weights[a['group']].sum());return a
    return dict(group=np.array([],int),power=np.zeros((2,0)),value=0.,mode='empty')

def reuse_value(cfg,mu,cov,payload,proxy,m,index):
    lib=library();masks=all_masks(cfg.n);b,cs=fantasy(cov,index,(cfg.measurement_db*np.log(10)/10)**2)
    kr,ko=quantiles(cfg.n,cfg.outage);pars=parameters(cfg,m+1);grid=np.linspace(-6,6,25);out=np.zeros(len(masks)*6)
    arrays=[np.ascontiguousarray(x,float) for x in [mu,cs,b,proxy]]
    lib.ivp_intervals(cfg.n,*map(ptr,arrays),ptr(pars),ptr(kr),ptr(ko),iptr(masks),len(masks),ptr(grid),len(grid),8,ptr(out))
    intervals=out.reshape(-1,2,3)[:,0,:].copy();weights=payload.sum(axis=0)/1000
    for row,mask in zip(intervals,masks):row[2]=sum(weights[i] for i in range(cfg.n) if mask&(1<<i)) if row[1]>row[0] else 0.
    return expected_envelope(intervals.tolist())

def run_fbl(cfg,sc,blocklength,reporting):
    mu=sc['mean'].copy();cov=sc['covariance'].copy();obs=[];used=np.zeros(len(mu),int);rv=(cfg.measurement_db*np.log(10)/10)**2
    targets=[coding_targets(cfg,sc['payload'],m,blocklength) for m in range(cfg.max_probes+1)]
    while True:
        m=len(obs);theta,blocks,rate,proxy=targets[m];act=reuse_schedule(cfg,mu,cov,sc['payload'],proxy,m)
        if not reporting or m==cfg.max_probes or act['value']>=sc['payload'].sum()/1000-1e-9:break
        qs=[reuse_value(cfg,mu,cov,sc['payload'],targets[m+1][3],m,int(k)) for k in sc['candidates']]
        if max(qs)<=act['value']+cfg.gain_tol_kbit:break
        k=int(sc['candidates'][np.argmax(qs)]);y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1
        mu,cov=posterior(mu,cov,k,y,rv);obs.append(dict(index=k,observation=y))
    group=act['group'];success=np.ones(len(group),bool);channel_ok=True;packet_fail=[];g=np.exp(sc['true'].reshape(2,cfg.n,cfg.n))
    uniform=np.random.default_rng(sc['seed']+25000000).random((2,cfg.n))
    for d in range(2):
        for ii,i in enumerate(group):
            interference=sum(act['power'][d,jj]*g[d,i,j] for jj,j in enumerate(group) if i!=j)
            sinr=act['power'][d,ii]*g[d,i,i]/(noise_power(cfg)+interference);channel_ok &= sinr>=theta[d,i]*(1-1e-9)
            dispersion=max(1-(1+sinr)**-2,1e-15)
            error=float(norm.sf((np.log2(1+sinr)-rate[d,i])*np.log(2)*np.sqrt(blocklength/dispersion)))
            complete_probability=float(np.exp(blocks[d]*np.log1p(-min(error,1-1e-15))))
            success[ii] &= uniform[d,i]<complete_probability;packet_fail.append(error)
    completed=float(sc['payload'][:,group[success]].sum()/1000)
    return dict(seed=sc['seed'],blocklength=blocklength,method='FBL-IVP' if reporting else 'FBL-NR',probes=len(obs),observations=obs,
        selected=group.tolist(),power=act['power'].tolist(),mode=act['mode'],admitted_kbit=act['value'],completed_kbit=completed,
        outage=int(len(group)>0 and not success.all()),target_outage=int(len(group)>0 and not channel_ok),active=int(len(group)>0),
        blocks=blocks.tolist(),thresholds=theta.tolist(),packet_error_probabilities=packet_fail)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output');a=p.parse_args()
    root=Path(a.root);out=Path(a.output) if a.output else root/'round3_results';out.mkdir(parents=True,exist_ok=True)
    cfg=replace(Config(),outage=.04,max_probes=2);library()
    with (out/'finite_blocklength_records.jsonl').open('w') as f:
        for L in [128,512]:
            for seed in range(350000,350256):
                s=scenario(cfg,seed)
                for reporting in [False,True]:write_record(f,run_fbl(cfg,s,L,reporting))
            f.flush();print('FBL',L,flush=True)

if __name__=='__main__':main()
