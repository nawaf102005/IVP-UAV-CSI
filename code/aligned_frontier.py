"""Frontier-aware acquisition diagnostic with a separate final validation stream.

Power scaling and OMA filling alone preserve certificate admission values.
This implementation evaluates the seven-margin finite family, including both
operations, inside its forecast. The sampled frontier's acceptance region is
not covered by the scalar interval theorem. Forecast normal-quantile midpoint
integration is approximate; no lower-value or nondegradation claim is made.
Planning draws are separate from the final 4096-draw validation gate, which
is never accessed during report selection.
"""
from pathlib import Path
from functools import lru_cache
from dataclasses import replace
import ctypes as ct,time
import numpy as np
from scipy.special import ndtri,ndtr
from scipy.stats import binom
from model import posterior
from fast_backend import library,ptr,iptr,all_masks,quantiles,parameters
from chance_benchmark import draw
from admission import execute
KNOBS=(.01,.025,.05,.1,.2,.4,.8)
PLAN_OFFSET=5600000
FINAL_OFFSET=4200000

@lru_cache(None)
def acceptance_count(n,samples=4096):
    family=2*(2**n-1)*len(KNOBS);alpha=.05/family
    legal=np.where(binom.cdf(np.arange(samples+1),samples,.05)<=alpha)[0]
    return int(legal[-1]) if len(legal) else -1

@lru_cache(None)
def arrays(n):
    kr=[];ko=[]
    for knob in KNOBS:
        a,b=quantiles(n,knob);kr.extend(a);ko.extend(b)
    return np.array(kr),np.array(ko)

def native_values(cfg,mean,cov,payload,count,seed,direction=None,nodes=None,reserve_s=0.,offset=PLAN_OFFSET,samples=4096):
    if cfg.n>8:raise ValueError('Frontier diagnostic is limited to small clusters')
    lib=library();dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int)
    lib.ivp_frontier_values.argtypes=[ct.c_int,dp,dp,dp,dp,dp,dp,dp,ct.c_int,ip,ct.c_int,dp,ct.c_int,ct.c_int,dp,ct.c_int,dp,dp];lib.ivp_frontier_values.restype=None
    nodes=np.array([0.]) if nodes is None else np.ascontiguousarray(nodes,float)
    direction=np.zeros_like(mean) if direction is None else np.ascontiguousarray(direction,float)
    mean=np.ascontiguousarray(mean,float);cov=np.ascontiguousarray(cov,float);payload=np.ascontiguousarray(payload,float)
    gains=np.ascontiguousarray(draw(mean,cov,samples,seed+offset,cfg.n));kr,ko=arrays(cfg.n);masks=all_masks(cfg.n);par=parameters(cfg,count,reserve_s=reserve_s);q=np.zeros(len(nodes));out=np.zeros((len(nodes),5+4*cfg.n))
    lib.ivp_frontier_values(cfg.n,ptr(mean),ptr(cov),ptr(direction),ptr(payload),ptr(par),ptr(kr),ptr(ko),len(KNOBS),iptr(masks),len(masks),ptr(gains),samples,acceptance_count(cfg.n,samples),ptr(nodes),len(nodes),ptr(q),ptr(out))
    return q,out

def action_from_row(cfg,row):
    g=np.array([i for i in range(cfg.n) if int(row[0])&(1<<i)],int);mode={-1:'empty',0:'reuse',1:'OMA'}[int(row[1])]
    a=dict(group=g,mode=mode,value=float(row[2]),power=row[5:5+2*cfg.n].reshape(2,cfg.n)[:,g],validation_bad=int(row[3]),knob_index=int(row[4]))
    if mode=='OMA':a['airtime']=row[5+2*cfg.n:].reshape(2,cfg.n)[:,g]
    return a

def terminal(cfg,sc,mean,cov,count,reserve_s=0.,offset=FINAL_OFFSET):
    q,out=native_values(cfg,mean,cov,sc['payload'],count,int(sc['seed']),reserve_s=reserve_s,offset=offset)
    return action_from_row(cfg,out[0])

def forecast(cfg,mean,post_cov,direction,payload,next_count,seed,reserve_s=0.,knots=49):
    # Equal-normal-mass midpoint rule on the same [-6,6] truncated domain.
    lo=ndtr(-6.);hi=ndtr(6.);nodes=ndtri(lo+(np.arange(knots)+.5)*(hi-lo)/knots)
    q,_=native_values(cfg,mean,post_cov,payload,next_count,seed,direction,nodes,reserve_s,PLAN_OFFSET)
    return float(q.mean()*(hi-lo))

def run_aligned(cfg,sc,jm,jc,cost_s,knots=49):
    mean=jm.copy();cov=jc.copy();D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(D,int);obs=[];history=[];tick=time.perf_counter()
    while True:
        m=len(obs)
        current=terminal(cfg,sc,mean[D:],cov[D:,D:],m,cost_s,offset=PLAN_OFFSET)
        if m==cfg.max_probes or current['value']>=sc['payload'].sum()/1000-1e-9 or cfg.frame_s-m*cfg.probe_s-(m+1)*cost_s<=0:break
        qs=[]
        for k in sc['candidates']:
            b=cov[D:,k]/np.sqrt(cov[k,k]+rv);post=cov[D:,D:]-np.outer(b,b)
            qs.append(forecast(cfg,mean[D:],post,b,sc['payload'],m+1,sc['seed'],cost_s,knots))
        k=int(sc['candidates'][np.argmax(qs)]);q=float(max(qs));history.append(dict(current=current['value'],forecast=q,index=k))
        if q<=current['value']+cfg.gain_tol_kbit:break
        y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1;mean,cov=posterior(mean,cov,k,y,rv);obs.append(dict(index=k,observation=y))
    # Final gate is independent of every planning draw and every forecast.
    m=len(obs);a=terminal(cfg,sc,mean[D:],cov[D:,D:],m,cost_s);reserved=(m+1)*cost_s
    r=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,a,m)
    r.update(method='Frontier-aware-IVP',seed=int(sc['seed']),probes=m,observations=obs,history=history,reserve_s=cost_s,reserved_s=reserved,enforce_elapsed=False,selected=a['group'].tolist(),mode=a['mode'],power=a['power'].tolist(),validation_bad=a['validation_bad'],knob_index=a['knob_index'],planner_wall_s=time.perf_counter()-tick)
    if 'airtime' in a:r['airtime']=a['airtime'].tolist()
    return r
