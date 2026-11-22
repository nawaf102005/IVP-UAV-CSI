"""Python interface to a small optional C++17 certificate kernel.

Build: python code/fast_backend.py. The original Python implementation remains
available. Report values returned here are conservative inner-interval values,
not a claim of exact equality to the reference interval minimizer.
"""
from pathlib import Path
import ctypes as ct
import functools,os,platform,subprocess
import numpy as np
from scipy.special import ndtri
from model import noise_power
from innovation import expected_envelope

ROOT=Path(__file__).resolve().parent
LIBRARY=None

def build():
    suffix='.dll' if os.name=='nt' else '.so'
    target=ROOT/('fast_kernel'+suffix);source=ROOT/'fast_kernel.cpp'
    if not target.exists() or target.stat().st_mtime<source.stat().st_mtime:
        command=[os.environ.get('CXX','g++'),'-std=c++17','-O3','-shared',str(source),'-o',str(target)]
        if os.name!='nt':command.insert(4,'-fPIC')
        subprocess.run(command,check=True)
    return target

def library():
    global LIBRARY
    if LIBRARY is None:
        LIBRARY=ct.CDLL(str(build()))
        dp=ct.POINTER(ct.c_double);ip=ct.POINTER(ct.c_int)
        LIBRARY.ivp_schedule.argtypes=[ct.c_int,dp,dp,dp,dp,dp,dp,ip,ct.c_int,ct.c_int,dp]
        LIBRARY.ivp_intervals.argtypes=[ct.c_int,dp,dp,dp,dp,dp,dp,dp,ip,ct.c_int,dp,ct.c_int,ct.c_int,dp]
        LIBRARY.ivp_scores.argtypes=[ct.c_int,dp,dp,dp,dp,dp,dp,ip,ct.c_int,dp]
        LIBRARY.ivp_integrate.argtypes=[dp,ct.c_int];LIBRARY.ivp_integrate.restype=ct.c_double
        LIBRARY.ivp_schedule.restype=LIBRARY.ivp_intervals.restype=None
    return LIBRARY

def ptr(a):return a.ctypes.data_as(ct.POINTER(ct.c_double))
def iptr(a):return a.ctypes.data_as(ct.POINTER(ct.c_int))

@functools.lru_cache(None)
def quantiles(n,delta,nominal=False):
    if nominal:return np.zeros(n+1),np.zeros(n+1)
    s=np.arange(1,n+1,dtype=float)
    return np.r_[0.,-ndtri(delta/(2*s*s))],np.r_[0.,-ndtri(delta/(2*s))]

@functools.lru_cache(None)
def all_masks(n):
    # Same cardinality/lexicographic tie ordering as the Python reference.
    from admission import subsets
    return np.array([sum(1<<int(i) for i in s) for s in subsets(n)],np.int32)

def parameters(cfg,count,guard_db=0.,reserve_s=0.):
    remaining=cfg.frame_s-count*cfg.probe_s-(count+1)*reserve_s
    return np.array([remaining,cfg.bandwidth_hz,cfg.ul_fraction,noise_power(cfg),
                     cfg.ul_power_w,cfg.dl_power_w,guard_db],float)

def schedule(cfg,mean,cov,payload,count,masks=None,nominal=False,orthogonal=False,guard_db=0.,reserve_s=0.,quantile_override=None):
    if not 1<=cfg.n<=24:raise ValueError('Compiled kernel supports 1..24 tasks')
    lib=library();masks=all_masks(cfg.n) if masks is None else np.ascontiguousarray(masks,np.int32)
    mean=np.ascontiguousarray(mean);cov=np.ascontiguousarray(cov);payload=np.ascontiguousarray(payload)
    kr,ko=quantiles(cfg.n,cfg.outage,nominal);pars=parameters(cfg,count,guard_db,reserve_s)
    if quantile_override is not None:kr=ko=np.full(cfg.n+1,float(quantile_override))
    out=np.zeros(5+4*cfg.n)
    lib.ivp_schedule(cfg.n,ptr(mean),ptr(cov),ptr(payload),ptr(pars),ptr(kr),ptr(ko),iptr(masks),len(masks),int(orthogonal),ptr(out))
    group=np.array([i for i in range(cfg.n) if int(out[0])&(1<<i)],int)
    answer=dict(group=group,value=float(out[2]),mode={-1:'empty',0:'reuse',1:'OMA'}[int(out[1])],
                power=out[5:5+2*cfg.n].reshape(2,cfg.n)[:,group],margins=out[3:5].tolist())
    if int(out[1])==1:answer['airtime']=out[5+2*cfg.n:].reshape(2,cfg.n)[:,group]
    return answer

def affine_value(cfg,mean,post_cov,direction,payload,next_count,masks=None,grid_size=25,refinements=8,reserve_s=0.):
    """Scalar report about a correlated target, including a future channel.

    The caller supplies the target's posterior covariance and its direction
    per STANDARD NORMAL innovation. The target need not be the measured state.
    """
    if not 1<=cfg.n<=24:raise ValueError('Compiled kernel supports 1..24 tasks')
    lib=library();masks=all_masks(cfg.n) if masks is None else np.ascontiguousarray(masks,np.int32)
    kr,ko=quantiles(cfg.n,cfg.outage);pars=parameters(cfg,next_count,reserve_s=reserve_s)
    grid=np.linspace(-6.,6.,grid_size);out=np.zeros(len(masks)*6)
    args=[np.ascontiguousarray(x,float) for x in [mean,post_cov,direction,payload]]
    lib.ivp_intervals(cfg.n,*map(ptr,args),ptr(pars),ptr(kr),ptr(ko),iptr(masks),len(masks),ptr(grid),len(grid),refinements,ptr(out))
    return float(lib.ivp_integrate(ptr(out),len(out)//3))

def value(cfg,mean,cov,payload,count,index,masks=None,grid_size=25,refinements=8,guard_db=0.,reserve_s=0.,return_intervals=False):
    from model import fantasy
    lib=library();masks=all_masks(cfg.n) if masks is None else np.ascontiguousarray(masks,np.int32)
    direction,new_cov=fantasy(cov,index,(cfg.measurement_db*np.log(10)/10)**2)
    kr,ko=quantiles(cfg.n,cfg.outage);pars=parameters(cfg,count+1,guard_db,reserve_s)
    grid=np.linspace(-6.,6.,grid_size);out=np.zeros(len(masks)*6)
    args=[np.ascontiguousarray(x,float) for x in [mean,new_cov,direction,payload]]
    lib.ivp_intervals(cfg.n,*map(ptr,args),ptr(pars),ptr(kr),ptr(ko),iptr(masks),len(masks),ptr(grid),len(grid),refinements,ptr(out))
    val=float(lib.ivp_integrate(ptr(out),len(out)//3))
    if return_intervals:
        intervals=out.reshape(-1,3);intervals=intervals[(intervals[:,2]>0)&(intervals[:,1]>intervals[:,0])]
        return val,intervals
    return val

def catalogue(cfg,mean,cov,payload,limit):
    """Deterministic prior-only row-bound ranking; catalogue stays fixed per frame."""
    masks=all_masks(cfg.n)
    if limit<=0 or limit>=len(masks):return masks
    lib=library();kr,ko=quantiles(cfg.n,cfg.outage);pars=parameters(cfg,0);out=np.zeros(len(masks))
    args=[np.ascontiguousarray(x,float) for x in [mean,cov,payload]]
    lib.ivp_scores(cfg.n,*map(ptr,args),ptr(pars),ptr(kr),ptr(ko),iptr(masks),len(masks),ptr(out))
    scored=list(zip(out,map(int,masks)))
    current=schedule(cfg,mean,cov,payload,0)
    keep={1<<i for i in range(cfg.n)}
    keep.add(sum(1<<int(i) for i in current['group']))
    keep.discard(0)
    for _,mask in sorted(scored,reverse=True):
        if len(keep)>=max(limit,cfg.n):break
        keep.add(mask)
    return np.array([x for x in masks if x in keep],np.int32)

if __name__=='__main__':print(build())
