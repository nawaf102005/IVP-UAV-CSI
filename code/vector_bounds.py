"""Lower/upper Gaussian value bounds for a fixed vector report.

Whitening makes innovation coordinates independent. Each disjoint box has an
exact product-CDF mass. Termwise exponential bounds produce sufficient robust
feasibility and necessary optimistic feasibility. Bounds are mathematical in
exact arithmetic; this implementation uses ordinary double precision.
"""
import heapq,time
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import norm
from admission import subsets
from model import thresholds,noise_power

class BoxEnvelope:
    def __init__(self,cfg,mean,cov,direction,payload,count):
        self.cfg=cfg;self.r=direction.shape[1];n=cfg.n;groups=subsets(n);K=len(groups)
        self.weights=np.array([payload[:,g].sum()/1000 for g in groups]);self.total=payload.sum()/1000
        self.lf=np.full((K,2,n,n),-np.inf);self.sf=np.zeros((K,2,n,n,self.r))
        self.lu=np.full((K,2,n),-np.inf);self.su=np.zeros((K,2,n,self.r))
        self.ls=np.zeros((K,2,n));self.ss=np.zeros((K,2,n,self.r));self.bits=np.zeros((K,2,n))
        self.cap=np.array([cfg.ul_power_w,cfg.dl_power_w])[None,:,None]
        self.durations=(cfg.frame_s-count*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
        theta=thresholds(cfg,payload,count);diag=np.diag(cov)
        for k,g in enumerate(groups):
            s=len(g);kr=norm.isf(cfg.outage/(2*s*s));ko=norm.isf(cfg.outage/(2*s))
            for d in range(2):
                for i in g:
                    ii=d*n*n+i*(n+1)
                    self.lu[k,d,i]=np.log(theta[d,i]*noise_power(cfg))-mean[ii]+kr*np.sqrt(max(diag[ii],0))
                    self.su[k,d,i]=-direction[ii]
                    self.ls[k,d,i]=np.log(self.cap[0,d,0]/noise_power(cfg))+mean[ii]-ko*np.sqrt(max(diag[ii],0))
                    self.ss[k,d,i]=direction[ii];self.bits[k,d,i]=payload[d,i]
                    for j in g:
                        if i==j:continue
                        ij=d*n*n+i*n+j;var=max(diag[ij]+diag[ii]-2*cov[ij,ii],0)
                        self.lf[k,d,i,j]=np.log(theta[d,i])+mean[ij]-mean[ii]+kr*np.sqrt(var)
                        self.sf[k,d,i,j]=direction[ij]-direction[ii]
        self.axis_scale=np.maximum(np.max(np.abs(self.sf),axis=(0,1,2,3)),np.max(np.abs(self.su),axis=(0,1,2)))
    def feasible_values(self,lf,lu,ls,optimistic=False):
        f=np.exp(lf);u=np.exp(lu);matrix=np.eye(self.cfg.n)-f
        try:p=np.linalg.solve(matrix,u[...,None])[...,0]
        except np.linalg.LinAlgError:
            p=np.full_like(u,np.nan)
            for k in range(len(u)):
                for d in range(2):
                    try:p[k,d]=np.linalg.solve(matrix[k,d],u[k,d])
                    except np.linalg.LinAlgError:pass
        tolerance=1+1e-10 if optimistic else 1-1e-12
        good=np.all(np.isfinite(p)&(p>=0)&(p<=self.cap*tolerance),axis=(1,2))
        residual=np.abs(np.einsum('kdij,kdj->kdi',matrix,p)-u)
        good &= np.all(residual<=1e-9*np.maximum(np.abs(p),1e-10),axis=(1,2))
        times=self.bits*np.log(2)/(self.cfg.bandwidth_hz*np.logaddexp(0,ls))
        oma=np.all(times.sum(axis=-1)<=self.durations*tolerance,axis=1)
        return float(np.max(np.where(good|oma,self.weights,0),initial=0))
    def bounds(self,lo,hi):
        center=(lo+hi)/2;radius=(hi-lo)/2
        f=self.lf+self.sf@center;df=np.abs(self.sf)@radius
        u=self.lu+self.su@center;du=np.abs(self.su)@radius
        s=self.ls+self.ss@center;ds=np.abs(self.ss)@radius
        lower=self.feasible_values(f+df,u+du,s-ds)
        upper=self.feasible_values(f-df,u-du,s+ds,optimistic=True)
        return lower,upper

def vector_fantasy(cov,indices,noise_var):
    ids=np.asarray(indices,int)
    R=noise_var*np.eye(len(ids)) if np.ndim(noise_var)==0 else np.asarray(noise_var,float)
    if R.shape!=(len(ids),len(ids)):raise ValueError('Observation covariance has wrong shape')
    S=cov[np.ix_(ids,ids)]+R
    chol=np.linalg.cholesky(S);direction=np.linalg.solve(chol,cov[ids,:]).T
    post=cov-direction@direction.T
    return direction,(post+post.T)/2

def value_bounds(cfg,mean,cov,payload,count,indices,budgets=(64,256,1024,4096),limit=6.,keep_leaves=False,noise_cov=None,next_count=None):
    started=time.perf_counter()
    R=(cfg.measurement_db*np.log(10)/10)**2 if noise_cov is None else noise_cov
    b,post=vector_fantasy(cov,indices,R)
    charged_count=count+len(indices) if next_count is None else next_count
    env=BoxEnvelope(cfg,mean,post,b,payload,charged_count);r=len(indices)
    heap=[];serial=0;lower=0.;upper=env.total*(1-(ndtr(limit)-ndtr(-limit))**r)
    def add(lo,hi):
        nonlocal serial,lower,upper
        l,u=env.bounds(lo,hi);mass=float(np.prod(ndtr(hi)-ndtr(lo)))
        lower+=mass*l;upper+=mass*u
        heapq.heappush(heap,(-mass*(u-l),serial,lo,hi,l,u,mass));serial+=1
    add(np.full(r,-limit),np.full(r,limit));checkpoints=[]
    for budget in budgets:
        while len(heap)<budget and heap[0][0]<-1e-12:
            _,_,lo,hi,l,u,mass=heapq.heappop(heap);lower-=mass*l;upper-=mass*u
            axis=int(np.argmax((hi-lo)*env.axis_scale))
            mid=float(ndtri((ndtr(lo[axis])+ndtr(hi[axis]))/2))
            left_hi=hi.copy();left_hi[axis]=mid;right_lo=lo.copy();right_lo[axis]=mid
            add(lo,left_hi);add(right_lo,hi)
        checkpoints.append(dict(budget=budget,leaves=len(heap),lower=lower,upper=upper,gap=upper-lower,wall_s=time.perf_counter()-started))
    result=dict(indices=list(map(int,indices)),checkpoints=checkpoints,tail_mass=1-(ndtr(limit)-ndtr(-limit))**r)
    if keep_leaves:result['leaves']=[dict(lo=x[2].tolist(),hi=x[3].tolist(),lower=x[4],upper=x[5],mass=x[6]) for x in heap]
    return result
