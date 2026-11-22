"""Quantify certificate conservatism with independent conditional channels.

SAA powers/time allocations are designed by a mixed-integer sample program.
Independent binomial validation controls false acceptance across the finite
candidate family; a third draw estimates risk. This is not a globally optimal
solution of the original continuous chance-constrained admission problem.
"""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from scipy.stats import beta
from model import Config,scenario,thresholds,noise_power
from admission import subsets,execute
from revised_policies import run_fast

def draw(mean,cov,L,seed,n):
    rng=np.random.default_rng(seed)
    x=mean+rng.normal(size=(L,len(mean)))@np.linalg.cholesky(cov+np.eye(len(mean))*1e-14).T
    return np.exp(x.reshape(L,2,n,n))

def failures(cfg,payload,gains,action,count=0):
    group=action['group']
    if not len(group):return np.zeros(len(gains),bool)
    g=gains[:,:,group][:,:,:,group];p=action['power']
    rec=g*p[None,:,None,:];desired=np.diagonal(rec,axis1=2,axis2=3)
    interference=rec.sum(axis=3)-desired
    if action['mode']=='OMA':
        sinr=desired/noise_power(cfg)
        passed=cfg.bandwidth_hz*action['airtime'][None,:,:]*np.log2(1+sinr)>=payload[:,group][None,:,:]*(1-1e-7)
    else:
        sinr=desired/(interference+noise_power(cfg))
        passed=sinr>=thresholds(cfg,payload,count)[None,:,group]*(1-1e-7)
    return ~passed.all(axis=(1,2))

def cp_upper(k,n,alpha):
    return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))

def design_action(cfg,payload,gains,group,allowed,mode):
    K=len(gains);s=len(group);size=2*s+K;cap=np.array([cfg.ul_power_w,cfg.dl_power_w])
    g=gains[:,:,group][:,:,:,group];theta=thresholds(cfg,payload,0)[:,group]
    rows=2*K*s+1+(2 if mode=='OMA' else 0)
    A=lil_matrix((rows,size));upper=np.zeros(rows);r=0
    if mode=='reuse':
        for k in range(K):
            for d in range(2):
                for i in range(s):
                    f=theta[d,i]*g[k,d,i]/g[k,d,i,i];f[i]=0
                    u=theta[d,i]*noise_power(cfg)/(g[k,d,i,i]*cap[d]);big=u+f.sum()
                    f[i]=-1;A[r,d*s:(d+1)*s]=f/big;A[r,2*s+k]=-1.;upper[r]=-u/big;r+=1
    else:
        duration=cfg.frame_s*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
        diag=np.diagonal(g,axis1=2,axis2=3)
        need=payload[:,group][None,:,:]/(cfg.bandwidth_hz*np.log2(1+cap[None,:,None]*diag/noise_power(cfg))*duration[None,:,None])
        for k in range(K):
            for d in range(2):
                for i in range(s):
                    A[r,d*s+i]=-1;A[r,2*s+k]=-need[k,d,i];upper[r]=-need[k,d,i];r+=1
        for d in range(2):A[r,d*s:(d+1)*s]=1;upper[r]=1;r+=1
    A[r,2*s:]=1;upper[r]=allowed
    objective=np.r_[np.ones(2*s),np.zeros(K)]
    result=milp(objective,integrality=np.r_[np.zeros(2*s),np.ones(K)],bounds=Bounds(np.zeros(size),np.ones(size)),
                constraints=LinearConstraint(A.tocsc(),np.full(rows,-np.inf),upper),
                options={'time_limit':1.,'mip_rel_gap':1e-6,'presolve':True})
    if result.x is None:return None,int(result.status)
    if np.max(A@result.x-upper)>1e-6:return None,int(result.status)
    q=result.x[:2*s].reshape(2,s)
    action=dict(group=group,mode=mode,value=float(payload[:,group].sum()/1000))
    if mode=='reuse':action['power']=q*cap[:,None]
    else:
        action['power']=np.broadcast_to(cap[:,None],(2,s)).copy();action['airtime']=q*duration[:,None]
    # Verify the actual sample program, independently of MILP solver tolerance.
    if int(failures(cfg,payload,gains,action).sum())>allowed:return None,int(result.status)
    return action,int(result.status)

def validated_saa(cfg,sc,design_count=64,validation_count=4096):
    n=cfg.n;groups=sorted(subsets(n),key=lambda s:-sc['payload'][:,s].sum())
    gd=draw(sc['mean'],sc['covariance'],design_count,sc['seed']+1000000,n)
    gv=draw(sc['mean'],sc['covariance'],validation_count,sc['seed']+2000000,n)
    allowances=[int(np.floor(.025*design_count)),0]
    family=2*len(groups)*len(allowances);alpha=.05/family
    attempted=[]
    for group in groups:
        for mode in ['reuse','OMA']:
            for allowed in allowances:
                action,status=design_action(cfg,sc['payload'],gd,group,allowed,mode)
                record=dict(group=list(map(int,group)),mode=mode,allowed=allowed,solver_status=status)
                if action is not None:
                    bad=int(failures(cfg,sc['payload'],gv,action).sum());upper=cp_upper(bad,validation_count,alpha)
                    record.update(validation_bad=bad,validation_upper=upper)
                    attempted.append(record)
                    if upper<=cfg.outage:return action,attempted
                else:attempted.append(record)
    return dict(group=np.array([],int),power=np.zeros((2,0)),value=0.,mode='empty'),attempted

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='../revision_results');p.add_argument('--count',type=int,default=128)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=Config()
    metadata=dict(snapshots=a.count,seed_start=190000,conditional_draws=16384,design_draws=64,validation_draws=4096,
                  validation_family_confidence=.95,risk_budget=.05,design_allowed_fraction=.025,config=asdict(cfg))
    (out/'chance_protocol.json').write_text(json.dumps(metadata,indent=2))
    t=time.perf_counter()
    with (out/'chance_records.jsonl').open('w') as f:
        for seed in range(190000,190000+a.count):
            sc=scenario(cfg,seed)
            for method in ['C-IVP','A-stop','No-report','OMA']:
                r,mean,cov,action=run_fast(cfg,sc,method,return_posterior=True)
                gg=draw(mean,cov,16384,seed+3000000,cfg.n)
                bad=int(failures(cfg,sc['payload'],gg,action,r['probes']).sum())
                r.update(conditional_bad=bad,conditional_draws=16384,conditional_risk=bad/16384)
                f.write(json.dumps(r)+'\n')
            start=time.perf_counter();action,trials=validated_saa(cfg,sc);elapsed=time.perf_counter()-start
            r=execute(cfg,sc,action,0);g=draw(sc['mean'],sc['covariance'],16384,seed+3000000,cfg.n)
            bad=int(failures(cfg,sc['payload'],g,action).sum())
            r.update(method='Validated-SAA',seed=seed,selected=list(map(int,action['group'])),mode=action['mode'],
                     power=action['power'].tolist(),wall_s=elapsed,probes=0,conditional_bad=bad,conditional_draws=16384,
                     conditional_risk=bad/16384,attempts=trials)
            if 'airtime' in action:r['airtime']=action['airtime'].tolist()
            f.write(json.dumps(r)+'\n');f.flush()
            if (seed-190000+1)%8==0:print(seed-190000+1,a.count,round(time.perf_counter()-t,2),flush=True)
    metadata['elapsed_s']=time.perf_counter()-t;(out/'chance_protocol.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
