"""Conservative fast IVP and fair fixed-count / multi-step comparators."""
import time,functools
from dataclasses import replace
import numpy as np
from numpy.polynomial.hermite import hermgauss
from model import posterior,fantasy
from admission import execute,coefficients
from fast_backend import schedule,value,catalogue,library

@functools.lru_cache(None)
def gaussian_rule(order):
    z,w=hermgauss(order)
    return z*np.sqrt(2),w/np.sqrt(np.pi)

def quadrature_value(cfg,mean,cov,payload,count,index,nodes,weights,guard_db=0.,reserve_s=0.):
    b,new_cov=fantasy(cov,index,(cfg.measurement_db*np.log(10)/10)**2)
    return sum(w*schedule(cfg,mean+b*z,new_cov,payload,count+1,guard_db=guard_db,reserve_s=reserve_s)['value'] for z,w in zip(nodes,weights))

def margin_scores(cfg,mean,cov,payload,count,candidates):
    """First-order variance reduction of the worst full-set row-power bound."""
    lf,_,lu,_=coefficients(cfg,mean,cov,payload,count,np.arange(cfg.n))
    f=np.exp(lf);u=np.exp(lu)/np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
    row=f.sum(axis=-1)+u;d,i=np.unravel_index(np.argmax(row),row.shape)
    gradient=np.zeros(2*cfg.n*cfg.n)
    gradient[d*cfg.n*cfg.n+i*cfg.n+np.arange(cfg.n)]=f[d,i]
    gradient[d*cfg.n*cfg.n+i*(cfg.n+1)]=-row[d,i]
    rv=(cfg.measurement_db*np.log(10)/10)**2
    return (gradient@cov[:,candidates])**2/(np.diag(cov)[candidates]+rv)

def run_fast(cfg,sc,method='C-IVP',fixed_count=None,grid_size=25,catalogue_limit=0,
             guard_db=0.,reserve_s=0.,return_posterior=False,candidate_limit=0):
    library() # compilation/loading explicitly outside measured, warmed online work
    if method=='GH-KG':gaussian_rule(17)
    if method=='Two-step':gaussian_rule(7)
    started=time.perf_counter();cpu_started=time.process_time();mean=sc['mean'].copy();cov=sc['covariance'].copy()
    rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(len(mean),int);observations=[];round_times=[]
    masks=catalogue(cfg,mean,cov,sc['payload'],catalogue_limit) if catalogue_limit else None
    count=0;rng=np.random.default_rng(sc['seed']+899999);history=[]
    if method=='Perfect-CSI':mean=sc.get('execution_true',sc['true']).copy();cov[:]=0
    no_probe=method in ['No-report','OMA','Nominal','Perfect-CSI']
    nominal=method in ['Nominal','Perfect-CSI']
    while True:
        tick=time.perf_counter()
        current=schedule(cfg,mean,cov,sc['payload'],count,nominal=nominal,orthogonal=method=='OMA',guard_db=guard_db,reserve_s=reserve_s)
        if no_probe or count>=cfg.max_probes or (fixed_count is not None and count>=fixed_count):
            round_times.append(time.perf_counter()-tick);break
        if fixed_count is None and current['value']>=sc['payload'].sum()/1000-1e-9:
            round_times.append(time.perf_counter()-tick);break
        candidates=sc['candidates']
        if candidate_limit>0:
            scores=margin_scores(cfg,mean,cov,sc['payload'],count,candidates)
            candidates=candidates[np.argsort(-scores,kind='stable')[:candidate_limit]]
        if method=='Margin-sensitivity':
            scores=margin_scores(cfg,mean,cov,sc['payload'],count,candidates)
            index=int(candidates[np.argmax(scores)])
            estimate=value(cfg,mean,cov,sc['payload'],count,index,masks,grid_size,guard_db=guard_db,reserve_s=reserve_s)
        elif method in ['A-stop','A-fixed','D-stop','D-fixed']:
            scores=np.sum(cov[:,candidates]**2,axis=0)/(np.diag(cov)[candidates]+rv) if method.startswith('A') else np.log1p(np.diag(cov)[candidates]/rv)
            index=int(candidates[np.argmax(scores)])
            estimate=value(cfg,mean,cov,sc['payload'],count,index,masks,grid_size,guard_db=guard_db,reserve_s=reserve_s) if fixed_count is None else np.nan
        elif method=='Random-fixed':index=int(rng.choice(candidates));estimate=np.nan
        else:
            if method=='Sensitivity-3pt':
                estimates=[quadrature_value(cfg,mean,cov,sc['payload'],count,int(k),[-np.sqrt(3),0,np.sqrt(3)],[1/6,2/3,1/6],guard_db,reserve_s) for k in candidates]
            elif method=='GH-KG':
                z,w=gaussian_rule(17)
                estimates=[quadrature_value(cfg,mean,cov,sc['payload'],count,int(k),z,w,guard_db,reserve_s) for k in candidates]
            elif method=='Two-step' and count+1<cfg.max_probes:
                estimates=[];z2,w2=gaussian_rule(7)
                for k in candidates:
                    b,cs=fantasy(cov,int(k),rv);estimate_k=0.
                    for zz,ww in zip(z2,w2):
                        mm=mean+b*zz
                        base=schedule(cfg,mm,cs,sc['payload'],count+1,guard_db=guard_db,reserve_s=reserve_s)['value']
                        if base<sc['payload'].sum()/1000-1e-9:
                            follow=max(value(cfg,mm,cs,sc['payload'],count+1,int(j),masks,grid_size,guard_db=guard_db,reserve_s=reserve_s) for j in candidates)
                            base=max(base,follow)
                        estimate_k+=ww*base
                    estimates.append(estimate_k)
            else:
                estimates=[value(cfg,mean,cov,sc['payload'],count,int(k),masks,grid_size,guard_db=guard_db,reserve_s=reserve_s) for k in candidates]
            index=int(candidates[np.argmax(estimates)]);estimate=float(max(estimates))
        history.append(dict(current=current['value'],forecast=None if not np.isfinite(estimate) else float(estimate),index=index))
        if fixed_count is None and estimate<=current['value']+cfg.gain_tol_kbit:
            round_times.append(time.perf_counter()-tick);break
        obs=sc['true'][index]+np.sqrt(rv)*sc['report_noise'][index,used[index]];used[index]+=1
        mean,cov=posterior(mean,cov,index,obs,rv);observations.append(dict(index=index,observation=float(obs)))
        count+=1;round_times.append(time.perf_counter()-tick)
    elapsed=time.perf_counter()-started;cpu_elapsed=time.process_time()-cpu_started
    reserved=(count+1)*reserve_s
    eval_cfg=replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved))
    result=execute(eval_cfg,sc,current,count)
    radio_completed=result['completed_kbit'];radio_outage=result['outage']
    overrun=reserve_s>0 and elapsed>reserved
    if overrun and result['active']:
        result['completed_kbit']=0.;result['completed_tasks']=0;result['outage']=1
    result.update(method=method,seed=int(sc['seed']),n=cfg.n,probes=count,wall_s=elapsed,
                  selected=list(map(int,current['group'])),power=current['power'].tolist(),mode=current['mode'],
                  observations=observations,history=history,margins=current.get('margins',[0,0]),
                  round_s=round_times,guard_db=guard_db,reserve_s=reserve_s,reserved_s=reserved,
                  deadline_overrun=int(overrun),radio_completed_kbit=radio_completed,radio_outage=radio_outage,
                  catalogue_limit=catalogue_limit,catalogue_size=len(masks) if masks is not None else 2**cfg.n-1,
                  grid_size=grid_size,fixed_count=fixed_count,demand_kbit=float(sc['payload'].sum()/1000))
    result['candidate_limit']=candidate_limit
    result['cpu_s']=cpu_elapsed
    if 'airtime' in current:result['airtime']=current['airtime'].tolist()
    if return_posterior:return result,mean,cov,current
    return result
