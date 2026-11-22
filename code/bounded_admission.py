"""Anytime branch-and-bound for the SAME finite certificate as enumeration.

The upper bound uses a partition into pair-conflict cliques. Feasible sets
are hereditary because deleting tasks reduces interference/airtime and relaxes
the cardinality-dependent risk quantile. This is an application of standard
branch-and-bound, not a new general optimization method. Worst case is exponential.
"""
import heapq,time
import numpy as np
from fast_backend import schedule,library

def bounded_schedule(cfg,mean,cov,payload,count,node_budget=256,collect_masks=False):
    library();started=time.perf_counter();n=cfg.n;weights=payload.sum(axis=0)/1000
    if n>24:raise ValueError('Compiled oracle supports N<=24')
    cache={};calls=0
    empty=dict(group=np.array([],int),value=0.,mode='empty',power=np.zeros((2,0)))
    def oracle(mask,mode):
        nonlocal calls
        if not mask:return empty
        key=(mask,mode)
        if key not in cache:
            act=schedule(cfg,mean,cov,payload,count,masks=np.array([mask],np.int32),orthogonal=mode=='OMA')
            cache[key]=act if act['mode']==mode else None;calls+=1
        return cache[key]
    incumbent=empty;graphs={};eligible={}
    for mode in ['reuse','OMA']:
        ids=[i for i in range(n) if oracle(1<<i,mode) is not None]
        conflict=np.zeros((n,n),bool)
        for ix,i in enumerate(ids):
            for j in ids[ix+1:]:
                conflict[i,j]=conflict[j,i]=oracle((1<<i)|(1<<j),mode) is None
        graphs[mode]=conflict;eligible[mode]=tuple(ids)
        # Deterministic feasible incumbent; every tested set uses the exact oracle.
        mask=0
        for i in sorted(ids,key=lambda j:(-weights[j],j)):
            act=oracle(mask|(1<<i),mode)
            if act is not None:
                mask|=1<<i
                if act['value']>incumbent['value']:incumbent=act
    def bound(chosen,candidates,mode):
        value=sum(weights[i] for i in range(n) if chosen&(1<<i))
        graph=graphs[mode];cliques=[]
        ordered=sorted(candidates,key=lambda i:(-sum(graph[i,j] for j in candidates),-weights[i],i))
        for i in ordered:
            for clique in cliques:
                if all(graph[i,j] for j in clique):clique.append(i);break
            else:cliques.append([i])
        return float(value+sum(max(weights[i] for i in c) for c in cliques))
    heap=[];serial=0
    def push(chosen,candidates,mode):
        nonlocal serial
        upper=bound(chosen,candidates,mode)
        if upper>incumbent['value']+1e-10:
            heapq.heappush(heap,(-upper,serial,chosen,candidates,mode));serial+=1
    for mode in ['reuse','OMA']:push(0,eligible[mode],mode)
    nodes=0
    while heap and nodes<node_budget:
        neg,_,chosen,candidates,mode=heapq.heappop(heap)
        if -neg<=incumbent['value']+1e-10:continue
        nodes+=1
        act=oracle(chosen,mode)
        if act is None:continue # Every superset is infeasible in this mode.
        if act['value']>incumbent['value']:incumbent=act
        whole=chosen|sum(1<<i for i in candidates)
        full=oracle(whole,mode)
        if full is not None:
            if full['value']>incumbent['value']:incumbent=full
            continue
        if not candidates:continue
        graph=graphs[mode]
        j=max(candidates,key=lambda i:(sum(graph[i,k] for k in candidates),weights[i],-i))
        rest=tuple(i for i in candidates if i!=j)
        push(chosen,rest,mode)
        push(chosen|(1<<j),tuple(i for i in rest if not graph[i,j]),mode)
    upper=max(incumbent['value'],max((-x[0] for x in heap),default=0.))
    info=dict(lower=incumbent['value'],upper=upper,gap=upper-incumbent['value'],
                         nodes=nodes,oracle_calls=calls,wall_s=time.perf_counter()-started,
                         ratio=incumbent['value']/upper if upper else 1.,node_budget=node_budget)
    if collect_masks:info['search_masks']=sorted(set(k[0] for k in cache))
    return incumbent,info

def run_bounded(cfg,sc,reporting=True,node_budget=256,catalogue_size=64):
    """Complete screened policy: no 2**N array is ever built.

    The forecast catalogue is fixed from the initial B&B search. Every terminal
    decision compares its B&B incumbent with ALL catalogue actions, so the
    forecast lower bound also lower-bounds the executed terminal value.
    """
    from fast_backend import value,ptr,iptr,parameters,quantiles
    from model import posterior
    from revised_policies import margin_scores
    from admission import execute
    lib=library();started=time.perf_counter();mu=sc['mean'].copy();cov=sc['covariance'].copy()
    current,info=bounded_schedule(cfg,mu,cov,sc['payload'],0,node_budget,True)
    candidates=np.array(info.pop('search_masks'),np.int32);score=np.zeros(len(candidates))
    kr,ko=quantiles(cfg.n,cfg.outage);pars=parameters(cfg,0)
    lib.ivp_scores(cfg.n,ptr(mu),ptr(cov),ptr(sc['payload']),ptr(pars),ptr(kr),ptr(ko),iptr(candidates),len(candidates),ptr(score))
    keep={1<<i for i in range(cfg.n)};keep.add(sum(1<<int(i) for i in current['group']));keep.discard(0)
    for j in np.argsort(-score,kind='stable'):
        if len(keep)>=max(cfg.n,catalogue_size):break
        keep.add(int(candidates[j]))
    masks=np.array(sorted(keep),np.int32);obs=[];history=[];rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(len(mu),int)
    while True:
        m=len(obs)
        if m:current,info=bounded_schedule(cfg,mu,cov,sc['payload'],m,node_budget)
        cat=schedule(cfg,mu,cov,sc['payload'],m,masks=masks)
        if cat['value']>current['value']:current=cat
        if not reporting or m>=cfg.max_probes or current['value']>=sc['payload'].sum()/1000-1e-9:break
        scores=margin_scores(cfg,mu,cov,sc['payload'],m,sc['candidates'])
        ids=sc['candidates'][np.argsort(-scores,kind='stable')[:2]]
        qs=[value(cfg,mu,cov,sc['payload'],m,int(k),masks=masks) for k in ids]
        k=int(ids[np.argmax(qs)]);q=max(qs);history.append(dict(current=current['value'],forecast=q,index=k))
        if q<=current['value']+cfg.gain_tol_kbit:break
        y=float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]]);used[k]+=1
        mu,cov=posterior(mu,cov,k,y,rv);obs.append(dict(index=k,observation=y))
    result=execute(cfg,sc,current,len(obs));result.update(method='B-IVP2' if reporting else 'B-NR',n=cfg.n,seed=sc['seed'],
        probes=len(obs),observations=obs,history=history,selected=current['group'].tolist(),power=current['power'].tolist(),mode=current['mode'],
        catalogue=masks.tolist(),wall_s=time.perf_counter()-started,terminal_upper=info['upper'],terminal_gap=max(0,info['upper']-current['value']))
    if 'airtime' in current:result['airtime']=current['airtime'].tolist()
    return result
