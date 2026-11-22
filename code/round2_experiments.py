"""Review-driven diagnostics; frozen original experiments are never overwritten.

Regularization is fixed at the known report-error standard deviation (1 dB).
Old ray tests are exploratory; new Munich positions use independent seed 967300.
Two-component batches charge the same 2*tau as sequential reports; no claim of
an optimized physical pilot or exact multivariate Gaussian integration is made.
"""
import argparse,itertools,json,time,platform,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy import stats
from model import Config,scenario,posterior,fantasy
from admission import execute
from innovation import probe_value,expected_envelope
from fast_backend import value,schedule,library
from revised_policies import run_fast,gaussian_rule,margin_scores
from ray_evaluation import fit,case,log_vectors
from conformal_report_filter import safe_schedule,conformal_quantile

def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read_records(path):return [json.loads(x) for x in path.read_text().splitlines()]
def write_record(f,r):f.write(json.dumps(r,allow_nan=False)+'\n')
def paired(a,b,alpha=.05):
    x=np.asarray(a)-np.asarray(b);n=len(x);mean=float(x.mean());se=float(stats.sem(x))
    half=float(stats.t.ppf(1-alpha/2,n-1)*se)
    p=float(stats.ttest_1samp(x,0).pvalue) if se else (1. if mean==0 else 0.)
    return dict(n=n,difference=mean,ci=[mean-half,mean+half],p=p,alpha=alpha)

def interval_audit(root,out):
    cfg=Config();rows=[];raw=[];started=time.perf_counter();library()
    with (out/'interval_reference.jsonl').open('w') as f:
        for seed in range(110000,110128):
            sc=scenario(cfg,seed)
            for ix in sc['candidates']:
                ref,ints=probe_value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(ix),return_intervals=True)
                write_record(f,dict(seed=seed,index=int(ix),reference=ref,intervals=ints))
                for grid_size in [13,25,49,97]:
                    grid=np.linspace(-6,6,grid_size)
                    hit=[x for x in ints if np.any((grid>=x[0])&(grid<=x[1]))]
                    t=time.process_time();inner=value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(ix),grid_size=grid_size)
                    cpu=time.process_time()-t
                    rows.append(dict(seed=seed,index=int(ix),grid=grid_size,nonempty=len(ints),missed=len(ints)-len(hit),
                                     reference=ref,inner=inner,gap=ref-inner,omission_gap=ref-expected_envelope(hit),cpu_s=cpu))
            if seed%16==15:print('intervals',seed-110000+1,128,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'interval_grid.csv',index=False);summary=[]
    for g,d in df.groupby('grid'):
        summary.append(dict(grid=int(g),candidate_reports=len(d),nonempty=int(d.nonempty.sum()),missed=int(d.missed.sum()),
                            missed_percent=100*float(d.missed.sum()/d.nonempty.sum()),reports_with_miss=int((d.missed>0).sum()),
                            mean_gap=float(d.gap.mean()),max_gap=float(d.gap.max()),min_gap=float(d.gap.min()),
                            mean_omission_gap=float(d.omission_gap.mean()),max_omission_gap=float(d.omission_gap.max()),
                            excess_above_002=int((d.gap<-.002).sum()),mean_cpu_ms=1000*float(d.cpu_s.mean())))
    dump(out/'interval_summary.json',dict(results=summary,elapsed_s=time.perf_counter()-started,
        note='Nonempty action intervals are reference-detected within [-6,6]; redundant singleton OMA intervals are omitted. Miss count is not a fraction of optimal decisions. CPU costs are diagnostic, not isolated online latency.'))

def joint_update(mean,cov,indices,y,noise_var):
    ids=np.array(indices,int);s=cov[np.ix_(ids,ids)]+np.eye(len(ids))*noise_var
    gain=np.linalg.solve(s,cov[ids,:]).T
    return mean+gain@(np.asarray(y)-mean[ids]),cov-gain@cov[ids,:]

def batch_value(cfg,sc,pair,order):
    ids=np.array(pair);cov=sc['covariance'];rv=(cfg.measurement_db*np.log(10)/10)**2
    ss=cov[np.ix_(ids,ids)]+rv*np.eye(2);chol=np.linalg.cholesky(ss)
    b=np.linalg.solve(chol,cov[ids,:]).T;cs=cov-b@b.T
    z,w=gaussian_rule(order);q=0.
    for i,zi in enumerate(z):
        for j,zj in enumerate(z):
            q+=w[i]*w[j]*schedule(cfg,sc['mean']+b[:,0]*zi+b[:,1]*zj,cs,sc['payload'],2)['value']
    return float(q)

def action_record(cfg,sc,act,obs,method,elapsed,**kw):
    r=execute(cfg,sc,act,len(obs));r.update(method=method,seed=int(sc['seed']),observations=obs,probes=len(obs),
        selected=list(map(int,act['group'])),power=act['power'].tolist(),mode=act['mode'],wall_s=elapsed,**kw)
    if 'airtime' in act:r['airtime']=act['airtime'].tolist()
    return r

def sequential_two(cfg,sc,method):
    mean=sc['mean'].copy();cov=sc['covariance'].copy();obs=[];rv=(cfg.measurement_db*np.log(10)/10)**2
    started=time.perf_counter()
    for m in range(2):
        candidates=[int(k) for k in sc['candidates'] if k not in [x['index'] for x in obs]]
        if method=='Sequential-CDF2':q=[value(cfg,mean,cov,sc['payload'],m,k) for k in candidates]
        elif method=='Margin-fixed2':q=margin_scores(cfg,mean,cov,sc['payload'],m,np.array(candidates))
        else:q=np.sum(cov[:,candidates]**2,axis=0)/(np.diag(cov)[candidates]+rv)
        k=candidates[int(np.argmax(q))];y=sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,0]
        mean,cov=posterior(mean,cov,k,y,rv);obs.append(dict(index=k,observation=float(y)))
    act=schedule(cfg,mean,cov,sc['payload'],2)
    return action_record(cfg,sc,act,obs,method,time.perf_counter()-started)

def vector_study(root,out):
    cfg=replace(Config(),max_probes=2);library();convergence=[];started=time.perf_counter()
    with (out/'vector_records.jsonl').open('w') as f:
        for seed in range(230000,230128):
            sc=scenario(cfg,seed);pairs=list(itertools.combinations(map(int,sc['candidates']),2))
            t=time.perf_counter();q=[batch_value(cfg,sc,p,9) for p in pairs];idx=int(np.argmax(q));pair=pairs[idx]
            rv=(cfg.measurement_db*np.log(10)/10)**2
            y=[float(sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,0]) for k in pair]
            mean,cov=joint_update(sc['mean'],sc['covariance'],pair,y,rv);act=schedule(cfg,mean,cov,sc['payload'],2)
            obs=[dict(index=k,observation=yy) for k,yy in zip(pair,y)]
            write_record(f,action_record(cfg,sc,act,obs,'Batch-KG9x9',time.perf_counter()-t,forecast=q[idx]))
            for method in ['Sequential-CDF2','A-fixed2','Margin-fixed2']:write_record(f,sequential_two(cfg,sc,method))
            if seed<230032:
                q17=[batch_value(cfg,sc,p,17) for p in pairs];i17=int(np.argmax(q17))
                convergence.append(dict(seed=seed,pair9=pair,pair17=pairs[i17],q9=q[idx],q17_at_pair9=q17[idx],
                                        best_q17=q17[i17],ranking_changed=idx!=i17))
            if seed%16==15:f.flush();print('vector',seed-230000+1,128,flush=True)
    dump(out/'vector_convergence.json',dict(records=convergence,elapsed_s=time.perf_counter()-started,
         interpretation='Finite-order 2D quadrature, not exact vector KG. Exactly two distinct coordinate observations, 2*tau overhead for every policy; both orders charge posterior data time T_2.'))

def regularized_covariance(cov,noise_db):
    floor=(noise_db*np.log(10)/10)**2
    return cov+np.diag(np.maximum(floor-np.diag(cov),0.))

def full_scores(x,mean,cov,n=4):
    e=x-mean;parts=[np.max(np.abs(e)/np.sqrt(np.diag(cov)),axis=1)]
    for d in range(2):
        for i in range(n):
            ii=d*n*n+i*(n+1)
            for j in range(n):
                if i==j:continue
                ij=d*n*n+i*n+j;v=cov[ij,ij]+cov[ii,ii]-2*cov[ij,ii]
                parts.append((e[:,ij]-e[:,ii])/np.sqrt(max(v,1e-12)))
    return np.max(parts,axis=0)

def score_quantile(prior,meta,cov):
    a=meta['train'];nc=meta['calibration'];rank=int(np.ceil((nc+1)*.96))
    return float(np.sort(full_scores(prior['x'][a:a+nc],prior['mean'],cov))[rank-1])

def fresh_case_data(root,name):
    old=np.load(root/'ray_data'/f'{name}.npz');new=np.load(root/'fresh_ray_data'/f'{name}.npz')
    meta=json.loads((root/'ray_data'/f'{name}.json').read_text());k=meta['train']+meta['calibration']
    data={s:old[s] for s in old.files}
    for key in ['positions','incoherent','coherent','path_count']:data[key]=np.concatenate([old[key][:k],new[key]])
    for key in ['future_positions','incoherent_future','coherent_future']:data[key]=new[key]
    return data,meta

def calibration_study(root,out,fresh=False):
    cfg=replace(Config(),ul_fraction=.5,ul_power_w=.2,dl_power_w=1.);settings=[]
    old=read_records(root/'revision_results/ray_records.jsonl')
    histories={(r['scenario'],r['ray_index']):r for r in old if r['method']=='C-IVP'}
    names=['munich'] if fresh else ['canyon_low','canyon_high','munich']
    target='fresh_calibration' if fresh else 'calibration'
    with (out/(target+'_records.jsonl')).open('w') as f:
        for name in names:
            if fresh:data,meta=fresh_case_data(root,name)
            else:data=np.load(root/'ray_data'/f'{name}.npz');meta=json.loads((root/'ray_data'/f'{name}.json').read_text())
            for kind,mobile in [('incoherent',False),('coherent',True)]:
                prior=fit(data,meta,kind);label=name+('-motion' if mobile else '-static')
                reg=regularized_covariance(prior['cov'],cfg.measurement_db);q=score_quantile(prior,meta,reg)
                qold=conformal_quantile(prior,meta);score=full_scores(prior['x'][meta['train']+meta['calibration']:],prior['mean'],reg)
                settings.append(dict(scenario=label,q=q,q_original=qold,floor_db=cfg.measurement_db,score_violations=int((score>q).sum()),test=meta['test']))
                for index in range(meta['test']):
                    seed=(250000 if fresh else 210000)+['canyon_low','canyon_high','munich'].index(name)*1000+index
                    sc=case(cfg,data,meta,prior,index,kind,mobile,True,seed)
                    if fresh:
                        history=run_fast(cfg,sc,'C-IVP');history.update(scenario=label,ray_index=index,fresh=True);write_record(f,history)
                    else:history=histories[label,index]
                    methods=['Regularized-NR','Regularized-IVP']+(['Safe-NR','Safe-IVP'] if fresh else [])
                    for method in methods:
                        obs=history['observations'] if method.endswith('IVP') else []
                        cov,qq=(prior['cov'],qold) if method.startswith('Safe') else (reg,q)
                        t=time.perf_counter();act=safe_schedule(cfg,sc,cov,qq,obs);r=action_record(cfg,sc,act,obs,method,time.perf_counter()-t,
                            scenario=label,ray_index=index,fresh=fresh,q=qq)
                        write_record(f,r)
                f.flush();print(target,label,'q',q,'old',qold,flush=True)
    dump(out/(target+'_protocol.json'),dict(settings=settings,regularization='Training covariance plus diagonal lift to report-noise variance; fixed before new test generation.',
        status='Independent new positions, frozen old fit/calibration.' if fresh else 'Exploratory reuse of previously inspected test frames.',
        guarantee='Marginal static exchangeability only; no new guarantee under motion or conditional on calibration.'))

def statistics(root,out):
    raw=read_records(root/'revision_results/revision_records.jsonl');df=pd.DataFrame(raw);ans={}
    def compare(setting,m,b,alpha=.05,column='completed_kbit'):
        d=df[df.scenario==setting].pivot(index='seed',columns='method',values=column)
        return paired(d[m],d[b],alpha)
    ans['guarded']=compare('blockage','Guarded-IVP','Guarded-no-report')
    ans['kg17']=compare('main','C-IVP','GH-KG')
    fam=[]
    for k in [1,2]:
        for b in ['A-fixed','D-fixed','Random-fixed']:
            r=compare('fixed-count',f'C-IVP-{k}',f'{b}-{k}',.05/6);r.update(reports=k,baseline=b);fam.append(r)
    order=np.argsort([r['p'] for r in fam]);prev=0.
    for j,ix in enumerate(order):prev=max(prev,min(1.,(len(fam)-j)*fam[ix]['p']));fam[ix]['p_holm']=prev
    ans['fixed_family']=fam
    d=df[df.scenario=='main'].pivot(index='seed',columns='method',values='outage');a=d['C-IVP'];b=d['No-report']
    n10=int(((a==1)&(b==0)).sum());n01=int(((a==0)&(b==1)).sum())
    ans['outage_pair']=dict(c_only=n10,nr_only=n01,p_exact=stats.binomtest(n10,n10+n01,.5).pvalue,
                            interpretation='Paired discordance test; no claim of equal reliability or equivalence.')
    old=read_records(root/'revision_results/timing_records.jsonl');tim=pd.DataFrame(old);trade=[]
    for n,d in tim.groupby('n'):
        d=d[d.method.isin(['C-IVP','Margin-sensitivity'])];p=d.pivot(index='seed',columns='method',values='completed_kbit')
        w=d.groupby('method').wall_s.mean()*1000;r=paired(p['C-IVP'],p['Margin-sensitivity'])
        r.update(n_pairs=int(n),c_ms=float(w['C-IVP']),margin_ms=float(w['Margin-sensitivity']))
        r['kbit_per_added_ms']=r['difference']/(r['c_ms']-r['margin_ms']);trade.append(r)
    ans['margin_tradeoff']=trade
    if (out/'vector_records.jsonl').exists():
        v=pd.DataFrame(read_records(out/'vector_records.jsonl'));vp=v.pivot(index='seed',columns='method',values='completed_kbit')
        ans['vector']={m:dict(completed=float(x.completed_kbit.mean()),outages=int(x.outage.sum()),n=len(x)) for m,x in v.groupby('method')}
        ans['vector']['sequential_minus_batch']=paired(vp['Sequential-CDF2'],vp['Batch-KG9x9'])
    dump(out/'statistics.json',ans)

def kg_timing_study(root,out):
    cfg=Config();library();methods=['C-IVP','GH-KG','Margin-sensitivity','No-report'];records=[]
    for seed in range(239980,239990):
        sc=scenario(cfg,seed)
        for m in methods:run_fast(cfg,sc,m)
    with (out/'kg_timing_records.jsonl').open('w') as f:
        for seed in range(240000,240128):
            sc=scenario(cfg,seed)
            for m in np.random.default_rng(seed).permutation(methods):
                r=run_fast(cfg,sc,str(m));write_record(f,r);records.append(r)
    assert len(read_records(out/'kg_timing_records.jsonl'))==512

def timing_study(root,out):
    kg_timing_study(root,out)
    cfg=Config()
    # Increase development size without tuning on held-out outcomes.
    methods=['C-IVP','Margin-sensitivity','No-report'];dev=[]
    for seed in range(940000,940256):
        sc=scenario(cfg,seed)
        for m in methods:
            r=run_fast(cfg,sc,m);dev.append(dict(seed=seed,method=m,per_round_s=r['wall_s']/(r['probes']+1)))
    dd=pd.DataFrame(dev);budgets={};summary={};rng=np.random.default_rng(949999)
    for m,d in dd.groupby('method'):
        x=d.per_round_s.to_numpy();values={}
        for n in [32,256]:
            sample=x[:n];quant=float(np.quantile(sample,.95));boot=np.quantile(sample[rng.integers(n,size=(4000,n))],.95,axis=1)
            values[str(n)]=dict(p95_s=quant,bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),reserve_s=float(np.ceil(2*quant/.00005)*.00005))
        budgets[m]=values['256']['reserve_s'];summary[m]=values
    dump(out/'reservation_protocol.json',dict(development=dev,summary=summary,reserve_per_round_s=budgets,
        rule='Twice warmed development p95 rounded upward to 0.05 ms; 256 disjoint development frames; 512 held-out frames; no WCET claim.',
        python=sys.version,platform=platform.platform(),test_seed_start=941000,test_frames=512))
    with (out/'reservation_records.jsonl').open('w') as f:
        for seed in range(941000,941512):
            sc=scenario(cfg,seed)
            for m in methods:write_record(f,run_fast(cfg,sc,m,reserve_s=budgets[m]))
    print('isolated timing complete',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['intervals','vector','calibration','fresh','statistics','timing','kg-timing'],required=True)
    p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output');a=p.parse_args();root=Path(a.root);out=Path(a.output) if a.output else root/'round2_results';out.mkdir(exist_ok=True,parents=True)
    {'intervals':interval_audit,'vector':vector_study,'calibration':calibration_study,
     'fresh':lambda r,o:calibration_study(r,o,True),'statistics':statistics,'timing':timing_study,'kg-timing':kg_timing_study}[a.mode](root,out)

if __name__=='__main__':main()
