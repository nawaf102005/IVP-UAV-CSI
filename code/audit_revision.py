"""Independent replay of every revised stored action and deterministic certificate.

Checks recorded execution, peak powers, phase airtime, posterior certificates,
fixed report counts, source observations and C-IVP's implemented stopping test.
Statistically validated methods receive physical and acceptance-log checks, not
an incorrectly asserted Gaussian union certificate. No policy is reoptimized.
"""
import argparse,json,functools,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.stats import norm
from model import Config,scenario,posterior,noise_power,thresholds
from run_revision import make_case
from ray_evaluation import fit,case
from conformal_report_filter import conformal_quantile,safe_schedule
from admission import coefficients,orthogonal_data,orthogonal_airtime,execute


def audit(root):
    res=root/'revision_results';proto=json.loads((res/'revision_protocol.json').read_text())
    config={s['name']:Config(**s['config']) for s in proto['settings']}
    raycfg=Config(**json.loads((res/'ray_protocol.json').read_text())['config'])
    raymeta={};raypriors={};rayarrays={}
    for name in ['canyon_low','canyon_high','munich']:
        rayarrays[name]=np.load(root/'ray_data'/f'{name}.npz');raymeta[name]=json.loads((root/'ray_data'/f'{name}.json').read_text())
        for kind in ['incoherent','coherent']:raypriors[name,kind]=fit(rayarrays[name],raymeta[name],kind)
    @functools.lru_cache(512)
    def synthetic(source,scn,seed):
        if source=='revision':cfg=config[scn];sc=make_case(scn,cfg,seed)
        elif source=='timing':cfg=replace(Config(),n=int(scn.split('-')[1]));sc=scenario(cfg,seed)
        elif source=='latency':
            duration=float(scn.split('-')[1][:-2])/1000;b=Config();cfg=replace(b,frame_s=duration,ul_bits=b.ul_bits*duration/b.frame_s,dl_bits=b.dl_bits*duration/b.frame_s);sc=scenario(cfg,seed)
        else:cfg=Config();sc=scenario(cfg,seed)
        return cfg,sc
    errors=[];counts={};maxres=0.;started=time.perf_counter()
    def error(r,message):
        if len(errors)<30:errors.append([r.get('scenario'),r['method'],r.get('seed'),message])
    for path in sorted(res.glob('*_records.jsonl')):
        source=path.stem.replace('_records','');counts[source]=0
        with path.open() as f:
            for line in f:
                r=json.loads(line);counts[source]+=1;scn=r.get('scenario',source);method=r['method'];seed=r['seed'];is_ray=source in ['ray','conformal']
                if is_ray:
                    name,which=scn.rsplit('-',1);mobile=which=='motion';kind='coherent' if mobile else 'incoherent';pr=raypriors[name,kind];cfg=raycfg
                    sc=case(cfg,rayarrays[name],raymeta[name],pr,r['ray_index'],kind,mobile,method not in ['Unscaled-IVP','Safe-IVP','Safe-NR'],seed)
                else:cfg,sc=synthetic(source,scn,seed)
                mean=sc['mean'].copy();cov=sc['covariance'].copy()
                if method=='Oracle-cov':cov=sc['truth_covariance'].copy()
                if method=='Perfect-CSI':mean=sc.get('execution_true',sc['true']).copy();cov[:]=0
                rv=(cfg.measurement_db*np.log(10)/10)**2;seen={};observations=r.get('observations',[])
                # Hybrid archive carries the original query observations as well.
                for obs in observations:
                    ix=obs['index'];rep=seen.get(ix,0);seen[ix]=rep+1
                    expected=sc['true'][ix]+np.sqrt(rv)*sc['report_noise'][ix,rep]
                    if abs(expected-obs['observation'])>1e-9:error(r,'measurement does not match frozen noise')
                    mean,cov=posterior(mean,cov,ix,obs['observation'],rv)
                if 'observations' in r and len(observations)!=r['probes']:error(r,'report count mismatch')
                if r.get('fixed_count') is not None and r['probes']!=r['fixed_count']:error(r,'fixed count not exact')
                if method.startswith('C-IVP') and r.get('fixed_count') is None:
                    for h in r.get('history',[])[:r['probes']]:
                        if h['forecast']<=h['current']+cfg.gain_tol_kbit:error(r,'nonpositive accepted lower value')
                group=np.array(r['selected'],int);power=np.array(r['power']);count=r['probes']
                act=dict(group=group,power=power,mode=r['mode'],value=r['admitted_kbit'])
                if 'airtime' in r:act['airtime']=np.array(r['airtime'])
                reserved=r.get('reserved_s',0.);ecfg=replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved))
                replay=execute(ecfg,sc,act,count)
                completed=replay['completed_kbit'];outage=replay['outage']
                if r.get('deadline_overrun') and replay['active']:completed=0.;outage=1
                if abs(completed-r['completed_kbit'])>1e-7 or outage!=r['outage']:error(r,'execution replay mismatch')
                if abs(sc['payload'][:,group].sum()/1000-r['admitted_kbit'])>1e-7:error(r,'admitted utility mismatch')
                if not len(group):continue
                cap=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
                if np.any(power<0) or np.any(power>cap*(1+1e-8)):error(r,'peak power violation')
                duration=(ecfg.frame_s-count*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
                if np.min(duration)<=0:error(r,'nonpositive data phase')
                if 'airtime' in act and np.any(act['airtime'].sum(axis=1)>duration+1e-10):error(r,'OMA duration violation')
                if source in ['frontier','hybrid'] or method=='Validated-SAA':
                    attempts=r.get('trials',r.get('attempts',[]))
                    if not attempts or attempts[-1].get('validation_upper',1)>.05:error(r,'missing successful held-out validation')
                    if abs(r['conditional_bad']/r['conditional_draws']-r['conditional_risk'])>1e-12:error(r,'risk tally')
                    continue
                if method.startswith('Safe-'):
                    q=conformal_quantile(pr,raymeta[name]);safe=safe_schedule(cfg,sc,pr['cov'],q,observations)
                    if abs(safe['value']-act['value'])>1e-7 or safe['mode']!=act['mode']:error(r,'conformal filter replay mismatch')
                    continue
                guard=r.get('guard_db',0.);nominal=method in ['Nominal','Perfect-CSI']
                if guard:
                    ix=np.r_[np.arange(cfg.n)*(cfg.n+1),cfg.n*cfg.n+np.arange(cfg.n)*(cfg.n+1)];mean[ix]-=guard*np.log(10)/10
                if method=='Conformal-NR':
                    # Independent direct calculation with a common calibrated quantile.
                    n=cfg.n;sd=np.diag(pr['cov']);q=pr['q'];mean=pr['mean'];cov=pr['cov'];theta=thresholds(cfg,sc['payload'],0)
                    fs=[];us=[]
                    for d in range(2):
                        ix=d*n*n+group[:,None]*n+group[None,:];di=d*n*n+group*(n+1)
                        v=sd[ix]+sd[di][:,None]-2*cov[ix,di[:,None]]
                        ff=theta[d,group][:,None]*np.exp(mean[ix]-mean[di][:,None]+q*np.sqrt(np.maximum(v,0)));np.fill_diagonal(ff,0);fs.append(ff)
                        us.append(theta[d,group]*noise_power(cfg)*np.exp(-mean[di]+q*np.sqrt(sd[di])))
                    fmat=np.array(fs);u=np.array(us)
                    di=np.array([group*(n+1),n*n+group*(n+1)])
                    min_time=sc['payload'][:,group]/(cfg.bandwidth_hz*np.logaddexp(0,np.log(cap/noise_power(cfg))+mean[di]-q*np.sqrt(sd[di]))/np.log(2))
                else:
                    cc=coefficients(ecfg,mean,cov,sc['payload'],count,group,nominal=nominal);fmat=np.exp(cc[0]);u=np.exp(cc[2])
                    data=orthogonal_data(ecfg,mean,cov,sc['payload'],count,group,nominal=nominal);min_time=orthogonal_airtime(data,0.)
                if act['mode']=='reuse':
                    violation=float(np.max(np.einsum('dij,dj->di',fmat,power)+u-power));maxres=max(maxres,violation)
                    if violation>1e-8:error(r,'posterior power certificate')
                elif np.any(act['airtime']<min_time-1e-10):error(r,'posterior OMA certificate')
        synthetic.cache_clear();print(source,counts[source],len(errors),flush=True)
    result=dict(decisions_checked=sum(counts.values()),counts=counts,errors=errors,maximum_reuse_residual_w=maxres,elapsed_s=time.perf_counter()-started,
        scope='Stored observations/actions, physical delivery, time/power bounds, applicable posterior certificates and finite-family acceptance logs. Does not prove propagation realism or novelty.')
    (res/'audit_revision.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if errors:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parent.parent));a=p.parse_args();audit(Path(a.root))
