"""Fresh, frozen 512-frame dense cost slice and matched frontier-aware test."""
import argparse,gzip,json,sys,platform,hashlib
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from model import Config,posterior
from mobility_cost_study import make_case,run_policy
from aligned_frontier import run_aligned,terminal,PLAN_OFFSET,FINAL_OFFSET
from admission import execute
from platform_cost_study import interval,compact
ALPHAS=[0.,.005,.01,.015]+[round(float(x),3) for x in np.arange(.02,.08001,.005)]
FRONT_ALPHAS=[0.,.01,.02]
START=97000000
FAMILY=3*len(ALPHAS)+3*len(FRONT_ALPHAS)

def final_record(cfg,sc,jm,jc,policy,cost):
    mu=jm.copy();cov=jc.copy();D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2
    for o in policy['observations']:mu,cov=posterior(mu,cov,o['index'],o['observation'],rv)
    m=policy['probes'];a=terminal(cfg,sc,mu[D:],cov[D:,D:],m,cost);reserved=(m+1)*cost
    r=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,a,m)
    r.update(method='No-report-F' if policy['method']=='Predictive-NR' else 'Certificate-forecast-F',seed=int(sc['seed']),probes=m,observations=policy['observations'],history=policy['history'],reserve_s=cost,reserved_s=reserved,enforce_elapsed=False,selected=a['group'].tolist(),mode=a['mode'],power=a['power'].tolist(),validation_bad=a['validation_bad'],knob_index=a['knob_index'])
    if 'airtime' in a:r['airtime']=a['airtime'].tolist()
    return r

def record(r,alpha,kind):
    return {**compact(r,alpha),'kind':kind,**{k:r[k] for k in ['validation_bad','knob_index','planner_wall_s'] if k in r}}

def run_frame(seed):
    cfg=Config();rho=float(np.exp(-20*cfg.frame_s/10));sc,jm,jc=make_case(cfg,seed,rho);rows=[]
    nr=run_policy(cfg,sc,jm,jc,'Predictive-NR',enforce_elapsed=False);rows.append(record(nr,0.,'dense'))
    nf=final_record(cfg,sc,jm,jc,nr,0.);rows.append(record(nf,0.,'frontier'))
    for alpha in ALPHAS:
        for method in ['Predictive-IVP','Predictive-Margin']:
            r=run_policy(cfg,sc,jm,jc,method,reserve_s=alpha*cfg.frame_s,enforce_elapsed=False);rows.append(record(r,alpha,'dense'))
            if method=='Predictive-IVP' and alpha in FRONT_ALPHAS:rows.append(record(final_record(cfg,sc,jm,jc,r,alpha*cfg.frame_s),alpha,'frontier'))
    for alpha in FRONT_ALPHAS:rows.append(record(run_aligned(cfg,sc,jm,jc,alpha*cfg.frame_s,49),alpha,'frontier'))
    return seed,rows

def summarize(rows,frames):
    result={}
    for kind,alphas,nr,methods,contrasts in [('dense',ALPHAS,'Predictive-NR',['Predictive-IVP','Predictive-Margin'],[('IVP-NR','Predictive-IVP','Predictive-NR'),('Margin-NR','Predictive-Margin','Predictive-NR'),('IVP-Margin','Predictive-IVP','Predictive-Margin')]),('frontier',FRONT_ALPHAS,'No-report-F',['Certificate-forecast-F','Frontier-aware-IVP'],[('Aware-NR','Frontier-aware-IVP','No-report-F'),('Certificate-NR','Certificate-forecast-F','No-report-F'),('Aware-Certificate','Frontier-aware-IVP','Certificate-forecast-F')])]:
        rr=[r for r in rows if r['kind']==kind];baseline=sorted([r for r in rr if r['method']==nr],key=lambda r:r['seed']);costs=[]
        for alpha in alphas:
            groups={nr:baseline,**{m:sorted([r for r in rr if r['method']==m and r['alpha']==alpha],key=lambda r:r['seed']) for m in methods}}
            assert all(len(rs)==frames for rs in groups.values());s=dict(alpha=alpha,methods={m:dict(completed=float(np.mean([r['completed_kbit'] for r in rs])),outages=sum(r['outage'] for r in rs),reports=float(np.mean([r['probes'] for r in rs]))) for m,rs in groups.items()})
            for label,a,b in contrasts:s[label]=interval([x['completed_kbit']-y['completed_kbit'] for x,y in zip(groups[a],groups[b])],FAMILY)
            costs.append(s)
        result[kind]=costs
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=512);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Use a fresh directory')
    cfg=Config();protocol=dict(study='Fresh dense cost and frontier-aware acquisition',frames=a.frames,seed_start=START,config=asdict(cfg),rho=float(np.exp(-20*cfg.frame_s/10)),dense_alphas=ALPHAS,frontier_alphas=FRONT_ALPHAS,simultaneous_contrasts=FAMILY,confidence='Bonferroni 95% paired t over 60 fresh-study contrasts, separate from previous 1194-contrast family',bootstrap=dict(samples=2000,seed=20261120,scope='Pointwise paired-frame percentiles for first crossings; excludes interpolation error'),frontier=dict(knobs=[.01,.025,.05,.1,.2,.4,.8],power='uniform per-direction scaling to first cap',oma='fill directional time budget proportionally',forecast_samples=4096,final_samples=4096,forecast_knots=49,forecast_rule='Equal-normal-mass midpoint rule over [-6,6]; no interval/lower-value claim',planning_seed_offset=PLAN_OFFSET,final_seed_offset=FINAL_OFFSET,contract='Final validation stream is not accessed until report selection has stopped; same candidate family and acceptance criterion in planning and final gate, separate random streams. No IVP nondegradation theorem asserted for sampled forecasts.'),numerical_audit=dict(states=32,seeds=[98000000,98000031],knots=[25,49,97],scope='Initial report/stop sensitivity; no uniform convergence claim'),python=sys.version,platform=platform.platform(),code_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['dense_alignment_study.py','aligned_frontier.py','fast_kernel.cpp']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_frame,s) for s in range(START,START+a.frames)]
        with gzip.open(out/'records.jsonl.gz','wt') as stream:
            for i,f in enumerate(as_completed(futures)):
                seed,rr=f.result();rows.extend(rr)
                for r in rr:stream.write(json.dumps(r)+'\n')
                if (i+1)%16==0:print('Completed fresh frames',i+1,'/',a.frames,flush=True)
    rows.sort(key=lambda r:(r['kind'],r['seed'],r['method'],r['alpha']));summary=summarize(rows,a.frames);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'complete.json').write_text(json.dumps(dict(frames=a.frames,records=len(rows),complete=True))+'\n')
    for kind in summary:print(kind,summary[kind][0],flush=True)
if __name__=='__main__':main()
