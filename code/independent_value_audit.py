"""Independent stratified scalar-value audit, with no headline speed claim.

Three independent 64-state seed blocks, distinct from every earlier block.
UL-heavy, balanced and DL-heavy traffic use a bidirectional 4N catalogue.
All tolerances and sample sizes are saved before evaluation. Timings remain
implementation diagnostics; the main comparison concerns errors and decisions.
"""
import argparse,csv,json,time,platform,sys
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from model import Config,scenario
from innovation import probe_value
from fast_backend import value,schedule,library
from revised_policies import quadrature_value,gaussian_rule
from numerical_audit import adaptive_value
from round3_experiments import report_catalogue

def cases():
    base=replace(Config(),endpoint_tol=1e-8)
    return [('UL-heavy',94000000,base),('Balanced',95000000,replace(base,ul_bits=18000.,dl_bits=18000.,ul_fraction=.5,ul_power_w=.2,dl_power_w=.2,reciprocity=.2)),('DL-heavy',96000000,replace(base,ul_bits=1800.,dl_bits=30000.,ul_fraction=.2,ul_power_w=1.,dl_power_w=.2,reciprocity=.2))]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--frames-per-case',type=int,default=64);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    if (out/'values.csv').exists():raise FileExistsError('Use a fresh result directory')
    protocol=dict(study='Independent stratified scalar-value audit',frames_per_case=a.frames_per_case,cases=[dict(name=n,seed_start=s,config=asdict(c)) for n,s,c in cases()],catalogue='All desired and strongest interfering coordinates in UL and DL, 4N=16 candidates',reference='Existing convex minimization and Brent roots with endpoint tolerance 1e-8 on [-6,6]',methods=['CDF25','KG17','Adaptive-GK'],adaptive=dict(epsabs=1e-4,epsrel=1e-7,limit=500,fixed_points=[-4,-2,0,2,4]),timing='Serial single-thread, warmed and randomized method order; descriptive mixed-implementation timings only, no algorithmic speed claim',sample_units=f'{3*a.frames_per_case} independent states in three disjoint fresh {a.frames_per_case}-state blocks; candidate values within a state are dependent',python=sys.version,platform=platform.platform())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');library();z,w=gaussian_rule(17);rows=[];decisions=[]
    fieldnames=['case','seed','index','method','q','reference','error','wall_s','reference_wall_s','estimated_error','evaluations','message']
    with (out/'values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
        for name,start,cfg in cases():
            warm=scenario(cfg,start-1);k=int(warm['candidates'][0]);value(cfg,warm['mean'],warm['covariance'],warm['payload'],0,k);quadrature_value(cfg,warm['mean'],warm['covariance'],warm['payload'],0,k,z,w);adaptive_value(cfg,warm['mean'],warm['covariance'],warm['payload'],0,k)
            for seed in range(start,start+a.frames_per_case):
                sc=scenario(cfg,seed);mu=sc['mean'];cov=sc['covariance'];payload=sc['payload'];ids=report_catalogue(mu,cfg.n,True);qs={m:[] for m in ['Reference','CDF25','KG17','Adaptive-GK']};current=schedule(cfg,mu,cov,payload,0)['value']
                for idx in ids:
                    k=int(idx);tick=time.perf_counter();reference=probe_value(cfg,mu,cov,payload,0,k);rt=time.perf_counter()-tick;qs['Reference'].append(reference)
                    for method in np.random.default_rng(seed*100+k).permutation(['CDF25','KG17','Adaptive-GK']):
                        detail=dict(estimated_error=0.,evaluations=0,message='');tick=time.perf_counter()
                        if method=='CDF25':q=value(cfg,mu,cov,payload,0,k)
                        elif method=='KG17':q=quadrature_value(cfg,mu,cov,payload,0,k,z,w)
                        else:q,detail=adaptive_value(cfg,mu,cov,payload,0,k)
                        elapsed=time.perf_counter()-tick;qs[method].append(q);r=dict(case=name,seed=seed,index=k,method=method,q=q,reference=reference,error=q-reference,wall_s=elapsed,reference_wall_s=rt,**detail);writer.writerow(r);rows.append(r)
                actions={m:int(ids[np.argmax(v)]) if max(v)>current+cfg.gain_tol_kbit else -1 for m,v in qs.items()}
                decisions.append(dict(case=name,seed=seed,current=current,candidates=ids.tolist(),q=qs,actions=actions,reference_action=actions['Reference']));f.flush();print(name,seed-start+1,'/',a.frames_per_case,flush=True)
    summary={}
    for case in ['All']+[x[0] for x in cases()]:
        selected=[r for r in rows if case=='All' or r['case']==case];ds=[r for r in decisions if case=='All' or r['case']==case];summary[case]={}
        for method in ['CDF25','KG17','Adaptive-GK']:
            rs=[r for r in selected if r['method']==method];e=np.array([r['error'] for r in rs]);summary[case][method]=dict(states=len(ds),candidate_values=len(rs),mean_abs_error=float(abs(e).mean()),p95_abs_error=float(np.quantile(abs(e),.95)),max_abs_error=float(abs(e).max()),mean_signed_error=float(e.mean()),decision_agreement=sum(r['actions'][method]==r['reference_action'] for r in ds),mean_ms=float(1000*np.mean([r['wall_s'] for r in rs])),integration_warnings=sum(bool(r['message']) for r in rs))
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['All'],indent=2),flush=True)
if __name__=='__main__':main()
