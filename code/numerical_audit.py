"""Post-review scalar Q audit: CDF grid, KG17 and adaptive Gauss--Kronrod.

Reuses the first 32 states of the existing value audit. It is not a
new independent service or reliability experiment. The reference is the
existing convex-minimization/Brent implementation at tighter endpoint tolerance.
All competitors use the same terminal certificate. The reference, grid and
adaptive integral truncate at [-6,6]; KG17 retains its full Gaussian rule. The
excluded tail utility is at most 2*Phi(-6) times total task utility. Timings
are descriptive; Python reference root timings are not a matched-language
speed benchmark. The integration tolerance is not a discontinuity miss bound.
"""
import argparse,csv,json,time,warnings,platform,sys,hashlib
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
from scipy.integrate import quad,IntegrationWarning
from model import Config,scenario,fantasy
from innovation import probe_value
from fast_backend import value,schedule,library
from revised_policies import quadrature_value,gaussian_rule

def adaptive_value(cfg,mean,cov,payload,count,index):
    b,cs=fantasy(cov,index,(cfg.measurement_db*np.log(10)/10)**2)
    def integrand(z):
        return schedule(cfg,mean+b*z,cs,payload,count+1)['value']*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    result=quad(integrand,-cfg.z_limit,cfg.z_limit,points=[-4.,-2.,0.,2.,4.],epsabs=1e-4,epsrel=1e-7,limit=500,full_output=1)
    q,error,info=result[:3]
    return float(q),dict(estimated_error=float(error),evaluations=int(info['neval']),message=result[3] if len(result)>3 else '')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--frames',type=int,default=32);args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    if (out/'values.csv').exists():raise FileExistsError('Use an empty output directory')
    cfg=replace(Config(),endpoint_tol=1e-8)
    protocol=dict(study='Post-review numerical value diagnostic',config=asdict(cfg),state_seeds=[110000,110000+args.frames-1],reuse='First states of the existing 128-state value audit; not a new independent sample',candidate_order='All default coordinates',competitors=['CDF25','KG17','Adaptive-GK'],adaptive=dict(epsabs=1e-4,epsrel=1e-7,limit=500,fixed_points=[-4,-2,0,2,4]),reference='Existing Python convex minimization and Brent roots, endpoint_tol=1e-8; finite precision, tail truncated at +/-6',timing='One warmed evaluation per method and candidate; deterministic randomized method order; sequential one-thread run. Root timing includes its Python implementation and cannot establish algorithmic speed superiority.',python=sys.version,platform=platform.platform())
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    library();z,w=gaussian_rule(17);warm=scenario(cfg,109999);k=int(warm['candidates'][0])
    value(cfg,warm['mean'],warm['covariance'],warm['payload'],0,k)
    quadrature_value(cfg,warm['mean'],warm['covariance'],warm['payload'],0,k,z,w)
    adaptive_value(cfg,warm['mean'],warm['covariance'],warm['payload'],0,k)
    rows=[];decisions=[]
    with (out/'values.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','index','method','q','reference','error','wall_s','reference_wall_s','estimated_error','evaluations','message']);writer.writeheader()
        for seed in range(110000,110000+args.frames):
            sc=scenario(cfg,seed);mu=sc['mean'];cs=sc['covariance'];payload=sc['payload'];qs={m:[] for m in ['Reference','CDF25','KG17','Adaptive-GK']}
            current=schedule(cfg,mu,cs,payload,0)['value']
            for idx in sc['candidates']:
                k=int(idx);tick=time.perf_counter();ref=probe_value(cfg,mu,cs,payload,0,k);ref_time=time.perf_counter()-tick;qs['Reference'].append(ref)
                for method in np.random.default_rng(seed*100+k).permutation(['CDF25','KG17','Adaptive-GK']):
                    tick=time.perf_counter();detail=dict(estimated_error=0.,evaluations=0,message='')
                    if method=='CDF25':q=value(cfg,mu,cs,payload,0,k)
                    elif method=='KG17':q=quadrature_value(cfg,mu,cs,payload,0,k,z,w)
                    else:q,detail=adaptive_value(cfg,mu,cs,payload,0,k)
                    wall=time.perf_counter()-tick;qs[method].append(q)
                    row=dict(seed=seed,index=k,method=method,q=q,reference=ref,error=q-ref,wall_s=wall,reference_wall_s=ref_time,**detail);writer.writerow(row);rows.append(row)
            actions={m:int(sc['candidates'][np.argmax(q)]) if max(q)>current+cfg.gain_tol_kbit else -1 for m,q in qs.items()}
            decisions.append(dict(seed=seed,current=current,reference_action=actions['Reference'],actions=actions,q=qs,candidates=sc['candidates'].tolist()))
            stream.flush();print('numerical states',seed-110000+1,flush=True)
    summary={}
    for method in ['CDF25','KG17','Adaptive-GK']:
        data=[x for x in rows if x['method']==method];err=np.array([x['error'] for x in data]);times=np.array([x['wall_s'] for x in data])
        summary[method]=dict(candidate_values=len(data),mean_abs_error=float(np.abs(err).mean()),p95_abs_error=float(np.quantile(np.abs(err),.95)),max_abs_error=float(np.abs(err).max()),mean_signed_error=float(err.mean()),max_overestimate=float(max(0,err.max())),mean_ms=float(1000*times.mean()),median_ms=float(1000*np.median(times)),decision_agreement=sum(x['actions'][method]==x['reference_action'] for x in decisions),states=len(decisions),integration_warnings=sum(bool(x['message']) for x in data))
    summary['reference']=dict(mean_ms=float(1000*np.mean([x['reference_wall_s'] for x in rows if x['method']=='CDF25'])),tail_probability=float(2*__import__('scipy').special.ndtr(-cfg.z_limit)))
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
