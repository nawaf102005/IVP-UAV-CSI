"""Count identical C++ certificate oracles on the existing independent Q audit.

Counts describe this implementation's logical work, not CPU instructions or
an algorithmic lower bound. Adaptive integrands stop on the first feasible
reward-ranked action; the grid scans all action/mode intervals. Both invoke
the same feasible() predicate. All previous Q values must reproduce.
"""
import argparse,csv,json,ctypes as ct
from pathlib import Path
import numpy as np
from model import scenario
from independent_value_audit import cases
from fast_backend import library,value
from revised_policies import gaussian_rule,quadrature_value
from numerical_audit import adaptive_value
from round3_experiments import report_catalogue

def reset():
    lib=library();lib.ivp_work_reset.argtypes=[ct.c_int];lib.ivp_work_reset.restype=None;lib.ivp_work_get.argtypes=[ct.POINTER(ct.c_ulonglong)];lib.ivp_work_get.restype=None;lib.ivp_work_reset(1)
def counts():
    buf=(ct.c_ulonglong*4)();library().ivp_work_get(buf);library().ivp_work_reset(0)
    return dict(certificate_tests=int(buf[0]),directional_linear_solves=int(buf[1]),action_preparations=int(buf[2]),oma_direction_tests=int(buf[3]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Use fresh output')
    (out/'protocol.json').write_text(json.dumps(dict(states=192,candidates_per_method=3072,reuse='Same fresh independent 192-state audit; no extra independent sample claim',metric='One action/mode feasible() invocation, with both directions and early rejection; linear() calls and coefficient preparations separately counted',methods=['CDF25','KG17','Adaptive-GK'],scope='Same C++ predicate across methods. Different action size and early exits have different arithmetic cost; no CPU-independent runtime guarantee.'),indent=2)+'\n')
    with (a.root/'independent_numerical_results/values.csv').open() as f:old={(int(r['seed']),int(r['index']),r['method']):float(r['q']) for r in csv.DictReader(f)}
    rows=[];z,w=gaussian_rule(17)
    with (out/'values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','seed','index','method','q','old_q','difference','certificate_tests','directional_linear_solves','action_preparations','oma_direction_tests']);writer.writeheader()
        for name,start,cfg in cases():
            for seed in range(start,start+64):
                sc=scenario(cfg,seed)
                for k in report_catalogue(sc['mean'],cfg.n,True):
                    for m in ['CDF25','KG17','Adaptive-GK']:
                        reset()
                        if m=='CDF25':q=value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(k))
                        elif m=='KG17':q=quadrature_value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(k),z,w)
                        else:q=adaptive_value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(k))[0]
                        c=counts();previous=old[seed,int(k),m];assert abs(q-previous)<1e-8,(seed,k,m,q,previous);r=dict(case=name,seed=seed,index=int(k),method=m,q=q,old_q=previous,difference=q-previous,**c);rows.append(r);writer.writerow(r)
                f.flush()
                if (seed-start+1)%16==0:print('Counted',name,seed-start+1,flush=True)
    summary={}
    for m in ['CDF25','KG17','Adaptive-GK']:
        rs=[r for r in rows if r['method']==m];summary[m]={k:dict(mean=float(np.mean([r[k] for r in rs])),median=float(np.median([r[k] for r in rs])),p95=float(np.quantile([r[k] for r in rs],.95)),maximum=max(r[k] for r in rs)) for k in ['certificate_tests','directional_linear_solves','action_preparations','oma_direction_tests']};summary[m]['max_q_change']=max(abs(r['difference']) for r in rs)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
