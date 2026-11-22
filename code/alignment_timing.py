"""Dedicated serial timing: no concurrent simulations, no execution checks."""
import argparse,json,time,platform,sys
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
from model import Config
from mobility_cost_study import make_case,run_policy
from aligned_frontier import run_aligned,terminal

def evaluate(cfg,sc,jm,jc,method):
    if method in ['Certificate-IVP','No-report-C']:
        r=run_policy(cfg,sc,jm,jc,'Predictive-IVP' if method=='Certificate-IVP' else 'Predictive-NR',enforce_elapsed=False);return r['wall_s'],r['probes']
    if method=='No-report-F':
        D=len(sc['mean']);tick=time.perf_counter();terminal(cfg,sc,jm[D:],jc[D:,D:],0);return time.perf_counter()-tick,0
    with patch('aligned_frontier.execute',return_value={}):r=run_aligned(cfg,sc,jm,jc,0.)
    return r['planner_wall_s'],r['probes']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if (out/'timing_protocol.json').exists():raise FileExistsError('Existing timing block')
    cfg=Config();methods=['Certificate-IVP','Frontier-aware-IVP','No-report-C','No-report-F'];protocol=dict(frames=32,seeds=[99000000,99000031],config=asdict(cfg),rho=float(np.exp(-.04)),methods=methods,clock='perf_counter; serial single-thread randomized method order after three warm frames, generation and execution checks excluded',planner='Includes native candidate design, sampled forecasts, posterior updates and final validation. Excludes offline compilation. No measured overruns affect service.',platform=platform.platform(),python=sys.version)
    (out/'timing_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for seed in range(98999997,99000000):
        sc,jm,jc=make_case(cfg,seed,protocol['rho'])
        for m in methods:evaluate(cfg,sc,jm,jc,m)
    records=[]
    for seed in range(99000000,99000032):
        sc,jm,jc=make_case(cfg,seed,protocol['rho'])
        for method in np.random.default_rng(seed).permutation(methods):
            wall,probes=evaluate(cfg,sc,jm,jc,method);records.append(dict(seed=seed,method=method,wall_s=wall,probes=probes))
    s={}
    for m in methods:
        rs=[r for r in records if r['method']==m];wall=float(np.mean([r['wall_s'] for r in rs]));rounds=float(np.mean([r['probes']+1 for r in rs]));s[m]=dict(mean_decision_ms=wall*1000,mean_rounds=rounds,ratio_of_means_ms=wall*1000/rounds,p95_decision_ms=float(np.quantile([r['wall_s']*1000 for r in rs],.95)))
    for m,nr in [('Certificate-IVP','No-report-C'),('Frontier-aware-IVP','No-report-F')]:s[m]['incremental_ratio_ms']=(s[m]['mean_decision_ms']-s[nr]['mean_decision_ms'])/s[m]['mean_rounds']
    (out/'timing_records.json').write_text(json.dumps(records,indent=2)+'\n');(out/'timing_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
if __name__=='__main__':main()
