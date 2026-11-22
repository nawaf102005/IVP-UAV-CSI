"""Frozen initial-decision sensitivity for sampled frontier forecasts."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from model import Config
from mobility_cost_study import make_case
from aligned_frontier import terminal,forecast,PLAN_OFFSET

def one(seed):
    cfg=Config();sc,jm,jc=make_case(cfg,seed,float(np.exp(-.04)));D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2;v=terminal(cfg,sc,jm[D:],jc[D:,D:],0,offset=PLAN_OFFSET)['value'];qs={}
    for knots in [25,49,97]:
        q=[]
        for k in sc['candidates']:
            b=jc[D:,k]/np.sqrt(jc[k,k]+rv);post=jc[D:,D:]-np.outer(b,b);q.append(forecast(cfg,jm[D:],post,b,sc['payload'],1,seed,knots=knots))
        qs[str(knots)]=q
    actions={k:int(sc['candidates'][np.argmax(vv)]) if max(vv)>v+cfg.gain_tol_kbit else -1 for k,vv in qs.items()}
    return dict(seed=seed,current=v,candidates=sc['candidates'].tolist(),q=qs,actions=actions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.output)
    if (out/'sensitivity_records.json').exists():raise FileExistsError('Existing sensitivity results')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(one,range(98000000,98000032)))
    summary={}
    for k in ['25','49']:
        gaps=np.concatenate([np.abs(np.array(r['q'][k])-np.array(r['q']['97'])) for r in rows]);summary[k]=dict(states=32,candidates=len(gaps),mean_abs_q_gap=float(gaps.mean()),max_abs_q_gap=float(gaps.max()),action_agreement=sum(r['actions'][k]==r['actions']['97'] for r in rows),reference='97-point midpoint sensitivity reference, not exact expected frontier value')
    (out/'sensitivity_records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
