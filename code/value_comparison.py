"""Distributional value audit across all candidate reports in 128 fresh frames."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from model import Config,scenario
from innovation import probe_value
from fast_backend import value

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='../revision_results');a=p.parse_args();out=Path(a.output);cfg=Config();rows=[];t=time.perf_counter()
    for seed in range(110000,110128):
        sc=scenario(cfg,seed)
        for index in sc['candidates']:
            lo=value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(index))
            ref=probe_value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,int(index))
            rows.append(dict(seed=seed,index=int(index),inner_value=lo,reference_value=ref,reference_minus_inner=ref-lo))
    with (out/'value_audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    error=np.array([r['reference_minus_inner'] for r in rows])
    (out/'value_audit.json').write_text(json.dumps(dict(count=len(rows),mean_gap=float(error.mean()),max_gap=float(error.max()),
          smallest_gap=float(error.min()),negative_beyond_reference_tolerance=int((error<-.002).sum()),elapsed_s=time.perf_counter()-t),indent=2))

if __name__=='__main__':main()
