"""Fresh-seed comparison against a true inexpensive margin-sensitivity rule."""
import argparse,json,time
from pathlib import Path
from model import Config,scenario
from revised_policies import run_fast

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='../revision_results');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    cfg=Config();start=time.perf_counter()
    specs=[('C-IVP','C-IVP',0),('C-IVP-S2','C-IVP',2),('C-IVP-S4','C-IVP',4),('Margin-sensitivity','Margin-sensitivity',0),('No-report','No-report',0)]
    with (out/'screening_records.jsonl').open('w') as f:
        for seed in range(220000,221024):
            sc=scenario(cfg,seed)
            for label,method,k in specs:
                r=run_fast(cfg,sc,method,candidate_limit=k);r.update(method=label,scenario='screening');f.write(json.dumps(r)+'\n')
            if (seed-219999)%128==0:f.flush();print(seed-219999,1024,round(time.perf_counter()-start,2),flush=True)
    (out/'screening_protocol.json').write_text(json.dumps(dict(seed_start=220000,count=1024,screening='top two/four reductions of linearized variance of worst full-set row-power bound',elapsed_s=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
