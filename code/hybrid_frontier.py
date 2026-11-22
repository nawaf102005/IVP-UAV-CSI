"""Same validated terminal power family, with or without preceding C-IVP reports."""
import json,time,argparse
from pathlib import Path
from dataclasses import replace
from model import Config,scenario
from revised_policies import run_fast
from power_frontier import reference
from admission import execute
from chance_benchmark import draw,failures

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='../revision_results');a=p.parse_args();out=Path(a.output);cfg=Config();start=time.perf_counter()
    with (out/'hybrid_records.jsonl').open('w') as f:
        for seed in range(190000,190128):
            sc=scenario(cfg,seed);t=time.perf_counter();r,mu,cov,_=run_fast(cfg,sc,'C-IVP',return_posterior=True)
            post={**sc,'mean':mu,'covariance':cov};short=replace(cfg,frame_s=cfg.frame_s-r['probes']*cfg.probe_s)
            action,trials=reference(short,post);elapsed=time.perf_counter()-t
            x=execute(cfg,sc,action,r['probes']);g=draw(mu,cov,16384,seed+3000000,cfg.n);bad=int(failures(cfg,sc['payload'],g,action,r['probes']).sum())
            x.update(method='IVP+frontier',seed=seed,selected=list(map(int,action['group'])),mode=action['mode'],power=action['power'].tolist(),
                     probes=r['probes'],observations=r['observations'],wall_s=elapsed,conditional_risk=bad/16384,conditional_bad=bad,
                     conditional_draws=16384,trials=trials)
            if 'airtime' in action:x['airtime']=action['airtime'].tolist()
            f.write(json.dumps(x)+'\n');f.flush()
    (out/'hybrid_protocol.json').write_text(json.dumps(dict(count=128,seed_start=190000,elapsed_s=time.perf_counter()-start,
        note='A terminal-allocator ablation with the same finite power family and held-out validation as Validated-frontier; not the certified IVP theorem.'),indent=2))

if __name__=='__main__':main()
