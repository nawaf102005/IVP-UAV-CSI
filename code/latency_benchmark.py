"""Sequential warm benchmark and causally reserved decision-time experiment.

Run when ray tracing and parallel experiments have finished. Reservations use
only disjoint development seeds. A deadline overrun earns zero completed service.
"""
import argparse,json,platform,time,sys,subprocess,os
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,scenario
from revised_policies import run_fast
from policies import run_policy
from fast_backend import build

SPECS=[('C-IVP','C-IVP',{}),('C-IVP-S2','C-IVP',{'candidate_limit':2}),
       ('Margin-sensitivity','Margin-sensitivity',{}),('A-stop','A-stop',{}),
       ('3pt-VoI','Sensitivity-3pt',{}),('No-report','No-report',{}),('OMA','OMA',{})]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='../revision_results');a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    build();base=Config();start=time.perf_counter();budgets={};dev=[]
    # Warm each code path before measuring. A shared host has scheduling stalls;
    # reserve twice its development 95th-percentile per-round wall time, rounded
    # up to 0.05 ms. Count every held-out overrun rather than claim a WCET bound.
    for label,method,kw in SPECS:
        for seed in range(929980,929990):run_fast(base,scenario(base,seed),method,**kw)
        costs=[]
        for seed in range(930000,930032):
            r=run_fast(base,scenario(base,seed),method,**kw);costs.append(r['wall_s']/(r['probes']+1));dev.append(dict(label=label,seed=seed,per_round_s=costs[-1],cpu_per_round_s=r['cpu_s']/(r['probes']+1)))
        budgets[label]=float(np.ceil(2*np.quantile(costs,.95)/.00005)*.00005)
    print('Frozen reservations',budgets,flush=True)
    cpu_model=platform.processor()
    cpu_file=Path('/proc/cpuinfo')
    if cpu_file.exists():
        cpu_model=next((line.split(':',1)[1].strip() for line in cpu_file.read_text().splitlines() if line.startswith('model name')),cpu_model)
    (out/'latency_protocol.json').write_text(json.dumps(dict(reserve_per_round_s=budgets,development=dev,platform=platform.platform(),
        reservation_rule='2 * development 95th-percentile per-round wall time, rounded upward to 0.05 ms, after 10 warm-up frames per method',
        warmup_seeds=[929980,929989],timing_seeds=[230000,230127],latency_seeds=[240000,240511],
        cpu_model=cpu_model,python=sys.version,compiler=subprocess.check_output([os.environ.get('CXX','g++'),'--version'],text=True).splitlines()[0],
        note='Single sequential process, BLAS/OpenMP one thread, warmed build. Reserve chosen before evaluating held-out seeds; overruns count as deadline failures.'),indent=2))
    with (out/'timing_records.jsonl').open('w') as f:
        for n in [4,6,8]:
            cfg=replace(base,n=n);specs=SPECS[:5]+[('C-IVP-C16','C-IVP',{'catalogue_limit':16}),('C-IVP-C32','C-IVP',{'catalogue_limit':32})]
            for seed in range(230000,230128):
                sc=scenario(cfg,seed)
                for label,method,kw in specs:
                    r=run_fast(cfg,sc,method,**kw);r.update(method=label,scenario=f'N-{n}');f.write(json.dumps(r)+'\n')
                if n<=6 and seed<230016:
                    r=run_policy(cfg,sc,'IVP');r.update(method='Python-IVP',scenario=f'N-{n}');f.write(json.dumps(r)+'\n')
            f.flush();print('Timing N',n,round(time.perf_counter()-start,1),flush=True)
    with (out/'latency_records.jsonl').open('w') as f:
        for duration in [.020,.050,.100]:
            cfg=replace(base,frame_s=duration,ul_bits=base.ul_bits*duration/base.frame_s,dl_bits=base.dl_bits*duration/base.frame_s)
            for seed in range(240000,240512):
                sc=scenario(cfg,seed)
                for label,method,kw in SPECS:
                    r=run_fast(cfg,sc,method,reserve_s=budgets[label],**kw)
                    r.update(method=label,scenario=f'T-{duration*1000:g}ms');f.write(json.dumps(r)+'\n')
            f.flush();print('Reserved T',duration,round(time.perf_counter()-start,1),flush=True)
    (out/'latency_complete.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-start)))

if __name__=='__main__':main()
