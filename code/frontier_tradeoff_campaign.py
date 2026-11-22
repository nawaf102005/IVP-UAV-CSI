"""Paired, single-CPU frontier planning resolution and end-to-end deadline study."""
import argparse
import gzip
import json
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from admission import execute
from aligned_frontier import PLAN_OFFSET, acceptance_count, action_from_row, native_values, terminal
from fast_backend import library
from mobility_cost_study import make_case
from model import posterior
from persistence_campaign import (CFG, PROFILE_DRAWS, PROFILE_NODES, PROFILE_N, PROFILE_START,
                                  ci, jsave, pin, write_csv)

@lru_cache(maxsize=None)
def nodes_for(knots):
    mass = ndtr(6.) - ndtr(-6.)
    return ndtri(ndtr(-6.) + (np.arange(knots) + .5) * mass / knots), mass

def select(cfg, sc, joint_mean, joint_cov, draws, knots):
    """Planning uses its private simulation stream. Only the selected report is observed."""
    mean, cov = joint_mean.copy(), joint_cov.copy()
    dim = len(sc['mean'])
    if draws == 0:
        return terminal(cfg, sc, mean[dim:], cov[dim:,dim:], 0), [], []
    nodes, mass = nodes_for(knots)
    rv = (cfg.measurement_db * np.log(10) / 10)**2
    used = np.zeros(dim, int)
    obs, history = [], []
    while True:
        count = len(obs)
        _, rows = native_values(cfg, mean[dim:], cov[dim:,dim:], sc['payload'], count, sc['seed'],
                                offset=PLAN_OFFSET, samples=draws)
        current = action_from_row(cfg, rows[0])['value']
        if (count == cfg.max_probes or current >= sc['payload'].sum()/1000 - 1e-9
                or cfg.frame_s - count * cfg.probe_s <= 0):
            break
        forecasts = []
        for index in sc['candidates']:
            direction = cov[dim:,index] / np.sqrt(cov[index,index] + rv)
            post = cov[dim:,dim:] - np.outer(direction, direction)
            values, _ = native_values(cfg, mean[dim:], post, sc['payload'], count + 1, sc['seed'],
                                      direction, nodes, 0., PLAN_OFFSET, draws)
            forecasts.append(float(values.mean() * mass))
        best = int(np.argmax(forecasts))
        index, value = int(sc['candidates'][best]), forecasts[best]
        history.append(dict(current=current, forecast=value, index=index))
        if value <= current + cfg.gain_tol_kbit:
            break
        y = float(sc['true'][index] + np.sqrt(rv)*sc['report_noise'][index,used[index]])
        used[index] += 1
        mean, cov = posterior(mean, cov, index, y, rv)
        obs.append(dict(index=index, observation=y))
    return terminal(cfg, sc, mean[dim:], cov[dim:,dim:], len(obs)), obs, history

def run_one(sc, jm, jc, profile):
    draws, knots, budget_ms = profile
    config = replace(CFG, frame_s=CFG.frame_s-budget_ms/1000.)
    tick = time.perf_counter_ns()
    action, obs, history = select(config, sc, jm, jc, draws, knots)
    elapsed = (time.perf_counter_ns()-tick)/1e6
    r = execute(config, sc, action, len(obs))
    missed = budget_ms > 0 and elapsed > budget_ms
    return dict(seed=int(sc['seed']), draws=draws, nodes=knots, budget_ms=budget_ms, wall_ms=elapsed,
                reports=len(obs), observations=obs, history=history, selected=action['group'].tolist(),
                mode=action['mode'], validation_bad=action['validation_bad'], knob_index=action['knob_index'],
                active=int(r['active']), radio_outage=r['outage'], active_overrun=int(missed and r['active']),
                deadline_miss=int(missed), combined_failure=int(r['active'] and (missed or r['outage'])),
                radio_kbit=r['completed_kbit'], net_kbit=0. if missed else r['completed_kbit'])

def run(out):
    protocol=json.loads((out/'protocol.json').read_text())['profile']
    assert not (out/'profile_frames.jsonl.gz').exists()
    profiles = [(l,j,b) for b in (0.,1.) for l,j in [(0,0)]+[(l,j) for l in PROFILE_DRAWS for j in PROFILE_NODES]]
    cpu=pin()
    library()
    rho = float(np.exp(-20*CFG.frame_s/10))
    for seed in protocol['warmup_seeds']:
        sc,jm,jc=make_case(CFG,seed,rho)
        for profile in profiles:
            run_one(sc,jm,jc,profile)
    records = {p:[] for p in profiles}
    with gzip.open(out/'profile_frames.jsonl.gz','wt') as stream:
        for seed in range(PROFILE_START,PROFILE_START+PROFILE_N):
            sc,jm,jc=make_case(CFG,seed,rho)
            order=np.random.default_rng(np.random.SeedSequence([283,seed])).permutation(len(profiles))
            for i in order:
                profile=profiles[i];row=run_one(sc,jm,jc,profile)
                records[profile].append(row)
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
            if (seed-PROFILE_START+1)%8 == 0:
                print('Planning fidelity frames',seed-PROFILE_START+1,'/',PROFILE_N,flush=True)
    summary=[]
    for (l,j,b),rows in records.items():
        baseline=records[0,0,b]
        net=np.array([r['net_kbit'] for r in rows]);nr=np.array([r['net_kbit'] for r in baseline])
        elapsed=np.array([r['wall_ms'] for r in rows])
        summary.append(dict(draws=l,nodes=j,budget_ms=b,frames=PROFILE_N,**ci(net-nr,18),
                            service_kbit=float(net.mean()),no_report_kbit=float(nr.mean()),
                            mean_wall_ms=float(elapsed.mean()),median_wall_ms=float(np.median(elapsed)),
                            p95_wall_ms=float(np.quantile(elapsed,.95)),reports=sum(r['reports'] for r in rows),
                            active_overruns=sum(r['active_overrun'] for r in rows),
                            deadline_misses=sum(r['deadline_miss'] for r in rows),
                            radio_outages=sum(r['radio_outage'] for r in rows),
                            combined_failures=sum(r['combined_failure'] for r in rows),
                            max_planning_bad=acceptance_count(CFG.n,l) if l else None))
    write_csv(out/'profile_summary.csv',summary)
    jsave(out/'profile_run.json',dict(cpu=cpu,records=sum(len(v) for v in records.values()),
                                       cells=18,test_frames=PROFILE_N,budget_ms=1.,
                                       operating_points=[(r['draws'],r['nodes']) for r in summary if
                                                         r['draws'] and r['budget_ms']==1 and r['p95_wall_ms']<=1 and r['lo']>0]))
    print('Planning fidelity complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
