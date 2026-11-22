"""Prospective, paired multiframe acquisition and independent host reservation check.

Start with --stage freeze, then --stage reuse. The two timing stages are serial.
All study inputs and inference families are set in the release protocol.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from admission import execute
from aligned_frontier import terminal
from fast_backend import affine_value, library, schedule
from mobility_cost_study import target_margin_scores
from model import Config, posterior, scenario

BASE = Path(__file__).resolve().parents[1]
CFG = Config()
H = 40
SPEEDS = (10,20,30)
KS = (1,5,10,20)
COST = (0.,.00172)
RULES = ('IVP','Sensitivity','Fixed-IVP')
START = 141000000
HOST_START = 151000000
DEV_START = 152000000
PROFILE_START = 161000000
NTRACKS = 128
HOST_TRACKS = 64
PROFILE_N = 128
PROFILE_DRAWS = (256,1024,4096)
PROFILE_NODES = (9,25,49)
FAMILY = len(SPEEDS)*len(KS)*len(COST)*2*4
FIELDS = ['speed','frame','K','cost_ms','rule','requested_index','observation','reserve_ms',
          'cert_service','cert_outage','cert_mask','cert_mode','front_service','front_outage','front_mask','front_mode']
def jsave(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def ci(x,family):
    x=np.asarray(x,float); mu=float(x.mean())
    half=float(student_t.ppf(1-.05/(2*family),len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mu,lo=mu-half,hi=mu+half)
def protocol(out):
    out.mkdir(parents=True,exist_ok=False)
    jsave(out/'protocol.json',dict(
        study='Prospective fixed-geometry large-scale shadow aging and reduced planning fidelity',
        config=vars(CFG),speeds_m_s=SPEEDS,decorrelation_m=10,frame_s=CFG.frame_s,horizon=H,
        refresh_every=KS,incremental_cost_s=COST,independent_tracks=NTRACKS,track_seed_start=START,
        geometry='Per-trajectory fixed path-loss mean, stationary covariance and pairing. New payloads at each frame uniform 0.85 to 1.15 times nominal, paired across all methods and speeds.',
        law='Stationary multivariate Gaussian AR(1); rho=exp(-v*T/10). Report x_t at start, serve true x_(t+1). Posterior persists across every frame and refresh; no execution labels used to learn. No small-scale fading or flight data.',
        rules='IVP and sensitivity make a one-step certificate-based report/stop decision on refresh frames only. Fixed-IVP always buys exactly one report, using the highest one-step certificate Q even if stopping would be preferred. Nonrefresh frames buy zero reports. Both terminal allocators are evaluated from the same history. No-report skips acquisition.',
        cost='Primary: whole report airtime and one or two hypothetical per-round costs on the refresh frame; no acquisition charge on other frames. Common terminal work excluded. 1.72 ms is the historical certificate ratio used as an assumed charge.',
        primary_family=FAMILY,primary_contrasts='IVP, sensitivity, fixed-IVP each minus no-report and IVP minus sensitivity for every speed, K, cost and terminal. Simultaneous Bonferroni paired t, two sided, 95%, paired mean kbit/frame per independent 40-frame trajectory; outages descriptive.',
        reuse_frame_columns=FIELDS,
        host=dict(dev_frames=64,dev_seed_start=DEV_START,test_seed_start=HOST_START,test_tracks=HOST_TRACKS,horizon=H,v_m_s=20,K=[1,10],methods=['No-report','IVP'],round_charge_ms=1.72,
                  terminal='4096-draw validated frontier for all methods',
                  common_reserve='1.25 times development 95th percentile of one warmed no-report terminal, rounded upward to 0.1 ms. Held fixed for all held-out frames. Evaluation is serial, one BLAS thread, warm-up excluded. Time includes acquisition, posterior update and final allocation, excludes scenario generation and execution scoring.',
                  deadline='Every frame pays common terminal reserve. Refresh frames additionally pay c per report/stop round and radio airtime if accepted. A late decision earns zero service. On an overrun a newly acquired observation is discarded before posterior carry-forward. No-report pays the common reserve and is penalized identically.',
                  family='Two direct IVP-minus-no-report trajectory contrasts, one per K, simultaneous 95%. No tuning of reservations on test data.'),
        profile=dict(test_seed_start=PROFILE_START,paired_frames=PROFILE_N,draws=PROFILE_DRAWS,nodes=PROFILE_NODES,cap=4,terminal_draws=4096,v_m_s=20,
                     cells='Full 3x3 planning grid. Same planning stream, each stopping step conditional on observed reports; final stream independent and shared across methods. Acceptance threshold recalculated for planning draw count.',
                     timing='Two runs per frame and cell on one pinned CPU, randomized within frame. (1) Radio-only service and warmed whole-decision wall time. (2) All methods including no-report receive a 1.0-ms total computation reservation on the 20-ms frame. Subtract this before planning; active overruns earn zero. Time stops after final validation, before truth-based service scoring.',
                     statistic='18 simultaneous Bonferroni 95% paired contrasts vs identical no-report terminal (9 each at 0 and 1 ms). A 1-ms operating point requires empirical whole-decision p95 at most 1 ms AND a positive charged-service lower limit.',
                     warmup_seeds=[PROFILE_START-9,PROFILE_START-8,PROFILE_START-7]),
        streams='NumPy independent SeedSequence namespaces for AR shocks, report errors and per-frame payloads. The same standardized input streams pair the speed scenarios. Frame-specific final-gate seeds are disjoint from all geometry and observation seeds.',
        python=sys.version,platform=platform.platform(),software={m:hashlib.sha256((BASE/'code'/m).read_bytes()).hexdigest() for m in ['persistence_campaign.py','frontier_tradeoff_campaign.py','aligned_frontier.py','fast_kernel.cpp','model.py']}))
def track_inputs(seed,cfg=CFG):
    sc=scenario(cfg,seed); base=sc['mean']; sigma=sc['covariance']; D=len(base)
    L=np.linalg.cholesky(sigma+1e-14*np.eye(D))
    shock=np.random.default_rng(np.random.SeedSequence([111,seed])).standard_normal((H,D))@L.T
    noise=np.random.default_rng(np.random.SeedSequence([127,seed])).standard_normal((H,D))
    mult=np.random.default_rng(np.random.SeedSequence([151,seed])).uniform(.85,1.15,(H,2,cfg.n))
    payload=mult*np.asarray([cfg.ul_bits,cfg.dl_bits])[None,:,None]
    return sc,shock,noise,payload
def true_path(sc,shock,rho):
    states=[sc['true']]
    for z in shock:
        states.append(sc['mean']+rho*(states[-1]-sc['mean'])+np.sqrt(1-rho*rho)*z)
    return states
def predict(mu,P,center,S,rho):
    if mu is center and P is S:return center,S
    return center+rho*(mu-center),rho*rho*P+(1-rho*rho)*S
def cache_key(mu,P,*extra):return (mu.tobytes(),P.tobytes(),*extra)
def select_report(cfg,mu,P,center,S,rho,sc,rule,c,cache=None):
    """One-step optional/myopic or fixed-one acquisition. No execution truth as input."""
    cache={} if cache is None else cache
    key=cache_key(mu,P,c)
    if key not in cache:
        me,Pe=predict(mu,P,center,S,rho)
        cur=schedule(cfg,me,Pe,sc['payload'],0,reserve_s=c)['value']
        cache[key]=(me,Pe,cur,{})
    me,Pe,cur,qs=cache[key]
    if rule!='Fixed-IVP' and cur>=sc['payload'].sum()/1000.-1e-9:return -1
    cand=sc['candidates'];rv=(cfg.measurement_db*np.log(10)/10)**2
    if rule=='Sensitivity':
        scores=target_margin_scores(replace(cfg,frame_s=cfg.frame_s-c),me,Pe,sc['payload'],0,cand,rho*P,P)
        cand=cand[[int(np.argmax(scores))]]
    for k in cand:
        if int(k) not in qs:
            b=rho*P[:,k]/np.sqrt(P[k,k]+rv)
            qs[int(k)]=affine_value(cfg,me,Pe-np.outer(b,b),b,sc['payload'],1,reserve_s=c)
    selected=int(cand[np.argmax([qs[int(k)] for k in cand])])
    return selected if rule=='Fixed-IVP' or qs[selected]>cur+cfg.gain_tol_kbit else -1
def observe_and_predict(cfg,belief,center,S,rho,idx,at_start,noise):
    mu,P=belief;y=None
    if idx>=0:
        rv=(cfg.measurement_db*np.log(10)/10)**2
        y=float(at_start[idx]+np.sqrt(rv)*noise[idx]);mu,P=posterior(mu,P,idx,y,rv)
    return predict(mu,P,center,S,rho),y
def public_input(sc,payload,seed,t):
    return dict(seed=400000000+(seed%200000000)*H+t,payload=payload,candidates=sc['candidates'])
def score(cfg,frame_input,a,m,reserve,truth):
    execution=dict(payload=frame_input['payload'],true=truth)
    return execute(replace(cfg,frame_s=cfg.frame_s-reserve),execution,a,m,true_override=truth)
def record_action(cfg,fi,me,Pe,m,c,truth):
    actions=[schedule(cfg,me,Pe,fi['payload'],m,reserve_s=c),terminal(cfg,fi,me,Pe,m,c)]
    vals=[]
    for a in actions:
        r=score(cfg,fi,a,m,(m+1)*c,truth)
        vals.extend([r['completed_kbit'],r['outage'],sum(1<<int(i) for i in a['group']),a['mode']])
    return vals
def reuse_track(job):
    seed,folder=job;sc,sh,noise,payload=track_inputs(seed);center=sc['mean'];S=sc['covariance']
    tactics=[('No-report',0,0.)]+[(rule,k,c) for k in KS for c in COST for rule in RULES]
    means=[]
    with gzip.open(folder/f'{seed}.jsonl.gz','wt',compresslevel=3) as stream:
        for v in SPEEDS:
            rho=float(np.exp(-v*CFG.frame_s/10));actual=true_path(sc,sh,rho)
            beliefs={opt:(center,S) for opt in tactics}; sums={opt:np.zeros(5) for opt in tactics}
            for t in range(H):
                fi=public_input(sc,payload[t],seed,t);terms={};look={}
                for opt in tactics:
                    rule,k,c=opt;refresh=k!=0 and t%k==0;charge=c if refresh else 0.
                    ix=select_report(CFG,*beliefs[opt],center,S,rho,fi,rule,charge,look) if refresh else -1
                    (me,Pe),y=observe_and_predict(CFG,beliefs[opt],center,S,rho,ix,actual[t],noise[t])
                    beliefs[opt]=(me,Pe);m=int(ix>=0);key=cache_key(me,Pe,m,charge)
                    if key not in terms:terms[key]=record_action(CFG,fi,me,Pe,m,charge,actual[t+1])
                    vals=terms[key]
                    stream.write(json.dumps([v,t,k,1000*c,rule,ix,y,1000*(m+1)*charge,*vals],separators=(',',':'))+'\n')
                    sums[opt]+=[m,vals[0],vals[1],vals[4],vals[5]]
            for (rule,k,c),s in sums.items():
                means.append(dict(seed=seed,speed=v,K=k,cost_ms=1000*c,rule=rule,reports=int(s[0]),cert_kbit=s[1]/H,
                                  cert_outages=int(s[2]),front_kbit=s[3]/H,front_outages=int(s[4])))
    return means
def reuse(out,workers):
    prot=json.loads((out/'protocol.json').read_text());assert prot['primary_family']==FAMILY
    directory=out/'trajectory_records';directory.mkdir(exist_ok=False)
    library();rows=[]
    with ProcessPoolExecutor(workers) as pool:
        for i,rr in enumerate(pool.map(reuse_track,[(s,directory) for s in range(START,START+NTRACKS)],chunksize=1)):
            rows.extend(rr)
            if (i+1)%8==0:print('Independent trajectories completed',i+1,'/',NTRACKS,flush=True)
    write_csv(out/'trajectory_means.csv',rows)
    lookup={(r['seed'],r['speed'],r['K'],r['cost_ms'],r['rule']):r for r in rows}
    results=[]
    for v in SPEEDS:
        for k in KS:
            for c in COST:
                for trm in ['cert','front']:
                    for l,r in [(s,'No-report') for s in RULES]+[('IVP','Sensitivity')]:
                        def entries(m):
                            return [lookup[s,v,0 if m=='No-report' else k,0. if m=='No-report' else 1000*c,m] for s in range(START,START+NTRACKS)]
                        aa,bb=entries(l),entries(r)
                        ya=np.array([x[trm+'_kbit'] for x in aa]);yb=np.array([x[trm+'_kbit'] for x in bb])
                        results.append(dict(speed=v,K=k,cost_ms=1000*c,terminal=trm,left=l,right=r,n_tracks=NTRACKS,**ci(ya-yb,FAMILY),
                                            left_kbit=float(ya.mean()),right_kbit=float(yb.mean()),left_reports=sum(z['reports'] for z in aa),
                                            left_outages=sum(z[trm+'_outages'] for z in aa),right_outages=sum(z[trm+'_outages'] for z in bb)))
    assert len(results)==FAMILY
    write_csv(out/'reuse_contrasts.csv',results);print('Reuse family complete',len(results),flush=True)
def pin():
    if hasattr(os,'sched_getaffinity'):
        cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu});return cpu
    return None
def host_terminal_budget():
    times=[]
    for i in range(-4,64):
        sc,sh,noise,payload=track_inputs(DEV_START+i);fi=public_input(sc,payload[0],DEV_START+i,0)
        tick=time.perf_counter_ns();terminal(CFG,fi,sc['mean'],sc['covariance'],0,0.);ms=(time.perf_counter_ns()-tick)/1e6
        if i>=0:times.append(ms)
    b=float(np.ceil(1.25*np.quantile(times,.95)/.1)*.1/1000)
    return b,times
def host(out):
    assert (out/'protocol.json').is_file() and not (out/'host_development.json').exists()
    cpu=pin();library();b,dev=host_terminal_budget();jsave(out/'host_development.json',dict(common_reserve_ms=1000*b,dev_ms=dev,cpu=cpu,frames=len(dev)))
    if b+2*COST[1]+CFG.probe_s>=CFG.frame_s:raise ValueError('Development reservation leaves insufficient time for a report.')
    cfg=replace(CFG,frame_s=CFG.frame_s-b);rho=float(np.exp(-20*CFG.frame_s/10));methods=[0,1,10];episodes=[]
    with gzip.open(out/'host_frames.jsonl.gz','wt') as f:
        for seed in range(HOST_START,HOST_START+HOST_TRACKS):
            sc,sh,noise,payload=track_inputs(seed);center=sc['mean'];S=sc['covariance'];truth=true_path(sc,sh,rho)
            belief={k:(center,S) for k in methods};totals={k:np.zeros(5) for k in methods}
            order_rng=np.random.default_rng(np.random.SeedSequence([217,seed]))
            for t in range(H):
                fi=public_input(sc,payload[t],seed,t)
                for val in order_rng.permutation(methods):
                    k=int(val);refresh=k>0 and t%k==0;c=COST[1] if refresh else 0.
                    tick=time.perf_counter_ns()
                    ix=select_report(cfg,*belief[k],center,S,rho,fi,'IVP',c) if refresh else -1
                    nextbel,y=observe_and_predict(cfg,belief[k],center,S,rho,ix,truth[t],noise[t]);m=int(ix>=0)
                    a=terminal(cfg,fi,*nextbel,m,c);wall=(time.perf_counter_ns()-tick)*1e-9
                    reserved=b+(m+1)*c;late=wall>reserved
                    r=score(CFG,fi,a,m,reserved,truth[t+1])
                    executed=0. if late else r['completed_kbit'];failure=int(r['active'] and (late or bool(r['outage'])))
                    belief[k]=predict(*belief[k],center,S,rho) if late and m else nextbel
                    totals[k]+=[executed,failure,r['outage'],int(late and r['active']),m]
                    f.write(json.dumps([seed,t,k,ix,y,wall,reserved,r['completed_kbit'],executed,r['outage'],int(late),int(r['active'])],separators=(',',':'))+'\n')
            for k,agg in totals.items():
                episodes.append(dict(seed=seed,K=k,net_kbit=agg[0]/H,combined_failures=int(agg[1]),radio_failures=int(agg[2]),active_overruns=int(agg[3]),reports=int(agg[4])))
            if (seed-HOST_START+1)%8==0:print('Host tracks',seed-HOST_START+1,'/',HOST_TRACKS,flush=True)
    write_csv(out/'host_trajectories.csv',episodes);res=[]
    base=np.array([r['net_kbit'] for r in episodes if r['K']==0])
    for k in (1,10):
        vals=[r for r in episodes if r['K']==k];ya=np.array([r['net_kbit'] for r in vals])
        res.append(dict(K=k,tracks=HOST_TRACKS,**ci(ya-base,2),ivp_kbit=float(ya.mean()),nr_kbit=float(base.mean()),common_ms=1000*b,
                        ivp_overruns=sum(x['active_overruns'] for x in vals),nr_overruns=sum(x['active_overruns'] for x in episodes if x['K']==0)))
    write_csv(out/'host_contrasts.csv',res);print('Host complete',res,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['freeze','reuse','host'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='freeze':protocol(a.out)
    elif a.stage=='reuse':reuse(a.out,a.workers)
    else:host(a.out)
