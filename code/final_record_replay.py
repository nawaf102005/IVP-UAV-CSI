"""Reanalyse released IVP records, without making new acquisition decisions.

Run from the package root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/final_record_replay.py --out reproduced_final --workers 4

The root audit instruments the original Python callback at one subset, one mode
and one innovation. Frontier re-evaluation uses recorded report histories and
the original independent terminal gate at the original seed. Host contrasts
use completed service after the archived reservation and overrun accounting.
"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import stats
import innovation
from dense_alignment_study import final_record
from fast_backend import library
from independent_value_audit import cases
from mobility_cost_study import make_case
from model import Config, scenario
from round3_experiments import report_catalogue

BASE = Path(__file__).resolve().parents[1]
METHOD = {"Predictive-IVP": "C-IVP", "Predictive-Margin": "Sensitivity", "Predictive-NR": "No-report"}
DURATIONS = (20, 25, 32, 40, 50, 75, 100)
FIELDS = ("completed_kbit","admitted_kbit","outage","active","probes","completed_tasks",
          "admitted_tasks","selected","power","mode","reserved_s","validation_bad","knob_index")
def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
def write_csv(path, data):
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=data[0]); w.writeheader(); w.writerows(data)
def load_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream]
def paired(a, b, family):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired samples have different sizes or insufficient frames")
    x = np.asarray(a) - np.asarray(b)
    half = stats.t.ppf(1 - .05 / (2 * family), len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    mean = float(np.mean(x))
    return dict(mean=mean, lo=float(mean-half), hi=float(mean+half))
def checked(condition, msg):
    if not condition:
        raise ValueError(msg)

def count_root(job):
    name, cfg, seed, old, original_actions = job
    sc = scenario(cfg, seed)
    candidates = list(map(int, report_catalogue(sc["mean"], cfg.n, True)))
    checked(set(candidates) == set(old), "Root input catalogue differs from released audit")
    rows = []
    for k in candidates:
        counts = defaultdict(int)
        def instrument(label, fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                counts[label] += 1
                if label == "reuse_evaluations":
                    counts["individual_eigenproblems"] += 2 * args[1][0].shape[-1]
                return fn(*args, **kwargs)
            return wrapper
        with ExitStack() as stack:
            for mod, attr, label in ((innovation,"convex_feasibility","reuse_evaluations"),
                                     (innovation,"orthogonal_feasibility","oma_evaluations"),
                                     (innovation,"coefficients","reuse_preparations"),
                                     (innovation,"orthogonal_data","oma_preparations"),
                                     (np.linalg,"solve","linear_solve_calls")):
                stack.enter_context(patch.object(mod, attr, instrument(label, getattr(mod,attr))))
            q = innovation.probe_value(cfg, sc["mean"], sc["covariance"], sc["payload"], 0, k)
        rows.append(dict(case=name,seed=seed,index=k,q_replay=q,q_archived=old[k],
                         absolute_difference=abs(q-old[k]),evaluations=counts["reuse_evaluations"]+counts["oma_evaluations"],
                         **{label:counts[label] for label in ("reuse_evaluations","oma_evaluations","individual_eigenproblems",
                                                             "linear_solve_calls","reuse_preparations","oma_preparations")}))
    current, reference_action = original_actions
    best = max(rows, key=lambda row: row["q_replay"])
    replay_action = best["index"] if best["q_replay"] > current + cfg.gain_tol_kbit else -1
    checked(replay_action == reference_action, f"Original root decision changed at seed {seed}")
    return rows

def replay_frame(job):
    seed, cfg, rho, records = job
    sc, jm, jc = make_case(cfg, seed, rho)
    old = {(r["method"], float(r["alpha"])): r for r in records if r["kind"] == "frontier"}
    rows, validations = [], 0
    for r in (x for x in records if x["kind"] == "dense"):
        name = METHOD[r["method"]]
        a = float(r["alpha"])
        checked(r["probes"] == len(r["observations"]), "Incomplete acquisition history")
        # All historical report values must match the original truth/noise realization.
        used = defaultdict(int)
        sigma = cfg.measurement_db * np.log(10) / 10
        for obs in r["observations"]:
            k = obs["index"]
            y = sc["true"][k] + sigma * sc["report_noise"][k, used[k]]
            checked(abs(y-obs["observation"]) < 1e-10, f"Stored report mismatch at {seed}")
            used[k] += 1
        out = final_record(cfg, sc, jm, jc, r, a*cfg.frame_s)
        old_key = ({"C-IVP":"Certificate-forecast-F","No-report":"No-report-F"}.get(name),a)
        if old_key in old:
            prev = old[old_key]
            for key in (*FIELDS,"airtime"):
                if key in ("power","airtime","completed_kbit","admitted_kbit","reserved_s"):
                    checked(np.allclose(out.get(key,[]), prev.get(key,[]),rtol=0,atol=1e-8),
                            f"Terminal replay mismatch at {seed} {a} {key}")
                else:
                    checked(out[key] == prev[key],f"Terminal replay mismatch at {seed} {a} {key}")
            validations += 1
        rows.append(dict(seed=seed,alpha=a,method=name,**{k:out[k] for k in FIELDS},
                         **({"airtime":out["airtime"]} if "airtime" in out else {})))
    return rows, validations

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args(); out=args.out
    out.mkdir(parents=True, exist_ok=False)
    source = {p:BASE/p for p in ["independent_numerical_results/values.csv","independent_numerical_results/decisions.json",
                                  "dense_alignment_results/records.jsonl.gz","dense_alignment_results/protocol.json",
                                  "round5_results/latency_records.jsonl","round5_results/protocol.json"]}
    original = json.loads(source["dense_alignment_results/protocol.json"].read_text())
    alphas = original["dense_alphas"]; family=3*len(alphas)+len(DURATIONS)
    checked(len(alphas)==17 and family==58, "Unexpected size of the analysis family")
    write_json(out/"protocol.json", dict(analysis="Retrospective reanalysis of archived cohorts A, F and G",
               cost_grid=alphas, durations_ms=DURATIONS, simultaneous_family=family,
               contrasts="Three paired contrasts at every archived cost (C-IVP minus no-report, sensitivity minus no-report, C-IVP minus sensitivity) and seven direct charged-rate contrasts (sensitivity minus C-IVP). Two-sided Bonferroni paired t, 95% familywise; retrospective.",
               accounting="F: no-report skips acquisition cost, F terminal gate shared, original 4096 final draws; G: causal method-specific reservations and historical overrun penalties; no host times are remeasured.",
               replay="Same original seeds, stored choices and report values; verify all 2048 overlapping frontier decisions and all 192 reference actions. No independent replication.",
               root_units="Python callbacks per subset/mode/innovation; count reuse and OMA callbacks, number of individual directional/cap eigenproblems, and separate preparations. Root time remains the original uninstrumented time.",
               inputs_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source.items()}))
    ref = defaultdict(dict)
    with source["independent_numerical_results/values.csv"].open() as f:
        for row in csv.DictReader(f):
            seed,k,val=int(row["seed"]),int(row["index"]),float(row["reference"])
            checked(ref[seed].setdefault(k,val)==val,"Inconsistent reference duplicates")
    acts={r["seed"]:(r["current"],r["reference_action"]) for r in json.loads(source["independent_numerical_results/decisions.json"].read_text())}
    jobs=[(name,cfg,seed,ref[seed],acts[seed]) for name,start,cfg in cases() for seed in range(start,start+64)]
    library()  # finish native compilation before workers open it
    rows=[]
    with ProcessPoolExecutor(args.workers) as pool:
        for batch in pool.map(count_root,jobs,chunksize=1): rows.extend(batch)
    checked(len(rows)==3072 and max(r["absolute_difference"] for r in rows)<1e-8,"Root reproduction failed")
    write_csv(out/"root_work.csv",rows)
    workcols=["evaluations","reuse_evaluations","oma_evaluations","individual_eigenproblems","linear_solve_calls","reuse_preparations","oma_preparations"]
    root_summary={key:float(np.mean([r[key] for r in rows])) for key in workcols}
    root_summary.update(candidates=len(rows),states=len(jobs),maximum_absolute_q_change=max(r["absolute_difference"] for r in rows),
                        decision_agreements=len(jobs))
    write_json(out/"root_summary.json",root_summary); print("Root",root_summary,flush=True)

    rc=defaultdict(list)
    for row in load_jsonl(source["dense_alignment_results/records.jsonl.gz"]): rc[row["seed"]].append(row)
    seeds=sorted(rc); checked(len(seeds)==512,"Incomplete cost cohort")
    cfg=Config(**original["config"]); rho=original["rho"]
    jobs=[(s,cfg,rho,rc[s]) for s in seeds]; matched={}; verified=0
    with ProcessPoolExecutor(args.workers) as pool, gzip.open(out/"frontier_records.jsonl.gz","wt") as f:
        for ix,(rs,n) in enumerate(pool.map(replay_frame,jobs,chunksize=2)):
            verified+=n
            for r in rs:
                key=(r["seed"],r["alpha"],r["method"])
                checked(key not in matched,"Duplicate frontier record");matched[key]=r
                f.write(json.dumps(r,allow_nan=False)+"\n")
            if (ix+1)%128==0: print("Frontier histories",ix+1,flush=True)
    checked(verified==2048 and len(matched)==512*(1+2*len(alphas)),"Incomplete terminal replay")
    contrasts=[]; desc=[]
    for a in alphas:
        names=["C-IVP","Sensitivity","No-report"]
        rs={m:[matched[s,0.0 if m=="No-report" else a,m] for s in seeds] for m in names}
        for m,rr in rs.items():
            desc.append(dict(alpha=a,method=m,mean_kbit=float(np.mean([r["completed_kbit"] for r in rr])),
                             outages=sum(r["outage"] for r in rr),reports=float(np.mean([r["probes"] for r in rr]))))
        for l,r in [(names[0],names[2]),(names[1],names[2]),(names[0],names[1])]:
            contrasts.append(dict(alpha=a,c_ms=1000*a*cfg.frame_s,left=l,right=r,**paired(
                [v["completed_kbit"] for v in rs[l]],[v["completed_kbit"] for v in rs[r]],family)))
    write_csv(out/"frontier_contrasts.csv",contrasts);write_csv(out/"frontier_descriptive.csv",desc)
    host=load_jsonl(source["round5_results/latency_records.jsonl"]);direct=[]
    for duration in DURATIONS:
        paired_rows={}
        for m in ["Margin","CDF","NR"]:
            subset=[r for r in host if r["duration_ms"]==duration and r["method"]==m]
            paired_rows[m]={r["seed"]:r for r in subset}
            checked(len(subset)==len(paired_rows[m])==256,"Incomplete host pairing")
        idx=sorted(paired_rows["NR"])
        checked(all(sorted(rr)==idx for rr in paired_rows.values()),"Host seed sets differ")
        rate={m:[rr[s]["completed_kbit"]/duration for s in idx] for m,rr in paired_rows.items()}
        direct.append(dict(duration_ms=duration,frames=len(idx),**paired(rate["Margin"],rate["CDF"],family),
                           margin_rate=float(np.mean(rate["Margin"])),ivp_rate=float(np.mean(rate["CDF"])),
                           nr_rate=float(np.mean(rate["NR"]))))
    write_csv(out/"host_direct.csv",direct)
    write_json(out/"complete.json",dict(root_values=len(rows),root_states=len(ref),terminal_records=len(matched),
                                       matched_frame_seeds=len(seeds),original_frontier_checks=verified,
                                       cost_contrasts=len(contrasts),host_contrasts=len(direct),family=family))
    print("Complete",out,flush=True)
if __name__=="__main__":
    main()
