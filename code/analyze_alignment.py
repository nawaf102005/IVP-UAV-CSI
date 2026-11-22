"""Apply frozen bootstrap display and translate historical decision timings."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from analyze_platform_cost import first_boundary,crossings

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--results',required=True);a=p.parse_args();out=Path(a.results);protocol=json.loads((out/'protocol.json').read_text());summ=json.loads((out/'summary.json').read_text())
    with gzip.open(out/'records.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    result={};draws=np.random.default_rng(protocol['bootstrap']['seed']).integers(0,protocol['frames'],(protocol['bootstrap']['samples'],protocol['frames']))
    rr=[r for r in rows if r['kind']=='dense'];nr={r['seed']:r['completed_kbit'] for r in rr if r['method']=='Predictive-NR'};seeds=sorted(nr);x=np.array(protocol['dense_alphas'])
    for method,label in [('Predictive-IVP','IVP-NR'),('Predictive-Margin','Margin-NR')]:
        lookup={(r['seed'],r['alpha']):r['completed_kbit'] for r in rr if r['method']==method};diff=np.array([[lookup[s,float(c)]-nr[s] for c in x] for s in seeds]);means=diff.mean(axis=0);est,status=first_boundary(x,means);boot=diff[draws].mean(axis=1);qs=np.array([first_boundary(x,y)[0] for y in boot]);prefix=0.
        for c in summ['dense']:
            if c[label]['low']<=0:break
            prefix=c['alpha']
        result[method]=dict(first_crossing=est,status=status,bootstrap95=np.quantile(qs,[.025,.975]).tolist(),supported_prefix=prefix,first_crossing_ms=est*20,supported_ms=prefix*20,all_crossings=crossings(x,means))
    (out/'boundary_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    timing=[]
    for n,label,method,nr in [('round5_results/scale_records.jsonl','Dedicated 256-frame N4 benchmark','CDF','NR'),('revision_results/timing_records.jsonl','Earlier 128-frame N4 benchmark','C-IVP',None)]:
        records=[json.loads(l) for l in (a.root/n).read_text().splitlines()];rs=[r for r in records if r.get('n')==4 and r['method']==method];wall=float(np.mean([r['wall_s'] for r in rs]));rounds=float(np.mean([r['probes']+1 for r in rs]));record=dict(source=n,label=label,frames=len(rs),mean_decision_ms=wall*1000,mean_rounds=rounds,ratio_of_means_ms=wall*1000/rounds,mean_frame_ratio_ms=float(np.mean([r['wall_s']*1000/(r['probes']+1) for r in rs])))
        if nr:
            ns=[r for r in records if r.get('n')==4 and r['method']==nr];record['nr_mean_ms']=float(np.mean([r['wall_s']*1000 for r in ns]));record['incremental_ratio_ms']=(wall*1000-record['nr_mean_ms'])/rounds
        timing.append(record)
    (out/'historical_time_translation.json').write_text(json.dumps(dict(records=timing,scope='Historical serial whole-decision times, frozen channels and old implementation. Ratios include terminal scheduling and are not measured c for the new frontier-aware planner. Subtracting NR is a descriptive incremental estimate, not a matched component timing or WCET guarantee.'),indent=2)+'\n');print(json.dumps(result,indent=2));print(timing)
if __name__=='__main__':main()
