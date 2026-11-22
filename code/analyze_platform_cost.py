"""Empirical first crossings, paired bootstrap bands, and discrete support.

Boundary estimates use piecewise-linear interpolation of observed mean gains.
All later crossings and positive islands are retained rather than assuming
global monotonicity of a completed-service policy.
"""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np

def crossings(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);roots=[]
    for i in range(len(x)-1):
        if y[i]==0:roots.append(float(x[i]))
        if y[i]*y[i+1]<0:roots.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
    if y[-1]==0:roots.append(float(x[-1]))
    return sorted(set(roots))

def first_boundary(x,y):
    if y[0]<=0:return 0.,'nonpositive_at_zero'
    for i in range(1,len(x)):
        if y[i]<=0:return float(x[i-1]+y[i-1]*(x[i]-x[i-1])/(y[i-1]-y[i])),'crossed'
    return float(x[-1]),'right_censored'

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();out=Path(a.results);protocol=json.loads((out/'protocol.json').read_text());summary=[];rows=[]
    analysis=dict(bootstrap=a.bootstrap,seed=20261120,definition='First zero of the piecewise-linear empirical mean gain, starting positive at alpha=0. All sign changes retained. Nonpositive origin is labelled no empirical positive entry, not a proven zero true threshold.',intervals='Pointwise paired frame-bootstrap percentiles; do not cover interpolation error. Primary grid contrasts retain their original global Bonferroni family.',stage='Post-run display transformation, recorded before calculating boundary summaries')
    (out/'analysis_protocol.json').write_text(json.dumps(analysis,indent=2)+'\n')
    for cell in protocol['cells']:
        s=json.loads((out/(cell['id']+'_summary.json')).read_text())
        with gzip.open(out/(cell['id']+'.jsonl.gz'),'rt') as f:records=[json.loads(l) for l in f]
        baseline={r['seed']:r['completed_kbit'] for r in records if r['method'].startswith('Predictive-NR')};seeds=sorted(baseline);x=np.array(cell['alphas']);result=dict(cell=cell,frames=s['frames'],costs=s['costs'],boundaries={})
        draws=np.random.default_rng(20261120).integers(0,len(seeds),(a.bootstrap,len(seeds)))
        for method,label in [('Predictive-IVP','IVP-NR'),('Predictive-Margin','Margin-NR')]:
            selected={(r['seed'],r['alpha']):r['completed_kbit'] for r in records if r['method'].startswith(method)}
            differences=np.array([[selected[seed,float(alpha)]-baseline[seed] for alpha in x] for seed in seeds]);mean=differences.mean(axis=0);estimate,status=first_boundary(x,mean)
            bootmeans=differences[draws].mean(axis=1);boot=np.array([first_boundary(x,y)[0] for y in bootmeans]);ci=np.quantile(boot,[.025,.975]);supported=0.;prefix=True
            for alpha,c in zip(x,s['costs']):
                prefix=prefix and c[label]['low']>0
                if prefix:supported=float(alpha)
            positive_ranges=[];edges=[float(x[0])]+crossings(x,mean)+[float(x[-1])]
            for left,right in zip(edges[:-1],edges[1:]):
                if right>left and np.interp((left+right)/2,x,mean)>0:positive_ranges.append([left,right])
            q=dict(estimate=estimate,status=status,bootstrap95=ci.tolist(),all_crossings=crossings(x,mean),positive_ranges=positive_ranges,supported_prefix_grid_budget=supported,mean_gain_at_zero=float(mean[0]),pointwise_bootstrap_nonpositive_origin=int(sum(y[0]<=0 for y in bootmeans)))
            result['boundaries'][method]=q
            rows.append(dict(cell=cell['id'],kind=cell['kind'],method=method,n=cell['config']['n'],duration_ms=cell['config']['frame_s']*1000,speed=cell['speed'],rho=cell['rho'],report_fraction=cell['report_fraction'],estimate=estimate,low=ci[0],high=ci[1],status=status,supported_prefix=supported,mean_gain_zero=mean[0],crossings=len(q['all_crossings'])))
        summary.append(result)
    (out/'boundary_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'boundaries.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Analyzed',len(summary),'cells; retained all crossing information')
    for d in summary:
        if d['cell']['kind']!='map':print(d['cell']['id'],d['boundaries'])
if __name__=='__main__':main()
