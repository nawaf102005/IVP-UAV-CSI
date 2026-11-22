"""Generate R3 summaries and camera-readable tables from frozen records."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import t,norm
from round2_experiments import read_records,dump

def ci(x):
    x=np.asarray(x,float);m=float(x.mean());h=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0
    return [m,m-h,m+h]
def summary(rows):
    n=len(rows);k=sum(r['outage'] for r in rows);z=norm.ppf(.975);p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(n=n,service_ci=ci([r['completed_kbit'] for r in rows]),outages=k,outage_wilson=[mid-half,mid+half],reports=float(np.mean([r['probes'] for r in rows])),
        mean_ms=float(np.mean([r.get('wall_s',0)*1000 for r in rows])))
def paired(rows,a,b):
    aa={r['seed']:r['completed_kbit'] for r in rows if r['method']==a};bb={r['seed']:r['completed_kbit'] for r in rows if r['method']==b}
    assert aa.keys()==bb.keys();return ci([aa[k]-bb[k] for k in sorted(aa)])
def analyze(root,out,artifact):
    data={k:read_records(out/(k+'_records.jsonl')) for k in ['branch','bounded_policy','vector_bound','temporal','endpoint','catalogue','catalogue_value','finite_blocklength']}
    result={};result['branch']={};result['bounded_policy']={};result['temporal']={};result['endpoint']={};result['catalogue']={};result['fbl']={};result['vector']={}
    for n in [4,8,12,16,24]:
        for budget in [64,256,1024]:
            rows=[r for r in data['branch'] if r['n']==n and r['node_budget']==budget];x=summary(rows)
            x.update(mean_ratio=float(np.mean([r['ratio'] for r in rows])),min_ratio=float(min(r['ratio'] for r in rows)),mean_gap=float(np.mean([r['gap'] for r in rows])),max_gap=float(max(r['gap'] for r in rows)),closed=sum(r['gap']<1e-8 for r in rows))
            result['branch'][f'{n}:{budget}']=x
    for n in [8,16,24]:
        rows=[r for r in data['bounded_policy'] if r['n']==n]
        result['bounded_policy'][str(n)]={m:summary([r for r in rows if r['method']==m]) for m in ['B-NR','B-IVP2']}
        result['bounded_policy'][str(n)]['gain_ci']=paired(rows,'B-IVP2','B-NR')
    for budget in [64,256,1024,4096]:
        xs=[next(c for c in r['checkpoints'] if c['budget']==budget) for r in data['vector_bound']]
        result['vector'][str(budget)]=dict(mean_gap=float(np.mean([r['gap'] for r in xs])),max_gap=float(max(r['gap'] for r in xs)),mean_ms=float(np.mean([r['wall_s']*1000 for r in xs])))
    for rho in [.95,.75,.4]:
        rows=[r for r in data['temporal'] if r['rho']==rho]
        result['temporal'][str(rho)]={m:summary([r for r in rows if r['method']==m]) for m in ['Frozen-IVP','Predictive-IVP','Predictive-NR']}
        result['temporal'][str(rho)]['gain_ci']=paired(rows,'Predictive-IVP','Predictive-NR')
    for m in ['Frozen-IVP','Frozen-Safe','Predictive-IVP','Endpoint-Safe','Endpoint-NR']:result['endpoint'][m]=summary([r for r in data['endpoint'] if r['method']==m])
    result['endpoint']['gain_ci']=paired(data['endpoint'],'Endpoint-Safe','Endpoint-NR')
    for m in ['UL-fixed','Bidirectional-fixed','Bidirectional-updated','All-coordinates']:result['catalogue'][m]=summary([r for r in data['catalogue'] if r['method']==m])
    result['catalogue']['all_minus_ul_ci']=paired(data['catalogue'],'All-coordinates','UL-fixed')
    loss=np.array([r['full_best']-r['default_best'] for r in data['catalogue_value']]);result['catalogue']['value_coverage']=dict(mean=float(loss.mean()),maximum=float(loss.max()),over_002=int(sum(loss>.002)))
    for length in [128,512]:
        rows=[r for r in data['finite_blocklength'] if r['blocklength']==length]
        result['fbl'][str(length)]={m:summary([r for r in rows if r['method']==m]) for m in ['FBL-IVP','FBL-NR']}
        result['fbl'][str(length)]['gain_ci']=paired(rows,'FBL-IVP','FBL-NR')
    dump(out/'summary_round3.json',result)
    ms=artifact/'manuscript';ms.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{table}[t]\centering\caption{Bounded search and vector integration. Means over 32 states; wall time includes bound construction.}\label{tab:bounds}\small',r'\begin{tabular}{@{}rrrrr@{}}\toprule',r'$N$ & Nodes & Mean $L/U$ & Gap (kbit) & Time (ms)\\\midrule']
    for n in [4,8,12,16,24]:
        for budget in ([256] if n in [4,8,12] else [256,1024]):
            x=result['branch'][f'{n}:{budget}'];lines.append(f"{n} & {budget} & {x['mean_ratio']:.3f} & {x['mean_gap']:.2f} & {x['mean_ms']:.2f} "+r'\\')
    lines.extend([r'\midrule',r'Boxes & \multicolumn{2}{c}{Mean/max gap (kbit)} & \multicolumn{2}{c}{Time (ms)}\\'])
    for budget in [64,256,1024,4096]:
        x=result['vector'][str(budget)];lines.append(f"{budget} & "+r'\multicolumn{2}{c}{'+f"{x['mean_gap']:.4f}/{x['max_gap']:.4f}"+r'} & \multicolumn{2}{c}{'+f"{x['mean_ms']:.2f}"+r'}\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}');(ms/'r3_bounds.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[t]\centering\caption{Execution-channel evaluation. Service in kbit; outages count any failed admitted task. Radio airtime only.}\label{tab:prediction}\small',r'\begin{tabular}{@{}llrr@{}}\toprule',r'Channel & Method & Service & Outages\\\midrule']
    for rho in [.95,.75,.4]:
        for m in ['Frozen-IVP','Predictive-IVP','Predictive-NR']:
            x=result['temporal'][str(rho)][m];lines.append(f"$\\rho={rho}$ & {m} & {x['service_ci'][0]:.2f} & {x['outages']}/512 "+r'\\')
    lines.append(r'\midrule')
    for m in ['Frozen-IVP','Frozen-Safe','Predictive-IVP','Endpoint-Safe','Endpoint-NR']:
        x=result['endpoint'][m];lines.append(f"Fresh canyon & {m} & {x['service_ci'][0]:.2f} & {x['outages']}/256 "+r'\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}');(ms/'r3_prediction.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results');p.add_argument('--artifact-root');a=p.parse_args();root=Path(a.root)
    analyze(root,Path(a.results) if a.results else root/'round3_results',Path(a.artifact_root) if a.artifact_root else root)
