"""Regenerate R4 pilot statistics and tables from frozen or reproduced records."""
import argparse,json
from pathlib import Path
import numpy as np
from physical_pilots import summarize

def analyze(out,target):
    summary=summarize(out);audit=[json.loads(x) for x in (out/'pilot_value_audit.jsonl').read_text().splitlines()]
    stats={}
    for length in sorted(summary,key=int):
        rows=[x for x in audit if x['length']==int(length)];delta=np.array([abs(x['cdf17']-x['gh33']) for x in rows])
        s=dict(states=len(rows),cdf17_gh33_mean_abs=float(delta.mean()),cdf17_gh33_max_abs=float(delta.max()),budgets={})
        for budget in [256,1024]:
            bb=[next(c for c in x['bounds']['checkpoints'] if c['budget']==budget) for x in rows]
            s['budgets'][str(budget)]=dict(mean_gap=float(np.mean([x['gap'] for x in bb])),max_gap=max(x['gap'] for x in bb),
                mean_wall_ms=1000*float(np.mean([x['wall_s'] for x in bb])),
                outside={key:sum(not b['lower']-1e-8<=r[key]<=b['upper']+1e-8 for r,b in zip(rows,bb)) for key in ['cdf17','gh17','cdf33','gh33']})
        stats[length]=s
    (out/'pilot_value_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    target.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{table}[t]\centering\caption{Physical two-pilot study: 256 frames per length. Service deducts sounding/feedback time; computation is timed separately.}\label{tab:pilots}\small',
        r'\begin{tabular}{@{}rlrrr@{}}\toprule',r'$L_{\rm p}$ & Rule & kbit & Outages & Wall ms\\\midrule']
    for length,s in summary.items():
        for method,label in [('Conditional-CDF','Cond.-CDF'),('Product-GH','Product-GH'),('No-report','No-report')]:
            d=s[method];lines.append(f"{length} & {label} & {d['completed_kbit']:.2f} & {d['outages']}/256 & {d['wall_ms']:.2f}"+r'\\')
        if length!='64':lines.append(r'\midrule')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (target/'r4_pilot_table.tex').write_text('\n'.join(lines)+'\n')
    a,b=summary['16'],summary['64'];da,db=a['observation_diagnostic'],b['observation_diagnostic']
    rows=[('Standardized log-error mean',f"{da['mean']:.4f}",f"{db['mean']:.4f}"),
        ('Standardized log-error variance',f"{da['variance']:.4f}",f"{db['variance']:.4f}"),
        (r'Outside nominal 95\% Gaussian interval',f"{100*da['outside_95']:.2f}\\%",f"{100*db['outside_95']:.2f}\\%"),
        ('Normalized LS mean-square error',f"{da['ls_normalized_mse']:.4f}",f"{db['ls_normalized_mse']:.4f}"),
        ('Minimum realized pilot SNR (dB)',f"{da['snr_db_quantiles'][0]:.2f}",f"{db['snr_db_quantiles'][0]:.2f}"),
        ('Same acquisition decision',f"{a['same_acquisition']}/256",f"{b['same_acquisition']}/256"),
        ('Mean / max $|Q_{\\rm CDF17}-Q_{\\rm GH33}|$ (kbit)',*[f"{stats[k]['cdf17_gh33_mean_abs']:.3f} / {stats[k]['cdf17_gh33_max_abs']:.3f}" for k in ['16','64']]),
        ('1,024-box mean / max value gap (kbit)',*[f"{stats[k]['budgets']['1024']['mean_gap']:.3f} / {stats[k]['budgets']['1024']['max_gap']:.3f}" for k in ['16','64']]),
        ('1,024-box mean wall time (ms)',*[f"{stats[k]['budgets']['1024']['mean_wall_ms']:.2f}" for k in ['16','64']])]
    lines=[r'\begin{table}[H]\centering\caption{Physical-pilot diagnostics. The value audit uses 16 prespecified states per length.}\label{tab:pilotdiag}\small',
           r'\begin{tabular}{@{}lrr@{}}\toprule',r'Diagnostic & $L_{\rm p}=16$ & $L_{\rm p}=64$\\\midrule']
    lines += [' & '.join(row)+r'\\' for row in rows];lines += [r'\bottomrule\end{tabular}\end{table}']
    (target/'r4_pilot_diagnostics_table.tex').write_text('\n'.join(lines)+'\n')
    text=(f"On this separately warmed run, mean full-decision wall costs are {a['Conditional-CDF']['wall_ms']:.2f}/{b['Conditional-CDF']['wall_ms']:.2f} ms "
          f"for conditional CDF at $L_{{\\rm p}}=16/64$, versus {a['Product-GH']['wall_ms']:.2f}/{b['Product-GH']['wall_ms']:.2f} ms for product quadrature "
          f"({a['Product-GH']['wall_ms']/a['Conditional-CDF']['wall_ms']:.2f}/{b['Product-GH']['wall_ms']/b['Conditional-CDF']['wall_ms']:.2f} times faster). "
          "Waveform generation and compilation are excluded. These batch times use a different experiment from Table~\\ref{tab:timecontext}; "
          "they are not substituted for historical scalar timings.\n")
    (target/'r4_pilot_timing.tex').write_text(text)
    print(json.dumps(dict(summary=summary,value_audit=stats),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results');p.add_argument('--artifact-root');a=p.parse_args()
    root=Path(a.root);analyze(Path(a.results) if a.results else root/'round4_results',(Path(a.artifact_root) if a.artifact_root else root)/'manuscript')
