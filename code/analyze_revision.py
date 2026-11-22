"""Regenerate all revised tables, figures and reported numbers from frozen logs."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'Sensitivity-3pt':'3-point VoI','GH-KG':'17-point KG','No-report':'No report',
       'Margin-sensitivity':'Margin sensitivity','Guarded-no-report':'Guarded no report',
       'Validated-frontier':'Validated frontier','IVP+frontier':'IVP + frontier'}
COLORS=['#145da0','#ca6f1e','#6b5b95','#2c8a6c','#af4852','#788b9e']

def label(x):return NAMES.get(x,x).replace('_',r'\_')
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h

def load(root):
    rows=[]
    for path in sorted(root.glob('*_records.jsonl')):
        source=path.stem.replace('_records','')
        with path.open() as f:
            for line in f:
                r=json.loads(line);keys=['method','seed','probes','wall_s','cpu_s','completed_kbit','admitted_kbit','outage','active','mode','conditional_risk','deadline_overrun','radio_completed_kbit','reserved_s','reserve_s','demand_kbit','catalogue_size']
                row={k:r.get(k,np.nan) for k in keys};row.update(source=source,scenario=r.get('scenario',source))
                obs=r.get('observations',[]);n=r.get('n',4)
                row['dl_reports']=sum(o['index']>=n*n for o in obs)
                row['cross_reports']=sum((o['index']%(n*n))//n!=(o['index']%(n*n))%n for o in obs)
                margins=r.get('margins',[np.nan,np.nan]);row['dl_tighter']=float(margins[1]>margins[0]) if r.get('active') else np.nan
                row['oma']=float(r.get('mode')=='OMA')
                row['history_key']=json.dumps(obs,sort_keys=True)
                rows.append(row)
    return pd.DataFrame(rows)

def summary(df):
    rows=[]
    for (src,sc,method),g in df.groupby(['source','scenario','method'],sort=False):
        n=len(g);k=int(g.outage.sum());lo,hi=wilson(k,n);mu=g.completed_kbit.mean();hw=t.ppf(.975,n-1)*g.completed_kbit.sem()
        v=dict(source=src,scenario=sc,method=method,n=n,completed=mu,ci_low=mu-hw,ci_high=mu+hw,admitted=g.admitted_kbit.mean(),
               reports=g.probes.mean(),outages=k,outage=k/n,outage_low=lo,outage_high=hi,mean_ms=g.wall_s.mean()*1000,p95_ms=g.wall_s.quantile(.95)*1000,
               max_ms=g.wall_s.max()*1000,cpu_ms=g.cpu_s.mean()*1000,oma=g.oma.mean(),dl_tighter=g.dl_tighter.mean(),
               dl_report_share=g.dl_reports.sum()/max(g.probes.sum(),1),cross_report_share=g.cross_reports.sum()/max(g.probes.sum(),1),
               risk_mean=g.conditional_risk.mean(),risk_max=g.conditional_risk.max(),overruns=g.deadline_overrun.sum(),reserved_ms=g.reserved_s.mean()*1000)
        rows.append(v)
    return pd.DataFrame(rows)

def pair(df,sc,method,base,source='revision',bootstrap=False):
    d=df[(df.source==source)&(df.scenario==sc)].pivot(index='seed',columns='method',values='completed_kbit')[[method,base]].dropna()
    x=d[method].to_numpy();y=d[base].to_numpy();a=x-y;h=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    r=dict(scenario=sc,source=source,method=method,base=base,n=len(a),difference=a.mean(),low=a.mean()-h,high=a.mean()+h,relative_percent=100*a.mean()/y.mean())
    if bootstrap:
        rng=np.random.default_rng(20261010);ix=rng.integers(len(a),size=(5000,len(a)));v=100*a[ix].mean(axis=1)/y[ix].mean(axis=1)
        r['relative_interval']=np.quantile(v,[.025,.975]).tolist()
    return r

def table(path,caption,tag,columns,rows,wide=False,fmt=None,note=''):
    env='table*' if wide else 'table';fmt=fmt or 'l'+'r'*(len(columns)-1)
    text='\\begin{'+env+'}[t]\n\\centering\\caption{'+caption+'}\\label{'+tag+'}\n\\small\n\\begin{tabular}{@{}'+fmt+'@{}}\\toprule\n'
    text+=' & '.join(columns)+r'\\\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)
    text+='\n\\bottomrule\\end{tabular}\n'
    if note:text+='\\par\\smallskip\\begin{minipage}{'+('\\textwidth' if wide else '\\columnwidth')+'}\\footnotesize '+note+'\\end{minipage}\n'
    text+='\\end{'+env+'}\n';path.write_text(text)

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='revision_results');p.add_argument('--root',default='.');a=p.parse_args()
    root=Path(a.root);out=Path(a.results);tex=root/'manuscript';fig=root/'figures';tex.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    df=load(out);ss=summary(df);ss.to_csv(out/'summary.csv',index=False)
    def get(sc,m,src='revision'):return ss[(ss.source==src)&(ss.scenario==sc)&(ss.method==m)].iloc[0]
    paired=[]
    for m in ['No-report','A-stop','Sensitivity-3pt','GH-KG']:
        paired.append(pair(df,'main','C-IVP',m,bootstrap=m in ['No-report','A-stop']))
    for c in [1,2]:
        for b in ['A-fixed','D-fixed','Random-fixed']:paired.append(pair(df,'fixed-count',f'C-IVP-{c}',f'{b}-{c}'))
    for b in ['C-IVP','A-stop','GH-KG']:paired.append(pair(df,'two-step','Two-step',b))
    # Chance, frontier, and hybrid share the same 128 base frames.
    joint=df[df.source.isin(['chance','frontier','hybrid'])].copy();joint['source']='chance_joint';joint['scenario']='chance_joint'
    for m,b in [('Validated-frontier','C-IVP'),('IVP+frontier','Validated-frontier')]:
        if m in set(joint.method):paired.append(pair(joint,'chance_joint',m,b,'chance_joint'))
    (out/'paired_gains.json').write_text(json.dumps(paired,indent=2))
    macros={'MainGainNR':paired[0]['relative_percent'],'MainGainA':paired[1]['relative_percent']}
    (tex/'revision_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.2f}'+'}' for k,v in macros.items())+'\n')
    rows=[]
    for m in ['C-IVP','A-stop','D-stop','Sensitivity-3pt','GH-KG','No-report','OMA','Nominal','Perfect-CSI']:
        r=get('main',m);rows.append([label(m),f'{r.admitted:.2f}',f'{r.completed:.2f}',f'[{r.ci_low:.2f},{r.ci_high:.2f}]',f'{r.reports:.2f}',f'{r.outages}/{r.n}',f'[{100*r.outage_low:.2f},{100*r.outage_high:.2f}]'])
    table(tex/'rev_main.tex','Main experiment: 1,024 independent frames, radio airtime charged; decision latency is evaluated separately. Payloads and mean intervals are in kbit/frame; outage intervals are percentages.','tab:main',['Policy','Admitted','Completed','95\\% mean CI','Reports','Outage count','95\\% Wilson CI'],rows,True)
    rows=[]
    for k in [1,2]:
        for m in ['C-IVP','A-fixed','D-fixed','Random-fixed']:
            r=get('fixed-count',f'{m}-{k}');rows.append([str(k),label(m),f'{r.completed:.2f}',f'{r.outages}/{r.n}'])
    table(tex/'rev_fixed.tex','Exactly matched report counts, including otherwise unnecessary reports (1,024 frames per count).','tab:fixed',['Reports','Policy','Completed (kbit)','Outages'],rows)
    # Conditional risk and stronger terminal allocators.
    rows=[]
    for src,m in [('chance','C-IVP'),('chance','A-stop'),('chance','No-report'),('chance','OMA'),('chance','Validated-SAA'),('frontier','Validated-frontier'),('hybrid','IVP+frontier')]:
        r=get(src,m,src);rows.append([label(m),f'{r.admitted:.2f}',f'{r.completed:.2f}',f'{100*r.risk_mean:.2f}',f'{100*r.risk_max:.2f}',f'{r.outages}/128'])
    table(tex/'rev_chance.tex','Conservatism and stronger allocators on the same 128 frames. Conditional risks are Monte Carlo estimates from 16,384 independent posterior draws per decision (percent).','tab:chance',['Policy','Admitted','Completed','Mean risk','Max. risk','Outages'],rows,True,note='Payloads are kbit/frame. Validated methods use separate 4,096-draw binomial validation with simultaneous 95\\% confidence over a finite candidate family; they are not exact chance-constrained optima or the conditional certificate of Theorem~\\ref{thm:reliability}.')
    rows=[]
    for sc,title in [('sigma_mismatch','SD: true 5 / assumed 3 dB'),('reciprocity_mismatch',r'$r_{UD}$: true 0.2 / assumed 0.9'),('structure_mismatch','Independent-edge truth'),('mean_mismatch','Desired mean bias: $-3$ dB'),('blockage','Post-report 8-dB loss')]:
        row=[title]
        for m in ['C-IVP','A-stop','No-report','Oracle-cov' if sc!='blockage' else 'Guarded-IVP']:
            r=get(sc,m);row.append(f'{r.completed:.2f} / {100*r.outage:.2f}')
        rows.append(row)
    table(tex/'rev_mismatch.tex','Mismatch: completed kbit/frame / any-task outage (\\%). The first four rows use 512 frames each; blockage uses 1,024.','tab:mismatch',['Truth versus assumed model','C-IVP','A-stop','No report','Oracle cov. / guard'],rows,True,note='Oracle covariance corrects covariance only and cannot correct the biased mean. In the final row the last column uses an 8-dB desired-loss guard with unchanged cross-links; unguarded post-report losses violate the frozen-channel assumption.')
    rows=[]
    for sc,title in [('UL-both','UL-heavy'),('balanced','Balanced'),('DL-heavy','DL-heavy')]:
        c=get(sc,'C-IVP');aa=get(sc,'A-stop');d=get(sc,'D-stop');nr=get(sc,'No-report')
        rows.append([title,f'{c.completed:.2f}',f'{aa.completed:.2f}',f'{d.completed:.2f}',f'{nr.completed:.2f}',f'{100*c.dl_tighter:.1f}',f'{100*c.dl_report_share:.1f}',f'{100*c.oma:.1f}'])
    table(tex/'rev_bidirectional.tex','Direction-balanced report catalogue ($4N$ coordinates), 512 frames per load. Completed service is kbit/frame.','tab:bidir',['Load','C-IVP','A-stop','D-stop','No report','DL tighter (\\%)','DL queries (\\%)','OMA (\\%)'],rows,True,note='DL tighter means the larger terminal directional peak-power utilization (reuse) or airtime fraction (OMA), over active decisions. It diagnoses the tighter bound rather than assigning a causal bottleneck to every rejected task.')
    # Six ray conditions, ten methods, static and motion side by side in separate tables.
    for suffix,title in [('static','Static incoherent path-power channels'),('motion','Coherent channels with 0.4-m end-of-frame displacement')]:
        rows=[]
        for m in ['C-IVP','A-stop','Sensitivity-3pt','No-report','OMA','Unscaled-IVP','Conformal-NR','Safe-IVP','Safe-NR','Perfect-CSI']:
            row=[label(m)]
            for name in ['canyon_low','canyon_high','munich']:
                r=get(name+'-'+suffix,m,'conformal' if m.startswith('Safe-') else 'ray')
                row.append(f'{r.completed:.2f} / {int(r.outages)}')
            rows.append(row)
        table(tex/f'rev_ray_{suffix}.tex',title+': completed kbit/frame / outage count out of 256 held-out frames per condition.','tab:ray'+suffix,['Policy','Canyon 35 m','Canyon 80 m','Munich 120 m'],rows,True,note='The two canyon heights share one layout. Perfect-CSI knows the execution gain and incurs no report overhead. Gaussian inflation is empirical, while conformal guarantees apply marginally to static exchangeable frames. All moving-channel rows are stress tests outside that static guarantee.')
    rows=[]
    for m in ['C-IVP','C-IVP-S2','Margin-sensitivity','A-stop','3pt-VoI']:
        for n in [4,6,8]:
            r=get(f'N-{n}',m,'timing');rows.append([label(m),n,f'{r.mean_ms:.2f}',f'{r.p95_ms:.2f}',f'{r.max_ms:.2f}',f'{r.completed:.2f}'])
    table(tex/'rev_timing.tex','Sequential warmed online decisions: 128 frames per $N$, one process and one BLAS thread. Compilation and scenario generation are excluded; Python orchestration is included.','tab:timing',['Policy','$N$','Mean ms','95th ms','Max. ms','kbit/frame'],rows,True,note='Wall time includes shared-host scheduling stalls. These are measured costs, not hard real-time bounds. Radio-only payload in this table is not the latency-charged service in Table~\\ref{tab:latency}.')
    budgets=json.loads((out/'latency_protocol.json').read_text())['reserve_per_round_s'];rows=[]
    for m in ['C-IVP','C-IVP-S2','Margin-sensitivity','A-stop','3pt-VoI','No-report','OMA']:
        row=[label(m),f'{budgets[m]*1000:.2f}']
        for dur in [20,50,100]:
            r=get(f'T-{dur}ms',m,'latency');row.append(f'{r.completed:.2f} / {int(r.overruns)}')
        rows.append(row)
    table(tex/'rev_latency.tex','Computation charged to the frame, 512 held-out frames per duration: completed kbit/frame / measured reservation overruns.','tab:latency',['Policy','$c$ (ms/round)','20 ms','50 ms','100 ms'],rows,True,note='Reserve $c$ is twice the development 95th-percentile per-round wall cost on 32 disjoint seeds, rounded up to 0.05 ms after warming. Report airtime and $(m+1)c$ are subtracted. Every active overrun earns zero completed service. Payloads scale with frame duration to keep offered bit rate fixed; guarantees are conditional on meeting the reserved time.')
    # Accuracy + performance figures.
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    audit=pd.read_csv(out/'value_audit.csv');gap=audit['reference_minus_inner']
    f,ax=plt.subplots(figsize=(3.45,2.15));v=np.sort(gap);ax.plot(np.maximum(v,0),np.arange(1,len(v)+1)/len(v),color=COLORS[0]);ax.set_xscale('symlog',linthresh=1e-5);ax.set_xlabel('Reference minus inner value (kbit)');ax.set_ylabel('Empirical CDF');ax.grid(alpha=.2);f.tight_layout();f.savefig(fig/'value_accuracy.pdf');plt.close(f)
    f,axs=plt.subplots(1,2,figsize=(7.1,2.5));methods=['C-IVP','A-stop','Sensitivity-3pt','No-report','OMA'];x=np.arange(len(methods))
    for j,m in enumerate(methods):
        r=get('main',m);axs[0].bar(j,r.completed,color=COLORS[j],width=.66);axs[0].errorbar(j,r.completed,[[r.completed-r.ci_low],[r.ci_high-r.completed]],color='black',capsize=2);axs[1].bar(j,r.reports,color=COLORS[j],width=.66)
    for ax in axs:ax.set_xticks(x,[label(m) for m in methods],rotation=24,ha='right');ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Completed payload (kbit/frame)');axs[0].set_ylim(0,125);axs[1].set_ylabel('Reports/frame');f.tight_layout();f.savefig(fig/'revision_main.pdf');plt.close(f)
    f,axs=plt.subplots(1,2,figsize=(7.1,2.6))
    for j,m in enumerate(['C-IVP','A-stop','No-report','OMA']):
        r=[get(f'delta-{d:g}',m) for d in [.01,.025,.05,.1,.2,.4]]
        xx=[100*t.outage for t in r];yy=[t.completed for t in r]
        axs[0].errorbar(xx,yy,xerr=[[100*(t.outage-t.outage_low) for t in r],[100*(t.outage_high-t.outage) for t in r]],fmt='o-',ms=3,capsize=1,lw=1,label=label(m),color=COLORS[j])
    axs[0].axvline(5,color='gray',ls='--',lw=.8);axs[0].set_xlabel('Observed any-task outage (%)');axs[0].set_ylabel('Completed payload (kbit/frame)');axs[0].legend(fontsize=7)
    for j,m in enumerate(['C-IVP','A-stop']):
        rr=[get(f'cap-{k}',m) for k in [0,1,2,4,8]];axs[1].plot([0,1,2,4,8],[r.completed for r in rr],'o-',label=label(m),color=COLORS[j])
    axs[1].set_xlabel('Maximum reports M');axs[1].set_ylabel('Completed payload (kbit/frame)');axs[1].legend(fontsize=7)
    for ax in axs:ax.grid(alpha=.2)
    f.tight_layout();f.savefig(fig/'risk_cap.pdf');plt.close(f)
    f,axs=plt.subplots(1,2,figsize=(7.1,2.5))
    for i,n in enumerate([6,8]):
        for j,m in enumerate(['C-IVP','C-IVP-C16','C-IVP-C32','C-IVP-S2','Margin-sensitivity']):
            r=get(f'N-{n}',m,'timing');axs[i].scatter(r.mean_ms,r.completed,color=COLORS[j],s=30,label=['Full C-IVP','C16','C32','S2','Margin sensitivity'][j])
        axs[i].set_title(f'N = {n}');axs[i].set_xlabel('Mean online wall time (ms)');axs[i].grid(alpha=.2)
    axs[0].set_ylabel('Completed payload (kbit/frame)');handles,labels=axs[0].get_legend_handles_labels();f.legend(handles,labels,loc='lower center',ncol=5,fontsize=8,frameon=False);f.tight_layout(rect=[0,.12,1,1]);f.savefig(fig/'catalogue_pareto.pdf');plt.close(f)
    # Independent calibration choice for no-report delta; simultaneous CP across six choices.
    cals=[]
    for d in [.01,.025,.05,.1,.2,.4]:
        r=get(f'cal-delta-{d:g}','No-report');upper=beta.ppf(1-.05/6,r.outages+1,r.n-r.outages) if r.outages<r.n else 1
        cals.append(dict(delta=d,upper=float(upper),completed=r.completed))
    eligible=[x for x in cals if x['upper']<=.05];chosen=max(eligible,key=lambda x:x['completed']) if eligible else None
    if chosen:
        r=get(f"delta-{chosen['delta']:g}",'No-report');chosen.update(test_completed=r.completed,test_outage=r.outage,test_ci=[r.outage_low,r.outage_high])
    (out/'calibrated_delta.json').write_text(json.dumps(dict(calibration=cals,choice=chosen),indent=2))
    # Exact A/D history coincidences and compact protocol totals.
    equality={}
    for sc in ['main','UL-both','balanced','DL-heavy']:
        d=df[(df.source=='revision')&(df.scenario==sc)].pivot(index='seed',columns='method',values='history_key');equality[sc]=int((d['A-stop']==d['D-stop']).sum())
    meta=dict(records=len(df),counts=df.groupby('source').size().to_dict(),history_equal=equality,
              value_audit=json.loads((out/'value_audit.json').read_text()))
    (out/'analysis_summary.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2));print('PAIRED',json.dumps(paired,indent=2));print('CALIBRATED_DELTA',chosen)
    show=ss[(ss.scenario.isin(['main','T-20ms','T-50ms','T-100ms','balanced','DL-heavy','UL-both']))|ss.source.isin(['chance','frontier','hybrid','conformal'])]
    print(show[['source','scenario','method','completed','reports','outages','n','mean_ms','overruns','risk_mean','dl_tighter','dl_report_share']].to_string(index=False))

if __name__=='__main__':main()
