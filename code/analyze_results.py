"""Regenerate every paper number and scientific figure from stored records."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t,norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Circle
from model import Config,scenario
from innovation import probe_value


COLORS={'IVP':'#176e91','A-opt-stop':'#9c5a28','D-opt-stop':'#b68b52','No-report':'#566677',
        'Orthogonal':'#699473','Perfect-CSI':'#393939','Nominal':'#bf4f45','Random-fixed':'#a089ac',
        'A-opt-fixed':'#8c846d','IVP-diagonal':'#60979b'}
ORDER=['IVP','A-opt-stop','D-opt-stop','A-opt-fixed','Random-fixed','No-report','Orthogonal','IVP-diagonal','Nominal','Perfect-CSI']


def mean_ci(x):
    x=np.asarray(x,float);m=x.mean()
    h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.
    return float(m),float(m-h),float(m+h)


def wilson(k,n):
    if n==0:return (0.,1.)
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-h),float(center+h)


def paired(d,reference):
    pivot=d.pivot(index='seed',columns='method',values='completed_kbit')
    difference=mean_ci(pivot['IVP']-pivot[reference])
    rng=np.random.default_rng(49318);ix=rng.integers(0,len(pivot),(5000,len(pivot)))
    a=pivot['IVP'].to_numpy()[ix].mean(axis=1);b=pivot[reference].to_numpy()[ix].mean(axis=1)
    pct=100*(pivot['IVP'].mean()/pivot[reference].mean()-1)
    ci=np.percentile(100*(a/b-1),[2.5,97.5])
    return dict(difference=difference,percent=float(pct),percent_ci=ci.tolist())


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--root',default='..');args=pa.parse_args()
    root=Path(args.root);res=root/'results';fig=root/'figures';tex=root/'manuscript';fig.mkdir(exist_ok=True);tex.mkdir(exist_ok=True)
    raw=[json.loads(x) for x in (res/'records.jsonl').read_text().splitlines()]
    df=pd.DataFrame(raw);df.drop(columns=['observations','power','selected','airtime'],errors='ignore').to_csv(res/'all_results.csv',index=False)
    protocol=json.loads((res/'protocol.json').read_text());main_df=df[df.scenario=='main']
    summaries=[]
    for (label,method),g in df.groupby(['scenario','method'],sort=False):
        mean,lo,hi=mean_ci(g.completed_kbit);active=int(g.active.sum());bad=int(g.outage.sum());wlo,whi=wilson(bad,active)
        summaries.append(dict(scenario=label,method=method,n=len(g),completed_mean=mean,ci_low=lo,ci_high=hi,
                              admitted_mean=g.admitted_kbit.mean(),probes_mean=g.probes.mean(),
                              failures=bad,active=active,outage_rate=bad/active if active else 0.,
                              outage_ci_low=wlo,outage_ci_high=whi,wall_ms=g.wall_s.mean()*1000,
                              radio_mj=g.radio_energy_mj.mean(),reuse_fraction=(g['mode']=='reuse').mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(res/'summary.csv',index=False)
    gains={r:paired(main_df,r) for r in ['No-report','A-opt-stop','Orthogonal','A-opt-fixed']}
    (res/'paired_gains.json').write_text(json.dumps(gains,indent=2))
    m=summary[summary.scenario=='main'].set_index('method');n=int(m.loc['IVP','n'])
    macros={'NumSnapshots':protocol['jobs'],'MainN':n,'GainNoReport':f"{gains['No-report']['percent']:.2f}",
            'GainAopt':f"{gains['A-opt-stop']['percent']:.2f}"}
    (tex/'result_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
    rows=[]
    for method in ORDER:
        z=m.loc[method]
        rows.append(f"{method} & {z.completed_mean:.2f} [{z.ci_low:.2f}, {z.ci_high:.2f}] & {z.admitted_mean:.2f} & {z.probes_mean:.2f} & {int(z.failures)}/{int(z.active)} & {100*z.outage_ci_low:.2f}--{100*z.outage_ci_high:.2f} & {z.wall_ms:.2f} \\\\")
    table=r'''\begin{table*}[t]
\centering\caption{Main experiment: 256 independent four-pair snapshots. Completed and admitted payloads are in kbit/frame; brackets denote a 95\% Student-$t$ interval for the mean. Outage is any admitted bidirectional task failing in an active frame; its interval is Wilson 95\%. Wall time is measured per complete policy execution, including reporting decisions, during a three-worker batch.}\label{tab:main}
\small\begin{tabular}{lrrrrrr}\toprule
Method & Completed [95\% CI] & Admitted & Reports & Outage frames & Outage CI (\%) & Wall time (ms)\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{table*}
'''
    (tex/'table_main.tex').write_text(table)
    scenario_order=list(protocol['scenarios'])
    name={'main':'Default','spacing_120':'Spacing 120 m','spacing_160':'Spacing 160 m','spacing_240':'Spacing 240 m',
          'overhead_0.25ms':'Report 0.25 ms','overhead_1.5ms':'Report 1.50 ms','reciprocity_0.2':'Reciprocity 0.2',
          'reciprocity_0.6':'Reciprocity 0.6','six_uav':'Six pairs','shadow_5dB':'Shadowing 5 dB',
          'low_load':'UL payload 13 kbit','post_report_blockage':'Post-report blockage'}
    cols=['IVP','A-opt-stop','D-opt-stop','Random-fixed','No-report','Orthogonal','Nominal','Perfect-CSI']
    rows=[]
    sensitivity=[]
    for label in scenario_order:
        q=summary[summary.scenario==label].set_index('method');g=df[df.scenario==label];gain=paired(g,'A-opt-stop');sensitivity.append(dict(scenario=label,**gain))
        rows.append(name[label]+f" & {int(q.loc['IVP','n'])} & "+' & '.join(f'{q.loc[c,"completed_mean"]:.2f}' for c in cols)+r' \\')
    (res/'sensitivity_gains.json').write_text(json.dumps(sensitivity,indent=2))
    (tex/'table_sweeps.tex').write_text(r'''\begin{table*}[t]\centering
\caption{Completed bidirectional payload (kbit/frame) for all evaluated scenarios and common policies. Each row changes only its named parameter from the default; the last row introduces unmodeled post-report desired-link blockage. Policies are paired on the same seeds within each row.}\label{tab:sweeps}
\small\begin{tabular}{lrrrrrrrrr}\toprule
Scenario & $n$ & IVP & A-stop & D-stop & Random & No-report & OMA & Nominal & Genie\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table*}'+'\n')
    q=summary[summary.scenario=='post_report_blockage'].set_index('method');rows=[]
    for method in cols:
        z=q.loc[method];rows.append(f'{method} & {z.completed_mean:.2f} & {int(z.failures)}/{int(z.active)} & {100*z.outage_ci_low:.1f}--{100*z.outage_ci_high:.1f}'+r' \\')
    (tex/'table_stress.tex').write_text(r'''\begin{table}[t]\centering
\caption{Unmodeled blockage stress: an 8-dB desired-link loss in both directions is applied independently to each pair with probability 0.1 after reporting. The guarantee's frozen-channel assumption is deliberately violated.}\label{tab:stress}
\small\begin{tabular}{lrrr}\toprule
Method & kbit/frame & Outage & 95\% CI (\%)\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}'+'\n')
    # Consistent vector publication figures; no generated raster illustrations.
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    fig1,axes=plt.subplots(2,1,figsize=(6.8,4.8),gridspec_kw={'height_ratios':[2,1]},sharex=True,layout='constrained')
    chosen=[x for x in ORDER if x!='D-opt-stop'];x=np.arange(len(chosen));means=m.loc[chosen,'completed_mean'].values
    errors=np.array([means-m.loc[chosen,'ci_low'].values,m.loc[chosen,'ci_high'].values-means])
    axes[0].bar(x,means,color=[COLORS[c] for c in chosen]);axes[0].errorbar(x,means,yerr=errors,fmt='none',ecolor='#18232c',capsize=3)
    axes[0].scatter(x,m.loc[chosen,'admitted_mean'],marker='_',s=100,color='black',label='Admitted mean')
    axes[0].set_ylabel('Completed payload (kbit/frame)');axes[0].legend(frameon=False,loc='upper left');axes[0].set_ylim(0,150)
    axes[1].bar(x,m.loc[chosen,'probes_mean'],color=[COLORS[c] for c in chosen]);axes[1].set_ylabel('CSI reports/frame')
    axes[1].set_xticks(x,chosen,rotation=35,ha='right');fig1.savefig(fig/'main_performance.pdf');plt.close(fig1)
    fig1,ax=plt.subplots(figsize=(6.8,4.4),layout='constrained')
    for i,g in enumerate(sensitivity):
        v,lo,hi=g['difference'];color='#176e91' if lo>0 else '#8b8c8e'
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=3)
    ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(np.arange(len(sensitivity)),[name[g['scenario']] for g in sensitivity]);ax.invert_yaxis()
    ax.set_xlabel('IVP minus A-opt-stop: paired completed payload (kbit/frame), 95% CI')
    fig1.savefig(fig/'sensitivity.pdf');plt.close(fig1)
    # Causal network / timing schematic.
    fig1,ax=plt.subplots(figsize=(3.5,3.3));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    ax.plot([.6,9.4],[7.3,7.3],color='#c1d8e4',lw=15,alpha=.5)
    for i,(xx,yy) in enumerate([(2,7.3),(7.8,7.3)]):
        ax.scatter([xx],[yy],s=130,marker='x',color='#176e91',lw=2);ax.text(xx,yy+.8,f'UAV {i+1}',ha='center')
        ax.scatter([xx-.6],[4.9],s=90,marker='^',color='#394b59');ax.text(xx-.6,4.3,f'BS {i+1}',ha='center')
        ax.add_patch(FancyArrowPatch((xx-.5,5.1),(xx-.05,7.05),arrowstyle='<->',mutation_scale=10,color='#176e91'))
    ax.add_patch(FancyArrowPatch((1.5,5.1),(7.55,7.05),arrowstyle='->',mutation_scale=9,color='#a76f4a',linestyle='--'))
    ax.add_patch(FancyArrowPatch((7.2,5.1),(2.2,7.05),arrowstyle='->',mutation_scale=9,color='#a76f4a',linestyle='--'))
    ax.text(5,9.4,'Directional BS antennas / shared spectrum',ha='center',fontsize=8)
    for x0,w,color,label in [(.5,2.2,'#d6dbe1','CSI reports'),(2.7,4.9,'#c0dce8','UL data'),(7.6,1.9,'#d5e3cf','DL control')]:
        ax.add_patch(Rectangle((x0,1.9),w,.9,facecolor=color,edgecolor='white'));ax.text(x0+w/2,2.35,label,ha='center',fontsize=7)
    ax.annotate('',xy=(9.5,1.5),xytext=(.5,1.5),arrowprops={'arrowstyle':'<->','color':'#444'})
    ax.text(5,.75,'Fixed deadline T: reporting reduces data time',ha='center',fontsize=7)
    ax.text(5,3.45,'Posterior -> probe or stop -> certify -> transmit',ha='center',fontsize=7)
    fig1.savefig(fig/'system_model.pdf',bbox_inches='tight');plt.close(fig1)
    # Concrete innovation envelope; no hypothetical performance curve.
    # Frozen illustrative seed/report, selected to expose multiple envelope levels.
    # It is not used to select or estimate any aggregate performance result.
    cfg=Config();sc=scenario(cfg,10002);idx=0
    val,intervals=probe_value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,idx,return_intervals=True)
    edges=np.array(sorted({-4.,4.}|{x for a,b,v in intervals for x in (a,b) if -4<x<4}))
    mids=(edges[:-1]+edges[1:])/2
    yy=np.array([max([v for a,b,v in intervals if a<=z<=b],default=0) for z in mids])
    fig1,ax=plt.subplots(figsize=(3.5,2.6),layout='constrained');ax.stairs(yy,edges,color='#176e91',lw=1.6)
    ax.stairs(yy,edges,fill=True,alpha=.12,color='#176e91');ax.set_xlabel('Standardized report innovation z');ax.set_ylabel('Certified payload (kbit)')
    ax.set_title(f'Expected payload after report: {val:.2f} kbit',fontsize=8);fig1.savefig(fig/'innovation_envelope.pdf');plt.close(fig1)
    # Final machine-readable values for prose assembly and audit.
    (res/'analysis_summary.json').write_text(json.dumps({'main':m.reset_index().to_dict(orient='records'),'gains':gains,
                                                       'stress':q.reset_index().to_dict(orient='records'),'record_count':len(df),
                                                       'snapshot_count':protocol['jobs']},indent=2))
    print(json.dumps({'records':len(df),'snapshots':protocol['jobs'],'gains':gains},indent=2))


if __name__=='__main__':main()
