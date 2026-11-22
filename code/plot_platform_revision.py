"""Independent IEEE-style plots of platform-neutral empirical cost tolerance."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ieee_figures import configure_ieee_style,style_axis

def export(fig,root,name,title,caption,source,height):
    out=root/'figures_r5';out.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=600)
    for directory in [root/'figures',root/'manuscript/figures']:
        directory.mkdir(exist_ok=True,parents=True);shutil.copyfile(out/(name+'.pdf'),directory/(name+'.pdf'))
    plt.close(fig)
    return dict(name=name,parent=name,panel=None,title=title,caption=caption,source=source,font='Times New Roman',font_size_pt=9.5,box_sides=['left','bottom','right','top'],width_in=3.5,height_in=height)

def main(root):
    root=Path(root);configure_ieee_style();path=root/'platform_cost_results';data=json.loads((path/'boundary_summary.json').read_text());byid={d['cell']['id']:d for d in data};catalogue=[];tables=root/'manuscript'
    maps=[d for d in data if d['cell']['kind']=='map'];speeds=sorted({d['cell']['speed'] for d in maps});fractions=sorted({d['cell']['report_fraction'] for d in maps})
    for method,name,supported in [('Predictive-IVP','cost_tolerance_map',False),('Predictive-Margin','cost_tolerance_margin',False),('Predictive-IVP','cost_support_map',True)]:
        arr=np.array([[100*next(d for d in maps if d['cell']['speed']==v and d['cell']['report_fraction']==f)['boundaries'][method]['supported_prefix_grid_budget' if supported else 'estimate'] for f in fractions] for v in speeds])
        vmax=max(1.,float(np.ceil(arr.max())));fig=plt.figure(figsize=(3.5,3.45));ax=fig.add_axes([.17,.34,.78,.61]);im=ax.imshow(arr,origin='lower',aspect='auto',cmap='Blues',vmin=0,vmax=vmax,interpolation='nearest')
        ax.set(xticks=range(len(fractions)),xticklabels=[f'{x:g}' for x in fractions],yticks=range(len(speeds)),yticklabels=[f'{v:g}' for v in speeds],xlabel=r'Report fraction $\tau/T$',ylabel=r'Speed $v$ (m/s)')
        for i,v in enumerate(speeds):
            for j,f in enumerate(fractions):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white' if arr[i,j]>.60*vmax else 'black',fontsize=9.5)
        for s in ax.spines.values():s.set_visible(True);s.set_color('black');s.set_linewidth(.8)
        ax.tick_params(direction='in',top=True,right=True)
        cb=fig.add_axes([.17,.14,.78,.04]);bar=fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[0,vmax/2,vmax]);bar.set_label('Supported tested budget (%)' if supported else 'First break-even estimate (%)',fontsize=9.5,labelpad=1);bar.outline.set_linewidth(.8)
        cap=('Largest initial tested cost prefix with positive simultaneous lower limits; no interpolation.' if supported else 'First zero of the piecewise-linear empirical mean gain; pointwise paired-bootstrap intervals are tabulated separately.')+' N=4, T=20 ms, 128 paired states per cell, and rho=exp(-v*T/10m). The 10-m 1/e length is illustrative, not a fitted UAV parameter.'
        catalogue.append(export(fig,root,name,'Platform-independent cost budget: '+method,cap,'platform_cost_results/boundary_summary.json; boundaries.csv',3.45))
    colors=['#145da0','#ca6f1e','#008b70']
    for method,name in [('Predictive-IVP','size_duration_boundaries'),('Predictive-Margin','margin_duration_boundaries')]:
        fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.18,right=.97,bottom=.23,top=.81)
        for n,color in zip([4,6,8],colors):
            ds=[byid[f'slice_N{n}_T{T}'] for T in [20,50,100]];q=[d['boundaries'][method] for d in ds];y=np.array([x['estimate']*100 for x in q]);lo=np.array([x['bootstrap95'][0]*100 for x in q]);hi=np.array([x['bootstrap95'][1]*100 for x in q]);ax.errorbar([20,50,100],y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color=color,label=f'N={n}')
        ax.set(xlabel='Frame duration T (ms)',ylabel='First break-even budget (%)',xticks=[20,50,100],xlim=(15,105));style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9.5,handlelength=1.2,columnspacing=1.0)
        catalogue.append(export(fig,root,name,'Size/duration cost boundaries: '+method,'128 paired states per setting. Pointwise 95% paired-bootstrap intervals for empirical first crossings, not intervals for an exact continuous threshold. Fixed 0.75-ms report airtime, v=20m/s and illustrative dcor=10m; payload scales with T.','platform_cost_results/boundary_summary.json',2.7))
    f=byid['frontier_N4_T20'];s=byid['slice_N4_T20'];fig,ax=plt.subplots(figsize=(3.5,2.85));fig.subplots_adjust(left=.18,right=.97,bottom=.21,top=.75)
    for ds,key,label,color,marker in [(s,'IVP-NR','IVP / certificate','#145da0','o'),(f,'IVP-NR','IVP / frontier','#ca6f1e','s'),(f,'Margin-NR','Margin / frontier','#008b70','^')]:
        cs=ds['costs'][:6];x=np.array([c['alpha']*100 for c in cs]);mean=np.array([c[key]['mean'] for c in cs])/20;lo=np.array([c[key]['low'] for c in cs])/20;hi=np.array([c[key]['high'] for c in cs])/20
        ax.errorbar(x,mean,yerr=[mean-lo,hi-mean],fmt=marker+'-',capsize=2.5,color=color,label=label)
    ax.axhline(0,color='#555',ls=':',lw=.9);ax.set(xlabel='Hypothetical decision cost c/T (%)',ylabel='Gain over no-report (Mbit/s)',xlim=(-.08,2.08));style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=1,frameon=False,fontsize=9.5,labelspacing=.2,handlelength=1.5)
    catalogue.append(export(fig,root,'frontier_cost_slice','Matched terminal-frontier ablation','N=4, T=20 ms, 128 paired states, v=20m/s and 0.75-ms reports. Bars are global simultaneous 95% paired-t intervals. Forecasts remain certificate-based; each reporting method and its no-report comparator use the same terminal allocator. Host elapsed time is excluded.','platform_cost_results/frontier_N4_T20_summary.json; slice_N4_T20_summary.json',2.85))
    with (root/'independent_numerical_results/values.csv').open() as f:values=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.19,right=.97,bottom=.23,top=.81)
    for method,color in zip(['CDF25','KG17','Adaptive-GK'],colors):
        e=np.sort([abs(float(r['error'])) for r in values if r['method']==method]);ax.plot(e,np.arange(1,len(e)+1)/len(e),label=method,color=color)
    ax.set_xscale('symlog',linthresh=1e-7);ax.set(xlabel='Absolute report-value error (kbit)',ylabel='Empirical CDF',xlim=(-1e-8,8),ylim=(0,1.03));ax.set_xticks([0,1e-6,1e-4,1e-2,1],['0',r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$','1']);style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9.5,handlelength=1.0,columnspacing=.8)
    catalogue.append(export(fig,root,'q_accuracy_independent','Independent stratified report-value audit','3,072 candidate values in 192 fresh states, three independent 64-state traffic blocks and a bidirectional 16-coordinate catalogue. Candidate errors within one state are dependent. The root reference uses 1e-8 endpoint tolerance.','independent_numerical_results/values.csv; summary.json; protocol.json',2.7))
    (root/'figures_r5/platform_catalogue.json').write_text(json.dumps(catalogue,indent=2)+'\n')

    summary=json.loads((root/'independent_numerical_results/summary.json').read_text());lines=[r'\begin{table}[t]\centering\footnotesize',r'\caption{Value errors and initial report/stop agreement: 3,072 candidates in 192 independent states. Errors are kbit.}\label{tab:independentnumerical}',r'\begin{tabular}{@{}lrrrr@{}}\toprule',r'Method & Mean $|\Delta Q|$ & 95th pct. & Max $|\Delta Q|$ & Agree\\\midrule']
    for method in ['CDF25','KG17','Adaptive-GK']:
        a=summary['All'][method];lines.append(f'{method} & {a["mean_abs_error"]:.3g} & {a["p95_abs_error"]:.3g} & {a["max_abs_error"]:.3g} & {a["decision_agreement"]}/192'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];(tables/'independent_numerical_table.tex').write_text('\n'.join(lines)+'\n')
    la=json.loads((path/'lookahead_summary.json').read_text());lines=[r'\begin{table}[t]\centering\small',r'\caption{Matched two-report-cap comparison: completed kbit/frame on 128 fresh paired states. LA2 adapts two-step look-ahead to the present belief and reward.}\label{tab:lookahead}',r'\begin{tabular}{@{}rrrr@{}}\toprule',r'$c/T$ (\%) & No report & IVP & LA2\\\midrule']
    for c in la['costs']:lines.append(f'{100*c["alpha"]:g} & {c["methods"]["Predictive-NR"]["completed"]:.2f} & {c["methods"]["Predictive-IVP"]["completed"]:.2f} & {c["methods"]["Two-step-LA"]["completed"]:.2f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];(tables/'lookahead_table.tex').write_text('\n'.join(lines)+'\n')
    for method,name in [('Predictive-IVP','boundary_bootstrap_ivp.tex'),('Predictive-Margin','boundary_bootstrap_margin.tex')]:
        lines=[r'\begin{table}[H]\centering\footnotesize\setlength{\tabcolsep}{3pt}',r'\caption{'+method+r': empirical first-crossing cost budget (\%) with pointwise paired-bootstrap 95\% interval. Fixed $N=4$, $T=20$ ms and illustrative $d_{\rm cor}=10$ m.}\label{tab:'+name[:-4]+r'}',r'\begin{tabular}{@{}r'+('r'*len(fractions))+r'@{}}\toprule',r'$v$ (m/s) / $\tau/T$ & '+' & '.join(f'{x:g}' for x in fractions)+r'\\\midrule']
        for v in speeds:
            row=[f'{v:g}']
            for f in fractions:
                q=next(d for d in maps if d['cell']['speed']==v and d['cell']['report_fraction']==f)['boundaries'][method];row.append(f'{100*q["estimate"]:.2f} [{100*q["bootstrap95"][0]:.2f},{100*q["bootstrap95"][1]:.2f}]')
            lines.append(' & '.join(row)+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}'];(tables/name).write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]\centering\small',r'\caption{Size/duration first-crossing budgets (\%) with pointwise paired-bootstrap 95\% intervals.}\label{tab:sliceboundaries}',r'\begin{tabular}{@{}rrlrr@{}}\toprule',r'$N$ & $T$ (ms) & $\rho_{\rm e}$ & IVP & Margin\\\midrule']
    for n in [4,6,8]:
        for T in [20,50,100]:
            d=byid[f'slice_N{n}_T{T}'];qs=[]
            for method in ['Predictive-IVP','Predictive-Margin']:
                q=d['boundaries'][method];qs.append(f'{100*q["estimate"]:.2f} [{100*q["bootstrap95"][0]:.2f},{100*q["bootstrap95"][1]:.2f}]')
            lines.append(f'{n} & {T} & {d["cell"]["rho"]:.4f} & '+' & '.join(qs)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];(tables/'boundary_slice_table.tex').write_text('\n'.join(lines)+'\n')
    print('Created',len(catalogue),'independent figures and five tables')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
