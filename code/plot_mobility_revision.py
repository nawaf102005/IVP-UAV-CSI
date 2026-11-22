"""Independent Times New Roman figures and tables for the additional studies."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ieee_figures import configure_ieee_style,style_axis

def save(fig,root,name,metadata):
    out=root/'figures_r5'
    for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=600)
    for folder in [root/'figures',root/'manuscript/figures']:
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(out/(name+'.pdf'),folder/(name+'.pdf'))
    plt.close(fig)
    return dict(name=name,parent=name,panel=None,font='Times New Roman',font_size_pt=9.5,box_sides=['left','bottom','right','top'],width_in=3.5,height_in=metadata.pop('height_in',3.2),**metadata)

def main(root):
    root=Path(root);configure_ieee_style();s=json.loads((root/'mobility_cost_results/summary.json').read_text());p=json.loads((root/'mobility_cost_results/protocol.json').read_text());new=[];points=[]
    for charged,name in [(False,'mobility_radio'),(True,'mobility_charged')]:
        rows=[r for r in s if r['charged']==charged];rhos=p['rhos'];costs=p['report_fractions'];array=np.array([[next(r for r in rows if r['rho']==rho and r['report_fraction']==cost)['IVP-minus-Predictive-NR']['mean'] for cost in costs] for rho in rhos])
        fig=plt.figure(figsize=(3.5,3.2));ax=fig.add_axes([.17,.31,.78,.64]);im=ax.imshow(array,origin='lower',aspect='auto',cmap='RdBu',vmin=-25,vmax=25,interpolation='nearest')
        ax.set(xticks=range(len(costs)),xticklabels=[f'{x:g}' for x in costs],yticks=range(len(rhos)),yticklabels=[f'{x:g}' for x in rhos],xlabel=r'Report fraction $\tau/T$',ylabel=r'Log-gain correlation $\rho_{\rm e}$')
        for i,rho in enumerate(rhos):
            for j,cost in enumerate(costs):
                item=next(r for r in rows if r['rho']==rho and r['report_fraction']==cost);q=item['IVP-minus-Predictive-NR'];star='*' if q['low']>0 else ''
                ax.text(j,i,f'{q["mean"]:.1f}'+star,ha='center',va='center',fontsize=9.5,color='white' if abs(q['mean'])>16 else 'black')
                points.append(dict(figure=name,rho=rho,report_fraction=cost,mean=q['mean'],low=q['low'],high=q['high'],ivp_minus_margin=item['IVP-minus-Predictive-Margin']['mean']))
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('black');spine.set_linewidth(.8)
        ax.tick_params(which='both',direction='in',top=True,right=True)
        cb=fig.add_axes([.17,.11,.78,.04]);bar=fig.colorbar(im,cax=cb,orientation='horizontal',ticks=[-25,0,25]);bar.set_label('IVP minus no-report (kbit/frame)',fontsize=9.5,labelpad=1);bar.outline.set_edgecolor('black');bar.outline.set_linewidth(.8)
        new.append(save(fig,root,name,dict(title='Joint aging and report-cost map'+(' with computation charged' if charged else ' with radio airtime charged'),caption='128 paired test frames per cell at N=4 and T=20 ms. Known jointly Gaussian initial/execution log gains. Stars mark a positive Bonferroni simultaneous 95% lower limit against no-report across 120 contrasts. No cell establishes superiority over predictive sensitivity. '+('Method-specific computation reservations are fixed on 128 separate development frames; observed overruns are charged.' if charged else 'Only radio report airtime is subtracted.'),source='mobility_cost_results/summary.json; protocol.json; reservations.json')))
    with (root/'numerical_results/values.csv').open() as f:rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(3.5,2.7));fig.subplots_adjust(left=.19,right=.97,bottom=.23,top=.80)
    for method,color in [('CDF25','#145da0'),('KG17','#ca6f1e'),('Adaptive-GK','#008b70')]:
        data=np.sort([abs(float(r['error'])) for r in rows if r['method']==method]);ax.plot(data,np.arange(1,len(data)+1)/len(data),label=method,color=color)
    ax.set(xlabel='Absolute report-value error (kbit)',ylabel='Empirical CDF',xscale='symlog',xlim=(-1e-8,8),ylim=(0,1.03));ax.set_xscale('symlog',linthresh=1e-7);ax.set_xticks([0,1e-6,1e-4,1e-2,1],['0',r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$','1']);style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False,fontsize=9.5,handlelength=1.1,columnspacing=.8)
    new.append(save(fig,root,'q_accuracy',dict(height_in=2.7,title='Report-value approximation accuracy',caption='Absolute Q errors for 256 candidate reports in 32 previously evaluated states. The reference uses convex minimization and Brent roots with 1e-8 endpoint tolerance. Candidates within a state are dependent. The horizontal scale is linear near zero. Adaptive integration tolerances are not uniform miss bounds.',source='numerical_results/values.csv; decisions.json; protocol.json')))
    out=root/'figures_r5';(out/'mobility_curve_points.csv').write_text('')
    with (out/'mobility_curve_points.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    (out/'additional_catalogue.json').write_text(json.dumps(new,indent=2)+'\n')
    summary=json.loads((root/'numerical_results/summary.json').read_text())
    lines=[r'\begin{table}[t]\centering\footnotesize',r'\caption{Report-value errors and initial report/stop agreement on 32 states (256 candidate reports).}\label{tab:numerical}',r'\begin{tabular}{@{}lrrrr@{}}\toprule',r'Method & Mean $|\Delta Q|$ & Max $|\Delta Q|$ & Agree & ms/value\\\midrule']
    for m in ['CDF25','KG17','Adaptive-GK']:
        a=summary[m];lines.append(f'{m} & {a["mean_abs_error"]:.3g} & {a["max_abs_error"]:.3g} & {a["decision_agreement"]}/32 & {a["mean_ms"]:.2f}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}',r'\par\smallskip\begin{minipage}{\columnwidth}\footnotesize Errors are kbit; agreement includes stopping and report identity. Mean root-reference time is '+f'{summary["reference"]["mean_ms"]:.2f}'+r' ms/value in Python. The compiled-grid and Python-root timings confound algorithm and implementation.\end{minipage}',r'\end{table}'])
    (root/'manuscript/numerical_table.tex').write_text('\n'.join(lines)+'\n');print('Created',len(new),'additional independent figures and one table')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
