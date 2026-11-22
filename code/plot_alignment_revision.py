"""Independent charts and main-text logical-work table for the fresh study."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ieee_figures import configure_ieee_style,style_axis
from plot_platform_revision import export

def main(root):
    root=Path(root);configure_ieee_style();path=root/'dense_alignment_results';s=json.loads((path/'summary.json').read_text());b=json.loads((path/'boundary_summary.json').read_text());cat=[];rows=[]
    for kind,name,methods,height in [('dense','dense_cost_refinement',[('IVP-NR','IVP / certificate','#145da0','o'),('Margin-NR','Sensitivity / certificate','#008b70','s')],2.95),('frontier','aligned_frontier_gain',[('Aware-NR','Frontier-aware forecast','#145da0','o'),('Certificate-NR','Certificate forecast','#ca6f1e','s')],2.85)]:
        fig,ax=plt.subplots(figsize=(3.5,height));fig.subplots_adjust(left=.19,right=.97,bottom=.23,top=.79)
        for label,title,color,marker in methods:
            cs=s[kind];x=np.array([c['alpha']*100 for c in cs]);y=np.array([c[label]['mean'] for c in cs]);lo=np.array([c[label]['low'] for c in cs]);hi=np.array([c[label]['high'] for c in cs]);ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=marker+'-',color=color,capsize=2,markersize=3.2,label=title,lw=1)
            rows.extend(dict(figure=name,method=label,cost_fraction=c['alpha'],gain=c[label]['mean'],low=c[label]['low'],high=c[label]['high']) for c in cs)
        ax.axhline(0,color='#555',ls=':',lw=.9);ax.set(xlabel='Decision cost c/T (%)',ylabel='Gain over no-report (kbit/frame)');style_axis(ax)
        if kind=='dense':
            ax.axvline(100*b['Predictive-IVP']['supported_prefix'],color='#555',ls='--',lw=.8);ax.axvline(100*b['Predictive-IVP']['first_crossing'],color='#777',ls=':',lw=.8);ax.set(xticks=[0,2,4,6,8],xlim=(-.2,8.2))
        else:ax.set(xticks=[0,1,2],xlim=(-.08,2.08))
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=1,frameon=False,fontsize=9.5,labelspacing=.2,handlelength=1.5)
        cap='512 fresh paired frames; 60-contrast simultaneous 95% paired-t intervals. N=4, T=20 ms, rho=exp(-0.04), tau=0.75 ms. '
        cap+=('17 cost points; dashed line is supported tested prefix, dotted line is empirical first crossing.' if kind=='dense' else 'All policies use the same finite terminal frontier with independent final validation. The frontier-aware forecast uses the scaled-power and filled-OMA candidate family, 4096 private planning draws and 49 approximate Gaussian midpoint nodes. No analytic interval guarantee is asserted for it.')
        cat.append(export(fig,root,name,'Fresh cost and allocation alignment: '+kind,cap,'dense_alignment_results/summary.json; boundary_summary.json',height))
    out=root/'figures_r5';(out/'alignment_catalogue.json').write_text(json.dumps(cat,indent=2)+'\n')
    with (out/'alignment_curve_points.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    work=json.loads((root/'certificate_work_results/summary.json').read_text());lines=[r'\begin{table}[t]\centering\footnotesize',r'\caption{Mean logical work per candidate value on the same 3,072-value audit. All methods call the same compiled certificate predicate.}\label{tab:workcounts}',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Method & Cert. tests & Linear solves & Coeff. builds\\\midrule']
    for method in ['CDF25','KG17','Adaptive-GK']:
        d=work[method];lines.append(f'{method} & {d["certificate_tests"]["mean"]:.1f} & {d["directional_linear_solves"]["mean"]:.1f} & {d["action_preparations"]["mean"]:.1f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];(root/'manuscript/work_count_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[H]\centering\small',r'\caption{Fresh matched-frontier slice: 512 paired frames at each cost. Reporting means and outage counts accompany service.}\label{tab:aligneddetail}',r'\begin{tabular}{@{}r l r r r@{}}\toprule',r'$c/T$ (\%) & Forecast & kbit/frame & Reports & Outages\\\midrule']
    for c in s['frontier']:
        for method,label in [('No-report-F','No report'),('Certificate-forecast-F','Certificate'),('Frontier-aware-IVP','Frontier-aware')]:
            d=c['methods'][method];lines.append(f'{100*c["alpha"]:g} & {label} & {d["completed"]:.2f} & {d["reports"]:.3f} & {d["outages"]}/512'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];(root/'manuscript/aligned_detail_table.tex').write_text('\n'.join(lines)+'\n');print('Created two independent figures and two tables')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
