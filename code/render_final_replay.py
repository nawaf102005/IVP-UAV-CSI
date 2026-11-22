"""Generate the final matched-frontier panel and retrospective contrast tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ieee_figures import configure_ieee_style, style_axis
from plot_platform_revision import export

NAME = 'frontier_dense_replay'
CAPTION = (r'Completed-service gain over frontier no-report across the 17 budgets of Fig.~\ref{fig:densecost}. '
           r'Both certificate-based acquisition rules use the same final frontier on the 512 archived cohort F frames. '
           r'Shading gives simultaneous 95\% paired-$t$ intervals from the retrospective 58-contrast family. '
           r'At $T=20$ ms, each 1\% on the horizontal axis is .20 ms per round.')
VALUE_CAPTION = (r'Scalar-value audit on 3,072 candidates in 192 states. Errors are in kbit. '
                 r'Work and time are descriptive means per candidate. The root row counts Python feasibility callbacks. '
                 r'The other rows count a shared compiled predicate.')
VALUE_NOTE = (r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize Ref. defines the numerical target. '
              r'Each root callback evaluates one action and mode at one innovation. The root uses spectral tests '
              r'(930.9 individual eigenproblems per value), with 15 reuse and 11 OMA preparations. '
              r'The compiled predicate uses directional solves and a different preparation routine. '
              r'Times are from the original uninstrumented run.\end{minipage}\end{table*}')

def records(p):
    with p.open() as f: return list(csv.DictReader(f))

def formatted_interval(row):
    return '$%.3f\\ [%.3f,%.3f]$' % tuple(float(row[k]) for k in ('mean','lo','hi'))

def generate(root):
    root=Path(root); data=root/'final_replay_results'; tex=root/'manuscript'
    curves=records(data/'frontier_contrasts.csv'); host=records(data/'host_direct.csv')
    configure_ieee_style()
    fig,ax=plt.subplots(figsize=(3.5,2.70)); fig.subplots_adjust(left=.188,right=.976,bottom=.208,top=.86)
    for name,color,shape,ls in [('C-IVP','#145da0','o','-'),('Sensitivity','#00836a','s','--')]:
        rows=[r for r in curves if r['left']==name and r['right']=='No-report']
        x=[100*float(r['alpha']) for r in rows]; y=[float(r['mean']) for r in rows]
        ax.fill_between(x,[float(r['lo']) for r in rows],[float(r['hi']) for r in rows],color=color,alpha=.16,linewidth=0)
        ax.plot(x,y,ls,marker=shape,color=color,markersize=3.4,markerfacecolor='white',label=name)
    ax.axhline(0,color='#454545',linewidth=.75,ls=':'); ax.set_xlim(-.12,8.12);ax.set_ylim(-9.1,2.6)
    ax.set(xticks=[0,2,4,6,8],yticks=[-8,-6,-4,-2,0,2],xlabel='Decision budget, c/T (%)',ylabel='Service gain (kbit/frame)')
    style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,
                               bbox_to_anchor=(.56,1.01),handletextpad=.34,columnspacing=1.15)
    item=export(fig,root,NAME,'Matched terminal frontier across decision budgets',CAPTION,
                'final_replay_results/frontier_contrasts.csv',2.70)
    cells={(float(r['alpha']),r['left'],r['right']):r for r in curves}
    rows=[r'\begin{table}[H]\centering\small',
          r'\caption{Completed-service differences in kbit/frame under the matched terminal frontier, 512 paired cohort F frames per budget. Intervals are simultaneous 95\% paired-$t$ intervals over the 58 retrospective contrasts.}\label{tab:replayfrontier}',
          r'\begin{tabular}{@{}rrrrr@{}}\toprule',
          r'$c/T$ (\%) & $c$ (ms) & C-IVP $-$ no-report & Sensitivity $-$ no-report & C-IVP $-$ sensitivity\\\midrule']
    for a in sorted({float(r['alpha']) for r in curves}):
        keys=[(a,'C-IVP','No-report'),(a,'Sensitivity','No-report'),(a,'C-IVP','Sensitivity')]
        rows.append(' & '.join([f'{100*a:g}',f"{float(cells[keys[0]]['c_ms']):.1f}",*[formatted_interval(cells[k]) for k in keys]])+r'\\')
    rows.append(r'\bottomrule\end{tabular}\end{table}');(tex/'replay_frontier_table.tex').write_text('\n'.join(rows)+'\n')
    rows=[r'\begin{table}[H]\centering\small',
          r'\caption{Completed rates in Mbit/s after method-specific causal reservations on 256 paired frozen-channel frames per duration (cohort G). The direct difference uses the retrospective simultaneous 95\% family of 58 paired contrasts.}\label{tab:replaylatency}',
          r'\begin{tabular}{@{}rrrrr@{}}\toprule',r'$T$ (ms) & C-IVP & Sensitivity & No-report & Sensitivity $-$ C-IVP\\\midrule']
    for r in host:
        rows.append(' & '.join([r['duration_ms'],*[f'{float(r[k]):.3f}' for k in ('ivp_rate','margin_rate','nr_rate')],formatted_interval(r)])+r'\\')
    rows.append(r'\bottomrule\end{tabular}\end{table}');(tex/'replay_host_table.tex').write_text('\n'.join(rows)+'\n')
    return item

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(generate(a.root),indent=2))
