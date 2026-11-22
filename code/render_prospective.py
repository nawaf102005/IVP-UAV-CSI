"""Build independent figures and complete tabulations from the registered frame studies."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from ieee_figures import configure_ieee_style, style_axis
from plot_platform_revision import export

REUSE='persistent_frontier_gain'
PROFILE='frontier_planning_resolution'
REUSE_CAP=(r'Paired service gain from optional C-IVP with a common terminal frontier and an assumed '
          r'1.72-ms charge per acquisition round. Bars give simultaneous 95\% intervals over 192 comparisons '
          r'on 128 independent 40-frame trajectories per speed. Horizontal offsets separate speeds at the same $K$.')
PROFILE_CAP=(r'Frontier-aware service gain over frontier no-report before charging computation, against the '
             r'95th percentile of whole-decision time on 128 paired frames. Lines join increasing midpoint counts '
             r'(9, 25, 49) at fixed planning draws. Bars give simultaneous 95\% intervals from the 18-contrast family. '
             r'The dotted vertical line marks 1 ms.')

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def interval(r,precision=2):
    return ('$%.*f$ $[%.*f,%.*f]$' % (precision,float(r['mean']),precision,float(r['lo']),precision,float(r['hi'])))

def maketable(dst,label,caption,columns,head,body):
    dst.write_text('\n'.join([r'\begin{table}[H]\centering\footnotesize',r'\caption{'+caption+r'}\label{'+label+'}',
                            r'\begin{tabular}{@{}'+columns+r'@{}}\toprule',head+r'\\\midrule',
                            *[row+r'\\' for row in body],r'\bottomrule\end{tabular}\end{table}'])+'\n')

def generate(root):
    root=Path(root);out=root/'prospective_results';tex=root/'manuscript'
    reuse=read(out/'reuse_contrasts.csv');prof=read(out/'profile_summary.csv')
    cost=read(out/'host_contrasts.csv');host=read(out/'host_trajectories.csv')
    configure_ieee_style()
    fig,ax=plt.subplots(figsize=(3.5,2.48));fig.subplots_adjust(left=.17,right=.974,bottom=.215,top=.83)
    for i,(v,color,marker) in enumerate([(10,'#145da0','o'),(20,'#00836a','s'),(30,'#ad4e00','^')]):
        rr=[r for r in reuse if int(r['speed'])==v and float(r['cost_ms'])>0 and r['terminal']=='front'
            and r['left']=='IVP' and r['right']=='No-report']
        rr.sort(key=lambda r:int(r['K']));x=np.arange(4)+(i-1)*.13;y=np.array([float(r['mean']) for r in rr])
        ax.errorbar(x,y,yerr=[y-np.array([float(r['lo']) for r in rr]),np.array([float(r['hi']) for r in rr])-y],
                    fmt=marker,ms=4,capsize=2,lw=1.,mfc='white',color=color,label=f'{v} m/s')
    ax.axhline(0,color='#333333',ls=':',lw=.8);ax.set(xticks=range(4),xticklabels=['1','5','10','20'],
            xlim=(-.45,3.45),xlabel='Frames between opportunities, K',ylabel='Service gain (kbit/frame)')
    style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
                              bbox_to_anchor=(.55,1.005),handletextpad=.15,columnspacing=.62)
    a=export(fig,root,REUSE,'Service from persistent reports with a common frontier',REUSE_CAP,
             'prospective_results/reuse_contrasts.csv',2.48)
    fig,ax=plt.subplots(figsize=(3.5,2.62));fig.subplots_adjust(left=.17,right=.974,bottom=.219,top=.825)
    for draws,color,marker,ls in [(256,'#145da0','o','-'),(1024,'#00836a','s','--'),(4096,'#ad4e00','^','-.')]:
        rr=[r for r in prof if int(r['draws'])==draws and float(r['budget_ms'])==0];rr.sort(key=lambda r:int(r['nodes']))
        y=np.array([float(r['mean']) for r in rr]);x=[float(r['p95_wall_ms']) for r in rr]
        ax.errorbar(x,y,yerr=[y-np.array([float(r['lo']) for r in rr]),np.array([float(r['hi']) for r in rr])-y],
                    fmt=marker,ls=ls,capsize=1.8,lw=1.,ms=3.8,mfc='white',color=color,label=f'{draws:,}')
    ax.axhline(0,color='#303030',ls=':',lw=.8);ax.axvline(1,color='#444',ls=':',lw=.8)
    ax.set(xscale='log',xlabel='Whole-decision 95th percentile (ms)',ylabel='Radio-only gain (kbit/frame)')
    xmax=max(float(r['p95_wall_ms']) for r in prof if r['draws']!='0' and float(r['budget_ms'])==0)
    ax.set_xlim(.75,max(5.,xmax*1.4));ts=[a for a in (1,3,10,30,100,300,1000,3000) if a<=xmax*1.4]
    ax.set_xticks(ts);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    style_axis(ax);fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
                              bbox_to_anchor=(.55,1.005),handletextpad=.2,columnspacing=.8)
    b=export(fig,root,PROFILE,'Planning draws, integration nodes and wall time',PROFILE_CAP,
             'prospective_results/profile_summary.csv',2.62)
    by={(r['terminal'],float(r['cost_ms']),int(r['speed']),int(r['K']),r['left'],r['right']):r for r in reuse}
    methods=('IVP','Sensitivity','Fixed-IVP')
    for trm,name in [('cert','sufficient certificate'),('front','finite terminal frontier')]:
        for c,tag in [(0.,'zero'),(1.72,'charged')]:
            token=trm+tag;pairrows=[];rawrows=[]
            for v in (10,20,30):
                for k in (1,5,10,20):
                    rr=[by[trm,c,v,k,m,'No-report'] for m in methods];direct=by[trm,c,v,k,'IVP','Sensitivity']
                    pairrows.append(' & '.join([str(v),str(k),*[interval(x) for x in [*rr,direct]]]))
                    baseline=f"{float(rr[0]['right_kbit']):.2f} / {rr[0]['right_outages']}"
                    rawrows.append(' & '.join([str(v),str(k),baseline,*[f"{float(r['left_kbit']):.2f} / {r['left_outages']} / {r['left_reports']}" for r in rr]]))
            prefix=f'Persistent reuse with the {name} and $c={c:g}$ ms per acquisition round. '
            maketable(tex/f'pros_gain_{token}.tex',f'tab:prosga{token}',
                      prefix+r'Entries are paired gains in kbit/frame over 128 independent 40-frame trajectories per speed. Brackets are simultaneous 95\% paired-$t$ intervals over all 192 contrasts. NR denotes no-report.',
                      'rrcccc',r'$v$ (m/s) & $K$ & IVP $-$ NR & Sensitivity $-$ NR & Fixed IVP $-$ NR & IVP $-$ sensitivity',pairrows)
            maketable(tex/f'pros_raw_{token}.tex',f'tab:prosraw{token}',
                      prefix+r'Entries give mean completed kbit/frame, total radio outages and total reports over 5,120 frames per setting. The no-report column omits its zero report count. Descriptive results for 128 independent trajectories.',
                      'rrcccc',r'$v$ (m/s) & $K$ & No-report & Optional IVP & Sensitivity & Fixed IVP',rawrows)
    baseline=[r for r in host if int(r['K'])==0];hc=[r for r in cost]
    def counts(rr):return [str(sum(int(z[field]) for z in rr)) for field in ('reports','radio_failures','active_overruns','combined_failures')]
    raw=[ ' & '.join(['No-report',f"{float(hc[0]['nr_kbit']):.2f}",'---',*counts(baseline)]) ]
    for r in hc:
        k=int(r['K']);rr=[x for x in host if int(x['K'])==k]
        raw.append(' & '.join([f'IVP, $K={k}$',f"{float(r['ivp_kbit']):.2f}",interval(r),*counts(rr)]))
    maketable(tex/'pros_host_table.tex','tab:proshost',r'Measured reservation at 20 m/s on 64 held-out 40-frame trajectories (2,560 frames per method). Net service and gains use kbit/frame. The two paired 95\% intervals are simultaneous. Failure counts are descriptive. A combined failure is the union of active overruns and radio failures.',
              'lrcrrrr',r'Method & Net service & Gain vs. no-report & Reports & Radio failures & Active overruns & Combined',raw)
    for c,tag in [(0.,'radio'),(1.,'one')]:
        rr=[r for r in prof if float(r['budget_ms'])==c];body=[]
        for r in rr:
            config= r['draws']+' / '+r['nodes'] if int(r['draws']) else 'No-report'
            body.append(' & '.join([config,f"{float(r['service_kbit']):.2f}",interval(r) if int(r['draws']) else '---',
               f"{float(r['mean_wall_ms']):.2f}",f"{float(r['median_wall_ms']):.2f}",f"{float(r['p95_wall_ms']):.2f}",
               r['reports'],r['radio_outages'],r['active_overruns'],r['combined_failures']]))
        prefix=('Radio-only frontier planning (no computation deducted). ' if not c else 'Whole-decision budget of 1 ms shared by frontier reporting and no-report. ')
        maketable(tex/f'pros_profile_{tag}.tex',f'tab:prosprof{tag}',prefix+r'Each row uses the same 128 independent frames. Service and gains are in kbit/frame. The 95\% gain intervals are simultaneous over 18 contrasts. Wall times are in ms. Counts give reports, potential radio outages, active overruns and their union.',
                  'lrrrrrrrrr',r'Draws / nodes & Service & Gain vs. no-report & Mean & Median & 95th pct. & Reports & Radio & Over & Union',body)
    return [a,b]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(generate(a.root),indent=2))
