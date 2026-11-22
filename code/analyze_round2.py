"""Regenerate the second-review tables, figures and numerical summary."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_revision import table,label,wilson
from round2_experiments import read_records,paired,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results');p.add_argument('--artifact-root');a=p.parse_args()
    root=Path(a.root);out=Path(a.results) if a.results else root/'round2_results'
    dest=Path(a.artifact_root) if a.artifact_root else root;tex=dest/'manuscript';fig=dest/'figures'
    ray=pd.DataFrame(read_records(root/'revision_results/ray_records.jsonl')+read_records(root/'revision_results/conformal_records.jsonl')+read_records(out/'calibration_records.jsonl'))
    rows=[];methods=['C-IVP','A-stop','Sensitivity-3pt','No-report','OMA','Unscaled-IVP','Conformal-NR','Safe-IVP','Safe-NR','Regularized-IVP','Regularized-NR','Perfect-CSI']
    for m in methods:
        row=[label(m)]
        for name in ['canyon_low','canyon_high','munich']:
            for kind in ['static','motion']:
                d=ray[(ray.scenario==name+'-'+kind)&(ray.method==m)]
                row.append(f'{d.completed_kbit.mean():.2f} / {d.outage.sum():.0f}')
        rows.append(row)
    table(tex/'r2_ray.tex','Ray-traced completed kbit/frame / outages in 256 frames per condition. S: static incoherent paths; M: coherent endpoint motion. Original and regularized filters are shown together.','tab:raycombined',
          ['Policy','Canyon 35 S','Canyon 35 M','Canyon 80 S','Canyon 80 M','Munich S','Munich M'],rows,True,
          note='Canyon heights share one layout. Regularized rows on these previously examined frames are exploratory; Table~\\ref{tab:fresh} uses new Munich positions. Marginal coverage concerns static exchangeable frames only. Perfect-CSI is a zero-cost execution-channel reference.')
    fresh=pd.DataFrame(read_records(out/'fresh_calibration_records.jsonl'));rows=[]
    for m in ['C-IVP','Safe-IVP','Safe-NR','Regularized-IVP','Regularized-NR']:
        row=[label(m)]
        for k in ['static','motion']:
            d=fresh[(fresh.method==m)&(fresh.scenario=='munich-'+k)]
            row.append(f'{d.completed_kbit.mean():.2f} / {d.outage.sum():.0f}')
        rows.append(row)
    table(tex/'r2_fresh.tex','New Munich test positions: kbit/frame / outages out of 256. Training, calibration, and the 1-dB scale floor are frozen.','tab:fresh',['Policy','Static','Motion stress'],rows)
    vec=pd.DataFrame(read_records(out/'vector_records.jsonl'));rows=[]
    for m in ['Batch-KG9x9','Sequential-CDF2','A-fixed2','Margin-fixed2']:
        d=vec[vec.method==m];rows.append([m.replace('9x9',r'$9\times9$'),f'{d.completed_kbit.mean():.2f}',f'{d.outage.sum():.0f}/128'])
    table(tex/'r2_vector.tex','Two distinct coordinates, fixed $2\\tau$ overhead: 128 paired synthetic frames. Sequential methods choose the second coordinate after the first observation.','tab:vector',['Policy','Completed (kbit)','Outages'],rows)
    summary=[]
    for filename in ['vector_records.jsonl','calibration_records.jsonl','fresh_calibration_records.jsonl','kg_timing_records.jsonl','reservation_records.jsonl']:
        df=pd.DataFrame(read_records(out/filename));df['scenario']=df.get('scenario','default')
        for (sc,m),d in df.groupby(['scenario','method']):
            n=len(d);sem=float(stats.sem(d.completed_kbit));half=stats.t.ppf(.975,n-1)*sem;k=int(d.outage.sum());lo,hi=wilson(k,n)
            row=dict(source=filename,scenario=sc,method=m,n=n,completed=float(d.completed_kbit.mean()),ci_low=float(d.completed_kbit.mean()-half),ci_high=float(d.completed_kbit.mean()+half),
                     outages=k,outage_low=lo,outage_high=hi,reports=float(d.probes.mean()),mean_ms=1000*float(d.wall_s.mean()),p95_ms=1000*float(d.wall_s.quantile(.95)))
            if 'deadline_overrun' in d:row['overruns']=int(d.deadline_overrun.sum())
            if 'cpu_s' in d:row['cpu_ms']=1000*float(d.cpu_s.mean())
            summary.append(row)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    x=json.loads((out/'statistics.json').read_text());fresh_pairs=[]
    for kind in ['static','motion']:
        d=fresh[fresh.scenario=='munich-'+kind].pivot(index='seed',columns='method',values='completed_kbit')
        for base in ['Safe-IVP','Regularized-NR']:
            r=paired(d['Regularized-IVP'],d[base]);r.update(scenario=kind,baseline=base);fresh_pairs.append(r)
    x['fresh_pairs']=fresh_pairs
    kg=pd.DataFrame(read_records(out/'kg_timing_records.jsonl'));d=kg.pivot(index='seed',columns='method',values='completed_kbit')
    x['new_kg_difference']=paired(d['C-IVP'],d['GH-KG'])
    conv=json.loads((out/'vector_convergence.json').read_text())['records']
    regret=np.array([r['best_q17']-r['q17_at_pair9'] for r in conv]);gap=np.array([abs(r['q9']-r['q17_at_pair9']) for r in conv])
    x['vector_order_sensitivity']=dict(states=len(conv),changed_pairs=sum(r['ranking_changed'] for r in conv),mean_selected_value_difference=float(gap.mean()),max_selected_value_difference=float(gap.max()),
                                       mean_selection_regret_under_order17=float(regret.mean()),max_selection_regret_under_order17=float(regret.max()))
    x['new_policy_records']=sum(len(read_records(out/n)) for n in ['vector_records.jsonl','calibration_records.jsonl','fresh_calibration_records.jsonl','kg_timing_records.jsonl','reservation_records.jsonl'])
    dump(out/'analysis_summary.json',x)
    ss=pd.DataFrame(summary)
    def one(source,method,scenario='default'):
        return ss[(ss.source==source)&(ss.method==method)&(ss.scenario==scenario)].iloc[0]
    a=one('fresh_calibration_records.jsonl','Regularized-IVP','munich-static')
    b=one('fresh_calibration_records.jsonl','Regularized-NR','munich-static')
    c=one('fresh_calibration_records.jsonl','Safe-IVP','munich-static')
    d=next(r for r in fresh_pairs if r['scenario']=='static' and r['baseline']=='Regularized-NR')
    text=(f"An independently generated 256-position Munich block (geometry seed 967300) retains the original fit and calibration sets. "
          f"Regularized IVP completes {a.completed:.2f} kbit with {a.outages}/256 outages, versus {c.completed:.2f} for the original Safe-IVP "
          f"and {b.completed:.2f} for regularized no-report. The last comparison has paired difference {d['difference']:.2f} kbit, "
          f"CI $[{d['ci'][0]:.2f},{d['ci'][1]:.2f}]$. "
          "This confirms a useful efficiency repair on new static positions, without establishing conditional coverage or eliminating motion risk.\n")
    (tex/'r2_fresh_text.tex').write_text(text)
    a=one('kg_timing_records.jsonl','C-IVP');b=one('kg_timing_records.jsonl','GH-KG');m=one('kg_timing_records.jsonl','Margin-sensitivity')
    text=(f"On a separate, sequentially timed 128-frame $N=4$ block with randomized method order, C-IVP/17-point KG/margin sensitivity "
          f"average {a.mean_ms:.2f}/{b.mean_ms:.2f}/{m.mean_ms:.2f} ms wall time and {a.cpu_ms:.2f}/{b.cpu_ms:.2f}/{m.cpu_ms:.2f} ms CPU time. "
          "This comparison measures computational cost, not a new acquisition objective; it supplies no hardware-independent speed or deadline bound.\n")
    (tex/'r2_timing_text.tex').write_text(text)
    dev=json.loads((out/'reservation_protocol.json').read_text());pieces=[]
    for method in ['C-IVP','Margin-sensitivity','No-report']:
        row=one('reservation_records.jsonl',method);dd=dev['summary'][method]
        pieces.append(f"{label(method)}: {1000*dd['32']['reserve_s']:.2f}/{1000*dd['256']['reserve_s']:.2f} ms, {row.completed:.2f} kbit, {row.outages}/{int(row.overruns)} failures/overruns")
    text=("A new reservation study increases development to 256 disjoint frames and retains 512 further evaluation frames. "
          "Using the first 32 versus all 256 development frames gives the following per-round reservations, followed by held-out 20-ms service and total failure/overrun counts: "
          + '; '.join(pieces)+'. ')
    dd=dev['summary']['C-IVP'];lo,hi=np.array(dd['32']['bootstrap95'])*1000;ll,hh=np.array(dd['256']['bootstrap95'])*1000
    text+=f"C-IVP's development 95th-percentile bootstrap interval is $[{lo:.2f},{hi:.2f}]$ ms with 32 frames and $[{ll:.2f},{hh:.2f}]$ with 256. "
    text+='These descriptive bootstrap intervals require representative timing samples; host scheduling can shift the distribution. Larger calibration does not turn an empirical reservation into a worst-case bound.\n'
    (tex/'r2_reservation_text.tex').write_text(text)
    # Separate overestimation (numerical) and underestimation (misses/refinement).
    grid=pd.read_csv(out/'interval_grid.csv');d=grid[grid.grid==25];summ=json.loads((out/'interval_summary.json').read_text())['results']
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    f,ax=plt.subplots(1,2,figsize=(7.0,2.15));e=np.sort(d.gap.to_numpy());ax[0].plot(e,np.arange(1,len(e)+1)/len(e),color='#145da0')
    ax[0].set_xscale('symlog',linthresh=1e-6)
    ax[0].set(xlabel='Reference minus inner value (kbit)',ylabel='Empirical CDF',xlim=(-5e-7,.4))
    ax[0].set_xticks([0,1e-5,1e-3,1e-1],['0',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$']);ax[0].grid(alpha=.2)
    xx=np.array([r['grid'] for r in summ]);ax[1].plot(xx,[r['max_gap'] for r in summ],'o-',label='Maximum shortfall')
    ax[1].plot(xx,[r['mean_gap'] for r in summ],'s-',label='Mean shortfall');ax[1].set(xlabel='Innovation knots',ylabel='Value shortfall (kbit)',yscale='log',xticks=xx);ax[1].grid(alpha=.2);ax[1].legend(fontsize=7)
    f.tight_layout();f.savefig(fig/'value_accuracy.pdf',bbox_inches='tight');plt.close(f)
    print('Round-two tables and analysis regenerated; records',x['new_policy_records'])

if __name__=='__main__':main()
