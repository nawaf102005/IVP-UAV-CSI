"""Twenty-eight independent scientific panels; PDF, SVG, 600-dpi PNG and CSV.

Pointwise intervals except the stated TOST/break-even families. Lines connect
actual tested settings; no interpolation is used for threshold inference.
"""
import argparse,json,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from r5_statistics import mean_ci,wilson,tost,holm
from ieee_figures import configure_ieee_style,make_panels,export_panels,style_axis,PANEL_DATA
C={'CDF':'#0072B2','KG17':'#D55E00','CDF-S2':'#009E73','KG17-S2':'#CC79A7','Margin':'#8064A2','NR':'#333333','Matched-CDF':'#009E73','Matched-NR':'#999999','C-IVP':'#0072B2','A-stop':'#D55E00','No-report':'#333333','OMA':'#009E73','Safe-IVP':'#009E73','Safe-NR':'#CC79A7'}
MARK=['o','s','^','v','D','x'];POINTS=[];CAT=[]
def read(root,path):return pd.DataFrame([json.loads(x) for x in (root/path).read_text().splitlines()])
def raw(root,path):return [json.loads(x) for x in (root/path).read_text().splitlines()]
def stat(df,key,kind='mean',alpha=.05):
    return wilson(int(df[key].sum()),len(df),alpha) if kind=='risk' else mean_ci(df[key],alpha)
def point(fig,panel,series,x,r,unit,source):POINTS.append(dict(figure=fig,panel=panel,series=series,x=x,unit=unit,source=source,**r))
def curve(ax,df,xkey,key,methods,fig,panel,source,mult=1,kind='mean',ylabel='',alpha=.05):
    for j,m in enumerate(methods):
        rows=[]
        for x,d in df[df.method==m].groupby(xkey,sort=True):
            r=stat(d,key,kind,alpha);r={k:v*mult if k in ['mean','lo','hi','se'] else v for k,v in r.items()};rows.append((x,r));point(fig,panel,m,x,r,ylabel,source)
        x=[a for a,b in rows];y=np.array([b['mean'] for a,b in rows]);lo=np.array([b['lo'] for a,b in rows]);hi=np.array([b['hi'] for a,b in rows])
        ax.errorbar(x,y,yerr=[np.maximum(0,y-lo),np.maximum(0,hi-y)],label=m,color=C.get(m),marker=MARK[j%6],ls='--' if 'Matched' in m or 'S2' in m else '-',capsize=2)
    ax.set_ylabel(ylabel)
def style(axs):
    for ax in np.asarray(axs).ravel():style_axis(ax)
def save(fig,name,title,caption,out,atlas,source):
    CAT.extend(export_panels(fig,name,out,atlas,source))
def legend(fig,ax,cols=4):
    # Each independent panel receives its own legend at export.
    pass

def main(root,results=None,artifact_root=None):
    POINTS.clear();CAT.clear();out=(artifact_root or root)/'figures_r5';out.mkdir(exist_ok=True,parents=True);res=results or root/'round5_results';summary={}
    configure_ieee_style()
    scale=read(res,'scale_records.jsonl');lat=read(res,'latency_records.jsonl');mis=read(res,'mismatch_records.jsonl')
    assert (len(scale),len(lat),len(mis))==(4608,5376,22528)
    summary['scale']=[]
    for (n,m),d in scale.groupby(['n','method']):summary['scale'].append(dict(n=int(n),method=m,service=mean_ci(d.completed_kbit),wall_ms=mean_ci(d.wall_s*1000),outage=wilson(int(d.outage.sum()),len(d))))
    tests=[]
    for n,d in scale.groupby('n'):
        p=d.pivot(index='seed',columns='method',values='completed_kbit')
        for suffix in ['', '-S2']:
            t=tost(p['CDF'+suffix]-p['KG17'+suffix]);t.update(n_pairs=int(n),family='full' if not suffix else 'S2');tests.append(t)
    for t,adj in zip(tests,holm([t['p'] for t in tests])):t.update(p_holm=float(adj),equivalent=bool(adj<.05))
    summary['tost']=tests
    lat['rate']=lat.completed_kbit/lat.duration_ms;lat['reserved_fraction']=100*lat.reserved_s/(lat.duration_ms/1000);gains=[]
    for T,d in lat.groupby('duration_ms'):
        p=d.pivot(index='seed',columns='method',values='rate')
        for m in ['CDF','Margin']:
            q=mean_ci(p[m]-p.NR,.05/14);q.update(duration_ms=int(T),method=m);gains.append(q)
    summary['break_even']=dict(gains=gains,first_positive_tested={m:next((q['duration_ms'] for q in gains if q['method']==m and q['lo']>0),None) for m in ['CDF','Margin']},reservations=json.loads((res/'reservations.json').read_text())['reserves'])
    summary['mismatch']=[dict(dimension=dim,true_parameter=float(x),method=m,service=mean_ci(d.completed_kbit),outage=wilson(int(d.outage.sum()),len(d))) for (dim,x,m),d in mis.groupby(['dimension','true_parameter','method'])]
    with PdfPages(out/'IVP_Figures.pdf') as atlas:
        name='r5_01_scaling';fig,axs=make_panels(1,3,figsize=(10.2,3.15));style(axs);methods=['CDF','KG17','CDF-S2','KG17-S2','Margin','NR']
        curve(axs[0],scale,'n','completed_kbit',methods,name,'a','R5 scale',ylabel='Completed service (kbit/frame)');curve(axs[1],scale,'n','wall_s',methods,name,'b','R5 scale',mult=1000,ylabel='Mean wall time (ms)');axs[1].set_yscale('log')
        for j,m in enumerate(methods):
            x=np.sort(scale[(scale.n==8)&(scale.method==m)].wall_s.to_numpy()*1000);y=np.arange(1,len(x)+1)/len(x);axs[2].step(x,y,where='post',color=C[m],label=m)
            for xx,yy in zip(x,y):point(name,'c',m,float(xx),dict(mean=float(yy)),'Empirical CDF','R5 scale N=8')
        for ax in axs[:2]:ax.set_xlabel('Number of UAV pairs N');ax.set_xticks([4,6,8])
        axs[2].set(xscale='log',xlabel='N=8 decision wall time (ms)',ylabel='Empirical CDF');legend(fig,axs[0],6)
        save(fig,name,'Network size, service and computation','256 paired frames per N; six policies. Bars: pointwise 95% t intervals. Radio-only service; runtime excludes scenario generation and compilation.',out,atlas,'round5_results/scale_records.jsonl')
        name='r5_02_equivalence';fig,axs=make_panels(1,2,figsize=(7.2,3.05));style(axs)
        for j,label in enumerate(['full','S2']):
            a=[t for t in tests if t['family']==label];x=np.array([t['n_pairs'] for t in a])+(-.07 if j==0 else .07);y=np.array([t['mean'] for t in a]);lo=np.array([t['lo'] for t in a]);hi=np.array([t['hi'] for t in a]);co=C['CDF' if j==0 else 'CDF-S2']
            axs[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt=MARK[j]+'-',color=co,label=label,capsize=3)
            for xx,t in zip(x,a):
                if t['equivalent']:axs[0].annotate('*',(xx,t['hi']),xytext=(0,3),textcoords='offset points',ha='center')
                point(name,'a',label,t['n_pairs'],t,'CDF minus KG17 (kbit)','R5 scale')
            xx=[];yy=[];hh=[];ll=[];rng=np.random.default_rng(995111+j)
            for n,d in scale.groupby('n'):
                p=d.pivot(index='seed',columns='method',values='wall_s');s='' if j==0 else '-S2';a0=p['CDF'+s].to_numpy();b=p['KG17'+s].to_numpy();idx=rng.integers(len(a0),size=(2000,len(a0)));rat=b[idx].mean(1)/a0[idx].mean(1);q=dict(mean=float(b.mean()/a0.mean()),lo=float(np.quantile(rat,.025)),hi=float(np.quantile(rat,.975)),n=len(a0));xx.append(n);yy.append(q['mean']);ll.append(q['lo']);hh.append(q['hi']);point(name,'b',label,n,q,'KG17/CDF runtime ratio','R5 scale')
            axs[1].errorbar(xx,yy,yerr=[np.array(yy)-ll,np.array(hh)-yy],fmt=MARK[j]+'-',color=co,label=label,capsize=3)
        axs[0].axhspan(-.5,.5,color='#e0efdf',alpha=.65);axs[0].axhline(0,color='grey',lw=.7);axs[0].axhline(.5,color='grey',ls='--');axs[0].axhline(-.5,color='grey',ls='--')
        axs[0].set_ylabel('CDF minus KG17 (kbit/frame)');axs[1].set_ylabel('KG17 / CDF mean runtime');axs[1].axhline(1,color='grey',ls='--')
        for ax in axs:ax.set_xlabel('Number of UAV pairs N');ax.set_xticks([4,6,8])
        legend(fig,axs[0],2);save(fig,name,'Equivalence and runtime ratio','Left: pointwise 90% paired-t intervals; preset equivalence margin +/-0.5 kbit; * means Holm-adjusted TOST passes across six tests. Right: paired-bootstrap 95% runtime-ratio intervals (2,000 resamples).',out,atlas,'R5 scale; protocol and r5_statistics.py')
        name='r5_03_breakeven';fig,axs=make_panels(1,3,figsize=(10.2,3.15));style(axs)
        curve(axs[0],lat,'duration_ms','rate',['CDF','Margin','NR'],name,'a','R5 latency',ylabel='Net delivered rate (Mbit/s)')
        for j,m in enumerate(['CDF','Margin']):
            rows=[r for r in gains if r['method']==m];x=[r['duration_ms'] for r in rows];y=np.array([r['mean'] for r in rows]);lo=np.array([r['lo'] for r in rows]);hi=np.array([r['hi'] for r in rows]);axs[1].errorbar(x,y,yerr=[y-lo,hi-y],fmt=MARK[j]+'-',color=C[m],capsize=3,label=m)
            for x,r in zip(x,rows):point(name,'b',m,x,r,'Rate gain over NR (Mbit/s)','R5 latency; simultaneous family of 14')
        axs[1].axhline(0,color='black',lw=.8);axs[1].set_ylabel('Rate gain over NR (Mbit/s)')
        curve(axs[2],lat,'duration_ms','reserved_fraction',['CDF','Margin','NR'],name,'c','R5 latency',ylabel='Mean reserved compute / frame (%)')
        for ax in axs:ax.set_xlabel('Frame duration T (ms)')
        legend(fig,axs[0],3);save(fig,name,'Measured break-even with charged computation','256 held-out paired frames per T; fixed reservations from 128 development frames. Active overruns earn zero. Panel b uses Bonferroni simultaneous 95% intervals over 14 contrasts. Other bars are pointwise. Payload scales with T.',out,atlas,'round5_results/latency_records.jsonl; reservations.json')
        name='r5_04_mismatch';fig,axs=make_panels(2,2,figsize=(7.2,5.05));style(axs);methods=['CDF','NR','Matched-CDF','Matched-NR']
        for j,dim in enumerate(['reciprocity','shadow_db']):
            d=mis[mis.dimension==dim];curve(axs[0,j],d,'true_parameter','completed_kbit',methods,name,'service '+dim,'R5 mismatch',ylabel='Completed service (kbit/frame)');curve(axs[1,j],d,'true_parameter','outage',methods,name,'outage '+dim,'R5 mismatch',kind='risk',mult=100,ylabel='Any-task outage (%)')
            axs[1,j].axhline(5,color='black',ls=':',lw=1);nom=.9 if j==0 else 3
            for ax in axs[:,j]:ax.axvline(nom,color='grey',ls=':',lw=.7);ax.set_xlabel('True UL-DL correlation' if j==0 else 'True shadowing SD (dB)')
        legend(fig,axs[0,0],4);save(fig,name,'Model sensitivity and matched-covariance diagnostic','512 paired frames per setting. Assumed correlation 0.9 and shadowing SD 3 dB; one changes at a time. Matched methods use the true covariance as an oracle, not an estimator. 95% t/Wilson bars are pointwise. Dotted horizontal line: 5% outage target.',out,atlas,'round5_results/mismatch_records.jsonl')
        old=read(root,'revision_results/revision_records.jsonl');name='r5_05_risk_budget';fig,axs=make_panels(1,2,figsize=(7.2,3.1));style(axs)
        d=old[old.scenario.str.startswith('delta-')].copy();d['delta']=d.scenario.str.replace('delta-','',regex=False).astype(float)
        for j,m in enumerate(['C-IVP','A-stop','No-report','OMA']):
            rows=[]
            for delta,q in d[d.method==m].groupby('delta'):
                x=wilson(int(q.outage.sum()),len(q));y=mean_ci(q.completed_kbit);rows.append((x,y));point(name,'a',m,delta,{**y,'outage':x['mean'],'outage_lo':x['lo'],'outage_hi':x['hi']},'Service kbit / outage proportion','R1 risk sweep')
            xx=np.array([x['mean']*100 for x,y in rows]);yy=np.array([y['mean'] for x,y in rows]);axs[0].errorbar(xx,yy,xerr=[xx-[x['lo']*100 for x,y in rows],[x['hi']*100 for x,y in rows]-xx],yerr=[yy-[y['lo'] for x,y in rows],[y['hi'] for x,y in rows]-yy],fmt=MARK[j]+'-',color=C[m],label=m,capsize=2)
        d=old[old.scenario.str.startswith('cap-')].copy();d['cap']=d.scenario.str.replace('cap-','',regex=False).astype(int)
        curve(axs[1],d,'cap','completed_kbit',['C-IVP','A-stop'],name,'b','R1 report cap',ylabel='Completed service (kbit/frame)');axs[0].set(xlabel='Observed any-task outage (%)',ylabel='Completed service (kbit/frame)');axs[1].set_xlabel('Maximum number of reports M');legend(fig,axs[0],4)
        save(fig,name,'Risk-service frontier and report budget','Frozen expanded-study data: 512 frames per point. Left sweeps design delta=0.01,0.025,0.05,0.10,0.20,0.40. Horizontal Wilson and vertical t intervals are pointwise. Right changes the report cap with paired seeds.',out,atlas,'revision_results/revision_records.jsonl')
        name='r5_06_raytracing';ray=pd.concat([read(root,'revision_results/ray_records.jsonl'),read(root,'revision_results/conformal_records.jsonl')]);fig,axs=make_panels(2,2,figsize=(7.2,5.05));style(axs);methods=['C-IVP','No-report','Safe-IVP','Safe-NR'];scenes=['canyon_low','canyon_high','munich']
        for row,kind in enumerate(['static','motion']):
            for j,m in enumerate(methods):
                for col,key in enumerate(['completed_kbit','outage']):
                    x=[];y=[];lo=[];hi=[]
                    for i,scene in enumerate(scenes):
                        d=ray[(ray.scenario==scene+'-'+kind)&(ray.method==m)];r=stat(d,key,'risk' if col else 'mean');mul=100 if col else 1;x.append(i+(j-1.5)*.09);y.append(r['mean']*mul);lo.append(r['lo']*mul);hi.append(r['hi']*mul);point(name,kind+' '+key,m,scene,r,'Outage proportion' if col else 'kbit/frame','Frozen R1 Sionna channels')
                    axs[row,col].errorbar(x,y,yerr=[np.array(y)-lo,np.array(hi)-y],fmt=MARK[j],color=C[m],label=m,capsize=2)
            for col in range(2):axs[row,col].set_xticks(range(3),['Canyon 35 m','Canyon 80 m','Munich']);axs[row,col].set_ylabel(('Static' if row==0 else 'Endpoint motion')+('\nservice (kbit/frame)' if col==0 else '\noutage (%)'))
            axs[row,1].axhline(5,color='black',ls=':',lw=.8)
        legend(fig,axs[0,0],4);save(fig,name,'Ray-traced propagation and endpoint aging','Existing Sionna RT 2.2.0 study: 128 training, 64 calibration, 256 test frames per condition. Static uses incoherent path sums; motion uses coherent sums and a 0.4-m endpoint displacement. Scene points are not joined. Simulation, not measured flights.',out,atlas,'revision_results/ray_records.jsonl; conformal_records.jsonl; ray_data/')
        name='r5_07_approximation';fig,axs=make_panels(1,3,figsize=(10.2,3.1));style(axs);grid=json.loads((root/'round2_results/interval_summary.json').read_text())['results'];v=raw(root,'round3_results/vector_bound_records.jsonl')
        for key,label,color in [('mean_gap','Mean','#0072B2'),('max_gap','Maximum','#D55E00')]:
            axs[0].plot([r['grid'] for r in grid],[r[key] for r in grid],'o-',color=color,label=label)
            for r in grid:point(name,'a',label,r['grid'],dict(mean=r[key]),'Reference minus inner (kbit)','R2 interval audit')
        for label,fun,color in [('Mean',np.mean,'#0072B2'),('Maximum',np.max,'#D55E00')]:
            x=[];y=[]
            for budget in [64,256,1024,4096]:
                q=[next(c for c in r['checkpoints'] if c['budget']==budget) for r in v];val=float(fun([a['gap'] for a in q]));x.append(budget);y.append(val);point(name,'b',label,budget,dict(mean=val,n=len(q)),'Upper minus lower (kbit)','R3 vector boxes')
            axs[1].plot(x,y,'o-',color=color,label=label)
        x=[];y=[]
        for budget in [64,256,1024,4096]:
            vals=[next(c for c in r['checkpoints'] if c['budget']==budget)['wall_s']*1000 for r in v];x.append(budget);y.append(np.mean(vals));point(name,'c','Mean wall',budget,mean_ci(vals),'ms','R3 vector boxes')
        axs[2].plot(x,y,'o-',color=C['CDF']);axs[0].set(xlabel='Scalar innovation knots',ylabel='Value shortfall (kbit)',yscale='log');axs[1].set(xlabel='Vector boxes',ylabel='Value enclosure gap (kbit)',xscale='log',yscale='log');axs[2].set(xlabel='Vector boxes',ylabel='Mean computation time (ms)',xscale='log',yscale='log');legend(fig,axs[0],2)
        save(fig,name,'Approximation error and the cost of certification','Scalar audit: 1,024 reports / 22,377 reference-detected intervals; 25 knots miss one interval. Vector audit: 32 two-component states. Mean/max curves are descriptive, not universal error bounds. Vector gaps bound the surrogate value.',out,atlas,'round2_results/interval_summary.json; round3_results/vector_bound_records.jsonl')
        name='r5_08_search';b=read(root,'round3_results/branch_records.jsonl');fig,axs=make_panels(1,3,figsize=(10.2,3.1));style(axs)
        for j,budget in enumerate([64,256,1024]):
            d=b[b.node_budget==budget]
            for col,(key,mul,ylab) in enumerate([('ratio',1,'Mean lower / upper bound'),('gap',1,'Mean unresolved gap (kbit)'),('wall_s',1000,'Mean wall time (ms)')]):
                x=[];y=[]
                for n,q in d.groupby('n'):r=mean_ci(q[key]*mul);x.append(n);y.append(r['mean']);point(name,str(col),str(budget),n,r,ylab,'R3 branch audit')
                axs[col].plot(x,y,MARK[j]+'-',label=str(budget)+' nodes');axs[col].set_xlabel('Number of UAV pairs N');axs[col].set_ylabel(ylab)
        axs[0].set_ylim(0,1.05);axs[2].set_yscale('log');legend(fig,axs[0],3);save(fig,name,'Interrupted search: scale, residual gap and cost','32 fixed states per N and node budget. Curves show sample means. The ratio is an instance-specific certificate, not a uniform approximation factor; large-N gaps remain substantial.',out,atlas,'round3_results/branch_records.jsonl')
        name='r5_09_pilots';fig,axs=make_panels(1,2,figsize=(7.2,3.1));style(axs);obs=raw(root,'round4_results/pilot_observations.jsonl');va=raw(root,'round4_results/pilot_value_audit.jsonl')
        for j,L in enumerate([16,64]):
            x=np.sort(np.array([o['standardized_error'] for r in obs if r['length']==L for o in r['batches']]).ravel());y=np.arange(1,len(x)+1)/len(x);axs[0].step(x,y,where='post',label=f'L={L}')
            for xx,yy in zip(x,y):point(name,'a',str(L),float(xx),dict(mean=float(yy)),'Empirical CDF','R4 physical observations')
            x=np.sort([abs(r['cdf17']-r['gh33']) for r in va if r['length']==L]);y=np.arange(1,len(x)+1)/len(x);axs[1].step(x,y,where='post',label=f'L={L}')
            for xx,yy in zip(x,y):point(name,'b',str(L),float(xx),dict(mean=float(yy)),'Empirical CDF','R4 16-state selected-batch audit')
        z=np.linspace(-3.5,3.5,300);axs[0].plot(z,stats.norm.cdf(z),'k--',label='Standard Gaussian');axs[0].set(xlim=(-3.5,3.5),xlabel='Standardized log-power error',ylabel='Empirical CDF');axs[1].set(xlabel='|CDF17 - GH33| (kbit)',ylabel='Empirical CDF');legend(fig,axs[0],3)
        save(fig,name,'Physical pilot model and surrogate quadrature discrepancy','Two complex orthogonal pilots; log-power Gaussian conditioning is approximate. Left: 2,048 residuals per length, clustered within 256 frames. Right: 16 prespecified selected batches per length. GH33 is a comparator, not exact truth.',out,atlas,'round4_results/pilot_observations.jsonl; pilot_value_audit.jsonl')
        name='r5_10_allocator';a=pd.concat([read(root,'revision_results/chance_records.jsonl'),read(root,'revision_results/frontier_records.jsonl'),read(root,'revision_results/hybrid_records.jsonl')]);a=a[a.method.isin(['No-report','C-IVP','Validated-frontier','IVP+frontier'])];p=a.pivot(index='seed',columns='method',values='completed_kbit');fig,axs=make_panels(1,2,figsize=(7.2,3.15));style(axs)
        methods=['No-report','C-IVP','Validated-frontier','IVP+frontier'];labels=['NR','CDF','Frontier\nNR','CDF +\nfrontier']
        for i,m in enumerate(methods):
            r=mean_ci(p[m]);axs[0].errorbar(i,r['mean'],yerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',capsize=3,color=C['CDF'] if 'IVP' in m else C['NR']);point(name,'a',m,i,r,'kbit/frame','R1 matched allocator block')
        axs[0].set_xticks(range(4),labels);axs[0].set_ylabel('Completed service (kbit/frame)');contrasts=[('C-IVP','No-report','Original allocator'),('IVP+frontier','Validated-frontier','Frontier allocator')];summary['allocator']=[]
        for i,(m,b,label) in enumerate(contrasts):
            r=mean_ci(p[m]-p[b]);axs[1].errorbar(i,r['mean'],yerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',capsize=4,color=C['CDF']);point(name,'b',label,i,r,'Paired gain (kbit/frame)','R1 matched allocator block');summary['allocator'].append(dict(label=label,**r))
        axs[1].set_xticks([0,1],['Original','Frontier']);axs[1].set_ylabel('Paired acquisition gain (kbit/frame)');axs[1].axhline(0,color='grey',ls='--')
        save(fig,name,'Separating acquisition gain from terminal allocation','Same 128 frames for all four methods; pointwise 95% t intervals. The finite validated frontier has a separate statistical validation guarantee. The gain shrinks after terminal allocation improves.',out,atlas,'revision_results/chance_records.jsonl; frontier_records.jsonl; hybrid_records.jsonl')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'figure_catalogue.json').write_text(json.dumps(CAT,indent=2)+'\n');pd.DataFrame(POINTS).to_csv(out/'curve_points.csv',index=False)
    (out/'panel_data.json').write_text(json.dumps(PANEL_DATA,sort_keys=True)+'\n')
    (out/'CAPTIONS.md').write_text('\n\n'.join('## '+x['name']+' — '+x['title']+'\n\n'+x['caption']+'\n\nSource: `'+x['source']+'`' for x in CAT)+'\n')
    print(json.dumps(dict(tost=tests,break_even=summary['break_even'],figures=len(CAT),curve_points=len(POINTS)),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results',type=Path);p.add_argument('--artifact-root',type=Path);a=p.parse_args();main(Path(a.root),a.results,a.artifact_root)
