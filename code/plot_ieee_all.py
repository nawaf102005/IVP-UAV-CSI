"""Regenerate all current manuscript/supplement figures, one panel per file.

Run: python code/plot_ieee_all.py --root .
Reads the frozen records and writes presentation artifacts only. No scientific
simulation or policy is rerun. Font: installed Times New Roman; optional path
IVP_TIMES_FONT_DIR. PDF embeds the font, PNG is 600 dpi, SVG keeps editable text.
"""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader,PdfWriter
import plot_round5
from ieee_figures import configure_ieee_style,make_panels,export_panels,PANEL_DATA

MAIN_ORDER=['system_model', 'q_accuracy_independent', 'aligned_frontier_gain', 'allocator_persistent_age', 'frontier_budget_6', 'frontier_budget_8', 'r5_03_breakeven_B', 'r5_06_raytracing_D']
SUPP_ORDER=['catalogue_pareto_A', 'catalogue_pareto_B', 'r5_05_risk_budget_A', 'r5_05_risk_budget_B', 'r5_10_allocator_A', 'r5_07_approximation_A', 'r5_07_approximation_B', 'r5_07_approximation_C', 'r5_08_search_B', 'r5_08_search_C', 'r5_09_pilots_A', 'r5_09_pilots_B', 'value_accuracy_A', 'r5_01_scaling_A', 'r5_01_scaling_C', 'r5_02_equivalence_B', 'r5_03_breakeven_A', 'r5_03_breakeven_C', 'r5_04_mismatch_A', 'r5_04_mismatch_B', 'r5_04_mismatch_D', 'r5_06_raytracing_A', 'r5_06_raytracing_C', 'value_accuracy_B', 'r5_10_allocator_B', 'r5_08_search_A', 'r5_02_equivalence_A', 'r5_04_mismatch_C', 'r5_06_raytracing_B', 'cost_tolerance_margin', 'cost_support_map', 'q_accuracy', 'mobility_radio', 'mobility_charged', 'cost_tolerance_map', 'margin_duration_boundaries', 'size_duration_boundaries', 'frontier_cost_break_even', 'frontier_cost_slice', 'dense_cost_refinement', 'frontier_dense_replay', 'r5_01_scaling_B', 'persistent_frontier_gain', 'frontier_planning_resolution']

def main(root):
 root=Path(root).resolve();out=root/'figures_r5';out.mkdir(exist_ok=True)
 plot_round5.main(root)
 configure_ieee_style();catalogue=json.loads((out/'figure_catalogue.json').read_text());extra=[]
 # Scalar accuracy: the original two plotting blocks and the same stored inputs.
 grid=pd.read_csv(root/'round2_results/interval_grid.csv');d=grid[grid.grid==25]
 summ=json.loads((root/'round2_results/interval_summary.json').read_text())['results']
 f,ax=make_panels(1,2);e=np.sort(d.gap.to_numpy());ax[0].plot(e,np.arange(1,len(e)+1)/len(e),color='#145da0')
 ax[0].set_xscale('symlog',linthresh=1e-6)
 ax[0].set(xlabel='Reference minus inner value (kbit)',ylabel='Empirical CDF',xlim=(-5e-7,.4))
 ax[0].set_xticks([0,1e-5,1e-3,1e-1],['0',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$'])
 xx=np.array([r['grid'] for r in summ]);ax[1].plot(xx,[r['max_gap'] for r in summ],'o-',label='Maximum shortfall')
 ax[1].plot(xx,[r['mean_gap'] for r in summ],'s-',label='Mean shortfall')
 ax[1].set(xlabel='Innovation knots',ylabel='Value shortfall (kbit)',yscale='log',xticks=xx)
 catalogue.extend(export_panels(f,'value_accuracy',out,None,'round2_results/interval_grid.csv; interval_summary.json'))
 extra += [dict(figure='value_accuracy_A',x=float(x),y=float(y),series='Empirical CDF') for x,y in zip(e,np.arange(1,len(e)+1)/len(e))]
 for series,key in [('Maximum shortfall','max_gap'),('Mean shortfall','mean_gap')]:
  extra += [dict(figure='value_accuracy_B',x=float(r['grid']),y=float(r[key]),series=series) for r in summ]
 # Catalogue tradeoff: unchanged frozen timing-summary values.
 ss=pd.read_csv(root/'revision_results/summary.csv');f,axs=make_panels(1,2)
 colors=['#145da0','#ca6f1e','#6b5b95','#2c8a6c','#af4852']
 for i,n in enumerate([6,8]):
  for j,m in enumerate(['C-IVP','C-IVP-C16','C-IVP-C32','C-IVP-S2','Margin-sensitivity']):
   r=ss[(ss.source=='timing')&(ss.scenario==f'N-{n}')&(ss.method==m)].iloc[0]
   label=['Full C-IVP','C16','C32','S2','Margin sensitivity'][j]
   axs[i].scatter(r.mean_ms,r.completed,color=colors[j],s=30,label=label)
   extra.append(dict(figure='catalogue_pareto_'+chr(65+i),x=float(r.mean_ms),y=float(r.completed),series=label))
  axs[i].set_title(f'N = {n}');axs[i].set_xlabel('Mean online wall time (ms)')
  axs[i].set_ylabel('Completed payload (kbit/frame)')
 catalogue.extend(export_panels(f,'catalogue_pareto',out,None,'revision_results/summary.csv: timing rows'))
 # The existing causal network schematic, with typography changed only.
 fig,ax=plt.subplots(figsize=(3.5,3.0));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
 ax.plot([.6,9.4],[7.3,7.3],color='#c1d8e4',lw=15,alpha=.5)
 for i,(xx,yy) in enumerate([(2,7.3),(7.8,7.3)]):
  ax.scatter([xx],[yy],s=130,marker='x',color='#176e91',lw=2);ax.text(xx,yy+.8,f'UAV {i+1}',ha='center')
  ax.scatter([xx-.6],[4.9],s=90,marker='^',color='#394b59');ax.text(xx-.6,4.3,f'BS {i+1}',ha='center')
  ax.add_patch(FancyArrowPatch((xx-.5,5.1),(xx-.05,7.05),arrowstyle='<->',mutation_scale=10,color='#176e91'))
 ax.add_patch(FancyArrowPatch((1.5,5.1),(7.55,7.05),arrowstyle='->',mutation_scale=9,color='#a76f4a',linestyle='--'))
 ax.add_patch(FancyArrowPatch((7.2,5.1),(2.2,7.05),arrowstyle='->',mutation_scale=9,color='#a76f4a',linestyle='--'))
 ax.text(5,9.4,'Directional BS antennas / shared spectrum',ha='center',fontsize=9.5)
 for x0,w,color,label in [(.5,2.2,'#d6dbe1','CSI reports'),(2.7,4.9,'#c0dce8','UL data'),(7.6,1.9,'#d5e3cf','DL control')]:
  ax.add_patch(Rectangle((x0,1.9),w,.9,facecolor=color,edgecolor='white'));ax.text(x0+w/2,2.35,label,ha='center',fontsize=9.5)
 ax.annotate('',xy=(9.5,1.5),xytext=(.5,1.5),arrowprops={'arrowstyle':'<->','color':'#444'})
 ax.text(5,.75,'Fixed deadline T: reporting reduces data time',ha='center',fontsize=9.5)
 ax.text(5,3.45,'Posterior -> probe or stop -> certify -> transmit',ha='center',fontsize=9.5)
 fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
 for ext in ['pdf','svg','png']:fig.savefig(out/('system_model.'+ext),dpi=600)
 plt.close(fig)
 catalogue.append(dict(name='system_model',parent='system_model',panel=None,title='System model',
  caption='Local corridor cluster and causal frame protocol. Dashed links carry reuse interference. Reports precede both data phases; no policy observes future channels.',
  source='Original analyze_results.py schematic',font='Times New Roman',font_size_pt=9.5,width_in=3.5,height_in=3.0))
 from plot_mobility_revision import main as plot_additional
 plot_additional(root)
 catalogue.extend(json.loads((out/'additional_catalogue.json').read_text()))
 from plot_platform_revision import main as plot_platform
 plot_platform(root)
 catalogue.extend(json.loads((out/'platform_catalogue.json').read_text()))
 from plot_alignment_revision import main as plot_alignment
 plot_alignment(root)
 catalogue.extend(json.loads((out/'alignment_catalogue.json').read_text()))
 from render_final_replay import generate as render_replay
 catalogue.append(render_replay(root))
 from render_prospective import generate as prospective
 catalogue.extend(prospective(root))
 from render_size_confirmation import generate as size_tables
 size_tables(root)
 from build_evidence_tables import main as build_evidence
 build_evidence(root)
 from render_break_even import generate as break_even_outputs
 catalogue.extend(break_even_outputs(root))
 order=MAIN_ORDER+SUPP_ORDER;byname={r['name']:r for r in catalogue};assert set(order)==set(byname)
 catalogue=[byname[name] for name in order]
 writer=PdfWriter()
 for name in order:writer.append(PdfReader(out/(name+'.pdf')))
 with (out/'IVP_Figures_Separated.pdf').open('wb') as f:writer.write(f)
 (out/'figure_catalogue.json').write_text(json.dumps(catalogue,indent=2)+'\n')
 with (out/'FIGURE_INDEX.csv').open('w',newline='') as idx:
  writer=csv.DictWriter(idx,fieldnames=['document','figure_number','file','old_group','old_panel','title'])
  writer.writeheader()
  for doc,names in [('main',MAIN_ORDER),('supplement',SUPP_ORDER)]:
   for i,stem in enumerate(names,1):
    item=byname[stem]
    writer.writerow(dict(document=doc,figure_number=('S' if doc=='supplement' else '')+str(i),file=stem,
                         old_group=item['parent'],old_panel=item['panel'] or '',title=item['title']))
 (out/'panel_data.json').write_text(json.dumps(PANEL_DATA,sort_keys=True)+'\n')
 pd.DataFrame(extra).to_csv(out/'extra_curve_points.csv',index=False)
 (out/'CAPTIONS.md').write_text('\n\n'.join('## '+r['name']+'\n\n'+r['caption']+'\n\nSource: `'+r['source']+'`' for r in catalogue)+'\n')
 # Current figure directories contain only current independent PDF assets.
 for dest in [root/'figures',root/'manuscript/figures']:
  dest.mkdir(parents=True,exist_ok=True)
  for p in dest.glob('*.pdf'):p.unlink()
  for name in order:shutil.copyfile(out/(name+'.pdf'),dest/(name+'.pdf'))
 # Remove superseded multi-panel presentation exports from the current atlas.
 keep={name+'.'+ext for name in order for ext in ['pdf','svg','png']}
 keep.update(['IVP_Figures_Separated.pdf'])
 for p in out.iterdir():
  if p.suffix.lower() in ['.pdf','.svg','.png'] and p.name not in keep:p.unlink()
 from refresh_captions import main as refresh_captions
 refresh_captions(root)
 print(json.dumps(dict(independent_figures=len(order),main_figures=len(MAIN_ORDER),supplement_figures=len(SUPP_ORDER),font='Times New Roman',full_axes_boxes=True,raw_results_unchanged=True),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();main(a.root)
