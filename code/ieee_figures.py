"""Independent IEEE-size plots with embedded Times New Roman and four spines.

Only presentation is changed. Raw line and error-bar coordinates are recorded
before styling so the independent-panel exports can be compared with the
original plotting code. Times New Roman must be installed, or supplied using
IVP_TIMES_FONT_DIR. Font binaries are not redistributed with this project.
"""
import os, math, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PANEL_DATA = {}
PANEL_INFO = {
'r5_01_scaling': [
 ('Completed service versus cluster size', 'Completed service on 256 paired frames per cluster size. Report airtime is charged; computation is timed separately. Bars: pointwise 95% t intervals.'),
 ('Decision time versus cluster size', 'Whole-decision wall time on 256 paired frames per cluster size. Compilation and scenario generation are excluded; bars are pointwise 95% t intervals.'),
 ('Runtime distribution at eight pairs', 'Empirical wall-time distributions at N=8 on the same 256 paired frames. Compilation and scenario generation are excluded.')],
'r5_02_equivalence': [
 ('Mean-service equivalence', 'CDF-minus-KG17 service differences with pointwise 90% paired-t intervals and the prespecified +/-0.5-kbit region. Stars denote Holm-adjusted TOST rejection across six tests. Identical N=6 S2 differences are not t-tested.'),
 ('Runtime ratio', 'Ratio of mean KG17 to CDF runtimes. Intervals are paired-bootstrap 95% intervals from 2,000 resamples.')],
'r5_03_breakeven': [
 ('Rate after computation charges', 'Completed rate after radio and computation charges on 256 held-out paired frames per duration. Bars: pointwise 95% t intervals. Payload scales with duration.'),
 ('Rate gain over no-report', 'Paired rate gain over no-report after computation charges. Simultaneous 95% intervals cover fourteen contrasts; reservations use 128 separate development frames.'),
 ('Reserved computation', 'Mean total reserved computation per frame relative to frame duration. Reservations are fixed using 128 development frames; payload scales with duration.')],
'r5_04_mismatch': [
 ('Service versus true reciprocity', 'Completed service versus true UL-DL correlation, with 512 paired frames per setting. Assumed correlation is 0.9; shadowing SD is 3 dB. Bars: pointwise 95% t intervals; matched methods know the true covariance.'),
 ('Service versus shadowing spread', 'Completed service versus true shadowing SD, with 512 paired frames per setting. The assumed SD is 3 dB and correlation is 0.9. Bars: pointwise 95% t intervals; matched methods know the true covariance.'),
 ('Outage versus true reciprocity', 'Any-task outage versus true UL-DL correlation on the same 512 paired frames per setting. Bars: pointwise 95% Wilson intervals; the horizontal dotted line marks 5%. Matched covariance is an oracle diagnostic.'),
 ('Outage versus shadowing spread', 'Any-task outage versus true shadowing SD on the same 512 paired frames per setting. Bars: pointwise 95% Wilson intervals; the horizontal dotted line marks 5%. Matched covariance is an oracle diagnostic.')],
'r5_05_risk_budget': [
 ('Service-risk tradeoff', 'Frozen design-risk sweep, 512 frames per setting. Horizontal Wilson and vertical t intervals are pointwise 95%; the sweep does not impose equal realized risk.'),
 ('Report budget', 'Completed service versus report cap on 512 paired frames per setting. Policies may stop early. Bars: pointwise 95% t intervals.')],
'r5_06_raytracing': [
 ('Static ray-traced service', 'Completed service for static incoherent ray-traced gains, with 256 test frames per scene. Bars: pointwise 95% t intervals. Safe methods use the marginal filter; these are simulations.'),
 ('Static ray-traced outage', 'Any-task outage for the same static ray-traced frames. Bars: pointwise 95% Wilson intervals; the dotted line marks 5%. Scene categories are not connected.'),
 ('Endpoint-aging service', 'Completed service under coherent endpoint-aging stress, with 256 test frames per scene. Bars: pointwise 95% t intervals. Changing combining and endpoint geometry does not isolate a pure motion effect.'),
 ('Endpoint-aging outage', 'Any-task outage under the same endpoint-aging stress. Bars: pointwise 95% Wilson intervals; the dotted line marks 5%. Initial-channel bounds need not bound the execution endpoint.')],
'r5_07_approximation': [
 ('Scalar-grid shortfall', 'Scalar reference-minus-inner shortfall on 1,024 report candidates and 22,377 detected intervals. One interval is missed at 25 knots. The 0.28574-kbit maximum is not bounded by the 0.002-kbit stopping margin; these are empirical diagnostics.'),
 ('Vector enclosure gap', 'Mean and maximum vector value enclosure gaps on 32 two-component states. The bounds concern the Gaussian surrogate under the stated arithmetic assumptions.'),
 ('Vector enclosure cost', 'Mean computation time for vector box budgets on the same 32 two-component states. The curve is descriptive; vector enclosure cost grows with the box budget.')],
'r5_08_search': [
 ('Interrupted-search value ratio', 'Mean incumbent-to-upper-bound ratio on 32 fixed states per cluster size and node budget. This is an instance-specific certificate, not a uniform approximation factor.'),
 ('Interrupted-search gap', 'Mean unresolved certified-utility gap on the same 32 states per cluster size and node budget. Substantial gaps remain at N=24.'),
 ('Interrupted-search time', 'Mean wall time on the same 32 states per cluster size and node budget. Timings do not establish a 20-ms deadline guarantee.')],
'r5_09_pilots': [
 ('Physical-pilot residuals', 'Standardized physical-pilot log-power errors: 2,048 residuals per pilot length, clustered in 256 frames. The Gaussian curve is a model reference, not a verified physical posterior.'),
 ('Pilot value discrepancy', 'Absolute conditional-CDF17 versus product-GH33 discrepancy on 16 prespecified selected batches per pilot length. GH33 is a comparator, not exact physical-posterior truth.')],
'r5_10_allocator': [
 ('Terminal allocator service', 'Completed service for four terminal allocators on 128 matched frames. Bars: pointwise 95% t intervals; finite-frontier validation differs from the Gaussian certificate.'),
 ('Acquisition gain by allocator', 'Paired acquisition gain with the original and stronger frontier allocators on the same 128 frames. Bars: pointwise 95% t intervals.')],
'value_accuracy': [
 ('Scalar-value error distribution', 'Reference-minus-inner value on 1,024 report candidates from 128 held-out initial states. Positive gaps are shortfalls; negative gaps are numerical overestimates. The abscissa is linear within 10^-6 kbit.'),
 ('Grid sensitivity', 'Scalar-grid sensitivity with eight boundary refinements. Maximum and mean shortfalls are empirical diagnostics, not guaranteed approximation bounds.')],
'catalogue_pareto': [
 ('Catalogue tradeoff at six pairs', 'Forecast-catalogue and screening tradeoff on paired N=6 timing frames. C16/C32 retain 16/32 subsets; S2 retains two reports. Terminal admission remains exhaustive.'),
 ('Catalogue tradeoff at eight pairs', 'The corresponding forecast-catalogue and screening tradeoff at N=8. Terminal admission remains exhaustive.')]
}

def configure_ieee_style():
    paths = [Path(os.environ['IVP_TIMES_FONT_DIR'])] if os.environ.get('IVP_TIMES_FONT_DIR') else []
    paths += [Path.home()/'.local/share/fonts/ivp_times']
    for path in paths:
        if path.exists():
            for p in path.iterdir():
                if p.suffix.lower() in ('.ttf','.otf'):font_manager.fontManager.addfont(str(p))
    font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
    family = font_manager.FontProperties(fname=font_path).get_name()
    if family != 'Times New Roman':raise RuntimeError('Install Times New Roman or set IVP_TIMES_FONT_DIR.')
    plt.rcParams.update({
      'font.family':'Times New Roman', 'font.size':9.5,
      'axes.labelsize':9.5, 'axes.titlesize':9.5,
      'xtick.labelsize':9.5, 'ytick.labelsize':9.5, 'legend.fontsize':9.5,
      'axes.spines.top':True, 'axes.spines.right':True,
      'axes.linewidth':.8, 'lines.linewidth':1.25, 'lines.markersize':4,
      'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
      'savefig.bbox':None, 'figure.figsize':(3.5,2.55),
      'mathtext.fontset':'custom', 'mathtext.rm':'Times New Roman',
      'mathtext.it':'Times New Roman:italic', 'mathtext.bf':'Times New Roman:bold',
      'mathtext.sf':'Times New Roman', 'mathtext.tt':'Times New Roman',
      'mathtext.cal':'Times New Roman:italic', 'mathtext.fallback':None,
    })
    return font_path

def make_panels(nrows=1,ncols=1,**kwargs):
    """Return independent figures while preserving the original axis indexing."""
    figs=[];axes=[]
    for _ in range(nrows*ncols):
        fig,ax=plt.subplots(figsize=(3.5,2.55))
        figs.append(fig);axes.append(ax)
    arr=np.array(axes,dtype=object).reshape(nrows,ncols)
    return figs,arr.squeeze()

def axis_data(ax):
    lines=[np.asarray(l.get_xydata(),dtype=float).tolist() for l in ax.lines]
    cols=[]
    for col in ax.collections:
        if hasattr(col,'get_segments'):cols.append({'segments':[np.asarray(s,dtype=float).tolist() for s in col.get_segments()]})
        elif hasattr(col,'get_offsets'):cols.append({'offsets':np.asarray(col.get_offsets(),dtype=float).tolist()})
    return {'lines':lines,'collections':cols}

def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True,color='#d7d7d7',lw=.45,alpha=.7)
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_linewidth(.8);spine.set_color('black');spine.set_zorder(10)
    ax.tick_params(which='both',direction='in',top=True,right=True)

def export_panels(figs,name,out,atlas,source):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);info=PANEL_INFO[name]
    if len(figs)!=len(info):raise ValueError((name,len(figs),len(info)))
    records=[];PANEL_DATA[name]=[]
    for index,(fig,(title,caption)) in enumerate(zip(figs,info)):
        ax=fig.axes[0];PANEL_DATA[name].append(axis_data(ax));style_axis(ax)
        letter=chr(65+index);stem=name+'_'+letter
        handles,labels=ax.get_legend_handles_labels()
        if ax.get_legend() is not None:ax.get_legend().remove()
        cols=min(3,len(labels));rows=math.ceil(len(labels)/cols) if cols else 0
        top=.94-.105*rows
        fig.subplots_adjust(left=.205,right=.975,bottom=.225,top=top)
        if handles:
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=cols,
                frameon=False,fontsize=9.5,handlelength=1.45,handletextpad=.4,
                columnspacing=.75,borderaxespad=.25,labelspacing=.28)
        if ax.get_title():ax.title.set_position((.5,.93))
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer();boundary=fig.bbox
        xlo,xhi=sorted(ax.get_xlim());ylo,yhi=sorted(ax.get_ylim())
        visible_ticks=[t for v,t in zip(ax.get_xticks(),ax.get_xticklabels()) if xlo<=v<=xhi]
        visible_ticks += [t for v,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ylo<=v<=yhi]
        for text in [ax.xaxis.label,ax.yaxis.label,*visible_ticks]:
            if text.get_visible() and text.get_text():
                bb=text.get_window_extent(renderer)
                if bb.x0 < boundary.x0-1 or bb.y0 < boundary.y0-1 or bb.x1 > boundary.x1+1 or bb.y1 > boundary.y1+1:
                    raise RuntimeError(f'Clipped label in {stem}: {text.get_text()}')
        for leg in fig.legends:
            bb=leg.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > boundary.x1+1 or bb.y1 > boundary.y1+1:
                raise RuntimeError(f'Clipped legend in {stem}')
        fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.svg'));fig.savefig(out/(stem+'.png'),dpi=600)
        if atlas is not None:atlas.savefig(fig)
        records.append(dict(name=stem,parent=name,panel=letter,title=title,caption=caption,source=source,
                            font='Times New Roman',font_size_pt=9.5,box_sides=['left','bottom','right','top'],
                            width_in=3.5,height_in=2.55))
        plt.close(fig)
    return records
