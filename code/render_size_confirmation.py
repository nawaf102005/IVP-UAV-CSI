"""Tabulate every preregistered size/duration contrast and underlying outcome count."""
from pathlib import Path
from render_prospective import read, interval, maketable

def generate(root):
    root=Path(root);data=read(root/'size_confirmation_results/size_contrasts.csv');tex=root/'manuscript'
    by={(int(r['n']),r['terminal'],float(r['alpha']),r['left'],r['right']):r for r in data}
    ivp,margin,nr='Predictive-IVP','Predictive-Margin','Predictive-NR'
    contrasts=[];means=[]
    for n,ms in [(6,50),(8,100)]:
        for terminal,label in [('cert','Certificate'),('front','Frontier')]:
            for alpha in [0.,.025]:
                x,y,d=[by[n,terminal,alpha,l,r] for l,r in [(ivp,nr),(margin,nr),(ivp,margin)]]
                prefix=[str(n),str(ms),label,f'{alpha*ms:g}']
                contrasts.append(' & '.join([*prefix,*[interval(z) for z in (x,y,d)]]))
                means.append(' & '.join([*prefix,f"{float(x['right_kbit']):.2f} / {x['right_outages']}",
                     *[f"{float(z['left_kbit']):.2f} / {z['left_outages']} / {z['left_reports']}" for z in (x,y)]]))
    maketable(tex/'size_confirm_diff.tex','tab:sizeconfdiff',
              r'Larger-cluster acquisition gains in kbit/frame on 512 independent paired frames per size/duration. Per-round cost $c$ is hypothetical. The 95\% paired intervals are simultaneous over 24 contrasts. NR denotes no-report.',
              'rrlrccc',r'$N$ & $T$ (ms) & Terminal & $c$ (ms) & C-IVP $-$ NR & Sensitivity $-$ NR & C-IVP $-$ sensitivity',contrasts)
    maketable(tex/'size_confirm_raw.tex','tab:sizeconfraw',
              r'Mean kbit/frame, total radio outages and total reports on 512 independent frames per size/duration. No-report has zero reports. Both terminal allocators score the same report histories. Values are descriptive.',
              'rrlrccc',r'$N$ & $T$ (ms) & Terminal & $c$ (ms) & No-report & C-IVP & Sensitivity',means)
if __name__=='__main__':generate(Path(__file__).resolve().parents[1])
