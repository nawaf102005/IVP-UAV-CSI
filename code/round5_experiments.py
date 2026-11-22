"""Fixed R5 protocol: larger-N quadrature, charged latency and model sensitivity.

Run serially with one BLAS/OpenMP thread. All frames/settings are fixed before
measurement. Matched-covariance methods are oracle diagnostics, not adaptation.
"""
import argparse,json,platform,sys
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from model import Config,scenario
from fast_backend import library
from revised_policies import run_fast
SCALE=[('CDF','C-IVP',{}),('KG17','GH-KG',{}),('CDF-S2','C-IVP',{'candidate_limit':2}),('KG17-S2','GH-KG',{'candidate_limit':2}),('Margin','Margin-sensitivity',{}),('NR','No-report',{})]
LAT=[('CDF','C-IVP',{}),('Margin','Margin-sensitivity',{}),('NR','No-report',{})]
def protocol():return dict(config=asdict(Config()),scale=dict(sizes=[4,6,8],frames=256,seed=991000,methods=SCALE),break_even=dict(durations_ms=[20,25,32,40,50,75,100],development_frames=128,development_seed=992000,test_frames=256,test_seed=993000,methods=LAT,reservation='Twice development p95 of wall_s/(reports+1), rounded up to 0.05 ms; fixed across T; active overruns earn zero.',offered_rate='Payloads scale with T; channels remain static.'),mismatch=dict(reciprocity=[0.,.2,.4,.6,.8,.9],shadow_db=[2.,3.,4.,5.,6.],frames=512,seed=994000,methods=['CDF','NR','Matched-CDF','Matched-NR']),statistics=dict(tost_margin_kbit=.5,tost_family='Six full/S2 comparisons at N=4,6,8; Holm across six TOST p-values, alpha .05. Engineering margin fixed before new results.',break_even_family='14 comparisons: two methods vs NR at seven T; Bonferroni simultaneous 95% intervals.',other='Pointwise 95% t/Wilson intervals; paired methods and sweep parameters.'),python=sys.version,platform=platform.platform())
def write(f,r):f.write(json.dumps(r,allow_nan=False)+'\n')
def runset(cfg,sc,specs,tag,extra=None,reserves=None):
    for ix in np.random.default_rng(sc['seed']+cfg.n*13+int(cfg.frame_s*1e6)).permutation(len(specs)):
        label,method,kw=specs[ix];r=run_fast(cfg,sc,method,reserve_s=0 if reserves is None else reserves[label],**kw)
        r.update(method=label,study=tag,**(extra or {}));yield r

def scale(out,p):
    s=p['scale']
    with (out/'scale_records.jsonl').open('w') as f:
        for n in s['sizes']:
            cfg=replace(Config(),n=n)
            for seed in range(s['seed']-3,s['seed']):list(runset(cfg,scenario(cfg,seed),SCALE,'warmup'))
            for i in range(s['frames']):
                for r in runset(cfg,scenario(cfg,s['seed']+i),SCALE,'scale'):write(f,r)
                if (i+1)%32==0:f.flush();print('scale',n,i+1,flush=True)

def latency(out,p):
    s=p['break_even'];cfg=Config();dev=[];costs={m[0]:[] for m in LAT}
    for seed in range(s['development_seed']-3,s['development_seed']):list(runset(cfg,scenario(cfg,seed),LAT,'warmup'))
    for seed in range(s['development_seed'],s['development_seed']+s['development_frames']):
        for r in runset(cfg,scenario(cfg,seed),LAT,'development'):
            dev.append(r);costs[r['method']].append(r['wall_s']/(r['probes']+1))
    reserves={m:float(np.ceil(2*np.quantile(x,.95)/.00005)*.00005) for m,x in costs.items()}
    (out/'reservations.json').write_text(json.dumps(dict(reserves=reserves,development=dev),indent=2)+'\n');print('Fixed reservations',reserves,flush=True)
    with (out/'latency_records.jsonl').open('w') as f:
        for T in s['durations_ms']:
            cfg=replace(Config(),frame_s=T/1000,ul_bits=30000*T/20,dl_bits=1800*T/20)
            for i in range(s['test_frames']):
                for r in runset(cfg,scenario(cfg,s['test_seed']+i),LAT,'latency',dict(duration_ms=T),reserves):write(f,r)
            f.flush();print('latency',T,flush=True)

def mismatch(out,p):
    s=p['mismatch'];cfg=Config()
    with (out/'mismatch_records.jsonl').open('w') as f:
        for dim in ['reciprocity','shadow_db']:
            for x in s[dim]:
                for i in range(s['frames']):
                    seed=s['seed']+i;assumed=scenario(cfg,seed);truth=scenario(replace(cfg,**{dim:x}),seed)
                    assumed['true']=truth['true'];matched={**assumed,'covariance':truth['covariance']}
                    for label,m,source in [('CDF','C-IVP',assumed),('NR','No-report',assumed),('Matched-CDF','C-IVP',matched),('Matched-NR','No-report',matched)]:
                        r=run_fast(cfg,source,m);r.update(method=label,study='mismatch',dimension=dim,true_parameter=x);write(f,r)
                f.flush();print('mismatch',dim,x,flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',default='round5_results');a.add_argument('--mode',choices=['all','scale','latency','mismatch'],default='all');a=a.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True);p=protocol()
    if (out/'protocol.json').exists():
        old=json.loads((out/'protocol.json').read_text())
        for k in ['scale','break_even','mismatch','statistics']:assert old[k]==json.loads(json.dumps(p[k]))
    else:(out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    library()
    for mode,fn in [('scale',scale),('latency',latency),('mismatch',mismatch)]:
        if a.mode in ['all',mode]:
            if (out/(mode+'_records.jsonl')).exists():raise FileExistsError('Choose fresh output or an unrun mode')
            fn(out,p)
    print('completed',a.mode,flush=True)
if __name__=='__main__':main()
