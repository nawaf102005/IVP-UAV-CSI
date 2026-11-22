"""Fit on ray training channels, calibrate separately, evaluate held-out rays."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from scipy.stats import norm
from model import Config,noise_power
from admission import execute
from fast_backend import schedule
from revised_policies import run_fast
from run_revision import both_catalogue

def log_vectors(dl):
    ul=np.swapaxes(dl,1,2)
    return np.log(np.maximum(np.stack([ul,dl],axis=1),1e-20)).reshape(len(dl),-1)

def fit(data,meta,kind):
    x=log_vectors(data[kind]);n=4;train=meta['train'];nc=meta['calibration']
    mean=x[:train].mean(axis=0);sample=np.cov(x[:train],rowvar=False)
    cov=.85*sample+.15*np.diag(np.diag(sample))+np.eye(32)*1e-6
    sd=np.sqrt(np.diag(cov));xc=x[train:train+nc]-mean;scores=[]
    for d in range(2):
        for i in range(n):
            ii=d*n*n+i*(n+1);scores.append(-xc[:,ii]/sd[ii])
            for j in range(n):
                if j==i:continue
                ij=d*n*n+i*n+j;var=cov[ij,ij]+cov[ii,ii]-2*cov[ij,ii]
                scores.append((xc[:,ij]-xc[:,ii])/np.sqrt(max(var,1e-12)))
    score=np.max(scores,axis=0);rank=int(np.ceil((nc+1)*.95));q=max(0.,float(np.sort(score)[rank-1]))
    # Gaussian inflation is empirically calibrated; posterior validity is not
    # distribution-free. The separate constant-q no-report policy is conformal.
    inflation=max(1.,q/norm.isf(.05/(2*n*n)))
    return dict(mean=mean,cov=cov,cal_cov=cov*inflation**2,q=q,inflation=inflation,x=x,cal_scores=score)

def case(cfg,data,meta,prior,index,kind,mobile,scale,seed):
    mean=prior['mean'];cov=prior['cal_cov'] if scale else prior['cov'];n=cfg.n
    start=meta['train']+meta['calibration'];sample=start+index
    true=prior['x'][sample]
    caps=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
    diag=np.diagonal(mean.reshape(2,n,n),axis1=1,axis2=2)
    duration=cfg.frame_s*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
    base_payload=1.15*cfg.bandwidth_hz*duration[:,None]*np.log2(1+caps*np.exp(diag)/noise_power(cfg))/n
    rng=np.random.default_rng(seed);payload=base_payload*rng.uniform(.85,1.15,(2,n))
    sc=dict(mean=mean,covariance=cov,true=true,payload=payload,seed=seed,
            report_noise=rng.normal(size=(32,cfg.max_probes+3)),
            base=data['base'],uav=data['positions'][sample])
    if mobile:sc['execution_true']=log_vectors(data[kind+'_future'][index:index+1])[0]
    both_catalogue(sc,n)
    return sc

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='../ray_data');p.add_argument('--output',default='../revision_results')
    a=p.parse_args();root=Path(a.data);out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    cfg=replace(Config(),ul_fraction=.5,ul_power_w=.2,dl_power_w=1.)
    summaries=[];start=time.perf_counter()
    with (out/'ray_records.jsonl').open('w') as f:
        for scene_index,name in enumerate(['canyon_low','canyon_high','munich']):
            data=np.load(root/f'{name}.npz');meta=json.loads((root/f'{name}.json').read_text())
            for kind,mobile in [('incoherent',False),('coherent',True)]:
                prior=fit(data,meta,kind);label=name+('-motion' if mobile else '-static')
                summaries.append(dict(scenario=label,q=prior['q'],inflation=prior['inflation'],train=meta['train'],calibration=meta['calibration'],test=meta['test'],
                                      train_mean=prior['mean'].tolist(),train_covariance=prior['cov'].tolist(),calibration_scores=prior['cal_scores'].tolist()))
                for index in range(meta['test']):
                    seed=210000+scene_index*1000+index;sc=case(cfg,data,meta,prior,index,kind,mobile,True,seed)
                    for method in ['C-IVP','A-stop','Sensitivity-3pt','No-report','OMA','Perfect-CSI']:
                        r=run_fast(cfg,sc,method);r.update(scenario=label,ray_index=index);f.write(json.dumps(r)+'\n')
                    raw=case(cfg,data,meta,prior,index,kind,mobile,False,seed)
                    r=run_fast(cfg,raw,'C-IVP');r.update(method='Unscaled-IVP',scenario=label,ray_index=index);f.write(json.dumps(r)+'\n')
                    # Marginal split-conformal guarantee only for exchangeable
                    # static test channels; no guarantee is carried to motion.
                    t=time.perf_counter();act=schedule(cfg,prior['mean'],prior['cov'],sc['payload'],0,quantile_override=prior['q'])
                    r=execute(cfg,sc,act,0);r.update(method='Conformal-NR',scenario=label,ray_index=index,seed=seed,probes=0,
                                                   wall_s=time.perf_counter()-t,mode=act['mode'],selected=list(map(int,act['group'])),
                                                   power=act['power'].tolist(),demand_kbit=float(sc['payload'].sum()/1000))
                    if 'airtime' in act:r['airtime']=act['airtime'].tolist()
                    f.write(json.dumps(r)+'\n')
                f.flush();print(label,meta['test'],'inflation',prior['inflation'],round(time.perf_counter()-start,2),flush=True)
    (out/'ray_protocol.json').write_text(json.dumps(dict(config=asdict(cfg),load_factor=1.15,covariance_shrinkage=.15,
        model_note='Gaussian fit on independent ray training samples; calibrated posterior is not distribution-free. Conformal-NR has marginal static-sample coverage only.',
        settings=summaries,elapsed_s=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
