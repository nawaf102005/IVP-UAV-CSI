"""Replay archived decisions and inspect constraints without rerunning policies."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
from model import Config,scenario,posterior
from admission import coefficients,execute,orthogonal_data,orthogonal_airtime


def main():
    root=Path(__file__).resolve().parent.parent;res=root/'results'
    protocol=json.loads((res/'protocol.json').read_text())
    records=[json.loads(s) for s in (res/'records.jsonl').read_text().splitlines()]
    cache={};errors=[];max_residual=-np.inf
    for rr in records:
        key=(rr['scenario'],rr['seed'])
        if key not in cache:
            cfg=Config(**protocol['scenarios'][key[0]]);sc=scenario(cfg,key[1])
            if key[0]=='post_report_blockage':
                rng=np.random.default_rng(key[1]+77);extra=np.zeros((2,cfg.n,cfg.n))
                for i in np.flatnonzero(rng.random(cfg.n)<.1):extra[:,i,i]=-8*np.log(10)/10
                sc['execution_true']=sc['true']+extra.ravel()
            cache[key]=(cfg,sc)
        cfg,sc=cache[key];mean,cov=sc['mean'].copy(),sc['covariance'].copy()
        if rr['method']=='Perfect-CSI':mean=sc.get('execution_true',sc['true']).copy();cov[:]=0
        for obs in rr['observations']:
            mean,cov=posterior(mean,cov,obs['index'],obs['observation'],(cfg.measurement_db*np.log(10)/10)**2)
        group=np.array(rr['selected'],int);power=np.array(rr['power'])
        sched=dict(group=group,power=power,value=rr['admitted_kbit'],mode=rr['mode'])
        if rr['mode']=='OMA':sched['airtime']=np.array(rr['airtime'])
        replay=execute(cfg,sc,sched,rr['probes'])
        if abs(replay['completed_kbit']-rr['completed_kbit'])>1e-8 or replay['outage']!=rr['outage']:
            errors.append((key,rr['method'],'execution mismatch'))
        if len(group):
            if rr['mode']=='reuse':
                coeff=coefficients(cfg,mean,cov,sc['payload'],rr['probes'],group,
                                   nominal=rr['method'] in ['Nominal','Perfect-CSI'],diagonal=rr['method']=='IVP-diagonal')
                f=np.exp(coeff[0]);u=np.exp(coeff[2]);violation=float(np.max(np.einsum('dij,dj->di',f,power)+u-power))
                max_residual=max(max_residual,violation)
                if violation>1e-8:errors.append((key,rr['method'],'power certificate violation'))
            else:
                data=orthogonal_data(cfg,mean,cov,sc['payload'],rr['probes'],group,
                                     nominal=rr['method'] in ['Nominal','Perfect-CSI'])
                minimum=orthogonal_airtime(data,0.)
                if np.any(sched['airtime']<minimum-1e-10) or np.any(sched['airtime'].sum(axis=1)>data['duration']+1e-10):
                    errors.append((key,rr['method'],'OMA time violation'))
            caps=np.array([cfg.ul_power_w,cfg.dl_power_w])[:,None]
            if np.any(power<0) or np.any(power>caps*(1+1e-8)):errors.append((key,rr['method'],'power cap violation'))
    test=subprocess.run([sys.executable,'-m','unittest','-v','test_theory'],cwd=root/'code',text=True,capture_output=True)
    (res/'tests.txt').write_text(test.stdout+test.stderr)
    result=dict(decisions_checked=len(records),snapshots=len(cache),constraint_or_replay_errors=errors,
                maximum_reuse_residual_w=max_residual,unit_test_exit_code=test.returncode,
                tests=7,scope='Stored-action replay, power/airtime certificates, and independent mathematical reduction checks; not external validation.')
    (res/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if errors or test.returncode:raise SystemExit(1)


if __name__=='__main__':main()
