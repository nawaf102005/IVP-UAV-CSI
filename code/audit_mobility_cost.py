"""Replay all additional aging/cost decisions without remeasuring host time."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from model import Config,posterior
from mobility_cost_study import make_case
from fast_backend import schedule,library
from admission import execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();root=Path(a.results);library();count=0;seen=set();test_seeds=set()
    for line in (root/'records.jsonl').open():
        r=json.loads(line);cfg=Config(**r['config']);sc,mean,cov=make_case(cfg,r['seed'],r['rho']);D=len(sc['mean']);rv=(cfg.measurement_db*np.log(10)/10)**2;used=np.zeros(D,int)
        key=(r['rho'],r['report_fraction'],r['charged'],r['seed'],r['method']);assert key not in seen;seen.add(key);test_seeds.add(r['seed'])
        for obs in r['observations']:
            k=obs['index'];expected=sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,used[k]];used[k]+=1
            assert abs(expected-obs['observation'])<1e-11
            mean,cov=posterior(mean,cov,k,obs['observation'],rv)
        action=schedule(cfg,mean[D:],cov[D:,D:],sc['payload'],r['probes'],reserve_s=r['reserve_s'])
        assert action['group'].tolist()==r['selected'] and action['mode']==r['mode']
        np.testing.assert_allclose(action['power'],r['power'],rtol=1e-10,atol=1e-12)
        if 'airtime' in action:np.testing.assert_allclose(action['airtime'],r['airtime'],rtol=1e-10,atol=1e-12)
        reserved=(r['probes']+1)*r['reserve_s'];assert abs(reserved-r['reserved_s'])<1e-12
        result=execute(replace(cfg,frame_s=max(1e-12,cfg.frame_s-reserved)),sc,action,r['probes'])
        assert abs(result['completed_kbit']-r['radio_completed_kbit'])<1e-9 and result['outage']==r['radio_outage']
        overrun=r['reserve_s']>0 and r['wall_s']>reserved;assert int(overrun)==r['deadline_overrun']
        if overrun and result['active']:result.update(completed_kbit=0.,completed_tasks=0,outage=1)
        for k in ['completed_kbit','admitted_kbit','completed_tasks','admitted_tasks','outage','active']:assert abs(result[k]-r[k])<1e-9,(key,k)
        count+=1
    protocol=json.loads((root/'protocol.json').read_text());assert count==len(protocol['rhos'])*len(protocol['report_fractions'])*2*protocol['test_frames']*3
    dev=json.loads((root/'reservations.json').read_text())['development'];dev_seeds={x['seed'] for x in dev};assert not dev_seeds&test_seeds
    report=dict(records_replayed=count,development_decisions=len(dev),test_seed_count=len(test_seeds),development_seed_count=len(dev_seeds),duplicate_records=0,mismatches=0,host_times_remeasured=False,forecast_values_recomputed=False)
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
