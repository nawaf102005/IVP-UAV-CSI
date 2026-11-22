"""Replay physical observations/decisions; diagnose duplicated Munich Motion rows.

Forecast values are checked for selection/stopping consistency; the full forecast
integration is reproduced by physical_pilots.py, not repeated by this audit.
"""
import argparse,json
from pathlib import Path
from dataclasses import replace
from collections import Counter
import numpy as np
from model import Config,scenario
from admission import execute
from fast_backend import schedule
from physical_pilots import physical_observations,noise_covariance,update
from round2_experiments import fresh_case_data,regularized_covariance,score_quantile,full_scores
from ray_evaluation import fit,case
from conformal_report_filter import safe_schedule,conformal_quantile

def read(path):return [json.loads(x) for x in path.read_text().splitlines()]

def same_action(cfg,sc,act,r,count):
    assert r['mode']==act['mode'] and r['selected']==act['group'].tolist()
    np.testing.assert_allclose(r['power'],act['power'],rtol=1e-8,atol=1e-12)
    if 'airtime' in act:np.testing.assert_allclose(r['airtime'],act['airtime'],rtol=1e-8,atol=1e-12)
    physical=execute(cfg,sc,act,count)
    for key in ['completed_kbit','completed_tasks','outage','active','admitted_kbit']:
        assert abs(physical[key]-r[key])<1e-7,(key,physical[key],r[key])

def audit(root,out):
    errors=[];raw=read(out/'pilot_records.jsonl');stored=read(out/'pilot_observations.jsonl')
    protocol=json.loads((out/'pilot_protocol.json').read_text());frames=protocol['frames_per_length']
    assert len(raw)==frames*len(protocol['lengths'])*len(protocol['methods'])
    assert len(stored)==frames*len(protocol['lengths'])
    cache={}
    for r in stored:
        cfg=replace(Config(),probe_s=.0015+r['length']/1e6,max_probes=1);sc=scenario(cfg,r['seed'])
        obs=physical_observations(cfg,sc,r['length'])
        for actual,saved in zip(obs,r['batches']):
            assert actual['indices']==saved['indices']
            for key in ['y','snr','standardized_error','ls_error']:
                np.testing.assert_allclose(actual[key],saved[key],rtol=1e-11,atol=1e-12)
        cache[r['seed'],r['length']]=(cfg,sc,obs)
    for r in raw:
        try:
            cfg,sc,obs=cache[r['seed'],r['length']];act=schedule(cfg,sc['mean'],sc['covariance'],sc['payload'],0)
            if r['method']=='No-report':assert r['probes']==0 and not r['forecasts']
            else:
                j=int(np.argmax(r['forecasts']));assert j==r['chosen']
                take=r['forecasts'][j]>act['value']+cfg.gain_tol_kbit;assert int(take)==r['probes']
                if take:
                    o=obs[j];np.testing.assert_allclose(o['y'],r['observation'],rtol=1e-11)
                    mean,cov=update(sc,o['indices'],o['y'],noise_covariance(cfg,sc,o['indices'],r['length']))
                    act=schedule(cfg,mean,cov,sc['payload'],1)
            same_action(cfg,sc,act,r,r['probes'])
        except (AssertionError,ValueError) as e:errors.append(dict(seed=r['seed'],length=r['length'],method=r['method'],error=str(e)))
    cfg=replace(Config(),ul_fraction=.5,ul_power_w=.2,dl_power_w=1.)
    data,meta=fresh_case_data(root,'munich');fresh=read(root/'round2_results/fresh_calibration_records.jsonl')
    calibration={};motion_count=0;paired_counts={};field_matches={}
    for kind,mobile in [('incoherent',False),('coherent',True)]:
        label='munich-motion' if mobile else 'munich-static';prior=fit(data,meta,kind)
        cov=prior['cov'];reg=regularized_covariance(cov,1.);qold=conformal_quantile(prior,meta);qreg=score_quantile(prior,meta,reg)
        lifted=np.flatnonzero(np.diag(reg)>np.diag(cov)+1e-14)
        a=meta['train'];b=a+meta['calibration'];x=prior['x'][a:b]
        calibration[label]=dict(lifted_indices=lifted.tolist(),q_original=qold,q_regularized=qreg,
            original_max_score=float(full_scores(x,prior['mean'],cov).max()),regularized_max_score=float(full_scores(x,prior['mean'],reg).max()),
            minimum_sd_db=float(np.sqrt(np.diag(cov)).min()*10/np.log(10)),lifted_means=prior['mean'][lifted].tolist())
        records={(r['ray_index'],r['method']):r for r in fresh if r['scenario']==label}
        for suffix in ['IVP','NR']:
            fields=['observations','mode','selected','power','airtime','q','completed_kbit','outage']
            matches=Counter();uses_lifted=0;identical=0
            for index in range(meta['test']):
                old=records[index,'Safe-'+suffix];new=records[index,'Regularized-'+suffix]
                all_same=True
                for key in fields:
                    eq=old.get(key)==new.get(key);matches[key]+=int(eq);all_same &= eq
                identical+=int(all_same)
                if new['mode']=='reuse':
                    g=new['selected'];coords={d*cfg.n*cfg.n+i*cfg.n+j for d in range(2) for i in g for j in g if i!=j}
                    uses_lifted+=bool(coords.intersection(lifted.tolist()))
                if mobile:
                    for r in [old,new]:
                        try:
                            sc=case(cfg,data,meta,prior,index,kind,True,True,r['seed'])
                            rcov,qq=(cov,qold) if r['method'].startswith('Safe') else (reg,qreg)
                            act=safe_schedule(cfg,sc,rcov,qq,r['observations'])
                            same_action(cfg,sc,act,r,len(r['observations']));motion_count+=1
                        except (AssertionError,ValueError) as e:errors.append(dict(index=index,method=r['method'],error=str(e)))
            paired_counts[label+'-'+suffix]=dict(identical=identical,frames=meta['test'],selected_reuse_includes_lifted=uses_lifted,
                regularized_modes=dict(Counter(records[i,'Regularized-'+suffix]['mode'] for i in range(meta['test']))),
                original_modes=dict(Counter(records[i,'Safe-'+suffix]['mode'] for i in range(meta['test']))))
            field_matches[label+'-'+suffix]=dict(matches)
    result=dict(pilot_decisions_checked=len(raw),pilot_observation_sets_checked=len(stored),motion_decisions_checked=motion_count,
        calibration=calibration,paired_rows=paired_counts,exact_field_matches=field_matches,errors=errors,
        scope='Regenerates physical observations and terminal actions; verifies stored forecast selection/stop consistency. Full candidate forecasts are reproduced by the experiment runner.')
    (out/'audit_round4.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if errors:raise RuntimeError('R4 audit mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--results');a=p.parse_args()
    root=Path(a.root);audit(root,Path(a.results) if a.results else root/'round4_results')
