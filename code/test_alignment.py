import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from model import Config,posterior,scenario
from mobility_cost_study import make_case
from aligned_frontier import terminal,forecast,run_aligned,PLAN_OFFSET,FINAL_OFFSET,acceptance_count
from power_frontier import reference
from chance_benchmark import cp_upper
from fast_backend import value
from certificate_work_audit import reset,counts

class AlignmentTests(unittest.TestCase):
    def test_native_frontier_matches_python_after_observation_and_cost(self):
        cfg=Config();cost=.00015
        for seed in range(96999200,96999204):
            sc,jm,jc=make_case(cfg,seed,.96);D=len(sc['mean']);k=int(sc['candidates'][0]);rv=(cfg.measurement_db*np.log(10)/10)**2;y=sc['true'][k]+np.sqrt(rv)*sc['report_noise'][k,0];mu,cov=posterior(jm,jc,k,y,rv)
            a=terminal(cfg,sc,mu[D:],cov[D:,D:],1,cost);remaining=cfg.frame_s-cfg.probe_s-2*cost;b,_=reference(replace(cfg,frame_s=remaining),{**sc,'mean':mu[D:],'covariance':cov[D:,D:]})
            self.assertEqual(a['group'].tolist(),b['group'].tolist());self.assertEqual(a['mode'],b['mode']);np.testing.assert_allclose(a['power'],b['power'],rtol=1e-8,atol=1e-12)
            if 'airtime' in a:np.testing.assert_allclose(a['airtime'],b['airtime'],rtol=1e-8,atol=1e-12)
    def test_final_validation_stream_never_used_in_planning(self):
        cfg=replace(Config(),max_probes=1);sc,jm,jc=make_case(cfg,96999210,.96);offsets=[];real=terminal
        def wrapped(*args,**kwargs):offsets.append(kwargs.get('offset',FINAL_OFFSET));return real(*args,**kwargs)
        with patch('aligned_frontier.terminal',side_effect=wrapped):run_aligned(cfg,sc,jm,jc,0.,knots=9)
        self.assertEqual(offsets[-1],FINAL_OFFSET);self.assertEqual(offsets.count(FINAL_OFFSET),1);self.assertTrue(all(x==PLAN_OFFSET for x in offsets[:-1]))
    def test_zero_direction_gives_constant_forecast(self):
        cfg=replace(Config(),probe_s=0.);sc=scenario(cfg,96999211);a=terminal(cfg,sc,sc['mean'],sc['covariance'],0,offset=PLAN_OFFSET);q=forecast(cfg,sc['mean'],sc['covariance'],np.zeros_like(sc['mean']),sc['payload'],1,sc['seed'],knots=9)
        self.assertAlmostEqual(q,a['value']*(1-1.973175290075389e-9),places=8)
    def test_binomial_cutoff_matches_independent_beta_upper_bound(self):
        k=acceptance_count(4);alpha=.05/(2*15*7)
        self.assertLessEqual(cp_upper(k,4096,alpha),.05);self.assertGreater(cp_upper(k+1,4096,alpha),.05)
    def test_work_counter_matches_constant_single_task_oracle(self):
        cfg=replace(Config(),n=1,ul_bits=100.,dl_bits=100.);sc=scenario(cfg,96999212);cov=np.zeros_like(sc['covariance']);reset();value(cfg,sc['mean'],cov,sc['payload'],0,0,grid_size=25);c=counts()
        self.assertEqual(c['certificate_tests'],50);self.assertEqual(c['directional_linear_solves'],50);self.assertEqual(c['action_preparations'],1);self.assertEqual(c['oma_direction_tests'],50)
    def test_work_recording_does_not_change_value(self):
        cfg=Config();sc=scenario(cfg,96999213);a=value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,0);reset();b=value(cfg,sc['mean'],sc['covariance'],sc['payload'],0,0);counts();self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
