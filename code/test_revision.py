"""Numerical equivalence, conservative interval and latency-accounting checks."""
import unittest
from dataclasses import replace
import numpy as np
from model import Config,scenario,fantasy
from admission import best_schedule,coefficients,minimum_power,orthogonal_data,orthogonal_feasibility,execute
from conformal_report_filter import safe_schedule
from innovation import probe_value,expected_envelope
from fast_backend import schedule,value,build
from revised_policies import run_fast

class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):build()

    def test_positive_solve_matches_spectral_reference(self):
        for n in [2,4,6]:
            cfg=replace(Config(),n=n)
            for seed in range(92100,92112):
                s=scenario(cfg,seed)
                for m in [0,2]:
                    a=schedule(cfg,s['mean'],s['covariance'],s['payload'],m)
                    b=best_schedule(cfg,s['mean'],s['covariance'],s['payload'],m)
                    self.assertAlmostEqual(a['value'],b['value'],8)
                    self.assertEqual(a['mode'],b['mode'])
                    np.testing.assert_allclose(a['power'],b['power'],rtol=1e-7,atol=1e-12)

    def test_inner_interval_value_is_a_lower_bound(self):
        cfg=Config()
        for seed in range(92200,92206):
            s=scenario(cfg,seed)
            for k in s['candidates']:
                a,ints=value(cfg,s['mean'],s['covariance'],s['payload'],0,int(k),return_intervals=True)
                b=probe_value(cfg,s['mean'],s['covariance'],s['payload'],0,int(k))
                self.assertLessEqual(a,b+2e-3) # reference root tolerance
                self.assertAlmostEqual(a,expected_envelope(ints.tolist()),9)

    def test_guard_equivalent_to_desired_mean_shift(self):
        cfg=Config();s=scenario(cfg,92300);mu=s['mean'].copy()
        ix=np.r_[np.arange(4)*5,16+np.arange(4)*5];mu[ix]-=8*np.log(10)/10
        a=schedule(cfg,s['mean'],s['covariance'],s['payload'],1,guard_db=8)
        b=best_schedule(cfg,mu,s['covariance'],s['payload'],1)
        self.assertAlmostEqual(a['value'],b['value'],8)
        np.testing.assert_allclose(a['power'],b['power'],rtol=1e-8)

    def test_reservation_equivalent_to_shorter_frame(self):
        cfg=Config();s=scenario(cfg,92400);m=2;c=.0005
        a=schedule(cfg,s['mean'],s['covariance'],s['payload'],m,reserve_s=c)
        b=best_schedule(replace(cfg,frame_s=cfg.frame_s-(m+1)*c),s['mean'],s['covariance'],s['payload'],m)
        self.assertAlmostEqual(a['value'],b['value'],8)

    def test_fixed_count_is_exact_even_if_service_already_feasible(self):
        cfg=replace(Config(),ul_bits=3000,dl_bits=300);s=scenario(cfg,92500)
        for method in ['C-IVP','A-fixed','D-fixed','Random-fixed']:
            r=run_fast(cfg,s,method,fixed_count=2)
            self.assertEqual(r['probes'],2)

    def test_conformal_event_implies_physical_delivery(self):
        cfg=replace(Config(),n=2,ul_bits=5000,dl_bits=1000)
        s=scenario(cfg,92501);cov=s['covariance'];std=np.sqrt(np.diag(cov))
        rng=np.random.default_rng(92502);active=0
        # All these residuals satisfy both coordinate and contrast bounds q=3.
        for _ in range(64):
            truth=s['mean']+rng.uniform(-.5,.5,len(std))*std
            sc={**s,'true':truth}
            obs=[dict(index=0,observation=float(truth[0]))]
            act=safe_schedule(cfg,sc,cov,3.,obs,reciprocal=False)
            rr=execute(cfg,sc,act,1);active+=rr['active']
            self.assertEqual(rr['outage'],0)
        self.assertGreater(active,0)

    def test_inconsistent_report_bounds_reject(self):
        cfg=Config();sc=scenario(cfg,92503)
        act=safe_schedule(cfg,sc,sc['covariance'],1.,[dict(index=0,observation=100.)],reciprocal=False)
        self.assertEqual(act['value'],0.)

if __name__=='__main__':unittest.main(verbosity=2)
