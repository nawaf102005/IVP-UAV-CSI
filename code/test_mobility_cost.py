"""Behavioral checks for execution-target acquisition and airtime accounting."""
import unittest
import numpy as np
from model import Config,scenario,fantasy
from fast_backend import affine_value,value
from revised_policies import margin_scores
from mobility_cost_study import make_case,run_policy,target_margin_scores

class TestExecutionTarget(unittest.TestCase):
    def test_independent_execution_makes_reports_useless(self):
        cfg=Config();sc,jm,jc=make_case(cfg,91999999,0.)
        nr=run_policy(cfg,sc,jm,jc,'Predictive-NR')
        for method in ['Predictive-IVP','Predictive-Margin']:
            r=run_policy(cfg,sc,jm,jc,method)
            self.assertEqual(r['probes'],0)
            self.assertEqual(r['selected'],nr['selected'])
            self.assertEqual(r['completed_kbit'],nr['completed_kbit'])
    def test_unit_correlation_recovers_existing_scalar_values(self):
        cfg=Config();sc=scenario(cfg,91999998);mu=sc['mean'];c=sc['covariance'];k=int(sc['candidates'][0]);b,post=fantasy(c,k,(cfg.measurement_db*np.log(10)/10)**2)
        self.assertAlmostEqual(value(cfg,mu,c,sc['payload'],0,k),affine_value(cfg,mu,post,b,sc['payload'],1),places=11)
        a=margin_scores(cfg,mu,c,sc['payload'],0,sc['candidates']);z=target_margin_scores(cfg,mu,c,sc['payload'],0,sc['candidates'],c,c)
        np.testing.assert_allclose(a,z,rtol=1e-13,atol=1e-13)
        charged=affine_value(cfg,mu,post,b,sc['payload'],1,reserve_s=.003)
        self.assertLessEqual(charged,value(cfg,mu,c,sc['payload'],0,k)+1e-8)
if __name__=='__main__':unittest.main()
