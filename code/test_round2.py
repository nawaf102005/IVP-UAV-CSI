"""Checks of joint/sequential Bayes equivalence and new calibration bounds."""
import unittest
from dataclasses import replace
import numpy as np
from model import Config,scenario,posterior
from innovation import feasibility_interval
from admission import execute
from conformal_report_filter import safe_schedule
from round2_experiments import joint_update,regularized_covariance,full_scores

class RoundTwoTests(unittest.TestCase):
    def test_joint_and_sequential_conditioning_agree_in_both_orders(self):
        s=scenario(Config(),92600);rv=(np.log(10)/10)**2;ids=[0,5];y=s['true'][ids]+[.1,-.2]
        mu,cs=joint_update(s['mean'],s['covariance'],ids,y,rv)
        for order in [[0,1],[1,0]]:
            m=s['mean'].copy();c=s['covariance'].copy()
            for j in order:m,c=posterior(m,c,ids[j],y[j],rv)
            np.testing.assert_allclose(mu,m,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(cs,c,rtol=1e-12,atol=1e-12)

    def test_reference_detects_interval_between_coarse_knots(self):
        lo,hi=feasibility_interval(Config(),function=lambda z:(z-.25)**2-.001)
        grid=np.linspace(-6,6,25)
        self.assertFalse(np.any((grid>=lo)&(grid<=hi)))
        self.assertAlmostEqual(lo,.25-np.sqrt(.001),places=4)
        self.assertAlmostEqual(hi,.25+np.sqrt(.001),places=4)

    def test_regularized_score_event_suffices_for_actual_delivery(self):
        cfg=replace(Config(),n=2,ul_bits=5000,dl_bits=1000);s=scenario(cfg,92601)
        cov=s['covariance'].copy();cov[0,:]*=.001;cov[:,0]*=.001
        reg=regularized_covariance(cov,cfg.measurement_db)
        self.assertGreaterEqual(np.linalg.eigvalsh(reg).min(),0)
        rng=np.random.default_rng(92602);active=0
        for _ in range(48):
            truth=s['mean']+rng.uniform(-.2,.2,len(reg))*np.sqrt(np.diag(reg))
            self.assertLessEqual(full_scores(truth[None,:],s['mean'],reg,n=2)[0],3.)
            sc={**s,'true':truth};obs=[dict(index=0,observation=float(truth[0]))]
            act=safe_schedule(cfg,sc,reg,3.,obs,reciprocal=False);r=execute(cfg,sc,act,1)
            active+=r['active'];self.assertEqual(r['outage'],0)
        self.assertGreater(active,0)

if __name__=='__main__':unittest.main(verbosity=2)
