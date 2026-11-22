"""Independent invariants for bounded search, vector bracketing and prediction."""
import unittest
from dataclasses import replace
import numpy as np
from model import Config,scenario,posterior,fantasy
from fast_backend import schedule,value,affine_value
from bounded_admission import bounded_schedule
from vector_bounds import BoxEnvelope,vector_fantasy,value_bounds

class RoundThreeTests(unittest.TestCase):
    def test_coding_targets_respect_time_and_decoder_budget(self):
        from finite_blocklength import coding_targets
        from scipy.stats import norm
        cfg=replace(Config(),outage=.04,max_probes=2);s=scenario(cfg,359900)
        for length in [128,512]:
            for m in range(3):
                target,blocks,rate,_=coding_targets(cfg,s['payload'],m,length)
                duration=(cfg.frame_s-m*cfg.probe_s)*np.array([cfg.ul_fraction,1-cfg.ul_fraction])
                self.assertTrue(np.all(blocks*length<=cfg.bandwidth_hz*duration))
                error=norm.sf((np.log2(1+target)-rate)*np.log(2)*np.sqrt(length/(1-(1+target)**-2)))
                self.assertLessEqual(float(np.sum(error*blocks[:,None])),.01+1e-10)
                self.assertTrue(np.all(target>2**rate-1))
        with self.assertRaises(ValueError):coding_targets(cfg,s['payload'],2,1000000)

    def test_small_sample_tolerance_limit(self):
        from scipy.stats import beta
        self.assertLess(max(beta.sf(.96,k,65-k) for k in range(1,65)),.95)
        self.assertGreaterEqual(beta.sf(.96,74,1),.95)

    def test_anytime_bounds_bracket_exhaustive_optimum(self):
        for n in [3,5,7]:
            cfg=replace(Config(),n=n)
            for seed in range(319900,319904):
                s=scenario(cfg,seed);args=(cfg,s['mean'],s['covariance'],s['payload'],0)
                exact=schedule(*args)['value']
                for budget in [0,8,10000]:
                    act,info=bounded_schedule(*args,budget)
                    self.assertLessEqual(act['value'],exact+1e-8);self.assertGreaterEqual(info['upper']+1e-8,exact)
                    if budget==10000:self.assertAlmostEqual(act['value'],exact,8)
    def test_vector_boxes_contain_actual_point_values(self):
        cfg=Config();s=scenario(cfg,329900);ids=s['candidates'][:2]
        b,c=vector_fantasy(s['covariance'],ids,(cfg.measurement_db*np.log(10)/10)**2)
        env=BoxEnvelope(cfg,s['mean'],c,b,s['payload'],2);rng=np.random.default_rng(329901)
        for _ in range(30):
            center=rng.normal(size=2);width=rng.uniform(.01,.8,2);lo=center-width;hi=center+width;l,u=env.bounds(lo,hi)
            for _ in range(8):
                z=rng.uniform(lo,hi);v=schedule(cfg,s['mean']+b@z,c,s['payload'],2)['value']
                self.assertLessEqual(l,v+1e-8);self.assertGreaterEqual(u+1e-8,v)
    def test_nested_vector_bounds_contract(self):
        cfg=Config();s=scenario(cfg,329902)
        r=value_bounds(cfg,s['mean'],s['covariance'],s['payload'],0,s['candidates'][:2],(16,64,256))
        checks=r['checkpoints']
        for a,b in zip(checks,checks[1:]):
            self.assertLessEqual(a['lower'],b['lower']+1e-8);self.assertGreaterEqual(a['upper']+1e-8,b['upper'])
    def test_prediction_conditioning_commutes_with_gaussian_transition(self):
        cfg=Config();s=scenario(cfg,339900);mu=s['mean'];c=s['covariance'];rho=.7;D=len(mu)
        joint=np.block([[c,rho*c],[rho*c,c]]);k=3;y=mu[k]+.2;rv=.03
        jm,jc=posterior(np.r_[mu,mu],joint,k,y,rv);pm,pc=posterior(mu,c,k,y,rv)
        np.testing.assert_allclose(jm[D:],mu+rho*(pm-mu),atol=1e-12)
        np.testing.assert_allclose(jc[D:,D:],rho*rho*pc+(1-rho*rho)*c,atol=1e-12)
    def test_affine_target_value_reduces_to_original_static_value(self):
        cfg=Config();s=scenario(cfg,339901);k=int(s['candidates'][1]);rv=(cfg.measurement_db*np.log(10)/10)**2
        b,c=fantasy(s['covariance'],k,rv)
        self.assertAlmostEqual(value(cfg,s['mean'],s['covariance'],s['payload'],0,k),
            affine_value(cfg,s['mean'],c,b,s['payload'],1),10)

if __name__=='__main__':unittest.main()
