"""Independent checks for complex pilot observations and surrogate conditioning."""
import unittest
from dataclasses import replace
import numpy as np
from model import Config, scenario
from physical_pilots import pilot_matrix, matched_filter, batches, noise_covariance, update
from vector_bounds import vector_fantasy, BoxEnvelope
from fast_backend import schedule

class RoundFourTests(unittest.TestCase):
    def test_orthogonal_noiseless_recovery(self):
        h=np.array([.3+.9j,-.7+.2j])
        for length in [16,64]:
            np.testing.assert_allclose(pilot_matrix(length).conj().T@pilot_matrix(length),length*np.eye(2),atol=1e-13)
            np.testing.assert_allclose(matched_filter(h,np.zeros(length),.2,length),h,atol=1e-14)

    def test_complex_noise_variance(self):
        rng=np.random.default_rng(874);length=16;power=.2;variance=.07
        noise=np.sqrt(variance/2)*(rng.normal(size=(20000,length))+1j*rng.normal(size=(20000,length)))
        error=noise@pilot_matrix(length).conj()/(length*np.sqrt(power))
        target=variance/(length*power)
        np.testing.assert_allclose(np.mean(abs(error)**2,axis=0),target,rtol=.025)
        self.assertLess(abs(np.mean(error[:,0]*error[:,1].conj()))/target,.025)

    def test_heteroscedastic_conditioning(self):
        cfg=Config();s=scenario(cfg,984321);ids=batches(cfg,s)[0];R=noise_covariance(cfg,s,ids,16)
        y=s['mean'][list(ids)]+[.1,-.3];mean,cov=update(s,ids,y,R)
        H=np.eye(len(s['mean']))[list(ids)];P=np.linalg.inv(s['covariance'])
        expected_cov=np.linalg.inv(P+H.T@np.linalg.solve(R,H))
        expected_mean=expected_cov@(P@s['mean']+H.T@np.linalg.solve(R,y))
        np.testing.assert_allclose(cov,expected_cov,atol=1e-11)
        np.testing.assert_allclose(mean,expected_mean,atol=1e-8)
        _,fantasy=vector_fantasy(s['covariance'],ids,R)
        np.testing.assert_allclose(cov,fantasy,atol=1e-12)

    def test_heteroscedastic_boxes_and_batch_cost(self):
        cfg=replace(Config(),max_probes=1,probe_s=.001516);s=scenario(cfg,984322)
        ids=batches(cfg,s)[0];R=noise_covariance(cfg,s,ids,16)
        B,post=vector_fantasy(s['covariance'],ids,R)
        env=BoxEnvelope(cfg,s['mean'],post,B,s['payload'],1);rng=np.random.default_rng(92)
        for center in [np.array([-2.,1.]),np.array([0.,0.]),np.array([2.,-1.])]:
            lo=center-.3;hi=center+.3;lower,upper=env.bounds(lo,hi)
            for _ in range(12):
                z=rng.uniform(lo,hi);v=schedule(cfg,s['mean']+B@z,post,s['payload'],1)['value']
                self.assertLessEqual(lower,v+1e-8);self.assertGreaterEqual(upper+1e-8,v)

if __name__=='__main__':unittest.main()
