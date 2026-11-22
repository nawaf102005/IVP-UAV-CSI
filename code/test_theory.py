"""Independent numerical checks of the mathematical reductions."""
import unittest
import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr
from model import Config, scenario, posterior, fantasy
from admission import coefficients, minimum_power, convex_feasibility, best_schedule, subsets
from innovation import feasibility_interval, probe_value, expected_envelope


class TheoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Config()
        self.sc=scenario(self.cfg,7301)

    def test_posterior_matches_information_form(self):
        s=self.sc; mu,cov=s['mean'],s['covariance']; k=3; r=.12; y=mu[k]+.3
        m,p=posterior(mu,cov,k,y,r)
        precision=np.linalg.inv(cov); precision[k,k]+=1/r
        ref=np.linalg.inv(precision)
        v=np.linalg.solve(cov,mu);v[k]+=y/r
        np.testing.assert_allclose(p,ref,atol=1e-12)
        np.testing.assert_allclose(m,ref@v,atol=1e-10)

    def test_augmented_radius_agrees_with_independent_lp(self):
        rng=np.random.default_rng(59)
        for scale in [.2,.5,1.,2.]:
            for size in [1,2,4]:
                f=rng.uniform(.01,.5,(2,size,size))*scale
                for a in f:np.fill_diagonal(a,0)
                u=rng.uniform(.01,.15,(2,size))
                with np.errstate(divide='ignore'):lf=np.log(f)
                coeff=(lf,np.zeros_like(f),np.log(u),np.zeros_like(u))
                feasible=True
                for d,cap in enumerate([self.cfg.ul_power_w,self.cfg.dl_power_w]):
                    out=linprog(np.ones(size),A_ub=f[d]-np.eye(size),b_ub=-u[d],
                                bounds=[(0,cap)]*size,method='highs')
                    feasible &= out.success
                self.assertEqual(feasible,convex_feasibility(self.cfg,coeff,0)<=0)
                self.assertEqual(feasible,minimum_power(self.cfg,coeff) is not None)

    def test_interval_has_no_holes(self):
        s=self.sc;direction,cov=fantasy(s['covariance'],int(s['candidates'][0]),.05)
        for group in subsets(self.cfg.n):
            coeff=coefficients(self.cfg,s['mean'],cov,s['payload'],1,group,direction)
            interval=feasibility_interval(self.cfg,coeff)
            for z in np.linspace(-5.9,5.9,35):
                flag=minimum_power(self.cfg,coeff,float(z)) is not None
                predicted=interval is not None and interval[0]<=z<=interval[1]
                self.assertEqual(flag,predicted)

    def test_log_spectral_function_is_convex(self):
        s=self.sc;direction,cov=fantasy(s['covariance'],3,.05)
        coeff=coefficients(self.cfg,s['mean'],cov,s['payload'],1,np.arange(4),direction)
        x=np.linspace(-5,5,101)
        y=np.array([convex_feasibility(self.cfg,coeff,float(z)) for z in x])
        self.assertGreaterEqual(np.diff(y,2).min(),-1e-10)

    def test_closed_form_envelope(self):
        intervals=[(-2,1,3),(0,3,5)]
        expected=3*(ndtr(0)-ndtr(-2))+5*(ndtr(3)-ndtr(0))
        self.assertAlmostEqual(expected,expected_envelope(intervals),places=12)

    def test_innovation_integral_against_independent_grid(self):
        s=self.sc;index=int(s['candidates'][2]);rv=(self.cfg.measurement_db*np.log(10)/10)**2
        value=probe_value(self.cfg,s['mean'],s['covariance'],s['payload'],0,index)
        direction,cov=fantasy(s['covariance'],index,rv)
        # Deterministic normal-quantile integration using the ordinary admission solver.
        from scipy.special import ndtri
        z=ndtri((np.arange(1200)+.5)/1200)
        reference=np.mean([best_schedule(self.cfg,s['mean']+direction*t,cov,s['payload'],1)['value'] for t in z])
        self.assertLess(abs(value-reference),.15)

    def test_covariance_is_psd_and_geometry_is_reciprocal(self):
        s=self.sc
        self.assertGreater(np.linalg.eigvalsh(s['covariance']).min(),0)
        a=s['mean'].reshape(2,4,4)
        np.testing.assert_allclose(a[0],a[1].T)


if __name__=='__main__':unittest.main(verbosity=2)
