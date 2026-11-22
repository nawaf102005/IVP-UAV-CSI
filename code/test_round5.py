import unittest
import numpy as np
from scipy import stats
from r5_statistics import tost,holm,mean_ci
class RoundFiveTests(unittest.TestCase):
    def test_tost_matches_ci_duality_and_sign_symmetry(self):
        rng=np.random.default_rng(887)
        for mean in [0,.3,.5,.9]:
            x=rng.normal(mean,.9,256);r=tost(x)
            self.assertEqual(r['p']<.05,r['lo']>-.5 and r['hi']<.5)
            self.assertAlmostEqual(r['p'],tost(-x)['p'],12)
    def test_boundary_cannot_pass(self):
        x=.5+np.linspace(-1,1,256);self.assertGreaterEqual(tost(x)['p'],.499999)
        self.assertEqual(tost(np.zeros(256))['p'],1.)
    def test_holm_known_family_and_permutation(self):
        p=[.01,.04,.03,.002];np.testing.assert_allclose(holm(p),[.03,.06,.06,.008])
        np.testing.assert_allclose(holm(p)[::-1],holm(p[::-1]))
    def test_bonferroni_intervals_expand(self):
        x=np.arange(256)/256;r=mean_ci(x);s=mean_ci(x,.05/14)
        self.assertLess(s['lo'],r['lo']);self.assertGreater(s['hi'],r['hi'])
if __name__=='__main__':unittest.main()
