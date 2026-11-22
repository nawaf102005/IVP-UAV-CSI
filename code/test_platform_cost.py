import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from model import Config
from mobility_cost_study import make_case,run_policy
from platform_cost_study import shadow_correlation,lookahead_policy
from analyze_platform_cost import crossings,first_boundary

class PlatformCostTests(unittest.TestCase):
    def test_nonmonotone_boundary_retains_later_crossings(self):
        x=[0.,1.,2.,3.];y=[2.,-2.,2.,-2.]
        self.assertEqual(crossings(x,y),[.5,1.5,2.5])
        self.assertEqual(first_boundary(x,y),(.5,'crossed'))
    def test_boundary_distinguishes_no_entry_and_censoring(self):
        self.assertEqual(first_boundary([0.,1.],[-1.,2.]),(0.,'nonpositive_at_zero'))
        self.assertEqual(first_boundary([0.,1.],[2.,1.]),(1.,'right_censored'))
    def test_hypothetical_cost_does_not_depend_on_host_clock(self):
        cfg=Config();sc,jm,jc=make_case(cfg,93999999,.96)
        with patch('mobility_cost_study.time.perf_counter',side_effect=[0.,100.]):a=run_policy(cfg,sc,jm,jc,'Predictive-IVP',reserve_s=.0001,enforce_elapsed=False)
        with patch('mobility_cost_study.time.perf_counter',side_effect=[0.,.000001]):b=run_policy(cfg,sc,jm,jc,'Predictive-IVP',reserve_s=.0001,enforce_elapsed=False)
        for k in ['selected','observations','probes','completed_kbit','outage','reserved_s']:self.assertEqual(a[k],b[k])
        self.assertEqual(a['deadline_overrun'],0);self.assertNotEqual(a['wall_s'],b['wall_s'])
    def test_shadow_age_uses_one_over_e_length(self):
        self.assertEqual(shadow_correlation(0.,.1,10.),1.)
        self.assertAlmostEqual(shadow_correlation(20.,.5,10.),np.exp(-1.),places=14)
        r=shadow_correlation(20.,.02,10.)
        self.assertAlmostEqual(r**10,shadow_correlation(20.,.2,10.),places=14)
    def test_normalized_time_and_load_scaling(self):
        cfg=replace(Config(),probe_s=.0005);scale=5.
        other=replace(cfg,frame_s=cfg.frame_s*scale,probe_s=cfg.probe_s*scale,ul_bits=cfg.ul_bits*scale,dl_bits=cfg.dl_bits*scale,gain_tol_kbit=cfg.gain_tol_kbit*scale)
        a=make_case(cfg,93999998,.96);b=make_case(other,93999998,.96)
        x=run_policy(cfg,*a,'Predictive-IVP',reserve_s=.0001,enforce_elapsed=False);y=run_policy(other,*b,'Predictive-IVP',reserve_s=.0001*scale,enforce_elapsed=False)
        self.assertEqual(x['selected'],y['selected']);self.assertEqual(x['observations'],y['observations']);self.assertAlmostEqual(x['completed_kbit']*scale,y['completed_kbit'],places=9)
    def test_two_step_has_no_value_for_independent_execution(self):
        cfg=replace(Config(),max_probes=2);sc,jm,jc=make_case(cfg,93999997,0.)
        x=lookahead_policy(cfg,sc,jm,jc,0.);y=run_policy(cfg,sc,jm,jc,'Predictive-NR',enforce_elapsed=False)
        self.assertEqual(x['probes'],0);self.assertEqual(x['selected'],y['selected']);self.assertEqual(x['completed_kbit'],y['completed_kbit'])
if __name__=='__main__':unittest.main()
