import csv
import math
import tempfile
import unittest
from pathlib import Path
from scipy.stats import fisher_exact
from abtest.core import plan, validate_plan, simulate, load_data, analyze_counts, analyze_file, monte_carlo


class ExperimentTests(unittest.TestCase):
    def test_sample_size_increases_for_smaller_effect(self):
        self.assertGreater(plan(mde=.01)['n_per_group'], plan(mde=.02)['n_per_group'])
        self.assertGreater(plan(power=.9)['n_per_group'], plan(power=.8)['n_per_group'])
    def test_invalid_plan(self):
        for kwargs in [dict(baseline=0), dict(mde=0), dict(baseline=.99,mde=.02), dict(alpha=1), dict(power=float('nan')), dict(daily_users=0)]:
            with self.assertRaises(ValueError): plan(**kwargs)
        p=plan();p['n_per_group']=1
        with self.assertRaises(ValueError): validate_plan(p)
    def test_simulation_reproducible_and_balanced(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.csv',Path(d)/'b.csv'
            simulate(a,100,seed=42);simulate(b,100,seed=42)
            self.assertEqual(a.read_bytes(),b.read_bytes())
            counts=load_data(a);self.assertEqual(counts['A'][0],100);self.assertEqual(counts['B'][0],100)
    def test_obvious_positive_and_null(self):
        positive=analyze_counts(1000,100,1000,300)
        self.assertLess(positive['p_value'],.05)
        self.assertGreater(positive['confidence_interval_difference'][0],0)
        self.assertTrue(positive['practical_threshold_confirmed'])
        same=analyze_counts(1000,100,1000,100)
        self.assertAlmostEqual(same['p_value'],1)
        self.assertFalse(same['statistically_significant'])
    def test_fisher_small_sample(self):
        r=analyze_counts(10,1,10,4)
        expected=fisher_exact([[4,6],[1,9]]).pvalue
        self.assertAlmostEqual(r['p_value'],expected)
        self.assertIn('Fisher',r['test_method'])
    def test_zero_and_full_conversion_finite(self):
        for successes in (0,10):
            r=analyze_counts(10,successes,10,successes)
            self.assertEqual(r['p_value'],1.)
            self.assertTrue(all(math.isfinite(x) for x in r['confidence_interval_difference']))
        self.assertIsNone(analyze_counts(10,0,10,1)['relative_uplift'])
    def test_srm_blocks_decision(self):
        r=analyze_counts(1000,100,100,30)
        self.assertTrue(r['srm_flag'])
        self.assertTrue(r['decision'].startswith('invalid_allocation'))
    def test_underpowered_sample_not_final(self):
        r=analyze_counts(100,10,100,40,planned_n=1000)
        self.assertFalse(r['planned_sample_reached'])
        self.assertTrue(r['decision'].startswith('incomplete'))
    def test_duplicate_or_invalid_outcome_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.csv'
            p.write_text('user_id,group,converted\nu1,A,0\nu1,B,1\n')
            with self.assertRaises(ValueError):load_data(p)
            p.write_text('user_id,group,converted\nu1,A,2\nu2,B,1\n')
            with self.assertRaises(ValueError):load_data(p)
    def test_missing_arm_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.csv';p.write_text('user_id,group,converted\nu1,A,0\n')
            with self.assertRaises(ValueError):load_data(p)
    def test_monte_carlo_rough_calibration(self):
        # Wide tolerances catch gross mistakes without asserting an exact random outcome.
        null=monte_carlo(4000,.1,.1,repetitions=500,seed=123)
        alt=monte_carlo(4000,.1,.12,repetitions=500,seed=456)
        self.assertLess(null['rejection_frequency'],.10)
        self.assertGreater(alt['rejection_frequency'],.65)
        self.assertGreater(alt['rejection_frequency'],null['rejection_frequency'])

if __name__=='__main__':unittest.main()
