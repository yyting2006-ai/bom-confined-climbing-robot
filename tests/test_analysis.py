"""Synthetic algebra fixtures for statistics, not scientific evidence."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_matrix import estimate, summarize


class Analysis(unittest.TestCase):
    def test_nesting(self):
        result=estimate([.1,.2,.3])
        self.assertEqual(result['n'],3)
        self.assertAlmostEqual(result['mean'],.2)
        self.assertAlmostEqual(result['sd'],.1)
        self.assertAlmostEqual(result['ci95_t'][1]-.2,4.302652729911275*.1/np.sqrt(3))

    def test_single_repetition_has_no_uncertainty_estimate(self):
        result=estimate([.4])
        self.assertIsNone(result['sd'])
        self.assertIsNone(result['ci95_t'])

    def test_paired_order(self):
        manifest={'config':{'scenes':['fixture'],'protocol':'synthetic'},'methods':['bom','fixed']}
        cells=[dict(scene='fixture',repeat=r,method=m,completion=v,
                    hold_success_fraction=v,discounted_return=10*v)
               for m,values in [('bom',[.3,.6,.9]),('fixed',[.2,.5,.8])]
               for r,v in enumerate(values,1)]
        cells.reverse()
        result=summarize(manifest,cells)
        np.testing.assert_allclose(result['paired_effects'][0]['completion_effect']['values'],[.1,.1,.1])
        self.assertEqual(result['paired_effects'][0]['completion_effect']['n'],3)


if __name__=='__main__':
    unittest.main()
