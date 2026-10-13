"""Synthetic software fixtures only; never used as experiment results."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,tempfile,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from search_proposers import bom_proposal,checked_history,text2robot_ea_proposal
from proposals import proposal,fixed_pair_proposal
from deepseek_client import completion,ApiFailure,validate_vector
import run
from unittest.mock import patch
import types
class Methods(unittest.TestCase):
    def setUp(self):self.history=[dict(candidate_id=0,rod_length_scales=[1.]*8,mean_reward=1.)]
    def test_cyclic_dimensions(self):
        for step in range(8):
            p=bom_proposal(self.history,step,23+step)
            self.assertEqual(p['active_rods'],[step]);self.assertTrue(np.array_equal(np.delete(p['rod_length_scales'],step),np.ones(7)))
    def test_global_dimensions(self):self.assertEqual(bom_proposal(self.history,0,77,True)['active_rods'],list(range(8)))
    def test_bad_input(self):
        with self.assertRaises(ValueError):checked_history([])
        with self.assertRaises(ValueError):checked_history([dict(rod_length_scales=[3.]*8,mean_reward=0)])
    def test_all_observations(self):
        self.history.append(dict(candidate_id=1,rod_length_scales=[.98]*8,mean_reward=-1.))
        p=bom_proposal(self.history,2,77);self.assertEqual(p['observations_used'],2)
        self.assertTrue(np.array_equal(np.delete(p['rod_length_scales'],2),np.ones(7)))
    def test_random_rod(self):
        with tempfile.TemporaryDirectory() as t:
            for step in range(10):
                p=proposal('random_rod_bo',self.history,step,23,0,[],[],Path(t)/f'{step}.json','parallel',[0,1])
                rod=int(np.random.default_rng(24+step).integers(8));self.assertEqual(p['active_rods'],[rod])
                self.assertTrue(np.array_equal(np.delete(p['rod_length_scales'],rod),np.ones(7)))
    def test_ea_and_fixed_pair(self):
        e=text2robot_ea_proposal(self.history,0,1);self.assertEqual(len(e['rod_length_scales']),8)
        p=fixed_pair_proposal(self.history,7,[0,1]);self.assertEqual(p['active_rods'],[0,1]);self.assertEqual(p['rod_length_scales'][2:],[1.]*6)
    def test_no_implicit_api(self):
        with self.assertRaises(ApiFailure):completion([])
        with self.assertRaises(ApiFailure):validate_vector({'rod_length_scales':[True]*8})
    def test_seeds_and_config(self):
        c=run.checked_config(ROOT/'configs/paper_retrain.json');self.assertNotIn('fixed_pair_2d',c['ablation_methods'])
        seeds=[]
        for scene in c['scenes']:
            for r in range(1,4):
                x=run.context(c,scene,r);seeds+=list(range(x['train_seed'],x['train_seed']+32))+x['search_seeds']+x['confirmation_seeds']
        self.assertEqual(len(seeds),len(set(seeds)))
    def check_loop(self,method,mode,expected_training_calls):
        config=run.checked_config(ROOT/'configs/software_smoke.json');config['controller_mode']=mode
        case=run.context(config,'parallel',1)
        def build(path,*args,**kwargs):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('<test-only/>')
        def evaluate(model,cp,dest,seeds,save_motion=False):
            return dict(mean_reward=-2. if 'candidate_' in str(dest) else 1.,total_trials=len(seeds))
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,{'paper_env':types.SimpleNamespace(build_model=build)}),patch('runtime.check_geometry',return_value=(True,{})),patch('runtime.train',return_value={'model':'test-model','normalizer':'test-normalizer'}) as training,patch('runtime.evaluate',side_effect=evaluate):
            result=run.run_cell(Path(t),config,case,method)
            self.assertEqual(training.call_count,expected_training_calls)
            self.assertEqual(result['selected_candidate'],0)
            self.assertEqual(len(result['search_observations']),1 if method=='fixed' else 2)
            if method!='fixed':self.assertEqual(result['search_observations'][1]['mean_reward'],-2.)
    def test_full_observation_conservative_loop(self):self.check_loop('bom','retrain',2)
    def test_frozen_loop(self):self.check_loop('bom','frozen',1)
    def test_fixed_loop(self):self.check_loop('fixed','retrain',1)
if __name__=='__main__':unittest.main()
