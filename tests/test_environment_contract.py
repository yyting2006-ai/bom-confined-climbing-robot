"""Numerical software tests; generated trajectories are not paper trials."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import run
from paper_env import PaperClimbEnv, build_model, load_model
from parallel_eight_rod_morphology import EightRodMorphology, NOMINAL_LENGTHS_M


class EnvironmentContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='bom_contract_')
        cls.path = Path(cls.temp.name) / 'nominal.xml'
        cls.physics = json.loads((ROOT / 'configs/contact_4ms.json').read_text())
        cls.metadata = build_model(cls.path, 'parallel', [1.]*8, physics=cls.physics)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_physics_and_actuator_contract(self):
        model = load_model(self.path)
        self.assertEqual(mujoco.__version__, '2.3.3')
        self.assertEqual(model.opt.timestep, .002)
        self.assertTrue(np.allclose(model.geom_solref, [.004, 1]))
        self.assertTrue(np.allclose(model.eq_solref, [.004, 1]))
        self.assertTrue(np.array_equal(model.opt.gravity, [0, 0, -9.81]))
        self.assertEqual((model.nq,model.nv,model.nu), (10,10,8))
        self.assertEqual(self.metadata['physics_overrides'], self.physics)

    def test_reward_identities(self):
        for height in ('root','com'):
            env = PaperClimbEnv(self.path, horizon_raw_steps=400,
                                reward_height=height, discount=.97)
            try:
                env.reset(seed=501)
                rewards = []
                for t in range(20):
                    _, r, terminal, truncated, info = env.step(np.zeros(4))
                    rewards.append(r)
                    self.assertFalse(terminal)
                    self.assertEqual(truncated,t==19)
                    self.assertTrue(np.array_equal(env.data.ctrl[[1,3,5,7]],np.zeros(4)))
                self.assertAlmostEqual(sum(rewards)*env.control_dt,info[f'{height}_gain_m'],places=11)
                self.assertAlmostEqual(sum(.97**i*r for i,r in enumerate(rewards)),info['discounted_return'],places=11)
                self.assertAlmostEqual(env.data.time,.8,places=10)
                self.assertAlmostEqual(info['completion_ratio'],np.clip(info['com_gain_m'],0,1),places=12)
            finally:
                env.close()

    def test_seed_partition_four_workers(self):
        config=run.checked_config(ROOT/'configs/paper_retrain.json')
        seeds=[]
        for scene in config['scenes']:
            for repeat in range(1,4):
                case=run.context(config,scene,repeat)
                seeds.extend(range(case['train_seed'],case['train_seed']+128))
                seeds.extend(case['search_seeds'])
                seeds.extend(case['confirmation_seeds'])
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertTrue(set(seeds).isdisjoint(range(41001,43000)))

    def test_morphology_lengths_and_closure(self):
        rng=np.random.default_rng(611)
        for _ in range(20):
            target=NOMINAL_LENGTHS_M*rng.uniform(.9,1.1,8)
            morphology=EightRodMorphology(*target)
            vertices=morphology.vertices()
            measured=np.linalg.norm(np.roll(vertices,-1,axis=0)-vertices,axis=1)
            np.testing.assert_allclose(measured,target,rtol=0,atol=1e-8)

    def test_bad_actions_rejected(self):
        env=PaperClimbEnv(self.path)
        try:
            env.reset(seed=511)
            for action in (np.zeros(8),np.full(4,np.nan)):
                with self.assertRaises(ValueError):
                    env.step(action)
        finally:
            env.close()

    def test_shifted_profile_onset(self):
        nominal=load_model(self.path)
        for scene in ('tilted','stepped','curved'):
            path=Path(self.temp.name)/f'{scene}_180.xml'
            metadata=build_model(path,scene,[1.]*8,physics=self.physics,terrain_start_m=1.8)
            self.assertAlmostEqual(metadata['right_wall_sections'][0]['end_xz_m'][1],1.8)
            self.assertAlmostEqual(metadata['right_wall_sections'][-1]['end_xz_m'][1],10.)
            self.assertLess(metadata['top_m'],1.8)
            np.testing.assert_array_equal(load_model(path).body_mass,nominal.body_mass)


if __name__ == '__main__':
    unittest.main()
