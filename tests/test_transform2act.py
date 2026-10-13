"""Bounded software tests; synthetic fixtures never enter research datasets."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from pathlib import Path
import tempfile
import json
import ctypes
import sys
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'src'))


def run(ppo_smoke=False):
    from runtime import memory
    mem = memory()
    if mem['physical_MiB'] < 2800 or mem['commit_MiB'] < 3100:
        print(json.dumps(dict(status='WAITING_FOR_MEMORY', memory=mem)))
        return 75
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from t2a_task_adapter import TransformControlEnv, TransformControlPolicy
    from paper_env import PaperClimbEnv, build_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(314159)
    root = PACKAGE
    nominal = root / 'preflight_output/parallel/nominal.xml'
    if not nominal.exists():build_model(nominal,'parallel',[1.]*8)
    env = TransformControlEnv('parallel', nominal, 2026090701)
    before, _ = env.reset(seed=2026090701)
    policy = TransformControlPolicy(env.observation_space, env.action_space, lambda _: .0003)
    # Mixed-stage batch: inactive actions and entropy must be truly masked.
    obs = {k: torch.as_tensor(np.stack([v] * 4)) for k, v in before.items()}
    obs['phase'][2:] = 1.
    action, value, logp = policy(obs)
    assert action.shape == (4, 12) and value.shape == (4, 1)
    assert torch.all(action[:2, 8:] == 0) and torch.all(action[2:, :8] == 0)
    alternate = action.detach().clone()
    alternate[:2, 8:] = 123.
    alternate[2:, :8] = -123.
    _, lp2, entropy = policy.evaluate_actions(obs, alternate)
    assert torch.allclose(logp, lp2)
    assert torch.allclose(entropy[:2], entropy[2:] * 2.)
    altered_obs = {k: v.clone() for k, v in obs.items()}
    altered_obs['nodes'][:, :, :7] += 20.
    m1 = policy.get_distribution(obs).distribution.mean
    m2 = policy.get_distribution(altered_obs).distribution.mean
    assert torch.allclose(m1[:, :8], m2[:, :8])  # attr actor cannot read physical state
    # Detach sampled actions, as PPO's stored rollout does, for score gradients.
    values, likelihood, _ = policy.evaluate_actions(obs, action.detach())
    policy.optimizer.zero_grad()
    (-(likelihood * torch.tensor([1., -.3, .7, -1.1])).mean() + values.square().mean()).backward()
    for head in (policy.attr_head, policy.control_head, policy.value_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all() and torch.count_nonzero(head.weight.grad) > 0
    # Exactly the common frozen morphology/environment compiler and physical step.
    design = np.zeros(12)
    after, reward, term, trunc, info = env.step(design)
    assert not term and not trunc and reward == 0. and after['phase'][0] == 1.
    reference = PaperClimbEnv(nominal)
    reference.reset(seed=2026090701)
    for name in ('body_mass', 'jnt_pos', 'geom_pos', 'actuator_trnid', 'actuator_gear'):
        assert np.allclose(getattr(env.core.model, name), getattr(reference.model, name))
    controls = np.array([.2, -.1, .3, -.4])
    _, r1, _, _, _ = env.step(np.r_[np.zeros(8), controls])
    _, r2, _, _, _ = reference.step(controls)
    assert r1 == r2 and np.array_equal(env.core.data.qpos, reference.data.qpos)
    assert env.total_control_steps == 1 and env.total_design_decisions == 1
    assert env.core.raw_step == 20
    assert np.all(env.core.data.ctrl[[1, 3, 5, 7]] == 0)
    reference.close()
    with tempfile.TemporaryDirectory(prefix='t2a_geometry_test_') as temp:
        # Non-nominal geometry must also equal the common builder exactly.
        path = Path(temp) / 'changed.xml'
        vector = np.ones(8); vector[3] = .96
        build_model(path, 'parallel', vector)
        reference = PaperClimbEnv(path); reference.reset(seed=2026090701)
        env.reset(seed=2026090701)
        _, _, term, _, _ = env.step(np.r_[(vector - 1.) / .1, np.zeros(4)])
        assert not term
        for name in ('body_mass', 'jnt_pos', 'geom_pos', 'qpos0'):
            assert np.allclose(getattr(env.core.model, name), getattr(reference.model, name))
        reference.close()
    env.close()
    result = dict(status='UNIT_TESTS_PASSED_NOT_RESEARCH_DATA',
                  stage_probability_mask=True, actor_gradient_checks=True,
                  common_geometry_and_dynamics=True, ppo_smoke=False)
    if ppo_smoke:
        # Caller must ensure there is no other active trainer before enabling.
        base = DummyVecEnv([lambda: TransformControlEnv('parallel', nominal, 2026092101)])
        norm = VecNormalize(base, norm_obs=True, norm_reward=False, norm_obs_keys=['nodes'])
        model = PPO(TransformControlPolicy, norm, n_steps=500, batch_size=250, n_epochs=2,
                    seed=2026092101, device='cpu', verbose=0)
        a0 = model.policy.attr_head.weight.detach().clone()
        c0 = model.policy.control_head.weight.detach().clone()
        model.learn(1000)
        assert not torch.equal(a0, model.policy.attr_head.weight)
        assert not torch.equal(c0, model.policy.control_head.weight)
        with tempfile.TemporaryDirectory(prefix='t2a_reload_test_') as temp:
            cp = Path(temp) / 'controller.zip'; normalizer = Path(temp) / 'norm.pkl'
            model.save(cp); norm.save(normalizer)
            observation = norm.reset()
            a1, _ = model.predict(observation, deterministic=True)
            loaded = PPO.load(cp, env=norm, device='cpu')
            a2, _ = loaded.predict(observation, deterministic=True)
            assert np.array_equal(a1, a2)
        core = base.envs[0]
        result.update(ppo_smoke=True, ppo_transitions=model.num_timesteps,
                      simulation_control_steps=core.total_control_steps,
                      design_decisions=core.total_design_decisions,
                      invalid_designs=core.total_invalid_designs,
                      save_reload_identity=True)
        norm.close()
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    if os.name=='nt':ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppo-smoke', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.ppo_smoke))
