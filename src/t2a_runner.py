"""Task-adapted two-stage attribute/control PPO; not official variable-topology T2A."""
import json,time
from pathlib import Path
import runtime
from runtime import memory,read,save,sha,event,MemoryPause,require_memory
OUT=Path('runs/transform2act')
NOMINAL=Path('models/parallel_original_8joint_4act_forceclean.xml')
SEED=2026090701
SEARCH=list(range(2026090901,2026090907))
CONFIRM=list(range(2026091001,2026091011))
GEOMETRY=list(range(SEED,SEED+32))+SEARCH
TARGET=27000000

def evaluate(checkpoint, destination, seeds, *, scene='parallel', nominal=NOMINAL, seed_base=SEED, geometry=GEOMETRY):
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from t2a_task_adapter import TransformControlEnv
    destination = Path(destination)
    if (destination / 'aggregate.json').exists():
        result = read(destination / 'aggregate.json')
        assert result['seeds'] == seeds and result['checkpoint_sha256'] == sha(checkpoint['model'])
        return result
    base = DummyVecEnv([lambda: TransformControlEnv(scene, nominal, seed_base, geometry)])
    norm = VecNormalize.load(checkpoint['normalizer'], base)
    norm.training = False; norm.norm_reward = False
    policy = PPO.load(checkpoint['model'], env=norm, device=runtime.DEVICE)
    core = TransformControlEnv(scene, nominal, seed_base, geometry)
    rows = []
    try:
        for seed in seeds:
            path = destination / f'seed_{seed}.json'
            if path.exists():
                rows.append(read(path)); continue
            obs, _ = core.reset(seed=seed)
            action, _ = policy.predict(norm.normalize_obs(obs), deterministic=True)
            obs, _, terminated, _, design = core.step(action)
            total = 0.; trace = []; actions = []; qpos = []; qvel = []
            if not terminated:
                qpos.append(core.core.data.qpos.copy()); qvel.append(core.core.data.qvel.copy())
                for k in range(225):
                    action, _ = policy.predict(norm.normalize_obs(obs), deterministic=True)
                    obs, reward, term, trunc, info = core.step(action)
                    total += reward; actions.append(action[8:])
                    qpos.append(core.core.data.qpos.copy()); qvel.append(core.core.data.qvel.copy())
                    trace.append(dict(time_s=(k + 1) * .04, com_gain_m=info['com_gain_m']))
                    if term or trunc:
                        break
                row = dict(seed=seed, valid_design=True, reward=total,
                           completion_ratio=info['completion_ratio'], com_gain_m=info['com_gain_m'],
                           wall_contact_fraction=info['wall_contact_fraction'],
                           sampled_height_hold_success=info['sampled_height_hold_success'],
                           rod_length_scales=design['rod_length_scales'])
                destination.mkdir(parents=True, exist_ok=True)
                save(destination / f'trace_{seed}.json', trace)
                np.savez_compressed(destination / f'motion_{seed}.npz', qpos=qpos, qvel=qvel, actions=actions, dt=.04)
            else:
                row = dict(seed=seed, valid_design=False, reward=0., completion_ratio=0.,
                           sampled_height_hold_success=False, com_gain_m=None,
                           wall_contact_fraction=None,
                           rod_length_scales=design['rod_length_scales'], reason=design['rejection'])
            save(path, row); rows.append(row)
        # The attribute actor cannot condition the morphology on evaluation state.
        assert all(np.allclose(r['rod_length_scales'], rows[0]['rod_length_scales']) for r in rows)
        result = dict(seeds=seeds, mean_reward=float(np.mean([r['reward'] for r in rows])),
                      mean_completion_ratio=float(np.mean([r['completion_ratio'] for r in rows])),
                      sampled_hold_successes=sum(r['sampled_height_hold_success'] for r in rows),
                      valid_trials=sum(r['valid_design'] for r in rows), total_trials=len(rows),
                      rod_length_scales=rows[0]['rod_length_scales'], checkpoint_sha256=sha(checkpoint['model']),
                      invalid_design_reward='zero termination reward per reference; not a successful simulation')
        save(destination / 'aggregate.json', result)
        return result
    finally:
        norm.close(); core.close()


def worker(*, out=OUT, nominal=NOMINAL, scene='parallel', seed=SEED,
           geometry=GEOMETRY, search=SEARCH, confirm=CONFIRM, target=TARGET, milestone_steps=3000000):
    require_memory()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from t2a_task_adapter import TransformControlEnv, TransformControlPolicy
    # Explicit per-process experiment context; defaults preserve the existing
    # Parallel pilot. The old running search source is never modified.
    OUT, NOMINAL, SEED, GEOMETRY, SEARCH, CONFIRM, TARGET = (
        Path(out), Path(nominal), int(seed), list(geometry), list(search), list(confirm), int(target))
    evaluation_args = dict(scene=scene, nominal=NOMINAL, seed_base=SEED, geometry=GEOMETRY)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    protocol = read(OUT / 'protocol.json')
    for path, expected in protocol['source_hashes'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen T2A source changed: ' + path)
    journal = OUT / 'committed_checkpoints.jsonl'
    old = [json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
    attempt = len(list(OUT.glob('attempt_*'))) + 1
    work = OUT / f'attempt_{attempt:02d}'; work.mkdir()
    def make_env():
        return TransformControlEnv(scene, NOMINAL, SEED, GEOMETRY,
                                   journal=lambda row: event(work / 'episodes.jsonl', **row))
    base = DummyVecEnv([make_env])
    offset_control = offset_design = offset_invalid = 0
    if old:
        cp = max(old, key=lambda r: r['steps'])
        norm = VecNormalize.load(cp['normalizer'], base); norm.training = True; norm.norm_reward = False
        policy = PPO.load(cp['model'], env=norm, device=runtime.DEVICE, force_reset=True)
        offset_control, offset_design, offset_invalid = cp['control_steps'], cp['design_decisions'], cp['invalid_designs']
    else:
        norm = VecNormalize(base, norm_obs=True, norm_reward=False, norm_obs_keys=['nodes'], clip_obs=10.)
        policy = PPO(TransformControlPolicy, norm, n_steps=500, batch_size=250, n_epochs=4,
                     learning_rate=.0003, gamma=.99, gae_lambda=.95, seed=SEED, device=runtime.DEVICE, verbose=0)
    save(work / 'provenance.json', dict(resumed=bool(old), simulator_rng_restored=False,
                                      official_full_reproduction=False, development_only=True))
    started = time.monotonic(); initial_steps = policy.num_timesteps
    committed_state=dict(steps=initial_steps,control_steps=offset_control,design_decisions=offset_design)
    class Guard(BaseCallback):
        def __init__(self):
            super().__init__(); self.low = False; self.last = initial_steps
        def _on_step(self):
            if self.num_timesteps - self.last >= 512:
                self.last = self.num_timesteps
                mem = memory()
                if mem['physical_MiB'] < 1024 or mem['commit_MiB'] < 1536:
                    core=base.envs[0]
                    event(OUT/'training_events.jsonl',status='stopped_for_memory',steps=self.num_timesteps,
                          last_committed_steps=committed_state['steps'],
                          discarded_steps=self.num_timesteps-committed_state['steps'],
                          discarded_control_steps=offset_control+core.total_control_steps-committed_state['control_steps'],
                          discarded_design_decisions=offset_design+core.total_design_decisions-committed_state['design_decisions'],
                          memory=mem)
                    self.low = True; return False
                if (self.num_timesteps - initial_steps) % 2048 == 0:
                    event(OUT / 'training_events.jsonl', steps=self.num_timesteps, target=TARGET, memory=mem,
                          steps_per_second=(self.num_timesteps-initial_steps)/max(time.monotonic()-started, .01))
            return True
    guard = Guard()
    try:
        while policy.num_timesteps < TARGET:
            until_milestone=milestone_steps-policy.num_timesteps%milestone_steps
            policy.learn(min(25000, TARGET-policy.num_timesteps, until_milestone), reset_num_timesteps=False, callback=guard)
            if guard.low:
                raise MemoryPause()
            step = int(policy.num_timesteps)
            model_file = work / f'controller_{step:08d}.zip'
            norm_file = work / f'normalizer_{step:08d}.pkl'
            policy.save(model_file); norm.save(norm_file)
            core = base.envs[0]
            cp = dict(steps=step, model=str(model_file), normalizer=str(norm_file),
                      control_steps=offset_control+core.total_control_steps,
                      design_decisions=offset_design+core.total_design_decisions,
                      invalid_designs=offset_invalid+core.total_invalid_designs)
            assert cp['steps'] == cp['control_steps'] + cp['design_decisions']
            event(journal, **cp)
            committed_state.update(cp)
            if step % milestone_steps == 0:
                evaluate(cp, OUT / 'search_evaluations' / f'step_{step:08d}', SEARCH, **evaluation_args)
        # Also finishes any missed evaluation if a process stopped after saving.
        rows = [json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()]
        milestones = {r['steps']: r for r in rows if r['steps'] % milestone_steps == 0}
        candidates = []
        for step in range(milestone_steps, TARGET + 1, milestone_steps):
            cp = milestones[step]
            result = evaluate(cp, OUT / 'search_evaluations' / f'step_{step:08d}', SEARCH, **evaluation_args)
            candidates.append(dict(checkpoint=cp, evaluation=result))
        best = max(candidates, key=lambda x: x['evaluation']['mean_reward'])
        selected = OUT / 'selected_endpoint.json'
        if selected.exists():
            assert read(selected) == best
        else:
            save(selected, best)
        confirmed = evaluate(best['checkpoint'], OUT / 'confirmation', CONFIRM, **evaluation_args)
        save(OUT / 'summary.json', dict(status='ONE_SCENE_TASK_ADAPTATION_RUN_COMPLETE_REQUIRES_SUITE_AUDIT', scene=scene,
             method='transform2act_attribute_control_ppo', ppo_transition_budget=TARGET,
             total_control_steps=cp['control_steps'], total_design_decisions=cp['design_decisions'],
             selected_training_step=best['checkpoint']['steps'], confirmation=confirmed,
             official_full_reproduction=False, independent_training_search_runs=1,
             comparison_axis='total PPO training transitions; not number of independently retrained morphologies'))
        return 0
    finally:
        norm.close()

