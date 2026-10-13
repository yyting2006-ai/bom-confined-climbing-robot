"""Portable evaluator/trainer extracted from the manuscript development implementation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import gc,hashlib,json,time,ctypes
from datetime import datetime,timezone
from pathlib import Path
TRAIN_SEED=2026090701
SEARCH_SEEDS=list(range(2026090901,2026090907))
CONFIRM_SEEDS=list(range(2026091001,2026091011))
DEVICE='cpu'
N_ENVS=1
ENV_KWARGS={}


def make_env(model_path,seed_base):
    from paper_env import PaperClimbEnv
    return PaperClimbEnv(model_path,seed_base=seed_base,**ENV_KWARGS)
def memory():
    if os.name=='nt':
        class Status(ctypes.Structure):
            _fields_=[('size',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','commit_limit','commit_available','virtual_total','virtual_available','extended')]
        m=Status();m.size=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return dict(physical_MiB=m.available//1048576,commit_MiB=m.commit_available//1048576)
    import psutil
    available=psutil.virtual_memory().available//1048576
    return dict(physical_MiB=available,commit_MiB=available+psutil.swap_memory().free//1048576)
def require_memory():
    m=memory()
    if m['physical_MiB']<2048 or m['commit_MiB']<2560:raise RuntimeError('Insufficient memory for training; free RAM and retry: '+str(m))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def event(path,**obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**obj),allow_nan=False)+'\n')


def bad_reset(env):
    return any(c.dist < -.001 and (int(c.geom1) in env.env_ids or int(c.geom2) in env.env_ids)
               for c in env.data.contact)


def check_geometry(model_path):
    from paper_env import PaperClimbEnv
    env=make_env(model_path,TRAIN_SEED)
    try:
        for seed in list(range(TRAIN_SEED,TRAIN_SEED+32*N_ENVS))+SEARCH_SEEDS:
            env.reset(seed=seed)
            if bad_reset(env):return False,dict(seed=seed,reason='Initial wall/floor penetration exceeds 1 mm')
        return True,{}
    finally:env.close()


def evaluate(model_path,checkpoint,dest,seeds,save_motion=False):
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from paper_env import PaperClimbEnv
    dest=Path(dest);aggregate=dest/'aggregate.json'
    if aggregate.exists():
        cached=read(aggregate)
        if (cached['seeds']!=seeds or cached['checkpoint_sha256']!=sha(checkpoint['model'])
            or cached.get('normalizer_sha256')!=sha(checkpoint['normalizer'])
            or cached['model_sha256']!=sha(model_path) or cached.get('environment')!=ENV_KWARGS):
            raise RuntimeError('Evaluation cache does not match requested model, checkpoint, or seeds')
        return cached
    dest.mkdir(parents=True,exist_ok=True)
    base=DummyVecEnv([lambda:Monitor(make_env(model_path,TRAIN_SEED))])
    norm=VecNormalize.load(checkpoint['normalizer'],base);norm.training=False;norm.norm_reward=False
    model=PPO.load(checkpoint['model'],env=norm,device=DEVICE);core=make_env(model_path,TRAIN_SEED)
    rows=[]
    try:
        for seed in seeds:
            path=dest/f'seed_{seed}.json'
            if path.exists():rows.append(read(path));continue
            obs,_=core.reset(seed=seed)
            if bad_reset(core):
                row=dict(seed=seed,valid_reset=False,completion_ratio=0.,reward=None,
                         sampled_height_hold_success=False,com_gain_m=None)
            else:
                total=0.;discounted=0.;trace=[];qpos=[core.data.qpos.copy()];qvel=[core.data.qvel.copy()];actions=[]
                for t in range(core.horizon_raw_steps//core.action_repeat):
                    action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
                    obs,r,term,trunc,info=core.step(action);total+=r
                    discounted+=core.discount**t*r
                    trace.append(dict(time_s=(t+1)*core.control_dt,reward=r,**info))
                    qpos.append(core.data.qpos.copy());qvel.append(core.data.qvel.copy());actions.append(action)
                    if term or trunc:break
                expected_gain=info['root_gain_m'] if core.reward_height=='root' else info['com_gain_m']
                if abs(total*core.control_dt-expected_gain)>1e-8 or abs(discounted-info['discounted_return'])>1e-8:
                    raise AssertionError('Reward/height or discount identity failed')
                row=dict(seed=seed,valid_reset=True,reward=discounted,undiscounted_return=total,
                         root_gain_m=info['root_gain_m'],completion_ratio=info['completion_ratio'],
                         com_gain_m=info['com_gain_m'],sampled_height_hold_success=info['sampled_height_hold_success'],
                         wall_contact_fraction=info['wall_contact_fraction'],
                         dual_wall_contact_fraction=info['dual_wall_contact_fraction'],
                         ground_contact_fraction=info['ground_contact_fraction'],terrain_contact_fraction=info['terrain_contact_fraction'],
                         max_control_sample_penetration_m=info['max_control_sample_penetration_m'],
                         max_control_sample_closure_m=info['max_control_sample_closure_m'],peak_com_gain_m=info['peak_com_gain_m'])
                if save_motion:
                    trace_file=dest/f'trace_{seed}.json'
                    if trace_file.exists():assert read(trace_file)==trace
                    else:save(trace_file,trace)
                    motion_file=dest/f'motion_{seed}.npz'
                    if not motion_file.exists():np.savez_compressed(motion_file,qpos=np.asarray(qpos),qvel=np.asarray(qvel),actions=np.asarray(actions),dt=core.control_dt)
            save(path,row);rows.append(row)
        valid=[r['reward'] for r in rows if r['valid_reset']]
        if not valid:raise RuntimeError('No valid evaluations; not imputed as a zero reward')
        result=dict(status='EVALUATED',seeds=seeds,valid_trials=len(valid),total_trials=len(rows),
                    mean_reward=float(np.mean(valid)),variance_of_mean=float(np.var(valid,ddof=1)/len(valid)) if len(valid)>1 else 0.,
                    mean_completion_ratio=float(np.mean([r['completion_ratio'] for r in rows])),
                    sampled_hold_successes=sum(r['sampled_height_hold_success'] for r in rows),
                    objective=f'discounted_{core.reward_height}_vertical_velocity_return',discount=core.discount,environment=ENV_KWARGS,
                    checkpoint_sha256=sha(checkpoint['model']),normalizer_sha256=sha(checkpoint['normalizer']),model_sha256=sha(model_path))
        save(aggregate,result);return result
    finally:
        core.close();norm.close();del model;gc.collect()


def train(model_path,dest,steps=3000000):
    require_memory()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize
    from functools import partial
    from paper_env import PaperClimbEnv
    dest=Path(dest)
    if N_ENVS not in (1,2,4):raise ValueError('Use 1, 2 or 4 environments to preserve disjoint reset seed sets')
    if steps<=0 or steps%(500*N_ENVS):raise ValueError('Use a positive whole-rollout step budget')
    if (dest/'complete.json').exists():
        cached=read(dest/'complete.json')
        if (cached['steps']!=steps or cached['model_sha256']!=sha(model_path)
            or cached.get('n_envs')!=N_ENVS or cached.get('environment')!=ENV_KWARGS):
            raise RuntimeError('Training cache does not match model, environment, or budget')
        return cached
    dest.mkdir(parents=True,exist_ok=True)
    checkpoints=dest/'checkpoints.jsonl'
    old=[json.loads(x) for x in checkpoints.read_text(encoding='utf-8').splitlines() if x.strip()] if checkpoints.exists() else []
    attempt=len(list(dest.glob('attempt_*')))+1;work=dest/f'attempt_{attempt:02d}';work.mkdir()
    factories=[partial(monitored_env,model_path,TRAIN_SEED+32*rank,ENV_KWARGS) for rank in range(N_ENVS)]
    vec=DummyVecEnv(factories) if N_ENVS==1 else SubprocVecEnv(factories,start_method='spawn')
    if old:
        chosen=max(old,key=lambda r:r['steps'])
        if chosen.get('environment')!=ENV_KWARGS or chosen.get('n_envs')!=N_ENVS:
            raise RuntimeError('Checkpoint belongs to another environment configuration')
        norm=VecNormalize.load(chosen['normalizer'],vec);norm.training=True;norm.norm_reward=False
        model=PPO.load(chosen['model'],env=norm,device=DEVICE,force_reset=True)
    else:
        norm=VecNormalize(vec,norm_obs=True,norm_reward=False,clip_obs=10.)
        model=PPO('MlpPolicy',norm,n_steps=500,batch_size=250,n_epochs=4,learning_rate=.0003,gamma=ENV_KWARGS.get('discount',.99),gae_lambda=.95,
                  seed=TRAIN_SEED,device=DEVICE,verbose=0,
                  policy_kwargs=dict(activation_fn=torch.nn.Tanh,net_arch=dict(pi=[128,128],vf=[128,128])))
    save(work/'provenance.json',dict(from_scratch=not old,resumed=bool(old),rng_and_simulator_state_restored=False,
                                   model_sha256=sha(model_path),seed=TRAIN_SEED,n_envs=N_ENVS,environment=ENV_KWARGS,
                                   checkpoint_resume='model/optimizer/normalizer; new simulator reset after process restart'))
    start=time.monotonic();base=model.num_timesteps
    class Guard(BaseCallback):
        def __init__(self):super().__init__();self.last=base;self.low=False
        def _on_step(self):
            for done,info in zip(self.locals.get('dones',[]),self.locals.get('infos',[])):
                if done:event(dest/'episodes.jsonl',steps=self.num_timesteps,training_attempt=attempt,reward=info.get('episode',{}).get('r'),
                    completion_ratio=info.get('completion_ratio'),com_gain_m=info.get('com_gain_m'),physical_seed=info.get('physical_seed'))
            if self.num_timesteps-self.last>=512:
                self.last=self.num_timesteps;mem=memory()
                if mem['physical_MiB']<1024 or mem['commit_MiB']<1536:self.low=True;return False
                if (self.num_timesteps-base)%2048==0:event(dest/'events.jsonl',status='training',steps=self.num_timesteps,target=steps,
                    memory=mem,steps_per_second=(self.num_timesteps-base)/max(time.monotonic()-start,.01))
            return True
    callback=Guard()
    try:
        while model.num_timesteps<steps:
            interval=(25000//(500*N_ENVS))*(500*N_ENVS)
            model.learn(total_timesteps=min(interval,steps-model.num_timesteps),reset_num_timesteps=False,callback=callback)
            if callback.low:raise MemoryPause()
            step=int(model.num_timesteps);cp=work/f'controller_{step:07d}.zip';np=work/f'normalizer_{step:07d}.pkl'
            model.save(cp);norm.save(np)
            record=dict(steps=step,raw_steps=step*ENV_KWARGS.get('action_repeat',20),model=str(cp),normalizer=str(np),
                        model_sha256=sha(model_path),n_envs=N_ENVS,environment=ENV_KWARGS)
            event(checkpoints,**record)
            print(json.dumps(dict(status='checkpoint',steps=step,target=steps,
                                  steps_per_second=(step-base)/max(time.monotonic()-start,.01)),allow_nan=False),flush=True)
        save(dest/'complete.json',record);return record
    finally:norm.close();del model;gc.collect()


class MemoryPause(Exception):pass


def monitored_env(model_path,seed_base,environment):
    from stable_baselines3.common.monitor import Monitor
    from paper_env import PaperClimbEnv
    return Monitor(PaperClimbEnv(model_path,seed_base=seed_base,**environment))

