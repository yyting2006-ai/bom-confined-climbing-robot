"""Train a nominal development policy and evaluate development reset seeds."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import sys
import time
from importlib.metadata import version

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',default='parallel',choices=['parallel','tilted','stepped','curved'])
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--envs',type=int,default=1,choices=[1,2,4])
    p.add_argument('--seed',type=int,default=41001)
    p.add_argument('--horizon-raw-steps',type=int,default=4500)
    p.add_argument('--target-gain-m',type=float,default=1.)
    p.add_argument('--discount',type=float,default=.99)
    p.add_argument('--reward-height',choices=['root','com'],default='root')
    p.add_argument('--physics',type=Path)
    p.add_argument('--terrain-start-m',type=float,default=2.1)
    a=p.parse_args()
    import torch
    import runtime
    from paper_env import build_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime.TRAIN_SEED=a.seed
    runtime.SEARCH_SEEDS=list(range(a.seed+1000,a.seed+1005))
    runtime.N_ENVS=a.envs
    runtime.ENV_KWARGS=dict(horizon_raw_steps=a.horizon_raw_steps,target_gain_m=a.target_gain_m,discount=a.discount,reward_height=a.reward_height)
    physics=json.loads(a.physics.read_text(encoding='utf-8')) if a.physics else None
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(stage='exploratory_development',scene=a.scene,steps=a.steps,n_envs=a.envs,seed=a.seed,
                  evaluation_seeds=runtime.SEARCH_SEEDS,environment=runtime.ENV_KWARGS,physics=physics,
                  scene_geometry=dict(terrain_start_m=a.terrain_start_m),
                  source_hashes={str(x.relative_to(ROOT)):runtime.sha(x) for x in [Path(__file__),*(ROOT/'src').glob('*.py')]},
                  versions={n:version(n) for n in ('numpy','scipy','mujoco','torch','gymnasium','stable-baselines3','scikit-learn')})
    spec=a.output/'protocol.json'
    if spec.exists():
        if runtime.read(spec)!=protocol:raise RuntimeError('Development output belongs to another source or configuration')
    else:runtime.save(spec,protocol)
    xml=a.output/'nominal.xml'
    if not xml.exists():build_model(xml,a.scene,[1.]*8,physics=physics,terrain_start_m=a.terrain_start_m)
    valid,reason=runtime.check_geometry(xml)
    if not valid:raise RuntimeError(str(reason))
    start=time.perf_counter()
    checkpoint=runtime.train(xml,a.output/'training',a.steps)
    training_seconds=time.perf_counter()-start
    result=runtime.evaluate(xml,checkpoint,a.output/'evaluation',runtime.SEARCH_SEEDS,save_motion=True)
    receipt=dict(status='DEVELOPMENT_TRAINING_COMPLETE',training_wall_seconds=training_seconds,
                 steps=a.steps,steps_per_second=a.steps/max(training_seconds,.01),evaluation=result)
    runtime.save(a.output/'receipt.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
