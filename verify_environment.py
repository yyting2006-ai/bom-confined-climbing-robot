"""Deterministic model, integration, metric and runtime checks."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from pathlib import Path
import sys
import time
import json
from importlib.metadata import version
import hashlib

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mujoco
from paper_env import PaperClimbEnv,build_model,load_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows=[]
    for scene in ('parallel','tilted','stepped','curved'):
        path=args.output/scene/'nominal.xml'
        meta=build_model(path,scene,[1.]*8)
        env=PaperClimbEnv(path)
        obs,_=env.reset(seed=1701)
        expected=np.random.default_rng(1701).normal(0,.002,env.model.nq)
        np.testing.assert_allclose(env.data.qpos,env.model.qpos0+expected)
        assert obs.shape==(56,) and np.isfinite(obs).all()
        assert env.action_space.shape==(4,) and (env.model.nq,env.model.nv,env.model.nu)==(10,10,8)
        assert abs(env.model.body_mass.sum()-56.16331083203798)<1e-8
        assert abs(meta['bottom_m']-.03)<1e-8 and abs(meta['top_m']-1.73)<1e-8
        # Identical state/control: native repeated stepping versus Python calls.
        reference=load_model(path);refdata=mujoco.MjData(reference)
        refdata.qpos[:]=env.data.qpos;refdata.qvel[:]=env.data.qvel
        mujoco.mj_forward(reference,refdata)
        action=np.array([.1,-.05,.03,-.1])
        refdata.ctrl[env.active_ids]=action
        for _ in range(env.action_repeat):mujoco.mj_step(reference,refdata)
        mujoco.mj_forward(reference,refdata)
        env.step(action)
        np.testing.assert_allclose(env.data.qpos,refdata.qpos,atol=1e-12,rtol=0)
        np.testing.assert_allclose(env.data.qvel,refdata.qvel,atol=1e-12,rtol=0)
        env.reset(seed=1701)
        root0=float(env.data.qpos[1]);rewards=[];start=time.perf_counter()
        for t in range(225):
            obs,r,term,trunc,info=env.step(np.zeros(4))
            assert not term and trunc==(t==224)
            assert np.isfinite(obs).all() and np.all(env.data.ctrl[[1,3,5,7]]==0)
            rewards.append(r)
        elapsed=time.perf_counter()-start
        assert abs(sum(rewards)*env.control_dt-(env.data.qpos[1]-root0))<1e-8
        discounted=sum(env.discount**t*r for t,r in enumerate(rewards))
        assert abs(discounted-info['discounted_return'])<1e-8
        assert abs(info['completion_ratio']-np.clip(info['com_gain_m'],0,1))<1e-10
        assert abs(env.data.time-9.)<1e-8
        rows.append(dict(scene=scene,control_steps=225,raw_steps=4500,wall_seconds=elapsed,
                         control_steps_per_second=225/elapsed,zero_action_outcome=info,
                         model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks_passed=True))
        print(json.dumps(rows[-1],allow_nan=False),flush=True)
        env.close()
    report=dict(status='ENVIRONMENT_CHECKS_PASSED',purpose='development validation and zero-action throughput; not trained performance',
                versions={n:version(n) for n in ('numpy','scipy','mujoco','torch','gymnasium','stable-baselines3','scikit-learn')},
                rows=rows,source_sha256=hashlib.sha256((ROOT/'src/paper_env.py').read_bytes()).hexdigest())
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
