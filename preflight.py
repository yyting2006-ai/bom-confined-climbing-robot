"""Short simulator checks, no PPO training and no API requests."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'preflight_output');a=p.parse_args()
import runtime
if runtime.memory()['physical_MiB']<1024:raise SystemExit('Below 1 GiB free RAM; preflight not started')
import numpy as np
from paper_env import PaperClimbEnv,build_model
results=[]
for scene in ['parallel','tilted','stepped','curved']:
    xml=a.output/scene/'nominal.xml'
    if not xml.exists():build_model(xml,scene,[1.]*8)
    env=PaperClimbEnv(xml);obs,_=env.reset(seed=123)
    assert obs.shape==(56,) and np.isfinite(obs).all() and env.action_space.shape==(4,)
    for t in range(225):
        before=float(env.data.qpos[1]);obs,r,term,trunc,info=env.step(np.zeros(4))
        assert np.isclose(r,(float(env.data.qpos[1])-before)/env.control_dt)
        assert np.all(env.data.ctrl[[1,3,5,7]]==0) and np.isfinite(obs).all()
    assert trunc and not term and 0<=info['completion_ratio']<=1
    results.append(dict(scene=scene,steps=225,observation_dim=56,action_dim=4,velocity_reward_verified=True,inactive_actuators_zero=True))
    env.close()
report=dict(status='SOFTWARE_PREFLIGHT_ONLY_NOT_EXPERIMENT_RESULTS',scenes=results,ppo_training_steps=0,api_calls=0)
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
