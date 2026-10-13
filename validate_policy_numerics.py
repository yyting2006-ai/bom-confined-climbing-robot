"""Closed-loop time-step sensitivity and raw-step contact audit of a saved PPO."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from paper_env import PaperClimbEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_variant(xml,checkpoint,environment,seeds,dest):
    core=PaperClimbEnv(xml,**environment)
    base=DummyVecEnv([lambda:PaperClimbEnv(xml,**environment)])
    norm=VecNormalize.load(checkpoint['normalizer'],base)
    norm.training=False
    norm.norm_reward=False
    policy=PPO.load(checkpoint['model'],device='cpu')
    native_step=mujoco.mj_step
    raw=[]

    def audited_step(model,data,nstep=1):
        for _ in range(nstep):
            native_step(model,data)
            contacts=[c for c in data.contact
                      if (int(c.geom1) in core.rod_index and int(c.geom2) in core.env_ids)
                      or (int(c.geom2) in core.rod_index and int(c.geom1) in core.env_ids)]
            penetration=max((max(0.,-float(c.dist)) for c in contacts),default=0.)
            raw.append((float(data.time),penetration))

    rows=[]
    try:
        mujoco.mj_step=audited_step
        for seed in seeds:
            obs,_=core.reset(seed=seed)
            raw.clear()
            trace=[]
            for t in range(core.horizon_raw_steps//core.action_repeat):
                action,_=policy.predict(norm.normalize_obs(obs.copy()),deterministic=True)
                obs,r,term,trunc,info=core.step(action)
                trace.append(dict(time_s=(t+1)*core.control_dt,reward=r,**info))
                if term or trunc:
                    break
            assert len(raw)==core.horizon_raw_steps
            raw_array=np.asarray(raw)
            np.savez_compressed(dest/f'raw_contacts_{seed}.npz',time_s=raw_array[:,0],penetration_m=raw_array[:,1])
            (dest/f'trace_{seed}.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
            rows.append(dict(seed=seed,raw_steps=len(raw),
                        raw_step_sample_max_penetration_m=float(raw_array[:,1].max()),
                        raw_step_sample_p99_penetration_m=float(np.quantile(raw_array[:,1],.99)),
                        warning_counts=[int(w.number) for w in core.data.warning],
                        **info))
    finally:
        mujoco.mj_step=native_step
        core.close()
        norm.close()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--factors',type=int,nargs='+',default=[1,2])
    a=p.parse_args()
    if a.factors[0]!=1 or len(set(a.factors))!=len(a.factors) or any(x not in (1,2,4) for x in a.factors):
        raise ValueError('Use distinct resolution factors starting with 1, drawn from 1,2,4')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol=json.loads((a.run/'protocol.json').read_text(encoding='utf-8'))
    checkpoint=json.loads((a.run/'training/complete.json').read_text(encoding='utf-8'))
    environment=protocol['environment']
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.run/'nominal.xml'
    variants={}
    for factor in a.factors:
        folder=a.output/f'resolution_{factor}'
        folder.mkdir()
        tree=ET.parse(source)
        dt=float(tree.getroot().find('option').get('timestep'))/factor
        tree.getroot().find('option').set('timestep',str(dt))
        xml=folder/'model.xml'
        tree.write(xml,encoding='utf-8',xml_declaration=True)
        env=dict(environment,action_repeat=20*factor,
                 horizon_raw_steps=environment['horizon_raw_steps']*factor)
        rows=evaluate_variant(xml,checkpoint,env,protocol['evaluation_seeds'],folder)
        variants[str(factor)]=dict(physics_step_s=dt,environment=env,model_sha256=sha(xml),rows=rows)
    pairs=[]
    for factor in a.factors[1:]:
        for coarse,fine in zip(variants['1']['rows'],variants[str(factor)]['rows']):
            assert coarse['physical_seed']==fine['physical_seed']
            pairs.append(dict(resolution_factor=factor,seed=coarse['physical_seed'],com_gain_difference_m=fine['com_gain_m']-coarse['com_gain_m'],
                              success_changed=coarse['sampled_height_hold_success']!=fine['sampled_height_hold_success']))
    result=dict(stage='development_closed_loop_numerical_audit',
                policy_sha256=sha(checkpoint['model']),normalizer_sha256=sha(checkpoint['normalizer']),
                source_model_sha256=sha(source),source_protocol_sha256=sha(a.run/'protocol.json'),
                policy_input='same saved policy and normalization; fresh deterministic feedback in each time-step condition',
                contact_sampling='contacts retained after each native mj_step; RK4 last-stage solver samples, not every internal substage',
                variants=variants,paired_time_step_effects=pairs)
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),paired_time_step_effects=pairs),indent=2))


if __name__=='__main__':
    main()
