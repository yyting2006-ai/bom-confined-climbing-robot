"""Probe a development policy's contact with each passage profile."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terrain-start-m',type=float,default=2.1)
    a=p.parse_args()
    import torch
    import runtime
    from paper_env import build_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol=runtime.read(a.run/'protocol.json')
    checkpoint=runtime.read(a.run/'training/complete.json')
    runtime.TRAIN_SEED=protocol['seed']
    runtime.ENV_KWARGS=protocol['environment']
    seeds=protocol['evaluation_seeds']
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for scene in ('parallel','tilted','stepped','curved'):
        folder=a.output/scene
        folder.mkdir()
        xml=folder/'nominal.xml'
        build_model(xml,scene,[1.]*8,physics=protocol['physics'],terrain_start_m=a.terrain_start_m)
        result=runtime.evaluate(xml,checkpoint,folder/'evaluation',seeds,save_motion=True)
        trials=[runtime.read(folder/'evaluation'/f'seed_{seed}.json') for seed in seeds]
        rows.append(dict(scene=scene,mean_completion=result['mean_completion_ratio'],
                         successes=result['sampled_hold_successes'],
                         exposed_trials=sum(r['terrain_contact_fraction']>0 for r in trials),
                         terrain_contact_fractions=[r['terrain_contact_fraction'] for r in trials]))
    result=dict(stage='development_cross_scene_transfer',training_scene=protocol['scene'],
                scene_geometry=dict(terrain_start_m=a.terrain_start_m),rows=rows)
    runtime.save(a.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
