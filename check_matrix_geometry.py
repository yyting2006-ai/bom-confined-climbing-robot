"""Check every nominal training/search reset before an experiment matrix."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import run
import runtime
from paper_env import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    c=run.checked_config(a.config)
    a.output.mkdir(parents=True,exist_ok=False)
    runtime.N_ENVS=c['parallel_envs']
    runtime.ENV_KWARGS=c['environment']
    rows=[]
    for scene in c['scenes']:
        xml=a.output/f'{scene}.xml'
        metadata=build_model(xml,scene,[1.]*8,physics=c['physics'],**c['scene_geometry'])
        for repetition in range(1,c['repeats']+1):
            case=run.context(c,scene,repetition)
            run.bind(case)
            valid,reason=runtime.check_geometry(xml)
            rows.append(dict(case=case['case_id'],valid=valid,reason=reason,model_sha256=metadata['model_sha256']))
    result=dict(protocol=c['protocol'],config_sha256=runtime.sha(a.config),
                stage='geometry_only_no_confirmation_outcomes',rows=rows,
                passed=all(r['valid'] for r in rows))
    runtime.save(a.output/'report.json',result)
    print(result,flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
