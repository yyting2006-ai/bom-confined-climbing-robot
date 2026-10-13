"""Serial paper comparison matrix; prints a plan unless --execute is explicit."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/paper_retrain.json')
    p.add_argument('--group',choices=['baseline','ablation','all','non-api'],default='ablation')
    p.add_argument('--output',type=Path,default=ROOT/'runs');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-api',action='store_true');p.add_argument('--max-api-calls',type=int,default=0)
    a=p.parse_args();c=json.loads(a.config.read_text())
    methods={'baseline':c['baseline_methods'],'ablation':['bom']+c['ablation_methods'],'all':c['baseline_methods']+c['ablation_methods'],'non-api':[m for m in c['baseline_methods']+c['ablation_methods'] if m!='robomorph_llm']}[a.group]
    jobs=[(s,r,m) for r in range(1,c['repeats']+1) for s in c['scenes'] for m in methods]
    print(json.dumps({'jobs':len(jobs),'methods':methods,'max_parallel_workers':1,'execute':a.execute},indent=2),flush=True)
    if not a.execute:return
    if 'robomorph_llm' in methods and (not a.allow_api or a.max_api_calls<1):p.error('Baseline/all includes live LLM. Select non-api or explicitly enable a bounded API budget.')
    for scene,repeat,method in jobs:
        if (a.output/'STOP_REQUESTED').exists():raise SystemExit('Stopped between jobs by STOP_REQUESTED')
        command=[sys.executable,str(ROOT/'run.py'),'--config',str(a.config.resolve()),'--output',str(a.output.resolve()),'--scene',scene,'--repeat',str(repeat),'--method',method,'--device',a.device,'--execute']
        if a.allow_api:command+=['--allow-api','--max-api-calls',str(a.max_api_calls)]
        print(f'Running {scene} repeat {repeat}: {method}',flush=True)
        subprocess.run(command,check=True,cwd=ROOT)
if __name__=='__main__':main()
