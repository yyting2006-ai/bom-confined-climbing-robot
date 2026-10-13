"""Paper-method portable entry point. Default is a plan, never automatic training."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import runtime

def context(config,scene,repeat):
    if scene not in config['scenes'] or not 1<=repeat<=config['repeats']:raise ValueError('Scene/repeat outside config')
    offset=config['scenes'].index(scene)*100000+(repeat-1)*10000+config.get('seed_namespace_offset',0)
    return dict(case_id=f'{scene}_r{repeat}',scene=scene,repeat=repeat,train_seed=2026090701+offset,
        search_seeds=list(range(2026090901+offset,2026090901+offset+config['search_rollouts'])),
        confirmation_seeds=list(range(2026091001+offset,2026091001+offset+config['confirmation_rollouts'])),
        proposer_seed_base=2026091200+offset)
def bind(case):
    runtime.TRAIN_SEED=case['train_seed'];runtime.SEARCH_SEEDS=case['search_seeds'];runtime.CONFIRM_SEEDS=case['confirmation_seeds']
def checked_config(p):
    c=json.loads(Path(p).read_text(encoding='utf-8'))
    assert c['rod_scale_bounds']==[.9,1.1] and c['observation_dim']==56 and c['action_dim']==4
    assert c['ppo_steps_per_candidate']>0 and c['ppo_steps_per_candidate']%500==0
    assert c['candidate_budget']>=1 and c['controller_mode'] in ['retrain','frozen']
    assert c.get('parallel_envs',1) in (1,2,4)
    assert c['ppo_steps_per_candidate']%(500*c.get('parallel_envs',1))==0
    assert 1<=c['search_rollouts']<=64 and 1<=c['confirmation_rollouts']<=64
    return c
def get_initial(folder,case,c):
    from paper_env import build_model
    record=folder/'observation.json'
    if record.exists():return runtime.read(record)
    model=folder/'nominal.xml'
    if not model.exists():build_model(model,case['scene'],[1.]*8,physics=c.get('physics'),**c.get('scene_geometry',{}))
    valid,reason=runtime.check_geometry(model)
    if not valid:raise RuntimeError('Invalid nominal geometry: '+str(reason))
    cp=runtime.train(model,folder/'training',c['ppo_steps_per_candidate'])
    result=runtime.evaluate(model,cp,folder/'search_evaluation',case['search_seeds'])
    row=dict(candidate_id=0,rod_length_scales=[1.]*8,mean_reward=result['mean_reward'],checkpoint=cp,model_path=str(model))
    runtime.save(record,row);return row
def run_cell(base,c,case,method):
    bind(case)
    runtime.N_ENVS=c.get('parallel_envs',1)
    runtime.ENV_KWARGS=c.get('environment',{})
    folder=base/case['case_id'];dest=folder/method;dest.mkdir(parents=True,exist_ok=True)
    summary=dest/'summary.json'
    if summary.exists():return runtime.read(summary)
    if method=='transform2act':
        if c['controller_mode']!='retrain':raise ValueError('Transform2Act joint-training adaptation cannot be called frozen-policy search')
        from paper_env import build_model
        from t2a_runner import worker
        nominal=folder/'initial'/'nominal.xml'
        if not nominal.exists():build_model(nominal,case['scene'],[1.]*8,physics=c.get('physics'),**c.get('scene_geometry',{}))
        protocol=dest/'protocol.json';hashes={str(p.resolve()):runtime.sha(p) for p in (ROOT/'src').glob('*.py')}
        if protocol.exists():
            if runtime.read(protocol)['source_hashes']!=hashes:raise RuntimeError('Code changed since this T2A run started')
        else:runtime.save(protocol,dict(source_hashes=hashes))
        worker(out=dest,nominal=nominal,scene=case['scene'],seed=case['train_seed'],
          geometry=list(range(case['train_seed'],case['train_seed']+32))+case['search_seeds'],search=case['search_seeds'],confirm=case['confirmation_seeds'],
          target=(c['candidate_budget']+1)*c['ppo_steps_per_candidate'],milestone_steps=c['ppo_steps_per_candidate'])
        return runtime.read(summary)
    from paper_env import build_model
    from proposals import proposal
    first=get_initial(folder/'initial',case,c);history=[first]
    if method!='fixed':
        for k in range(1,c['candidate_budget']+1):
            work=dest/f'candidate_{k:03d}';work.mkdir(exist_ok=True);observed=work/'observation.json'
            if observed.exists():history.append(runtime.read(observed));continue
            selected=work/'selected_proposal.json'
            if selected.exists():chosen=runtime.read(selected)
            else:
                excluded=[];feedback=[];chosen=None
                for attempt in range(32):
                    p=proposal(method,history,k-1,case['proposer_seed_base'],attempt,excluded,feedback,work/f'proposal_{attempt:02d}.json',case['scene'],c['fixed_pair'])
                    model=work/f'geometry_{attempt:02d}.xml'
                    try:
                        if not model.exists():build_model(model,case['scene'],p['rod_length_scales'],physics=c.get('physics'),**c.get('scene_geometry',{}))
                        valid,why=runtime.check_geometry(model)
                    except ValueError:valid=False;why={'reason':'closed linkage generator rejected candidate'}
                    if valid:
                        chosen=dict(model_path=str(model),rod_length_scales=p['rod_length_scales']);runtime.save(selected,chosen);break
                    excluded.append(p['rod_length_scales']);feedback.append(dict(rod_length_scales=p['rod_length_scales'],reason=why))
                    runtime.event(work/'rejections.jsonl',attempt=attempt,**why)
                if chosen is None:raise RuntimeError('No feasible proposal; method not substituted')
            cp=first['checkpoint'] if c['controller_mode']=='frozen' else runtime.train(chosen['model_path'],work/'training',c['ppo_steps_per_candidate'])
            result=runtime.evaluate(chosen['model_path'],cp,work/'search_evaluation',case['search_seeds'])
            row=dict(candidate_id=k,rod_length_scales=chosen['rod_length_scales'],model_path=chosen['model_path'],checkpoint=cp,mean_reward=result['mean_reward'])
            runtime.save(observed,row);history.append(row)
    best=max(history,key=lambda r:r['mean_reward'])
    selected=dest/'selected_endpoint.json'
    if selected.exists():assert runtime.read(selected)==best
    else:runtime.save(selected,best)
    confirmation=runtime.evaluate(best['model_path'],best['checkpoint'],dest/'confirmation',case['confirmation_seeds'],save_motion=True)
    result=dict(status='MEASURED_RUN_COMPLETE_NOT_FULL_PAPER',method=method,case=case,protocol=c['protocol'],controller_mode=c['controller_mode'],
      selected_candidate=best['candidate_id'],confirmation=confirmation,search_observations=history,official_full_external_reproduction=False)
    runtime.save(summary,result);return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper_retrain.json')
    p.add_argument('--scene',choices=['parallel','tilted','stepped','curved'],default='parallel');p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--method',choices=['bom','robomorph_llm','text2robot_ea','transform2act','fixed','random_rod_bo','global_bo','fixed_pair_2d'],default='bom')
    p.add_argument('--controller-mode',choices=['retrain','frozen'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,default=ROOT/'runs')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-api',action='store_true');p.add_argument('--max-api-calls',type=int,default=0)
    a=p.parse_args();c=checked_config(a.config)
    if a.controller_mode:
        c['controller_mode']=a.controller_mode;c['protocol']+='_'+a.controller_mode
    case=context(c,a.scene,a.repeat)
    plan=dict(config=c,case=case,method=a.method,device=a.device,training_started=False,api_calls=0)
    if not a.execute:print(json.dumps(plan,indent=2));return 0
    if a.method=='robomorph_llm' and (not a.allow_api or a.max_api_calls<1):p.error('RoboMorph requires --allow-api --max-api-calls N and a locally configured key')
    if a.method=='transform2act' and c['controller_mode']=='frozen':p.error('T2A joint-training requires retrain mode')
    base=a.output.resolve();base.mkdir(parents=True,exist_ok=True)
    lock=base/'.runner.lock'
    try:fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise RuntimeError('One runner at a time; check the existing lock before resuming') from None
    try:
        os.write(fd,str(os.getpid()).encode());runtime.require_memory()
        import torch
        torch.set_num_threads(1)
        if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable; device not silently changed')
        runtime.DEVICE=a.device
        spec=base/'run_config.json';config_record=dict(config=c,device=a.device,source_hashes={str(f.relative_to(ROOT)):runtime.sha(f) for f in [Path(__file__),*(ROOT/'src').glob('*.py')]})
        if spec.exists():
            if runtime.read(spec)!=config_record:raise RuntimeError('Output belongs to another config/code revision; choose a new output directory')
        else:runtime.save(spec,config_record)
        import deepseek_client
        deepseek_client.OUT=base/'api_log';deepseek_client.REQUEST_LIMIT=a.max_api_calls if a.allow_api else 0
        result=run_cell(base,c,case,a.method);print(json.dumps(result,indent=2))
    finally:os.close(fd);lock.unlink()
    return 0
if __name__=='__main__':raise SystemExit(main())
