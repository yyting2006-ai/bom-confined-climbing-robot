"""Run a finite experiment matrix continuously, with per-case subprocesses."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from threading import Lock

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3,choices=(1,2,3))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    config=read(a.config)
    methods=config['baseline_methods']+config['ablation_methods']
    allowed={'bom','fixed','random_rod_bo','global_bo'}
    if not set(methods)<=allowed or len(set(methods))!=len(methods):
        raise ValueError('This matrix runner accepts distinct internal search comparisons only')
    scene_order=[s for s in ('stepped','curved','parallel','tilted') if s in config['scenes']]
    cases=[(s,r) for r in range(1,config['repeats']+1) for s in scene_order]
    execution_methods=([m for m in methods if m=='fixed']+
                       [m for m in methods if m!='fixed'])
    source=[ROOT/'run.py',Path(__file__),*(ROOT/'src').glob('*.py')]
    models=list((ROOT/'models').glob('*.xml'))
    manifest=dict(config=config,config_sha256=sha(a.config),
                  sources={str(f.relative_to(ROOT)):sha(f) for f in source+models},
                  workers=a.workers,methods=methods,execution_methods=execution_methods,cases=cases,
                  total_policies=len(cases)*(1+sum(config['candidate_budget'] for m in methods if m!='fixed')),
                  invocation_mode='finite continuous subprocess matrix; no scheduled jobs')
    print(json.dumps(dict(event='matrix_plan',**manifest),indent=2),flush=True)
    if not a.execute:
        return
    a.output.mkdir(parents=True,exist_ok=True)
    spec=a.output/'matrix_manifest.json'
    # JSON maps case tuples to arrays; compare normalized records when resuming.
    normalized=json.loads(json.dumps(manifest))
    if spec.exists():
        if read(spec)!=normalized:
            raise RuntimeError('Matrix code/configuration changed; use a separate output')
    else:
        spec.write_text(json.dumps(normalized,indent=2),encoding='utf-8')
    import psutil
    from uuid import uuid4
    run_id=uuid4().hex[:12]
    events=a.output/f'matrix_events_{run_id}.jsonl'
    event_lock=Lock()

    def emit(event,**details):
        record=dict(utc=datetime.now(timezone.utc).isoformat(),event=event,**details)
        line=json.dumps(record)
        with event_lock:
            with events.open('a',encoding='utf-8') as stream:
                stream.write(line+'\n')
            print(line,flush=True)

    def worker(scene,repeat):
        case=f'{scene}_r{repeat}'
        shard=a.output/'shards'/case
        shard.mkdir(parents=True,exist_ok=True)
        lock=shard/'.runner.lock'
        if lock.exists():
            pid=int(lock.read_text())
            if psutil.pid_exists(pid):
                raise RuntimeError(f'{case}: lock PID {pid} exists; inspect before retrying')
            recovered=shard/f'.runner.lock.stale-{run_id}'
            lock.rename(recovered)
            emit('stale_lock_preserved',case=case,previous_pid=pid,path=str(recovered))
        for method in execution_methods:
            if (a.output/'STOP_REQUESTED').exists():
                return dict(case=case,status='STOPPED_BETWEEN_CELLS')
            completed=shard/case/method/'summary.json'
            if completed.exists():
                emit('cell_cached',case=case,method=method)
                continue
            log=shard/f'{method}_{run_id}.log'
            command=[sys.executable,'-B','-X','utf8',str(ROOT/'run.py'),
                     '--config',str(a.config.resolve()),'--output',str(shard.resolve()),
                     '--scene',scene,'--repeat',str(repeat),'--method',method,'--execute']
            start=time.monotonic()
            emit('cell_started',case=case,method=method,log=str(log))
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                         text=True,encoding='utf-8',cwd=str(ROOT))
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if '"status": "checkpoint"' in line:
                        checkpoint=json.loads(line)
                        emit('training_checkpoint',case=case,method=method,**checkpoint)
                code=process.wait()
            if code:
                emit('cell_failed',case=case,method=method,exit_code=code,log=str(log))
                return dict(case=case,status='FAILED',method=method,log=str(log))
            if not completed.exists():
                raise RuntimeError(f'{case}/{method} exited without a summary')
            emit('cell_completed',case=case,method=method,wall_seconds=time.monotonic()-start)
        return dict(case=case,status='COMPLETE')

    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,*case):case for case in cases}
        for future in as_completed(futures):
            try:
                result=future.result()
            except Exception as error:
                result=dict(case=futures[future],status='FAILED',error=repr(error))
            results.append(result)
            emit('case_finished',**result)
    successful=all(r['status']=='COMPLETE' for r in results)
    receipt=a.output/f'matrix_receipt_{run_id}.json'
    receipt.write_text(json.dumps(dict(complete=successful,cases=results),indent=2),encoding='utf-8')
    emit('matrix_finished',complete=successful,receipt=str(receipt))
    if not successful:
        raise SystemExit(1)


if __name__=='__main__':
    main()
