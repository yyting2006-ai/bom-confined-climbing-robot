"""Collect only existing measured per-run summaries, never fill missing cells."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run_directory',type=Path);p.add_argument('--output',type=Path)
a=p.parse_args();dest=a.output or a.run_directory/'results.json'
rows=[]
for file in sorted(a.run_directory.glob('*_r*/*/summary.json')):
    d=json.loads(file.read_text());v=d.get('confirmation')
    if not v:continue
    rows.append(dict(case=file.parent.parent.name,method=file.parent.name,status=d['status'],mean_reward=v['mean_reward'],completion_ratio=v['mean_completion_ratio'],sampled_hold_successes=v['sampled_hold_successes'],total_trials=v['total_trials'],source=str(file.relative_to(a.run_directory))))
dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps({'measured_runs':rows,'count':len(rows),'missing_values_imputed':False},indent=2),encoding='utf-8')
if rows:
    with dest.with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(f'{len(rows)} measured runs collected; no missing cells imputed.')
