"""Create paper statistics only from a complete, provenance-matched matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


LABELS={'bom':'BOM','fixed':'Fixed','random_rod_bo':'Random-Rod BO','global_bo':'Global BO'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(values):
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite independent-repetition values required')
    n=len(values)
    mean=float(values.mean())
    sd=float(values.std(ddof=1)) if n>1 else None
    interval=None
    if n>1:
        half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n))
        interval=[mean-half,mean+half]
    return dict(n=n,mean=mean,sd=sd,ci95_t=interval,values=values.tolist())


def load_cells(root):
    manifest=read(root/'matrix_manifest.json')
    config=manifest['config']
    cells=[]
    for scene,repeat in manifest['cases']:
        case=f'{scene}_r{repeat}'
        folder=root/'shards'/case/case
        for method in manifest['methods']:
            source=folder/method/'summary.json'
            if not source.exists():
                raise RuntimeError(f'Incomplete experiment: {case}/{method}')
            summary=read(source)
            confirmation=read(folder/method/'confirmation/aggregate.json')
            if summary['confirmation']!=confirmation or summary['method']!=method:
                raise AssertionError('Summary/confirmation mismatch')
            if len(summary['search_observations'])!=(1 if method=='fixed' else config['candidate_budget']+1):
                raise AssertionError('Candidate count differs from the frozen budget')
            selected=max(summary['search_observations'],key=lambda x:x['mean_reward'])
            if selected['candidate_id']!=summary['selected_candidate']:
                raise AssertionError('Confirmation policy was not selected by search score')
            if sha(selected['model_path'])!=confirmation['model_sha256']:
                raise AssertionError('Selected morphology hash mismatch')
            if sha(selected['checkpoint']['model'])!=confirmation['checkpoint_sha256']:
                raise AssertionError('Selected policy hash mismatch')
            if sha(selected['checkpoint']['normalizer'])!=confirmation['normalizer_sha256']:
                raise AssertionError('Normalizer hash mismatch')
            if confirmation['environment']!=config['environment']:
                raise AssertionError('Evaluation environment differs from protocol')
            seeds=summary['case']['confirmation_seeds']
            if confirmation['seeds']!=seeds or len(seeds)!=config['confirmation_rollouts']:
                raise AssertionError('Confirmation reset set mismatch')
            rows=[read(folder/method/'confirmation'/f'seed_{seed}.json') for seed in seeds]
            if [row['seed'] for row in rows]!=seeds:
                raise AssertionError('Per-seed identity mismatch')
            completion=float(np.mean([row['completion_ratio'] for row in rows]))
            success=float(np.mean([row['sampled_height_hold_success'] for row in rows]))
            if abs(completion-confirmation['mean_completion_ratio'])>1e-12:
                raise AssertionError('Completion aggregation mismatch')
            values=[row['reward'] for row in rows if row['valid_reset']]
            if abs(float(np.mean(values))-confirmation['mean_reward'])>1e-12:
                raise AssertionError('Score aggregation mismatch')
            histories=summary['search_observations']
            cells.append(dict(scene=scene,repeat=repeat,method=method,
                         completion=completion,hold_success_fraction=success,
                         discounted_return=confirmation['mean_reward'],
                         valid_trials=confirmation['valid_trials'],total_trials=len(rows),
                         selected_candidate=selected['candidate_id'],
                         selected_scales=selected['rod_length_scales'],
                         search_observed=[x['mean_reward'] for x in histories],
                         search_incumbent=np.maximum.accumulate([x['mean_reward'] for x in histories]).tolist(),
                         seed_rows=rows,source=str(source),source_sha256=sha(source)))
    return manifest,cells


def summarize(manifest,cells):
    grouped=[]
    paired=[]
    for scene in manifest['config']['scenes']:
        for method in manifest['methods']:
            rows=sorted([c for c in cells if c['scene']==scene and c['method']==method],key=lambda c:c['repeat'])
            grouped.append(dict(scene=scene,method=method,
                           completion=estimate([r['completion'] for r in rows]),
                           hold_success_fraction=estimate([r['hold_success_fraction'] for r in rows]),
                           discounted_return=estimate([r['discounted_return'] for r in rows])))
        bom={c['repeat']:c for c in cells if c['scene']==scene and c['method']=='bom'}
        for other in [m for m in manifest['methods'] if m!='bom']:
            control={c['repeat']:c for c in cells if c['scene']==scene and c['method']==other}
            if set(bom)!=set(control):
                raise AssertionError('Unpaired independent repetitions')
            paired.append(dict(scene=scene,comparison=f'bom_minus_{other}',
                          completion_effect=estimate([bom[r]['completion']-control[r]['completion'] for r in sorted(bom)]),
                          success_fraction_effect=estimate([bom[r]['hold_success_fraction']-control[r]['hold_success_fraction'] for r in sorted(bom)])))
    return dict(protocol=manifest['config']['protocol'],
                statistical_unit='independent search/controller-training repetition; reset rollouts are nested measurements',
                interval_definition='two-sided Student t interval over paired or unpaired repetition-level observations; descriptive with small n',
                fixed_reference_budget='one trained nominal policy; adaptive searches train the nominal plus new candidate policies',
                cells=cells,scene_method_estimates=grouped,paired_effects=paired)


def render_table(summary,scenes,methods):
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Held-out endpoint COM completion ratio. Entries are mean $\pm$ sample standard deviation across independent repetitions. Reset rollouts are averaged within each repetition.}',r'\label{tab:completion}',r'\begin{tabular}{l'+'c'*len(scenes)+'}',r'\toprule',
           'Method & '+' & '.join(s.title() for s in scenes)+r' \\',r'\midrule']
    for method in methods:
        columns=[]
        for scene in scenes:
            estimate_row=next(x['completion'] for x in summary['scene_method_estimates'] if x['scene']==scene and x['method']==method)
            mean,sd=estimate_row['mean'],estimate_row['sd']
            columns.append(f'${mean:.3f}$' if sd is None else f'${mean:.3f} \\pm {sd:.3f}$')
        lines.append(LABELS[method]+' & '+' & '.join(columns)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table*}',''])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest,cells=load_cells(a.matrix)
    summary=summarize(manifest,cells)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (a.output/'completion_table.tex').write_text(render_table(summary,manifest['config']['scenes'],manifest['methods']),encoding='utf-8')
    print(json.dumps(dict(cells=len(cells),output=str(a.output),protocol=summary['protocol'])))


if __name__=='__main__':
    main()
