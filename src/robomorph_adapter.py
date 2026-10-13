"""Real LLM-guided candidate generation for the manuscript's fixed-topology adaptation.

This is not an official full RoboMorph reproduction: robot grammar mutation is
replaced with the manuscript's eight continuous rod lengths. A proposal is not
an evaluated result. The shared PPO evaluator must supply every fitness value.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from deepseek_client import ApiFailure, OUT, completion, validate_vector

SCENES = {
    'parallel': 'Two vertical parallel walls with a constant gap.',
    'tilted': 'A confined wall corridor whose gap gradually narrows with height.',
    'stepped': 'A confined wall corridor with abrupt reductions in gap width.',
    'curved': 'A confined wall corridor with a smooth inward wall bulge.',
}


def messages(scene, observations, constraint_feedback=None):
    if scene not in SCENES:
        raise ValueError('Unknown scene')
    checked = []
    for record in observations:
        vector = validate_vector({'rod_length_scales':record['rod_length_scales']})
        reward = record['mean_reward']
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError('Only finite, measured reward observations are permitted')
        checked.append(dict(candidate_id=record['candidate_id'], rod_length_scales=vector, mean_reward=float(reward)))
    # Fixed best-shot history rule; never obtain rewards from the LLM.
    examples = sorted(checked, key=lambda r:(-r['mean_reward'],r['candidate_id']))[:8]
    system = ('You propose robot morphologies; you do not evaluate their reward. '
              'The robot has a fixed closed chain of eight rods and four motors. '
              'Only the eight rod lengths may change. Return a JSON object with '
              'rod_length_scales (exactly eight numbers, each in [0.9,1.1]) and a short rationale. '
              'Do not change topology, motors, controller, gravity, terrain, or reward. '
              'Do not claim that a proposal has already succeeded.')
    user = json.dumps(dict(scene=scene, description=SCENES[scene],
                           objective='Improve measured climbing reward after equal-budget PPO training.',
                           initial_rod_length_scales=[1.0]*8,
                           constraints=dict(minimum_scale=.9, maximum_scale=1.1, dimensions=8),
                           measured_best_examples=examples,
                           rejected_geometries=(constraint_feedback or [])[-4:],
                           request='Propose one distinct candidate. Do not invent measured examples or fitness values.'),
                      ensure_ascii=False)
    return [{'role':'system','content':system},{'role':'user','content':user}]


def propose(scene, observations, destination: Path, *, preflight=False, constraint_feedback=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Proposal record already exists; never overwrite or silently regenerate')
    prompt = messages(scene, observations, constraint_feedback)
    result, usage, model = completion(prompt,
        purpose='proposal_transport_preflight' if preflight else 'morphology_proposal',
        max_tokens=256 if preflight else 768)
    vector = validate_vector(result)
    if any(all(abs(a-b)<1e-8 for a,b in zip(vector,r['rod_length_scales'])) for r in observations):
        raise ApiFailure('Duplicate observed morphology; no silent regeneration or perturbation')
    record = dict(status='TRANSPORT_PREFLIGHT_NOT_AN_EXPERIMENT' if preflight else 'PROPOSED_NOT_EVALUATED',
                  created_utc=datetime.now(timezone.utc).isoformat(), scene=scene,
                  model=model, usage=usage, prompt=prompt,
                  prompt_sha256=hashlib.sha256(json.dumps(prompt,sort_keys=True).encode()).hexdigest(),
                  rod_length_scales=vector, rationale=result.get('rationale',''),
                  measured_reward=None, simulation_evaluated=False, ppo_trained=False,
                  official_full_robomorph_reproduction=False,
                  adaptation='Manuscript fixed-topology continuous rod-length LLM baseline')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if not args.preflight:
        parser.error('Only explicit preflight is exposed; experiment runner will pass measured observations')
    result=propose('parallel',[],OUT/'llm_transport_preflight/proposal.json',preflight=True)
    print(json.dumps({k:result[k] for k in ('status','model','usage','rod_length_scales','simulation_evaluated')},ensure_ascii=False))


if __name__=='__main__':main()
