"""Recompute labels for the exact existing integration dataset; no resampling."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

from control import HERE, PROJECT, TRAIN

sys.path.insert(0, str(PROJECT / 'src'))
import pyarrow.parquet as pq
from cpt_world import CPTWorldEnvironment, compute_query_truth
from cpt_world.query_truth import nearest_backdoor_adjustment_set


def clean(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


records = []
for index, data in enumerate(pq.read_table(TRAIN).to_pylist()):
    row = json.loads(data['extra_info']['tools_kwargs']['act']['create_kwargs']['row_json'])
    env = CPTWorldEnvironment()
    env.reset(**row)
    episode = env.episode
    world, seed = episode.world, episode.seed
    truth = compute_query_truth(world, seed, counterfactual_endpoint_time_limit_seconds=5)
    cached = json.loads(row['terminal_truth_json']) if row['terminal_truth_json'] else None
    family = row['query_type']
    labels = seed['visible_schema']['variable_labels']
    if family == 'individual_counterfactual_probability':
        assert cached is not None
        tolerance = float(cached['endpoint_error']) + float(truth['endpoint_error']) + 1e-8
        assert abs(float(cached['lower']) - float(truth['lower'])) <= tolerance
        assert abs(float(cached['upper']) - float(truth['upper'])) <= tolerance
        answer = {'type': 'answer', 'lower': max(0, min(1, float(cached['lower']))),
                  'upper': max(0, min(1, float(cached['upper'])))}
    elif family == 'ate':
        answer = {'type': 'answer', 'effect': {f'state_{i}': float(x) for i, x in enumerate(truth['effect'])}}
    elif family == 'best_intervention':
        answer = {'type': 'answer', 'value': f'state_{truth["value"]}'}
    elif family == 'backadj_minimal_sets':
        inverse = {label: node for node, label in labels.items()}
        x, y = (inverse[seed['query'][k]] for k in ('treatment', 'outcome'))
        _, nearest = nearest_backdoor_adjustment_set(world, x, y, frozenset())
        answer = {'type': 'answer', 'adjustment_set': [labels[n] for n in nearest]}
    elif family == 'mediator_set':
        answer = {'type': 'answer', 'mediators': [labels[n] for n in truth['mediators']],
                  'order': [[labels[a], labels[b]] for a, b in truth['order']]}
    else:
        raise AssertionError(family)
    feedback = env.act(answer)
    assert episode.completed, (index, feedback)
    assert abs(env.get_reward() - 1) < 1e-10, (index, episode.terminal_score)
    entry = {'index': index, 'family': family, 'sample_index': row['sample_index'],
             'tape_key': row['tape_key'], 'truth': clean(truth), 'cached': cached,
             'oracle_reward': env.get_reward(), 'terminal_score': clean(episode.terminal_score)}
    records.append(entry)
    print(json.dumps({'index': index, 'family': family, 'passed': True}), flush=True)
result = {'passed': True, 'data_sha256': hashlib.sha256(Path(TRAIN).read_bytes()).hexdigest(),
          'counts': dict(Counter(r['family'] for r in records)), 'records': records,
          'interpretation': 'Known diagnostic data. Oracle answers use private truth solely to audit scoring, never as rollout actions.'}
(HERE/'data-acceptance.json').write_text(json.dumps(result, indent=2))
