"""Prepare production-law certified tasks and audit every accepted terminal label."""

import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
ROOT = Path('/home/chen/runs/training-submission-20260907')
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT)]
from cpt_world import CPTWorldEnvironment, compute_query_truth
from cpt_world import trl_environment as owner
from cpt_world.query_truth import nearest_backdoor_adjustment_set
from cpt_world.world_space import _best_intervention_observational_relation
from scripts.prepare_verl_cpt_data import convert_row


def clean(x):
    if isinstance(x, Fraction):
        return float(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    return x


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    data = ROOT/'data'
    data.mkdir(exist_ok=False)
    (data/'truth').mkdir()
    os.environ['CPT_WORLD_RESOURCE_DIAGNOSTIC_DIR'] = str(ROOT/'cf-diagnostics')
    original = owner.compute_counterfactual_truth_isolated

    def observed(*args, **kwargs):
        # Observation only: call the existing isolated solver exactly once and
        # return its unchanged result or re-raise its unchanged exception.
        started = time.monotonic()
        record = {'seed_id': args[1]['seed_id']}
        try:
            result = original(*args, **kwargs)
            record.update(status='accepted', truth=result)
            return result
        except Exception as error:
            record.update(status='rejected', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record['seconds'] = time.monotonic()-started
            with (ROOT/'cf-attempts.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(record, default=str)+'\n')

    owner.compute_counterfactual_truth_isolated = observed
    summaries = []
    seed_sets = {}
    for split, count, start in [('train', 500, 2000000), ('validation', 25, 3000000)]:
        source = owner.iter_random_balanced_training_rows(start_seed=start)
        converted = []
        seed_sets[split] = set()
        try:
            for i in range(count):
                row = next(source)
                env = CPTWorldEnvironment()
                env.reset(**row)
                ep = env.episode
                world, seed = ep.world, ep.seed
                family = row['query_type']
                seed_sets[split].add(seed['seed_id'])
                if family == 'individual_counterfactual_probability':
                    # This was just computed by the unchanged current isolated
                    # solver. Keep its certified endpoints, never substitute a guess.
                    truth = json.loads(row['terminal_truth_json'])
                    assert truth.get('certification') in ('exact', 'epsilon_sharp')
                    answer = {'type': 'answer', 'lower': max(0, min(1, float(truth['lower']))),
                              'upper': max(0, min(1, float(truth['upper'])))}
                else:
                    truth = compute_query_truth(world, seed)
                labels = seed['visible_schema']['variable_labels']
                if family == 'ate':
                    answer = {'type': 'answer', 'effect': {f'state_{j}': float(x)
                              for j, x in enumerate(truth['effect'])}}
                elif family == 'best_intervention':
                    answer = {'type': 'answer', 'value': f'state_{truth["value"]}'}
                elif family == 'backadj_minimal_sets':
                    inv = {v: k for k, v in labels.items()}
                    _, nearest = nearest_backdoor_adjustment_set(
                        world, inv[seed['query']['treatment']], inv[seed['query']['outcome']], frozenset())
                    answer = {'type': 'answer', 'adjustment_set': [labels[n] for n in nearest]}
                elif family == 'mediator_set':
                    answer = {'type': 'answer', 'mediators': [labels[n] for n in truth['mediators']],
                              'order': [[labels[a], labels[b]] for a, b in truth['order']]}
                path = data/'truth'/f'{split}-{i:04}.pkl'
                path.write_bytes(pickle.dumps((row, world, seed, truth)))
                env.act(answer)
                assert ep.completed and abs(env.get_reward()-1) < 1e-10, (split, i, ep.terminal_score)
                item = {'split': split, 'index': i, 'query_type': family, 'seed_id': seed['seed_id'],
                        'tape_key': row['tape_key'], 'truth': clean(truth),
                        'oracle_score': clean(ep.terminal_score), 'world_file_sha256': sha(path)}
                if family == 'best_intervention':
                    inv = {v: world.variables.index(k) for k, v in labels.items()}
                    q = seed['query']
                    item['discordant'], item['observational_causal_gap'] = _best_intervention_observational_relation(
                        world, {'decision_target': inv[q['decision_target']], 'outcome': inv[q['outcome']],
                                'outcome_state': int(q['outcome_state'].removeprefix('state_')),
                                'objective': q['objective']})
                summaries.append(item)
                converted.append(convert_row(row, i))
                with (data/'accepted.jsonl').open('a', encoding='utf-8') as output:
                    output.write(json.dumps(item)+'\n')
                if (i+1) % 25 == 0:
                    print(json.dumps({'split': split, 'accepted': i+1}), flush=True)
        finally:
            source.close()
        assert len(seed_sets[split]) == count
        assert set(Counter(r['extra_info']['query_type'] for r in converted).values()) == {count//5}
        selected = [r for r in summaries if r['split'] == split and r['query_type'] == 'best_intervention']
        assert sum(r['discordant'] for r in selected) == 4*count//25
        pq.write_table(pa.Table.from_pylist(converted), data/f'{split}.parquet')
    owner.compute_counterfactual_truth_isolated = original
    assert not seed_sets['train'].intersection(seed_sets['validation'])
    manifest = {'passed': True, 'counts': {'train': 500, 'validation': 25},
                'sha256': {p.name: sha(p) for p in data.glob('*.parquet')},
                'labels_sha256': sha(data/'accepted.jsonl'),
                'scope': 'Existing production generator and certification admission unchanged; rejected CF attempts logged.'}
    (data/'acceptance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
