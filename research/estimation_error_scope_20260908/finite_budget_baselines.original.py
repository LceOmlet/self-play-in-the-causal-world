"""Legal-feedback-only diagnostic policies. Hidden state is confined to evaluation."""

import argparse
import json
import math
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audit_environment import clean, score
from cpt_world import OutcomeTape, WorldSpecEpisode


def passive_answer(query, counts):
    # Only the public query and observed histogram enter this policy.
    law = (counts + 0.5) / (counts.sum(1)[:, None] + 0.5 * counts.shape[1])
    state = lambda text: int(text.removeprefix('state_'))
    if query['type'] == 'ate':
        vector = law[state(query['treatment_value'])] - law[state(query['baseline_value'])]
        return {'type': 'answer', 'effect': {f'state_{i}': float(v) for i, v in enumerate(vector)}}
    if query['type'] == 'best_intervention':
        target = state(query.get('outcome_state', query.get('target_state')))
        values = law[:, target]
        best = np.argmax(values) if query['objective'] == 'maximize' else np.argmin(values)
        return {'type': 'answer', 'value': f'state_{best}'}
    factual = law[state(query['factual_value']), state(query['factual_outcome_state'])]
    other = law[state(query['counterfactual_value']), state(query['outcome_state'])]
    return {'type': 'answer', 'lower': float(max(0, factual + other - 1) / factual),
            'upper': float(min(factual, other) / factual)}


def execute_feedback(episode, command):
    step = episode.step(json.dumps(command))
    assert step.kind == 'batch'
    return json.loads(step.message.splitlines()[0])


def worker(path_string):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (10 * 1024**3, 10 * 1024**3))
    path = Path(path_string)
    started = time.monotonic()
    with path.open('rb') as f:
        row, world, seed, truth = pickle.load(f)
    family = row['query_type']
    query = seed['query']
    labels = seed['visible_schema']['variable_labels']
    domains = {labels[n]: world.domains[i] for i, n in enumerate(world.variables)}
    x = query.get('treatment', query.get('decision_target'))
    y = query['outcome']
    # These quantities are all disclosed in the rendered task prompt.
    public_budget = seed['observation_budget']
    legal_targets = [labels[n] for n in world.variables if seed['manipulability'][n]]
    records = []
    for replicate in range(2):
        tape = OutcomeTape(f'finite-budget-audit:{seed["seed_id"]}:{replicate}')
        episode = WorldSpecEpisode(world, seed, tape, terminal_truth=truth)
        if family != 'backadj_minimal_sets':
            counts = np.zeros((domains[x], domains[y]))
            consumed = 0
            checkpoints = sorted(set([min(128, public_budget // 2), min(2048, public_budget // 2), public_budget // 2]))
            for target_n in checkpoints:
                payload = execute_feedback(episode, {'type': 'observe', 'measure': [x, y], 'batch_size': target_n - consumed})
                for values, count in payload['batch']['joint_histogram']['rows']:
                    counts[tuple(values)] += count
                consumed = target_n
                answer = passive_answer(query, counts)
                evaluated = score(answer, world, seed, truth)
                records.append({'policy': 'passive_conditional_plugin', 'replicate': replicate,
                                'sample_count': consumed, 'scalar_cost': 2 * consumed,
                                'full_budget': target_n == public_budget // 2, **evaluated})
            terminal = episode.step(json.dumps(answer))
            assert abs(float(terminal.reward) - records[-1]['reward']) < 1e-12
        else:
            rng = np.random.default_rng(row['sample_index'] + replicate)
            n = public_budget // sum(domains[z] for z in legal_targets)
            chosen = []
            decisions = []
            for z in legal_targets:
                counts = np.zeros((domains[z], domains[x]))
                for state in range(domains[z]):
                    payload = execute_feedback(episode, {'type': 'intervene', 'target': z,
                                'value': f'state_{state}', 'measure': [x], 'batch_size': n})
                    for values, count in payload['batch']['joint_histogram']['rows']:
                        counts[state, values[0]] += count
                pooled = counts.sum(0) / counts.sum()
                expected = n * pooled
                stat = np.sum((counts - expected)**2 / (expected + 1e-15))
                samples = rng.multinomial(n, pooled, size=(3999, domains[z]))
                expected_mc = samples.sum(1)[:, None, :] / domains[z]
                mc = ((samples - expected_mc)**2 / (expected_mc + 1e-15)).sum((1, 2))
                pvalue = float((1 + np.sum(mc >= stat)) / 4000)
                if pvalue <= 0.05 / len(legal_targets):
                    chosen.append(z)
                decisions.append({'target': z, 'p': pvalue})
            answer = {'type': 'answer', 'adjustment_set': chosen}
            terminal = episode.step(json.dumps(answer))
            records.append({'policy': 'active_marginal_ancestor_screen', 'replicate': replicate,
                            'scalar_cost': episode.observations_used, 'samples_per_arm': n,
                            'screening': decisions, 'answer': answer,
                            'reward': float(terminal.reward), 'diagnostic': clean(terminal.score)})
        assert episode.observations_used <= public_budget
    return {'task_file': str(path), 'seed_id': seed['seed_id'], 'query_type': family,
            'scalar_budget': public_budget, 'records': records,
            'seconds': time.monotonic() - started,
            'limitations': 'Two tapes per task. Diagnostic heuristics are not optimal causal solvers; marginal screening can miss cancellation. Checkpoints share each legal sample stream; intermediate scores never enter policy decisions.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    output = args.root / 'finite-baselines.jsonl'
    completed = set()
    if output.exists():
        completed = {json.loads(line)['task_file'] for line in output.read_text().splitlines()}
    tasks = []
    for cohort in ['cohort-a', 'cohort-b']:
        for path in sorted((args.root / cohort).glob('task-*.pkl'), key=lambda p: int(p.stem.split('-')[1])):
            with path.open('rb') as f:
                row, _, _, _ = pickle.load(f)
            if row['query_type'] != 'mediator_set' and str(path) not in completed:
                tasks.append(str(path))
    print('TASKS', len(tasks), 'ALREADY_DONE', len(completed), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, p): p for p in tasks}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as error:
                with (args.root / 'finite-baseline-errors.jsonl').open('a') as f:
                    f.write(json.dumps({'task_file': path, 'error': repr(error)}) + '\n')
                print('FAILED', path, repr(error), flush=True)
                continue
            with output.open('a') as f:
                f.write(json.dumps(clean(result), allow_nan=False) + '\n')
            print('DONE', result['query_type'], result['seed_id'], round(result['seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
