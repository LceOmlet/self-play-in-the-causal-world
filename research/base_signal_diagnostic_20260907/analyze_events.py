"""Summarize actual environment events without treating a partial run as complete.

Task errors come from the production terminal scorer. This analysis never
supplies actions or rewards to the model. No-action trajectories are counted
only when an official complete validation dump makes their denominator known.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def feedback_of(event):
    if event['completed']:
        assert event['feedback'] == 'Terminal answer accepted. Episode complete; make no more tool calls.'
        return {'type': 'terminal'}
    return json.loads(event['feedback'].splitlines()[0])


def read_events(run):
    events, fingerprints = [], {}
    for file in sorted((run / 'environment').glob('*.jsonl')):
        data = file.read_bytes()
        # Snapshot only complete appended lines; record the exact byte prefix.
        last = data.rfind(b'\n') + 1
        data = data[:last]
        fingerprints[file.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        events.extend(json.loads(line) for line in data.splitlines())
    return events, fingerprints


def summarize_trajectory(events):
    first, last = events[0], events[-1]
    errors = Counter()
    experiments = Counter()
    observations = 0
    measured = Counter()
    targets = Counter()
    terminal = []
    for event in events:
        feedback = feedback_of(event)
        command = event['command']
        if feedback['type'] == 'protocol_error':
            errors[feedback['error']['message']] += 1
        elif feedback['type'] == 'batch_result':
            columns = feedback['batch']['joint_histogram']['columns']
            n = feedback['batch']['n']
            observations += n * len(columns)
            experiments[command['type']] += 1
            for label in columns:
                measured[label] += n
            if command['type'] == 'intervene':
                targets[command['target']] += 1
        elif not event['completed']:
            raise AssertionError(('Unknown feedback', feedback))
        assert observations == event['observations_used'], event['trajectory_id']
        if event['completed']:
            terminal.append(event)
    assert len(terminal) <= 1
    assert not terminal or terminal[0] is last
    score = last['terminal_score'] if terminal else None
    error_fields = (
        'total_variation_error', 'mean_absolute_endpoint_error', 'regret',
        'normalized_regret', 'optimal_action', 'edit_distance', 'valid_adjustment_set',
        'mediator_f1', 'order_f1', 'mediators_exact_match', 'order_exact_match',
    )
    return {
        'trajectory_id': first['trajectory_id'], 'tape_key': first['tape_key'],
        'sample_index': first['sample_index'], 'query_type': first['query_type'],
        'actions': len(events), 'protocol_errors': dict(errors),
        'valid_experiments': dict(experiments), 'observations_used': observations,
        'measured_scalar_counts': dict(measured), 'intervention_calls_by_target': dict(targets),
        'completed': bool(terminal), 'quality': last['raw_reward'] if terminal else None,
        'answer': last['command'] if terminal else None,
        'task_metrics': {k: score[k] for k in error_fields if k in score} if score else None,
        'terminal_score': score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events, fingerprints = read_events(args.run)
    launch = json.loads((args.run / 'launch.json').read_text())
    accepted = json.loads((args.run / 'data-acceptance.json').read_text())
    groups = defaultdict(list)
    for event in events:
        if event['event'] == 'act':
            groups[event['trajectory_id']].append(event)
    trajectories = [summarize_trajectory(group) for group in groups.values()]
    by_tape = defaultdict(list)
    for trajectory in trajectories:
        by_tape[trajectory['tape_key']].append(trajectory)
    rows = []
    for task in accepted['records']:
        observed = by_tape[task['tape_key']]
        complete = [t for t in observed if t['completed']]
        qualities = [t['quality'] for t in complete]
        rows.append({
            'index': task['index'], 'query_type': task['query_type'],
            'seed_id': task['seed_id'], 'tape_key': task['tape_key'],
            'trajectories_with_actions': len(observed), 'completed': len(complete),
            'completed_qualities': qualities,
            'completed_quality_std': statistics.pstdev(qualities) if qualities else None,
            'distinct_completed_answers': len({json.dumps(t['answer'], sort_keys=True) for t in complete}),
            'discordant': task.get('discordant'),
        })
    dumps = []
    for file in sorted((args.run / 'validation').glob('*.jsonl')):
        data = file.read_bytes()
        dumps.extend(json.loads(line) for line in data[:data.rfind(b'\n') + 1].splitlines())
    expected = launch['validation_worlds'] * launch['trajectories_per_world']
    assert len(dumps) <= expected, ('Unexpected dump layout', len(dumps))
    exit_file = args.run / 'exit.json'
    exit_record = json.loads(exit_file.read_text()) if exit_file.exists() else None
    if exit_record and exit_record['exit_code'] == 0:
        assert len(dumps) == expected, 'Successful validation must preserve every official output'
    report = {
        'scope': 'Actual original-base validation; partial event snapshots are not completed-evaluation denominators.',
        'exit': exit_record, 'expected_trajectories': expected, 'official_dump_rows': len(dumps),
        'trajectories_with_actions': len(trajectories),
        'completed_trajectories': sum(t['completed'] for t in trajectories),
        'event_counts': dict(Counter(e['event'] for e in events)),
        'event_prefix_fingerprints': fingerprints, 'tasks': rows, 'trajectories': trajectories,
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('exit', 'expected_trajectories', 'official_dump_rows',
                                           'trajectories_with_actions', 'completed_trajectories')}))
    for task in rows:
        if task['trajectories_with_actions']:
            print(json.dumps(task))


if __name__ == '__main__':
    main()
