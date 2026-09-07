"""Freeze and analyze existing official output without changing the running job.

This is a reader, not a trainer or reward implementation. The official rollout
writer can overwrite its shaped `score` column with raw score metadata; use
raw_reward + overlong_reward and retain both original columns explicitly.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time


def read_complete(path):
    data = path.read_bytes()
    return data[:data.rfind(b'\n') + 1]


def metrics_from_log(data):
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', data.decode(errors='replace'))
    records = {}
    for line in text.replace('\r', '\n').splitlines():
        match = re.search(r'\bstep:(\d+) - (.*)', line)
        if not match:
            continue
        row = records.setdefault(int(match[1]), {})
        for item in match[2].split(' - '):
            key, separator, value = item.partition(':')
            if separator:
                # Official LocalLogger serializes numeric values only.
                # NumPy 2.x pprint includes a scalar type wrapper; never eval logs.
                wrapper = re.fullmatch(r'np\.(?:float(?:16|32|64)|int(?:8|16|32|64))\(([^()]*)\)', value)
                number = float(wrapper[1] if wrapper else value)
                if key in row:
                    assert row[key] == number, (match[1], key)
                row[key] = number
    return [{'step': step, **values} for step, values in sorted(records.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    analysis_path = Path(__file__).resolve().parent.parent / 'base_signal_diagnostic_20260907/analyze_events.py'
    spec = importlib.util.spec_from_file_location('existing_event_analysis', analysis_path)
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)

    files = [args.run / name for name in ('launch.json', 'process.json', 'exit.json', 'train.log')]
    files += list((args.run / 'environment').glob('*.jsonl'))
    files += list((args.run / 'rollouts').glob('*.jsonl'))
    files += list((args.run / 'validation').glob('*.jsonl'))
    fingerprints = {}
    for source in sorted(files):
        if not source.exists():
            continue
        relative = source.relative_to(args.run)
        data = read_complete(source) if source.suffix in ('.jsonl', '.log') else source.read_bytes()
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        fingerprints[str(relative)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    events, _ = existing.read_events(args.output)
    by_request = defaultdict(list)
    for event in events:
        if event['event'] == 'act':
            by_request[event['trajectory_id']].append(event)
            assert event['completed'] or event['raw_reward'] == 0
    trajectories = {key: existing.summarize_trajectory(value) for key, value in by_request.items()}
    raw_scores = [e for e in events if e['event'] == 'raw_score']
    for event in raw_scores:
        assert event['score'] == event['acc'] == event['raw_reward']
        assert event['completed'] or event['raw_reward'] == 0

    retained = []
    for source in sorted((args.output / 'rollouts').glob('*.jsonl'), key=lambda p: int(p.stem)):
        for line in source.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            request = row['request_id']
            trajectory = trajectories.get(request)
            raw = float(row['raw_reward'])
            penalty = float(row['overlong_reward'])
            assert math.isfinite(raw) and math.isfinite(penalty) and penalty <= 0
            assert row['score'] == raw == row['acc']
            if trajectory and trajectory['completed']:
                assert raw == trajectory['quality']
            elif trajectory:
                assert not row['completed'] and raw == 0
            retained.append({
                'step': row['step'], 'request_id': request,
                'raw_quality': raw, 'official_overlong_reward': penalty,
                'shaped_reward_from_components': raw + penalty,
                'original_dump_score': row['score'], 'completed': bool(row['completed']),
                'query_type': trajectory['query_type'] if trajectory else None,
                'tape_key': trajectory['tape_key'] if trajectory else None,
                'task_metrics': trajectory['task_metrics'] if trajectory else None,
            })

    logs = metrics_from_log((args.output / 'train.log').read_bytes())
    updates = [row for row in logs if 'actor/grad_norm' in row]
    nonfinite = [{'step': row['step'], 'metric': key, 'value': str(value)}
                 for row in logs for key, value in row.items() if not math.isfinite(value)]
    by_step = defaultdict(list)
    for row in retained:
        by_step[row['step']].append(row)
    groups = []
    for step, rows in sorted(by_step.items()):
        groups.append({
            'step': step, 'dumped_rows': len(rows),
            'families': dict(Counter(row['query_type'] for row in rows)),
            'raw_qualities': [row['raw_quality'] for row in rows],
            'shaped_rewards': [row['shaped_reward_from_components'] for row in rows],
            'raw_group_varies': len({row['raw_quality'] for row in rows}) > 1,
        })

    process = json.loads((args.output / 'process.json').read_text())
    import psutil
    try:
        actual = psutil.Process(process['pid'])
        same_live = abs(actual.create_time() - process['create_time']) < .01 and actual.is_running()
    except psutil.NoSuchProcess:
        same_live = False
    report = {
        'snapshot_time': time.time(), 'run': str(args.run), 'same_process_live': same_live,
        'process': process, 'source_prefixes': fingerprints,
        'event_counts': dict(Counter(e['event'] for e in events)),
        'trajectories': list(trajectories.values()), 'raw_score_events': raw_scores,
        'official_metrics': logs, 'official_updates_logged': [row['step'] for row in updates],
        'nonfinite_metrics': nonfinite, 'retained_rollouts': retained, 'retained_groups': groups,
        'scope': 'Read-only append-prefix snapshot. Update logs are execution evidence, not post-update capability measurements. Missing/in-flight records are not failed trajectories. Raw-score events lack request IDs and are not individually joined.',
        'dump_score_semantics': 'Pinned official writer overwrites shaped score with raw score metadata. Shaped rewards here are reconstructed from recorded raw_reward and overlong_reward, not from decoded-token length.',
    }
    (args.output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('same_process_live', 'event_counts',
                     'official_updates_logged', 'nonfinite_metrics', 'retained_groups')}))


if __name__ == '__main__':
    main()
