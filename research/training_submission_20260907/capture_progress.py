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
import os
from pathlib import Path
import re
import time


def read_complete(path):
    data = path.read_bytes()
    return data[:data.rfind(b'\n') + 1]


def commands_from_output(output):
    """Keep malformed command strings exactly as the tool audit records them."""
    calls = re.findall(
        r'<tool_call>\s*<function=act>\s*<parameter=command>\s*'
        r'(.*?)\s*</parameter>\s*</function>\s*</tool_call>', output, re.DOTALL)
    commands = []
    for call in calls:
        try:
            commands.append(json.loads(call))
        except json.JSONDecodeError:
            # The actual tool passes malformed JSON through as a string; the
            # environment records it and returns a protocol error. Dropping it
            # would change the executed sequence used for the unique join.
            commands.append(call)
    return commands


def metrics_from_log(data):
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', data.decode(errors='replace'))
    records = {}
    for line in text.replace('\r', '\n').splitlines():
        match = re.search(r'\bstep:(\d+) - (.*)', line)
        if not match:
            continue
        row = records.setdefault(int(match[1]), {})
        # Ray can append the next worker's message without a separating newline.
        # Its explicit process prefix ends this metric record; keep the preceding
        # numeric value intact and do not parse the worker's warning as metrics.
        # Ray's native signal handler can also append its shutdown banner to
        # the last metric without a newline. Cut only explicit log boundaries.
        boundary = r'\([A-Za-z_][\w:.]* pid=\d+\)|\*\*\* SIG[A-Z0-9]+ received at time='
        body = re.split(boundary, match[2], maxsplit=1)[0]
        for item in body.split(' - '):
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


def capture_task_runner_stdout(run, output):
    """Read the verified live trainer's own log, before Ray driver forwarding."""
    import psutil
    saved = json.loads((run / 'process.json').read_text())
    info = {'available': False, 'main_process': saved}
    try:
        main = psutil.Process(saved['pid'])
        if abs(main.create_time() - saved['create_time']) >= .01 or not main.is_running():
            return {**info, 'reason': 'Recorded main process is not the same live process'}
        candidates = []
        for child in main.children(recursive=True):
            try:
                if child.name().startswith('ray::DAPOTaskRunner'):
                    candidates.append(child)
            except psutil.NoSuchProcess:
                continue
        if len(candidates) != 1:
            return {**info, 'reason': 'Need one live DAPOTaskRunner child', 'candidates': len(candidates)}
        child = candidates[0]
        source = Path(os.readlink(f'/proc/{child.pid}/fd/1'))
        if not source.is_file():
            return {**info, 'reason': 'Task runner stdout is not a regular file'}
        data = source.read_bytes()
        (output / 'task-runner-stdout.log').write_bytes(data)
        return {**info, 'available': True, 'pid': child.pid, 'create_time': child.create_time(),
                'source': str(source), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    except (OSError, psutil.Error) as error:
        return {**info, 'reason': repr(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-snapshot', action='store_true')
    args = parser.parse_args()
    if args.reuse_snapshot:
        assert args.output.is_dir()
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    analysis_path = Path(__file__).resolve().parent.parent / 'base_signal_diagnostic_20260907/analyze_events.py'
    spec = importlib.util.spec_from_file_location('existing_event_analysis', analysis_path)
    existing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(existing)

    if not args.reuse_snapshot:
        runner_source = capture_task_runner_stdout(args.run, args.output)
        (args.output / 'task-runner-source.json').write_text(json.dumps(runner_source, indent=2) + '\n')

    source_root = args.output if args.reuse_snapshot else args.run
    files = [source_root / name for name in ('launch.json', 'process.json', 'exit.json', 'train.log')]
    files += list((source_root / 'environment').glob('*.jsonl'))
    files += list((source_root / 'rollouts').glob('*.jsonl'))
    files += list((source_root / 'validation').glob('*.jsonl'))
    fingerprints = {}
    for source in sorted(files):
        if not source.exists():
            continue
        relative = source.relative_to(source_root)
        # Ray may leave a fully printed metric line without a trailing newline.
        # JSONL needs complete records; the console log must preserve every byte.
        data = read_complete(source) if source.suffix == '.jsonl' else source.read_bytes()
        target = args.output / relative
        if not args.reuse_snapshot:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        fingerprints[str(relative)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    for name in ('task-runner-stdout.log', 'task-runner-source.json'):
        path = args.output / name
        if path.exists():
            data = path.read_bytes()
            fingerprints[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

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
            request = row.get('request_id')
            join_method = 'official_request_id'
            if request is None:
                # This pinned runtime's actual dump omits request IDs. Match the
                # entire executed JSON command sequence, never score/order alone.
                commands = commands_from_output(row['output'])
                candidates = [key for key, trace in by_request.items()
                              if commands and commands == [event['command'] for event in trace]]
                request = candidates[0] if len(candidates) == 1 else None
                join_method = 'unique_full_command_sequence' if request else 'unresolved'
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
                'event_join_method': join_method,
                'raw_quality': raw, 'official_overlong_reward': penalty,
                'shaped_reward_from_components': raw + penalty,
                'original_dump_score': row['score'], 'completed': bool(row['completed']),
                'query_type': trajectory['query_type'] if trajectory else None,
                'tape_key': trajectory['tape_key'] if trajectory else None,
                'task_metrics': trajectory['task_metrics'] if trajectory else None,
            })

    merged_metrics = {}
    metric_sources = defaultdict(list)
    for name in ('train.log', 'task-runner-stdout.log'):
        path = args.output / name
        if not path.exists():
            continue
        for row in metrics_from_log(path.read_bytes()):
            combined = merged_metrics.setdefault(row['step'], {})
            for key, value in row.items():
                assert key not in combined or combined[key] == value, (name, row['step'], key)
                combined[key] = value
            metric_sources[row['step']].append(name)
    logs = [merged_metrics[step] for step in sorted(merged_metrics)]
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
        'official_metric_sources': dict(metric_sources),
        'nonfinite_metrics': nonfinite, 'retained_rollouts': retained, 'retained_groups': groups,
        'scope': 'Read-only append-prefix snapshot. Update logs are execution evidence, not post-update capability measurements. Missing/in-flight records are not failed trajectories. Raw-score events lack request IDs and are not individually joined.',
        'dump_score_semantics': 'Pinned official writer overwrites shaped score with raw score metadata. Shaped rewards here are reconstructed from recorded raw_reward and overlong_reward, not from decoded-token length.',
    }
    (args.output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('same_process_live', 'event_counts',
                     'official_updates_logged', 'nonfinite_metrics', 'retained_groups')}))


if __name__ == '__main__':
    main()
