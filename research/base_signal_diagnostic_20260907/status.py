"""Read this diagnostic's exact process identity and completed tool events."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import psutil

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path,
                    default=Path('/home/chen/runs/base-signal-diagnostic-20260907/run-01'))
args = parser.parse_args()
run = args.run
launch = json.loads((run/'launch.json').read_text())
result = {'elapsed_seconds': time.time()-launch['start_time'],
          'exit': json.loads((run/'exit.json').read_text()) if (run/'exit.json').exists() else None}
if (run/'process.json').exists():
    identity = json.loads((run/'process.json').read_text())
    result['process'] = identity
    try:
        process = psutil.Process(identity['pid'])
        result['same_process_live'] = abs(process.create_time()-identity['create_time']) < .01
        result['process_status'] = process.status()
        result['descendants'] = len(process.children(recursive=True))
    except psutil.NoSuchProcess:
        result['same_process_live'] = False
events = []
for file in (run/'environment').glob('*.jsonl'):
    lines = file.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # Only an unfinished final append may be transient during a live read.
            if i != len(lines)-1 or result.get('exit') is not None:
                raise
acts = [e for e in events if e['event'] == 'act']
result['events'] = dict(Counter(e['event'] for e in events))
result['trajectories_with_actions'] = len({e['trajectory_id'] for e in acts})
result['completed_trajectories'] = len({e['trajectory_id'] for e in acts if e['completed']})
result['scored_by_family'] = dict(Counter(e['data_source'] for e in events if e['event'] == 'raw_score'))
errors = Counter()
for event in acts:
    feedback = json.loads(event['feedback'].splitlines()[0])
    if feedback.get('type') == 'protocol_error':
        errors[feedback['error']['message']] += 1
result['protocol_errors'] = dict(errors)
result['validation_files'] = [p.name for p in (run/'validation').glob('*.jsonl')]
print(json.dumps(result, indent=2))
