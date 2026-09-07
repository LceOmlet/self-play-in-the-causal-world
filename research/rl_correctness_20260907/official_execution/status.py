"""Read only this audit's state; never infer completion from an empty log."""
from collections import Counter
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent/'run-02-mask-v1-audit-only'
events = [json.loads(line) for path in (ROOT/'observation').glob('*/events.jsonl')
          for line in path.read_text().splitlines()]
env_events = [json.loads(line) for path in (ROOT/'environment').glob('*.jsonl')
              for line in path.read_text().splitlines()]
result = {'time': time.time(), 'elapsed_seconds': time.time()-json.loads((ROOT/'launch.json').read_text())['start_time'],
          'exit': json.loads((ROOT/'exit.json').read_text()) if (ROOT/'exit.json').exists() else None,
          'observer_events': dict(Counter(e['event'] for e in events)),
          'observer_errors': [str(p) for p in (ROOT/'observation').glob('observer-errors-*')],
          'environment_events': dict(Counter(e['event'] for e in env_events)),
          'completed_trajectories': len({e['trajectory_id'] for e in env_events if e['event']=='act' and e['completed']}),
          'latest_environment': [{k: e.get(k) for k in ('event', 'query_type', 'raw_reward', 'completed', 'observations_used', 'queries_used')}
                                 for e in env_events[-4:]]}
print(json.dumps(result, indent=2))
