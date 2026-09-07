"""Compare the candidate with the last accepted production kernel."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'kernel_integration_20260907/evidence/boundary'
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'response-envelope'
NEW = ROOT / 'evidence' / VARIANT
CANDIDATE = 'diagonal_candidate' if VARIANT == 'diagonal-transport' else 'candidate'
BACKEND = ('indirect_mediator_diagonal_transport' if VARIANT == 'diagonal-transport'
           else 'one_mediator_response_envelope_attainment')


def read(folder, index):
    parent = json.loads((folder / f'{index}.parent.json').read_text())
    events = [json.loads(line) for line in (folder / f'{index}.events.jsonl').read_text().splitlines()]
    result = next((e for e in reversed(events) if e['event'] == 'result'), {})
    certificates = [e for e in events if e['event'] == 'certificate']
    return {'status': 'timeout' if parent['hard_timeout'] else result.get('status', 'no_result'),
            'result': result, 'parent': parent,
            'backend': certificates[-1].get('backend') if certificates else None}


records = []
regressions = []
old_counts = Counter()
new_counts = Counter()
for index in range(74):
    old, new = read(OLD, index), read(NEW, index)
    old_counts[old['status']] += 1
    new_counts[new['status']] += 1
    row = {'index': index, 'before': old['status'], 'after': new['status'],
           'before_seconds': old['result'].get('seconds'),
           'after_seconds': new['result'].get('seconds'), 'backend': new['backend']}
    if old['status'] == 'ok':
        if new['status'] != 'ok':
            regressions.append(index)
            records.append(row)
            continue
        a, b = old['result']['truth'], new['result']['truth']
        difference = max(abs(a[k] - b[k]) for k in ('lower', 'upper'))
        error_sum = a['endpoint_error'] + b['endpoint_error']
        assert difference <= error_sum + 1e-8, (index, difference, error_sum)
        row.update(endpoint_difference=difference, reported_error_sum=error_sum)
    records.append(row)
direct = [r for r in records if r['backend'] == BACKEND]
summary = {'before': dict(old_counts), 'after': dict(new_counts),
           'old_accepted_preserved': not regressions, 'regressions': regressions,
           'old_accepted_bounds_compatible': True,
           'newly_accepted': [r['index'] for r in records if r['before'] != 'ok' and r['after'] == 'ok'],
           'max_endpoint_difference': max(r.get('endpoint_difference', 0) for r in records),
           'direct_lp_certificates': direct,
           'direct_before_seconds': sum(r['before_seconds'] for r in direct),
           'direct_after_seconds': sum(r['after_seconds'] for r in direct),
           'candidate_sha256': hashlib.sha256((ROOT / CANDIDATE / 'src/cpt_world/counterfactual_solver.py').read_bytes()).hexdigest(),
           'records': records}
(ROOT / f'{VARIANT}-acceptance-summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k != 'records'}, indent=2))
