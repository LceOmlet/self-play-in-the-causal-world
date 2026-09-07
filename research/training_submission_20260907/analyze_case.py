"""Verify the actual completed pilot answer against its first public histogram."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
events = [json.loads(line) for p in (args.run/'environment').glob('*.jsonl')
          for line in p.read_text(encoding='utf-8').splitlines()]
end = next(e for e in events if e['event'] == 'act' and e['completed'])
assert end['trajectory_id'] == '7a8a9289e2c74de280e3fdc41521411c'
same = [e for e in events if e.get('trajectory_id') == end['trajectory_id']]
prediction = end['command']['effect']
experiments = []
for i, event in enumerate(same):
    if event['completed']:
        continue
    feedback = json.loads(event['feedback'].splitlines()[0])
    if feedback['type'] != 'batch_result':
        continue
    histogram = feedback['batch']['joint_histogram']
    columns = histogram['columns']
    if 'GED' not in columns or 'VPK' not in columns:
        continue
    counts = [[0]*5 for _ in range(2)]
    for values, n in histogram['rows']:
        counts[values[columns.index('GED')]][values[columns.index('VPK')]] += n
    sizes = [sum(row) for row in counts]
    effect = [float(Fraction(counts[1][j], sizes[1])-Fraction(counts[0][j], sizes[0]))
              for j in range(5)] if all(sizes) else None
    experiments.append({
        'action': i, 'command': event['command'], 'X_counts': sizes, 'XY_counts': counts,
        'empirical_difference': effect,
        'max_prediction_difference': max(abs(effect[j]-prediction[f'state_{j}'])
                                         for j in range(5)) if effect else None,
    })
assert experiments[0]['X_counts'] == [4, 124]
assert experiments[0]['max_prediction_difference'] < 5e-9
assert experiments[1]['X_counts'] == [20, 44]
assert experiments[1]['command']['target'] == 'SUM'
assert len(same) == 24 and end['queries_used'] == 23
report = {
    'trajectory_id': end['trajectory_id'], 'terminal': end,
    'experiments': experiments, 'total_actions': len(same),
    'scope': 'Numerical equality to the first passive estimator; not a claim about hidden model reasoning.',
}
args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'matched_to_output_precision': True,
                  'quality': end['raw_reward'],
                  'TV_error': end['terminal_score']['total_variation_error']}))
