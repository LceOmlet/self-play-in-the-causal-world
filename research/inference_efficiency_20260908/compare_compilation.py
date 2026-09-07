"""Compare completed equal-work GPU runs without claiming RL acceptance."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    invocation = json.loads((path/'invocation.json').read_text())
    complete = json.loads((path/'completion.json').read_text())
    assert complete['all_lengths_and_logprobs_valid']
    assert complete['measurement_sha256'] == sha(path/'measurements.json')
    measured = json.loads((path/'measurements.json').read_text())
    records = {(r['case'], r['repeat'], r['phase']): r for r in measured['records']}
    expected = {(f'p{length}_b{batch}', repeat, phase)
                for length in (2048, 24576) for batch in (1, 4)
                for repeat in (0, 1) for phase in ('cold_prefix', 'reused_prefix')}
    assert set(records) == expected and len(measured['records']) == 16
    for record in records.values():
        assert record['wall_seconds'] > 0 and math.isfinite(record['wall_seconds'])
        assert sum(len(r['token_ids']) for r in record['requests']) == record['output_tokens']
        for request in record['requests']:
            assert len(request['token_ids']) == len(request['selected_token_logprobs']) == 256
            assert all(math.isfinite(p) and p <= 1e-6 for p in request['selected_token_logprobs'])
    return invocation, measured, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eager', type=Path, required=True)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    eager, eager_data, eager_rows = read_run(args.eager)
    compiled, compiled_data, compiled_rows = read_run(args.compiled)
    for key in ('inputs_sha256', 'fingerprints', 'versions', 'sampling_options', 'script_sha256'):
        assert eager[key] == compiled[key], key
    a, b = eager['engine_options'], compiled['engine_options']
    changes = {key: {'eager': a.get(key), 'compiled': b.get(key)}
               for key in set(a) | set(b) if a.get(key) != b.get(key)}
    assert set(changes) == {'enforce_eager', 'compilation_config'}, changes
    summary = []
    for length in (2048, 24576):
        for batch in (1, 4):
            case = f'p{length}_b{batch}'
            for phase in ('cold_prefix', 'reused_prefix'):
                ea = [eager_rows[case, r, phase] for r in (0, 1)]
                co = [compiled_rows[case, r, phase] for r in (0, 1)]
                assert all(x['output_tokens'] == y['output_tokens'] == 256*batch
                           for x, y in zip(ea, co, strict=True))
                et = statistics.mean(x['wall_seconds'] for x in ea)
                ct = statistics.mean(x['wall_seconds'] for x in co)
                cached = {}
                for name, rows in (('eager', ea), ('compiled', co)):
                    values = [r['cached_prompt_tokens'] for row in rows for r in row['requests']]
                    cached[f'{name}_cached_prompt_tokens_mean'] = (
                        statistics.mean(values) if all(v is not None for v in values) else None)
                summary.append({'case': case, 'phase': phase, 'repeats': 2,
                                'eager_seconds_mean': et, 'compiled_seconds_mean': ct,
                                'eager_to_compiled_wall_ratio': et/ct,
                                'eager_output_tokens_per_wall_second': 256*batch/et,
                                'compiled_output_tokens_per_wall_second': 256*batch/ct, **cached})
    probabilities = []
    for key in sorted(eager_rows):
        for index, (a, b) in enumerate(zip(eager_rows[key]['requests'],
                                           compiled_rows[key]['requests'], strict=True)):
            token_pairs = zip(a['token_ids'], b['token_ids'], strict=True)
            common = next((i for i, (x, y) in enumerate(token_pairs) if x != y), 256)
            deltas = [abs(x-y) for x, y in zip(a['selected_token_logprobs'][:common],
                                             b['selected_token_logprobs'][:common], strict=True)]
            probabilities.append({'case': key[0], 'repeat': key[1], 'phase': key[2],
                                  'request': index, 'common_output_prefix_tokens': common,
                                  'max_abs_logprob_difference_on_common_history':
                                      max(deltas) if deltas else None})
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'timing.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = {'identical_inputs_weights_and_sampling': True, 'engine_option_changes': changes,
              'startup_seconds': {'eager': eager_data['startup_seconds'],
                                  'compiled': compiled_data['startup_seconds']},
              'timing': summary, 'common_history_probability_comparisons': probabilities,
              'inputs': {str(p/f): sha(p/f) for p in (args.eager, args.compiled)
                         for f in ('invocation.json', 'measurements.json', 'completion.json')},
              'scope': 'Two warmed repetitions per fixed workload. Wall time includes prefill '
              'and host overhead. Sampled probabilities are comparable only before the first '
              'token divergence. This does not establish full distribution equality, historical '
              'slowdown attribution, verl weight-sync behavior, official TIS or learning.'}
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'startup_seconds': report['startup_seconds'], 'timing': summary}))


if __name__ == '__main__':
    main()
