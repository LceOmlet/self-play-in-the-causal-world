"""Compare recorded workloads and timings; do not infer a controlled speedup."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics


def aggregate(rows):
    return {
        'groups': len(rows),
        'step_seconds_mean': statistics.mean(r['step_time'] for r in rows),
        'step_seconds_median': statistics.median(r['step_time'] for r in rows),
        'model_generated_tokens_mean': statistics.mean(r['completions/mean_length'] for r in rows),
        'tool_calls_mean': statistics.mean(r['tools/call_frequency'] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-log', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.legacy_log.read_bytes()
    rows = []
    for line in data.decode().replace('\r', '\n').splitlines():
        if not line.startswith("{'loss':"):
            continue
        row = {key: float(value) for key, value in ast.literal_eval(line).items()}
        row['task'] = next(key.split('/')[1] for key in row
                           if key.startswith('task/') and key.endswith('/reward_raw'))
        rows.append(row)
    assert len(rows) == 649
    backdoor = [r for r in rows if r['task'] == 'backadj_minimal_sets']
    progress = json.loads(args.progress.read_text())
    first = next(r for r in progress['official_metrics'] if r['step'] == 1)
    retained = [r for r in progress['retained_rollouts'] if r['step'] == 1]
    assert len(retained) == 4 and all(r['query_type'] == 'backadj_minimal_sets' for r in retained)
    report = {
        'legacy_log_sha256': hashlib.sha256(data).hexdigest(),
        'progress_sha256': hashlib.sha256(args.progress.read_bytes()).hexdigest(),
        'legacy_all': aggregate(rows), 'legacy_backdoor': aggregate(backdoor),
        'legacy_backdoor_nonzero_gradient': aggregate([r for r in backdoor if r['grad_norm'] > 0]),
        'legacy_last100_backdoor': aggregate([r for r in rows[-100:] if r['task'] == 'backadj_minimal_sets']),
        'current_first_update': {key: first[key] for key in (
            'timing_s/step', 'timing_s/gen', 'timing_s/old_log_prob',
            'timing_s/update_actor', 'timing_s/update_weights',
            'timing_s/agent_loop/tool_calls/mean', 'timing_s/agent_loop/tool_calls/max',
            'response_length/mean', 'train/num_gen_batches', 'actor/grad_norm')},
        'current_generation_share': first['timing_s/gen'] / first['timing_s/step'],
        'current_to_legacy_backdoor_wall_ratio': first['timing_s/step'] / aggregate(backdoor)['step_seconds_mean'],
        'scope': 'Observed first current update versus historical task-family averages; different worlds, policy weights and runtimes. Not a causal attribution or steady-state throughput estimate.',
        'token_warning': 'Historical completions/mean_length counts model-generated tokens; current response_length/mean uses attention-mask response length, including tool text. Do not divide these to claim a generation-length or tokens-per-second ratio.',
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
