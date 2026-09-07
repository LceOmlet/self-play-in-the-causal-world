"""Derive concurrency bounds from the actual official generation timers.

No model, environment or loss implementation is used here. Each trajectory's
timer sums its nonoverlapping awaits of server_manager.generate. If I_i is that
set of intervals, |I_max \\ union(other I_i)| >= max(0, 2 max_i |I_i| - sum_i |I_i|).
This bounds time with only one pending trajectory, not GPU kernel utilization.
"""

import argparse
import hashlib
import json
from pathlib import Path


def analyze(snapshot):
    source = snapshot / 'progress.json'
    progress = json.loads(source.read_text())
    groups = {row['step']: row for row in progress['retained_groups']}
    results = []
    for metrics in progress['official_metrics']:
        step = metrics['step']
        if step not in groups or 'actor/grad_norm' not in metrics:
            continue
        group = groups[step]
        # Generation timers can include rejected groups; require the observed
        # unfiltered single generation batch before using the retained count.
        assert metrics['train/num_gen_batches'] == 1, step
        n = group['dumped_rows']
        wall = metrics['timing_s/gen']
        longest = metrics['timing_s/agent_loop/generate_sequences/max']
        cumulative = n * metrics['timing_s/agent_loop/generate_sequences/mean']
        assert 0 < longest <= wall and longest <= cumulative <= n * wall
        single_lower_bound = max(0.0, 2 * longest - cumulative)
        retained = [row for row in progress['retained_rollouts'] if row['step'] == step]
        results.append({
            'step': step, 'trajectories': n, 'family_counts': group['families'],
            'step_seconds': metrics['timing_s/step'], 'generation_seconds': wall,
            'generation_fraction': wall / metrics['timing_s/step'],
            'shortest_cumulative_generation_seconds': metrics['timing_s/agent_loop/generate_sequences/min'],
            'longest_cumulative_generation_seconds': longest,
            'sum_trajectory_generation_seconds': cumulative,
            'mean_pending_generation_requests': cumulative / wall,
            'single_pending_generation_seconds_lower_bound': single_lower_bound,
            'single_pending_generation_fraction_lower_bound': single_lower_bound / wall,
            'max_cumulative_tool_seconds': metrics['timing_s/agent_loop/tool_calls/max'],
            'grad_norm': metrics['actor/grad_norm'],
            'tis_ess_fraction': metrics['rollout_corr/rollout_is_eff_sample_size'],
            'tis_clipped_high_fraction': metrics['rollout_corr/rollout_is_ratio_fraction_high'],
            'qualities': group['raw_qualities'],
            'valid_adjustment_sets': sum(row['task_metrics']['valid_adjustment_set'] for row in retained)
            if set(group['families']) == {'backadj_minimal_sets'} else None,
        })
    return {
        'snapshot': str(snapshot), 'progress_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'steps': results,
        'interpretation': 'Pending generation includes Ray/server/engine waiting and prefill/decode. It is not measured GPU utilization or decode throughput. The bound does not require equal trajectory start times.',
        'limits': [
            'No per-token vLLM metrics in the current live endpoint: disable_log_stats is true.',
            'Response length includes tool text and cannot be compared directly with old model-only token counts.',
            'Two different worlds do not measure learning progress or isolate an inference optimization effect.',
            'This bound is for the current four trajectories, not a speed comparison against the historical policy.',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.snapshot)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
