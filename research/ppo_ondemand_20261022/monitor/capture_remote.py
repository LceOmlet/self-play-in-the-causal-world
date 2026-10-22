"""CPU-only, read-only training capture with exact native terminal diagnostics."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(x, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    temp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lines(p):
    if not p.exists():
        return []
    result = []
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A concurrent final append is not a completed observation.
    return result


def first_json(text):
    try:
        result = json.loads(text.splitlines()[0])
        return result if isinstance(result, dict) else {}
    except (ValueError, IndexError):
        return {}


def case_record(folder, row, environment):
    rollout = read(folder / 'rollout.json')
    events = lines(folder / 'events.jsonl')
    environment.reset(**row)
    ep = environment.episode
    budget = ep.budget.max_observations
    remaining = budget
    receipts = []
    requests = []
    checks = {'receipt_only': True, 'budget_ledger': True, 'native_reward_matches': None}
    accepted = []
    numerics_errors = protocol_errors = state_losses = exceptions = 0
    pending = None
    for event in events:
        if event['kind'] == 'tool_request':
            pending = event
            requests.append(event)
        elif event['kind'] == 'tool_exception':
            exceptions += 1
        elif event['kind'] == 'tool_response':
            output = event['output']
            data = first_json(output)
            if event['tool'] == 'act':
                if data.get('type') == 'batch_stored':
                    checks['receipt_only'] &= set(data) == {'type', 'observation_id', 'n', 'cost', 'remaining_budget'}
                    cmd = pending['arguments']['command'] if pending and pending['tool'] == 'act' else {}
                    checks['budget_ledger'] &= (data['cost'] == data['n'] * len(cmd.get('measure', []))
                        and data['remaining_budget'] == remaining - data['cost'])
                    remaining = data['remaining_budget']
                    receipts.append(data)
                elif data.get('type') == 'protocol_error' or 'error' in data:
                    protocol_errors += 1
                if output.startswith('Terminal answer accepted.'):
                    assert pending and pending['tool'] == 'act'
                    accepted.append(pending['arguments']['command'])
            else:
                numerics_errors += bool(re.search(r'(?m)^[A-Za-z_][\w.]*(?:Error|Exception):', output))
                state_losses += data.get('state_lost') is True
    checks['budget_ledger'] &= (sum(r['cost'] for r in receipts) == rollout['observations_used']
                                and remaining == budget - rollout['observations_used'])
    score = rollout.get('terminal_score')
    source = 'live owner terminal_score' if score is not None else None
    exact_reward = rollout.get('terminal_reward_exact')
    if rollout['submitted'] and score is None:
        assert len(accepted) == 1, 'Cannot unambiguously identify the accepted terminal answer'
        environment.act(accepted[0])
        assert environment.episode.completed
        score = environment.episode.terminal_score
        exact_reward = str(environment.episode.terminal_reward)
        source = 'same-version native owner scorer, recorded accepted answer and original task row'
        checks['native_reward_matches'] = environment.get_reward() == rollout['reward']
        assert checks['native_reward_matches'], 'Native score reconstruction disagrees with recorded reward'
    elif rollout['submitted']:
        from fractions import Fraction
        checks['native_reward_matches'] = float(Fraction(exact_reward)) == rollout['reward']
    else:
        checks['native_reward_matches'] = rollout['reward'] == 0.0
    prompt = len(rollout['prompt_ids'])
    completion = len(rollout['completion_ids'])
    active = sum(rollout['tool_mask'])
    checks.update(context_within_32768=prompt + completion <= 32768,
                  mask_matches_completion=len(rollout['tool_mask']) == completion,
                  mask_binary=all(x in (0, 1) for x in rollout['tool_mask']),
                  native_tool_call_count=len(requests) == rollout['tool_calls'])
    return {'case': folder.name, 'reward': rollout['reward'], 'submitted': rollout['submitted'],
        'terminal_score': score, 'terminal_score_source': source, 'terminal_reward_exact': exact_reward,
        'prompt_tokens': prompt, 'completion_tokens_including_tools': completion, 'model_tokens': active,
        'feedback_tokens': completion - active, 'total_context_tokens': prompt + completion,
        'at_context_limit': prompt + completion == 32768, 'budget': budget,
        'observations_used': rollout['observations_used'], 'remaining_budget': remaining,
        'budget_fraction': rollout['observations_used'] / budget, 'queries_used': len(receipts),
        'tool_calls': len(requests), 'act_calls': sum(x['tool'] == 'act' for x in requests),
        'numerics_calls': sum(x['tool'] == 'numerics' for x in requests),
        'explicit_numerics_error_outputs': numerics_errors, 'protocol_error_outputs': protocol_errors,
        'state_lost_outputs': state_losses, 'tool_exceptions': exceptions, 'checks': checks}


def step_record(runtime, metric, environment):
    folder = runtime / 'rollouts' / f"step-{metric['step']:06d}"
    row = read(folder / 'task-row.json')
    cases = [case_record(folder / f'case-{i}', row, environment) for i in range(4)]
    requests = responses = model_tokens = 0
    seconds = 0.
    finite_logprobs = True
    trace = folder / 'generation-batches.jsonl.gz'
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            if item['kind'] == 'generation_request':
                requests += 1
            else:
                responses += 1
                seconds += item['seconds']
                for output in item['outputs']:
                    for choice in output['choices']:
                        model_tokens += len(choice['token_ids'])
                        lp = choice['sampled_token_logprobs']
                        finite_logprobs &= len(lp) == len(choice['token_ids']) and all(math.isfinite(v) for v in lp)
    checks = {'four_cases': len(cases) == 4, 'generation_calls_paired': requests == responses,
        'sampled_logprobs_finite': finite_logprobs,
        'generated_tokens_match_masks': model_tokens == sum(c['model_tokens'] for c in cases),
        'active_policy_tokens_match': metric['active_policy_tokens'] == sum(c['model_tokens'] for c in cases),
        'rewards_match': metric['rewards'] == [c['reward'] for c in cases],
        'finite_losses_and_grad_norm': math.isfinite(metric['grad_norm']) and all(
            math.isfinite(v) for m in metric['micro_batches'] for v in m.values()),
        'lora_changed': metric['first_lora_B_changed'], 'gradient_accumulation_4': metric['gradient_accumulation_steps'] == 4}
    # The loss functions already normalize by global policy-token count: sum, never mean, microbatch losses.
    micro = metric['micro_batches']
    pg = sum(x['policy_loss'] for x in micro)
    vf = sum(x['value_loss'] for x in micro)
    lengths = [c['model_tokens'] for c in cases]
    def token_weighted(key):
        return sum(x[key] * n for x, n in zip(micro, lengths, strict=True)) / sum(lengths)
    return {'step': metric['step'], 'unix': metric['unix'], 'family': row['query_type'],
        'metrics': metric, 'cases': cases, 'checks': checks,
        'policy_loss_global': pg, 'value_loss_global': vf, 'combined_loss_global': pg + .1 * vf,
        'ppo_kl_token_weighted': token_weighted('actor/ppo_kl'),
        'policy_clipfrac_token_weighted': token_weighted('actor/pg_clipfrac'),
        'value_clipfrac_token_weighted': token_weighted('value_clipfrac'),
        'generation_calls': responses, 'generation_seconds': seconds, 'generated_model_tokens': model_tokens,
        'generation_tokens_per_second': model_tokens / seconds if seconds else None,
        'update_policy_tokens_per_second': metric['active_policy_tokens'] / metric['update_seconds'],
        'weight_sync': read(folder / 'weight-sync.json') if (folder / 'weight-sync.json').exists() else None,
        'input_sha256': {str(p.relative_to(runtime)): sha(p) for p in [folder / 'task-row.json', trace,
            *sorted(folder.glob('case-*/events.jsonl')), *sorted(folder.glob('case-*/rollout.json'))]}}


def main(root, output):
    started = time.time()
    project = root / 'project'
    receipt = read(root / 'launch-receipt.json')
    source_ok = all(sha(project / name) == expected for name, expected in receipt['source_sha256'].items())
    assert source_ok, 'Launched source changed; do not silently recompute scores under a different owner'
    for path in (root / 'deps', project / 'src'):
        sys.path.insert(0, str(path))
    from cpt_world import CPTWorldEnvironment
    environment = CPTWorldEnvironment()
    runtime = root / 'training'
    cache = root / 'monitor' / 'diagnostics'
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in lines(runtime / 'metrics.jsonl'):
        path = cache / f"step-{metric['step']:06d}.json"
        if path.exists():
            record = read(path)
            assert record['metrics'] == metric, 'Completed step mutated; retain evidence and inspect'
            current_stats = {name: {'bytes': (runtime / name).stat().st_size,
                'mtime_ns': (runtime / name).stat().st_mtime_ns} for name in record['input_sha256']}
            if 'input_stat' in record:
                assert record['input_stat'] == current_stats, 'Completed trajectory files changed'
            else:
                assert all(sha(runtime / name) == value for name, value in record['input_sha256'].items())
                record['input_stat'] = current_stats
                write(path, record)
        else:
            record = step_record(runtime, metric, environment)
            record['input_stat'] = {name: {'bytes': (runtime / name).stat().st_size,
                'mtime_ns': (runtime / name).stat().st_mtime_ns} for name in record['input_sha256']}
            write(path, record)
        rows.append(record)
    unit = 'cpt-numerics-ppo-ondemand-from-base-20261022.service'
    fields = ['ActiveState', 'SubState', 'MainPID', 'InvocationID', 'ExecMainStartTimestamp',
              'ExecMainStatus', 'CPUUsageNSec', 'MemoryCurrent', 'MemoryPeak', 'ControlGroup']
    command = ['systemctl', '--user', 'show', unit]
    for key in fields:
        command.extend(['-p', key])
    service = dict(x.split('=', 1) for x in subprocess.check_output(command, text=True).splitlines() if '=' in x)
    resource = {}
    for name in ('memory.current', 'memory.peak', 'memory.events', 'cpu.stat', 'io.stat'):
        path = Path('/sys/fs/cgroup' + service.get('ControlGroup', '')) / name
        try:
            resource[name] = path.read_text().strip()
        except OSError as error:
            resource[name] = {'unavailable': str(error)}
    pid = service.get('MainPID', '0')
    process = {}
    if pid != '0':
        try:
            process = {'cmdline': Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' '),
                       'cwd': os.readlink(f'/proc/{pid}/cwd'), 'stat': Path(f'/proc/{pid}/stat').read_text()}
        except OSError as error:
            process['error'] = str(error)
    status = read(runtime / 'status.json')
    active = runtime / 'rollouts' / f"step-{status.get('active_step', 0):06d}"
    newest = max((p.stat().st_mtime for p in active.glob('**/*') if p.is_file()), default=None)
    configuration = read(runtime / 'configuration.json')
    load = configuration['load']
    load_ok = all(load[k] == v for k, v in {'context_length': 32768, 'generation_batch_size': 4,
        'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 4}.items())
    checkpoints = []
    for path in sorted((runtime / 'checkpoints').glob('checkpoint-*'), key=lambda x: int(x.name.split('-')[-1])):
        required = ['adapter_model.safetensors', 'adapter_config.json', 'ppo_state.pt', 'training-state.json']
        checkpoints.append({'name': path.name, 'step': int(path.name.split('-')[-1]),
            'complete_files': all((path / n).is_file() and (path / n).stat().st_size > 0 for n in required),
            'files': {n: {'bytes': (path / n).stat().st_size, 'mtime_unix': (path / n).stat().st_mtime}
                      for n in required if (path / n).is_file()}})
    captured = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'unix': time.time(),
        'capture_seconds': time.time() - started, 'service': service, 'process': process,
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'status': status, 'source_matches_launch': source_ok, 'learning_load_unchanged': load_ok,
        'configuration': configuration, 'resources': resource, 'checkpoints': checkpoints,
        'last_active_artifact_unix': newest, 'active_artifact_age_seconds': time.time() - newest if newest else None,
        'failure': read(runtime / 'failure.json') if (runtime / 'failure.json').exists() else None,
        'gpu_physical_memory_bytes': None, 'gpu_utilization': None, 'gpu_power_watts': None,
        'gpu_missing_reason': 'Host NVML unavailable; vLLM sleep allocator counters are not physical residency',
        'completed_steps': len(rows), 'captured_global_step': max((r['step'] for r in rows), default=0),
        'native_score_source': 'owner fields, no reward inversion',
        'source_receipt_sha256': sha(root / 'launch-receipt.json'), 'monitor_source_sha256': sha(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'capture.json', captured)
    with (output / 'series.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as target:
        for path in sorted(output.iterdir()):
            target.add(path, arcname=path.name)
    print(json.dumps({'archive': str(archive), 'sha256': sha(archive), 'capture': captured}, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
