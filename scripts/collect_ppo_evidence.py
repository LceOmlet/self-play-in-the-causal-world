"""Snapshot launch sources and PPO evidence without modifying the running task."""
import argparse
import hashlib
import gzip
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    project = root / 'project'
    receipt = json.loads((root / 'launch-receipt.json').read_text())
    hashes = receipt['source_sha256']
    for name, expected in hashes.items():
        assert sha(project / name) == expected, name
    with (root / 'source.tar.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name in sorted(hashes):
                    path = project / name
                    info = archive.gettarinfo(path, arcname=name)
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        archive.addfile(info, stream)
    (root / 'source-files-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    runtime = root / 'training'
    versions = {}
    dependency_paths = [str(root / 'kernel-compat'), str(root / 'deps'), *sys.path]
    for distribution in importlib.metadata.distributions(path=dependency_paths):
        name = distribution.metadata.get('Name', '').lower()
        if name in {'torch', 'transformers', 'trl', 'vllm', 'peft', 'liger-kernel', 'flash-linear-attention',
                    'kernels', 'kernels-data', 'tensordict', 'omegaconf', 'ray'}:
            versions.setdefault(name, distribution.version)
    (root / 'dependency-versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    files = set(root.glob('*.json')) | set(root.glob('*.xml')) | set(root.glob('*.log'))
    files |= set((root / 'benchmark-trials').glob('*.json'))
    files |= set(root.glob('preflight*-run_ppo_ondemand.py'))
    files |= {runtime / name for name in ('configuration.json', 'fresh-start.json', 'status.json', 'metrics.jsonl', 'failure.json')}
    for step in (1, 2):
        folder = runtime / 'rollouts' / f'step-{step:06d}'
        files |= {p for p in folder.rglob('*') if p.is_file()}
    files |= set((runtime / 'checkpoints').glob('*/training-state.json'))
    checkpoints = {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in (runtime / 'checkpoints' / 'checkpoint-1').glob('*') if p.is_file()}
    checkpoint = runtime / 'checkpoints' / 'checkpoint-1'
    if (checkpoint / 'ppo_state.pt').exists():
        import torch
        from safetensors.torch import load_file
        torch.set_num_threads(1)
        state = torch.load(checkpoint / 'ppo_state.pt', map_location='cpu', weights_only=True)
        assert state['global_step'] == 1
        assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state['value_head'].values())
        assert all(int(v['step'].item()) == 1 for v in state['optimizer']['state'].values())
        adapter = load_file(checkpoint / 'adapter_model.safetensors', device='cpu')
        assert all(torch.isfinite(v).all() for v in adapter.values())
        nonzero_b = sum(int(torch.count_nonzero(v)) > 0 for k, v in adapter.items() if '.lora_B.' in k)
        assert nonzero_b > 0
        validation = {'passed': True, 'global_step': 1, 'value_head_finite_fp32': True,
            'optimizer_states_at_step_one': len(state['optimizer']['state']),
            'adapter_all_finite': True, 'nonzero_lora_B_layers': nonzero_b,
            'checkpoint_files': checkpoints}
        (root / 'checkpoint-validation.json').write_text(json.dumps(validation, indent=2) + '\n')
        files.add(root / 'checkpoint-validation.json')
    unit = 'cpt-numerics-ppo-ondemand-from-base-20261022.service'
    service = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'InvocationID', '-p', 'ExecMainStartTimestamp'], text=True)
    snapshot = {'service': service, 'checkpoint_files': checkpoints, 'source_tar_sha256': sha(root / 'source.tar.gz')}
    summaries = {}
    for step in (1, 2):
        folder = runtime / 'rollouts' / f'step-{step:06d}'
        trace = folder / 'generation-batches.jsonl.gz'
        if not trace.exists():
            continue
        responses = []
        latest_request = {}
        try:
            with gzip.open(trace, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    if record['kind'] == 'generation_request':
                        latest_request = record
                    if record['kind'] == 'generation_response':
                        responses.append(record)
        except (EOFError, json.JSONDecodeError):
            pass  # A live append may be incomplete; completed response records remain valid.
        tokens = sum(len(c['token_ids']) for r in responses for o in r['outputs'] for c in o['choices'])
        seconds = sum(r['seconds'] for r in responses)
        receipts = 0
        for path in folder.glob('case-*/events.jsonl'):
            for line in path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('kind') == 'tool_response' and r.get('tool') == 'act':
                    try:
                        output = json.loads(r['output'].splitlines()[0])
                    except (json.JSONDecodeError, IndexError):
                        continue  # Original terminal acknowledgements / protocol errors.
                    if output.get('type') == 'batch_stored':
                        assert set(output) == {'type', 'observation_id', 'n', 'cost', 'remaining_budget'}
                        receipts += 1
        summaries[folder.name] = {'completed_generation_calls': len(responses), 'generated_model_tokens': tokens,
            'generation_seconds': seconds, 'generation_tokens_per_second': tokens / seconds if seconds else None,
            'verified_receipt_only_queries': receipts, 'completed_rollouts': len(list(folder.glob('case-*/rollout.json'))),
            'latest_generation_prompt_lengths': latest_request.get('prompt_lengths')}
    snapshot['rollout_summary'] = summaries
    (root / 'evidence-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    files.add(root / 'evidence-snapshot.json')
    with tarfile.open(root / 'evidence.tar.gz', 'w:gz') as archive:
        for path in sorted(files):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(root)))
    print(json.dumps({'source_files': len(hashes), 'source_sha256': sha(root / 'source.tar.gz'),
                      'evidence_sha256': sha(root / 'evidence.tar.gz'), 'snapshot': snapshot}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
