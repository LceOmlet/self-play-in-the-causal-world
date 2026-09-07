"""Audit a saved real update and export its LoRA through upstream methods on CPU.

This compares saved parameters/moments with the real post-Adam observation.
It does not construct a 9B model, resume a trainer or implement an optimizer.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
RUN = Path('/home/chen/runs/rl-correctness-goal-20260907/official_execution/run-02-mask-v1-audit-only')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(VERL))
    import torch
    from safetensors.torch import load_file
    from torch.distributed.tensor import DTensor
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    def local(tensor):
        if isinstance(tensor, DTensor):
            assert tensor.device_mesh.mesh.numel() == 1
            tensor = tensor.to_local()
        assert tensor.device.type == 'cpu'
        return tensor

    acceptance_path = RUN / 'execution-acceptance.json'
    accepted = json.loads(acceptance_path.read_text())
    assert accepted['passed'] and accepted['official_updates'] == 1
    checkpoint = RUN / 'checkpoints/global_step_1'
    actor = checkpoint / 'actor'
    observation, = (RUN / 'observation').glob('*/adam_return-*.pt')
    post = torch.load(observation, map_location='cpu', weights_only=True)
    model_path = actor / 'model_world_size_1_rank_0.pt'
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    optimizer_path = actor / 'optim_world_size_1_rank_0.pt'
    optimizer = torch.load(optimizer_path, map_location='cpu', weights_only=False)
    extra_path = actor / 'extra_state_world_size_1_rank_0.pt'
    extra = torch.load(extra_path, map_location='cpu', weights_only=False)
    data_path = checkpoint / 'data.pt'
    data = torch.load(data_path, map_location='cpu', weights_only=False)

    assert set(model) == set(post['trainable']) and len(model) == 496
    for name, tensor in model.items():
        assert torch.equal(local(tensor), post['trainable'][name]['value']), name
        assert torch.isfinite(local(tensor)).all(), name
    id_to_name = {}
    for saved, observed in zip(optimizer['param_groups'], post['param_groups'], strict=True):
        assert len(saved['params']) == len(observed['params'])
        id_to_name.update(zip(saved['params'], observed['params'], strict=True))
        assert {k: v for k, v in saved.items() if k != 'params'} == {
            k: v for k, v in observed.items() if k != 'params'}
    frozen_placeholders = []
    moment_tensors = 0
    for index, state in optimizer['state'].items():
        name = id_to_name[index]
        if name in post['state']:
            for key, value in state.items():
                expected = post['state'][name][key]
                assert torch.equal(local(value), expected), (name, key)
                moment_tensors += key in ('exp_avg', 'exp_avg_sq')
        else:
            # Upstream checkpoint serialization adds empty frozen-state entries.
            assert name in {item['name'] for item in post['frozen']}
            assert float(local(state['step'])) == 0
            assert local(state['exp_avg']).numel() == local(state['exp_avg_sq']).numel() == 0
            frozen_placeholders.append(name)
    assert set(post['state']) == set(model) and moment_tensors == 992
    assert len(frozen_placeholders) == 760
    assert extra['lr_scheduler']['last_epoch'] == 1
    assert data['_num_yielded'] == 1
    assert set(extra['rng']) == {'cpu', 'numpy', 'random', 'cuda'}

    # Execute upstream shard loading/merging and upstream adapter serialization.
    # The full-HF export path allocates a CPU base model; it is unnecessary for
    # this LoRA-only checkpoint and is deliberately not called.
    config = ModelMergerConfig(operation='merge', backend='fsdp', local_dir=str(actor),
                               target_dir=str(args.output), hf_model_config_path=str(actor / 'huggingface'))
    merger = FSDPModelMerger(config)
    world_size = merger._get_world_size()
    assert world_size == 1
    rank_zero = merger._load_rank_zero_state_dict(world_size)
    mesh, names = merger._extract_device_mesh_info(rank_zero, world_size)
    shards, shape = merger._calculate_shard_configuration(mesh, names)
    merged = merger._load_and_merge_state_dicts(world_size, shards, shape, names)
    assert set(merged) == set(model)
    for name, value in merged.items():
        # The upstream merger casts to BF16; this particular real checkpoint is
        # already BF16, so export must preserve every value exactly.
        assert model[name].dtype == torch.bfloat16 and torch.equal(value, local(model[name]))
    adapter = Path(merger.save_lora_adapter(merged))
    assert not merged
    exported = load_file(adapter / 'adapter_model.safetensors')
    assert len(exported) == len(model)
    for name, value in model.items():
        assert torch.equal(exported[name.replace('.default.weight', '.weight')], local(value)), name
    adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
    assert adapter_config['r'] == 8 and adapter_config['lora_alpha'] == 32
    assert adapter_config['task_type'] == 'CAUSAL_LM'
    assert not torch.cuda.is_initialized()
    inputs = [acceptance_path, observation, model_path, optimizer_path, extra_path, data_path,
              actor / 'lora_train_meta.json', actor / 'fsdp_config.json', actor / 'huggingface/config.json']
    sources = [VERL / f'verl/model_merger/{name}.py' for name in ('base_model_merger', 'fsdp_model_merger')]
    report = {
        'passed_saved_state_and_export': True,
        'model_tensors_identical_to_post_adam': len(model),
        'optimizer_moment_tensors_identical_to_post_adam': moment_tensors,
        'frozen_zero_sized_optimizer_placeholders': len(frozen_placeholders),
        'nonzero_lora_B_tensors': sum(bool(local(t).count_nonzero()) for n, t in model.items() if 'lora_B' in n),
        'dtypes': dict(Counter(str(t.dtype) for t in model.values())),
        'lr_scheduler_last_epoch': extra['lr_scheduler']['last_epoch'],
        'dataloader_yielded': data['_num_yielded'], 'saved_training_rng_keys': sorted(extra['rng']),
        'adapter_path': str(adapter), 'adapter_tensors_bitwise_identical': len(exported),
        'cuda_initialized': torch.cuda.is_initialized(),
        'inputs': {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in inputs},
        'official_sources': {str(p): digest(p) for p in sources},
        'export_files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in adapter.iterdir()},
        'limits': ['No actual GPU checkpoint load or resumed optimizer step in this audit.',
                   'Training RNG files do not establish persistence of the separate vLLM sampling RNG.',
                   'This is the verified run-02 checkpoint, not the current live run or a disqualified checkpoint.',
                   'No claim of exact shuffled-data continuation from file presence alone.'],
    }
    (args.output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('inputs', 'official_sources', 'export_files')}))


if __name__ == '__main__':
    main()
