"""Equal-work vLLM timing on the real 9B model; no trainer or loss implementation.

Prepare token fixtures on CPU while training runs. GPU execution refuses to run
alongside any compute process. Compilation is the only variant; adapter merging
and the actual verl checkpoint/sync/update lifecycle need separate acceptance.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

MODEL = Path('/home/chen/models/Qwen/Qwen3.5-9B')
ADAPTER = Path('/home/chen/runs/checkpoint-continuity-20260908/saved-state-v2/lora_adapter')
ROLLOUT = Path('/home/chen/runs/training-submission-20260907/run-01/rollouts/2.jsonl')
CONFIG = Path('/home/chen/runs/inference-efficiency-20260908/configuration-v2/current.yaml')
EXPECTED = {
    'model_config': 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05',
    'adapter_weights': '7ceddcfef4e4b56e5febd21b34571baad6b67343a8af5bb64a0c8f50d2cfa12f',
    'adapter_config': 'f27058964c6dfb8a13bba725aeffd1befafcac005c6e3ec650500f4110ed6e61',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fingerprints():
    paths = {'model_config': MODEL/'config.json',
             'adapter_weights': ADAPTER/'adapter_model.safetensors',
             'adapter_config': ADAPTER/'adapter_config.json',
             'tokenizer': MODEL/'tokenizer.json',
             'tokenizer_config': MODEL/'tokenizer_config.json',
             'resolved_training_config': CONFIG, 'source_rollout': ROLLOUT}
    hashes = {key: sha(path) for key, path in paths.items()}
    for key, expected in EXPECTED.items():
        assert hashes[key] == expected, (key, hashes[key])
    return hashes


def prepare(output):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from transformers import AutoTokenizer

    hashes = fingerprints()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    rows = [json.loads(line) for line in ROLLOUT.read_text().splitlines() if line.strip()]
    encoded = [tokenizer.encode(row['input'] + row['output'], add_special_tokens=False)
               for row in rows]
    index = max(range(len(encoded)), key=lambda i: len(encoded[i]))
    tokens = encoded[index]
    assert len(tokens) >= 24576 + 768, len(tokens)
    cases = []
    for length in (2048, 24576):
        prompts = [tokens[offset:offset+length] for offset in (0, 256, 512, 768)]
        # Separate first cache blocks prevent accidental within-batch sharing.
        assert len({tuple(prompt[:256]) for prompt in prompts}) == 4
        for batch in (1, 4):
            cases.append({'name': f'p{length}_b{batch}', 'prompt_length': length,
                          'batch': batch, 'prompt_token_ids': prompts[:batch]})
    assert not torch.cuda.is_initialized()
    output.mkdir(parents=True, exist_ok=False)
    write(output/'inputs.json', {'scope': 'Retokenized sliding excerpts of actual rollout text; '
           'not original training token IDs or new environment trajectories.',
           'fingerprints': hashes, 'source_row': index,
           'encoded_row_lengths': [len(row) for row in encoded], 'cases': cases})
    write(output/'preparation.json', {'time': time.time(), 'cuda_initialized': False,
          'script_sha256': sha(__file__), 'inputs_sha256': sha(output/'inputs.json'),
          'shape_summary': [{k: case[k] for k in ('name', 'prompt_length', 'batch')}
                            for case in cases]})
    print(json.dumps({'prepared': str(output), 'row_lengths': [len(x) for x in encoded]}))


def engine_options(variant):
    import yaml
    config = yaml.safe_load(CONFIG.read_text())
    rollout = config['actor_rollout_ref']['rollout']
    keys = ('dtype', 'load_format', 'max_model_len', 'max_num_seqs',
            'enable_chunked_prefill', 'max_num_batched_tokens',
            'enable_prefix_caching', 'logprobs_mode', 'gpu_memory_utilization',
            'seed', 'scheduling_policy')
    options = {key: rollout[key] for key in keys}
    options.update(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=False,
                   tensor_parallel_size=1, distributed_executor_backend='mp',
                   enable_lora=True, max_loras=1, max_lora_rank=8,
                   disable_log_stats=False, enforce_eager=variant == 'eager',
                   enable_sleep_mode=True, generation_config='vllm')
    options['compilation_config'] = {'cudagraph_mode': 'FULL_AND_PIECEWISE'}
    if variant == 'compiled':
        options['compilation_config']['cudagraph_capture_sizes'] = [1, 2, 4]
    assert options['max_model_len'] == 32768 and options['max_num_seqs'] == 4
    assert options['dtype'] == 'bfloat16'
    return options


def sampling_options(seed):
    # Fixed output work is solely for timing. EOS ignoring is not a trainer edit.
    return dict(temperature=1.0, top_p=1.0, top_k=-1, seed=seed,
                max_tokens=256, min_tokens=256, ignore_eos=True,
                logprobs=0, detokenize=False)


def benchmark(inputs_path, output, variant):
    gpu_processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not gpu_processes, f'GPU occupied; preserve training first: {gpu_processes}'
    fixture = json.loads(inputs_path.read_text())
    assert fixture['fingerprints'] == fingerprints()
    preparation = json.loads((inputs_path.parent/'preparation.json').read_text())
    assert sha(inputs_path) == preparation['inputs_sha256']
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    options = engine_options(variant)
    write(output/'invocation.json', {'variant': variant, 'time': time.time(),
          'engine_options': options, 'sampling_options': sampling_options(777),
          'script_sha256': sha(__file__), 'inputs_sha256': sha(inputs_path),
          'fingerprints': fixture['fingerprints'],
          'versions': {name: importlib.metadata.version(name)
                       for name in ('vllm', 'torch', 'transformers')}})
    start = time.perf_counter()
    llm = LLM(**options)
    startup = time.perf_counter() - start
    adapter = LoRARequest('verified-nonzero-adapter', 1, str(ADAPTER))
    records = []
    for case in fixture['cases']:
        prompts = [{'prompt_token_ids': ids} for ids in case['prompt_token_ids']]
        params = [SamplingParams(**sampling_options(777+i)) for i in range(case['batch'])]
        # Warm this exact shape, then measure cache-empty and cache-reused calls.
        llm.generate(prompts, params, lora_request=adapter, use_tqdm=False)
        for repeat in range(2):
            assert llm.reset_prefix_cache()
            for phase in ('cold_prefix', 'reused_prefix'):
                start = time.perf_counter()
                results = llm.generate(prompts, params, lora_request=adapter, use_tqdm=False)
                elapsed = time.perf_counter() - start
                assert len(results) == case['batch']
                requests = []
                for expected_prompt, result in zip(case['prompt_token_ids'], results, strict=True):
                    assert list(result.prompt_token_ids) == expected_prompt
                    assert result.finished and len(result.outputs) == 1
                    completion = result.outputs[0]
                    ids = list(completion.token_ids)
                    assert len(ids) == 256 and completion.finish_reason == 'length'
                    assert completion.logprobs is not None and len(completion.logprobs) == 256
                    probs = [row[token].logprob for token, row
                             in zip(ids, completion.logprobs, strict=True)]
                    assert all(math.isfinite(p) for p in probs)
                    requests.append({'request_id': result.request_id, 'token_ids': ids,
                                     'selected_token_logprobs': probs,
                                     'cached_prompt_tokens': result.num_cached_tokens})
                record = {'case': case['name'], 'repeat': repeat, 'phase': phase,
                          'wall_seconds': elapsed, 'output_tokens': 256*case['batch'],
                          'requests': requests}
                records.append(record)
                write(output/'measurements.json', {'startup_seconds': startup, 'records': records})
                print(json.dumps({k: v for k, v in record.items() if k != 'requests'}), flush=True)
    write(output/'completion.json', {'time': time.time(), 'all_lengths_and_logprobs_valid': True,
          'measurement_sha256': sha(output/'measurements.json'),
          'scope': 'Equal-work official vLLM inference only. No merge/sync, training loss, '
          'official TIS, GPU checkpoint resume, or learning acceptance is implied.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'benchmark'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--variant', choices=['eager', 'compiled'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    else:
        assert args.inputs is not None and args.variant is not None
        benchmark(args.inputs, args.output, args.variant)
