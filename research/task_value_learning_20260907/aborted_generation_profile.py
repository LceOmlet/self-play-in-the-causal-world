"""Bounded inference-only replay: real prefixes, pristine LoRA, worker profiler off/on/off."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/home/chen/runs/generation-priority-20260907')
MODEL = '/home/chen/models/Qwen/Qwen3.5-9B'
AUDIT = Path('/home/chen/runs/rl-correctness-goal-20260907/official_execution')
RUN = AUDIT / 'run-02-mask-v1-audit-only'


def toggle_worker_profiler(worker, enabled):
    import importlib.util
    import os
    import sys
    import threading
    from pathlib import Path

    if enabled:
        os.environ['CPT_DAPO_OBSERVATION_DIR'] = '/home/chen/runs/generation-priority-20260907/worker-observer'
        os.environ['VERL_ROOT'] = '/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1'
        path = Path('/home/chen/runs/rl-correctness-goal-20260907/official_execution/observer/execution_observer.py')
        spec = importlib.util.spec_from_file_location('generation_audit_observer', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.install()
    else:
        sys.setprofile(None)
        threading.setprofile(None)
    return {'pid': os.getpid(), 'enabled': sys.getprofile() is not None}


def main():
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    ROOT.mkdir(parents=True, exist_ok=True)
    assert sys.getprofile() is None
    initial = RUN / 'observation/75071/adam_call-0001.pt'
    before = torch.load(initial, map_location='cpu', weights_only=False)
    assert not before['state']
    state = {name.replace('.default.', '.'): record['value'].contiguous()
             for name, record in before['trainable'].items()}
    b_names = [name for name in state if '.lora_B.' in name]
    assert len(b_names) == 248 and all(torch.count_nonzero(state[name]).item() == 0 for name in b_names)
    adapter = ROOT / 'pristine-adapter'
    adapter.mkdir(exist_ok=True)
    LoraConfig(r=8, lora_alpha=32, target_modules='all-linear',
               exclude_modules='.*(?:visual|vision|aligner|multi_token_predictor|mtp|lm_head).*',
               task_type='CAUSAL_LM', base_model_name_or_path=MODEL).save_pretrained(adapter)
    save_file(state, str(adapter / 'adapter_model.safetensors'))
    del before, state
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in (RUN / 'rollouts/1.jsonl').read_text().splitlines()]
    prompts = [{'prompt_token_ids': tokenizer.encode(row['input'] + row['output'], add_special_tokens=False)[:2048]}
               for row in rows]
    assert len(prompts) == 4 and all(len(p['prompt_token_ids']) == 2048 for p in prompts)
    (ROOT / 'prompts.json').write_text(json.dumps(prompts))
    report = {'purpose': 'Inference-only cost localization, no optimizer or training',
              'pristine_lora_B_all_zero': True, 'before_optimizer_sha256': hashlib.sha256(initial.read_bytes()).hexdigest(),
              'prompt_sha256': hashlib.sha256((ROOT / 'prompts.json').read_bytes()).hexdigest(),
              'max_model_len': 8192, 'actual_prefix_tokens': 2048, 'generated_tokens_per_request': 64,
              'enforce_eager': True, 'worker_only_profiler_toggle': True, 'trials': []}
    started = time.perf_counter()
    llm = LLM(model=MODEL, dtype='bfloat16', max_model_len=8192, max_num_seqs=4,
              max_num_batched_tokens=8192, gpu_memory_utilization=0.5,
              enforce_eager=True, enable_prefix_caching=True, enable_lora=True,
              max_lora_rank=8, max_loras=1, seed=42,
              limit_mm_per_prompt={'image': 0, 'video': 0})
    report['load_seconds'] = time.perf_counter() - started
    lora = LoRARequest('pristine', 1, str(adapter))
    params = SamplingParams(temperature=0.0, max_tokens=64, min_tokens=64,
                            ignore_eos=True, logprobs=1)
    for label, enabled in [('warmup', False), ('off-1', False), ('on', True), ('off-2', False)]:
        worker_status = llm.collective_rpc(toggle_worker_profiler, args=(enabled,))
        assert all(status['enabled'] == enabled for status in worker_status)
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
        elapsed = time.perf_counter() - started
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        assert all(len(row) == 64 for row in tokens)
        record = {'label': label, 'seconds': elapsed, 'tokens_per_second': 256 / elapsed,
                  'token_ids': tokens, 'worker_status': worker_status}
        report['trials'].append(record)
        (ROOT / 'generation-profile.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in record.items() if k != 'token_ids'}), flush=True)
    assert len({json.dumps(trial['token_ids']) for trial in report['trials']}) == 1
    assert llm.collective_rpc(toggle_worker_profiler, args=(False,))[0]['enabled'] is False
    report['identical_greedy_tokens'] = True
    report['passed'] = True
    (ROOT / 'generation-profile.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
