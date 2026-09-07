"""Save read-only source/config evidence for the historical speed comparison."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/chen/runs/training-submission-20260907')
OLD = Path('/home/chen/.venvs/dolens-rl/lib/python3.12/site-packages')
NEW = Path('/home/chen/.venvs/dolens-dapo-official/lib/python3.12/site-packages')
VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
ARGS = Path('/home/chen/runs/dolens-grpo-rewardv10-cfisolated-85cb577-20260902-195713-resume250-10k/checkpoints.disqualified-20260906/checkpoint-850/training_args.bin')

sources = {}
for path, needles in (
    (OLD/'trl/generation/vllm_generation.py', ('quantization = None', 'model.merge_adapter()', 'self.llm = LLM(')),
    (OLD/'trl/trainer/grpo_trainer.py', ('count only model-generated tokens',)),
    (OLD/'vllm/config/model.py', ('enforce_eager: bool',)),
    (NEW/'vllm/config/vllm.py', ('disabling torch.compile and CUDAGraphs',)),
    (VERL/'verl/workers/rollout/vllm_rollout/vllm_async_server.py', ('"enable_lora": True',)),
    (VERL/'verl/trainer/ppo/metric_utils.py', ('response_mask = batch.batch["attention_mask"]',)),
):
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    excerpts = []
    for needle in needles:
        matches = [i for i, line in enumerate(lines) if needle in line]
        assert matches, (path, needle)
        for index in matches:
            excerpts.append({'needle': needle, 'line': index+1,
                             'source': '\n'.join(lines[max(0,index-3):index+16])})
    sources[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'excerpts': excerpts}

fields = ['num_generations', 'generation_batch_size', 'max_completion_length',
          'vllm_max_model_length', 'vllm_enable_prefix_caching', 'vllm_speculative_config',
          'vllm_sleep_level', 'bf16', 'model_init_kwargs', 'vllm_mode',
          'vllm_gpu_memory_utilization', 'gradient_accumulation_steps', 'num_iterations']
# This is trusted, locally produced TrainingArguments metadata, not model weights.
code = ('import torch,json; a=torch.load(' + repr(str(ARGS))
        + ', map_location="cpu", weights_only=False); print(json.dumps({k:getattr(a,k,None) '
        + 'for k in ' + repr(fields) + '},default=str))')
loaded = subprocess.run(['/home/chen/.venvs/dolens-rl/bin/python', '-c', code],
                        check=True, capture_output=True, text=True)
metadata = json.loads(loaded.stdout)
assert metadata['num_generations'] == metadata['generation_batch_size'] == 4
report = {'inspection_time': time.time(), 'sources': sources,
          'historical_training_args': metadata,
          'historical_training_args_sha256': hashlib.sha256(ARGS.read_bytes()).hexdigest(),
          'scope': 'Read-only inspection. Historical arguments are archived metadata; installed dependency sources are as inspected now, not proof of historical graph hit rates. No checkpoint weights loaded.'}
(ROOT/'speed-source-evidence.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'historical_training_args': metadata, 'source_files': len(sources)}))
