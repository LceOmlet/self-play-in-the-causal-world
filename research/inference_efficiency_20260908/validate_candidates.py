"""Resolve candidates through official Hydra and configuration validation on CPU.

This neither generates model output nor updates parameters. It verifies that
only the declared inference settings differ from the actual submitted command.
It is not a GPU compatibility, speed, probability, or learning acceptance test.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
RECIPE = Path('/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1')
RUN = Path('/home/chen/runs/training-submission-20260907/run-01')
OUT = Path('/home/chen/runs/inference-efficiency-20260908/configuration')


def flatten(value, prefix=''):
    if isinstance(value, dict):
        return {key: item for name, child in value.items()
                for key, item in flatten(child, prefix + ('.' if prefix else '') + name).items()}
    return {prefix: value}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    script = PROJECT / 'research/training_submission_20260907/control.py'
    spec = importlib.util.spec_from_file_location('submitted_controller', script)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    env = controller.environment()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['CPT_WORLD_RUN_DIR'] = str(OUT / 'unused-run-path')
    env['CPT_WORLD_VERL_AUDIT_DIR'] = str(OUT / 'unused-events')
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    # Existing submission overrides set the actual training horizon and sampling.
    # --cfg job exits at Hydra composition before the trainer is constructed.
    launch = json.loads((RUN / 'launch.json').read_text())
    submitted_overrides = [arg for arg in launch['command'][2:]
                           if not arg.startswith('++ray_kwargs.ray_init.runtime_env.env_vars.')]
    command = [sys.executable, '-m', 'dapo.main_dapo',
               f'hydra.searchpath=[file://{VERL}/verl/trainer/config,file://{PROJECT}/configs/verl]',
               '+profiles@_global_=cpt_world_dapo', *submitted_overrides, '--cfg', 'job', '--resolve']
    variants_path = Path(__file__).with_name('variants.json')
    variants = json.loads(variants_path.read_text())['variants']
    subprocess.run([sys.executable, str(PROJECT/'scripts/verify_official_dapo.py'),
                    '--output', str(OUT/'source-verification.json')],
                   env=env, check=True, stdout=(OUT/'source-verification.log').open('w'))
    from omegaconf import OmegaConf
    configs = {}
    commands = {}
    for name, overrides in variants.items():
        invoked = command + overrides
        completed = subprocess.run(invoked, env=env, cwd=VERL, capture_output=True, text=True)
        (OUT/f'{name}.yaml').write_text(completed.stdout)
        (OUT/f'{name}.stderr').write_text(completed.stderr)
        completed.check_returncode()
        configs[name] = OmegaConf.to_container(OmegaConf.create(completed.stdout), resolve=True)
        commands[name] = invoked

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(VERL), str(RECIPE)]
    from verl.utils.config import validate_config
    for name, config in configs.items():
        validate_config(OmegaConf.create(config), use_reference_policy=False, use_critic=False)
        assert config['actor_rollout_ref']['model']['path'] == '/home/chen/models/Qwen/Qwen3.5-9B'
        assert config['actor_rollout_ref']['model']['lora_rank'] == 8
        assert config['actor_rollout_ref']['rollout']['max_num_seqs'] == 4

    baseline = flatten(configs['current'])
    allowed = {
        'actor_rollout_ref.rollout.enforce_eager',
        'actor_rollout_ref.rollout.cudagraph_capture_sizes',
        'actor_rollout_ref.model.lora.merge',
    }
    differences = {}
    for name, config in configs.items():
        flat = flatten(config)
        changed = {key: {'before': baseline.get(key), 'after': flat.get(key)}
                   for key in sorted(set(baseline) | set(flat)) if baseline.get(key) != flat.get(key)}
        assert set(changed) <= allowed, (name, changed)
        assert len(changed) == {'current': 0, 'compiled_adapter': 2, 'compiled_merged': 3}[name]
        differences[name] = changed

    model_path = Path('/home/chen/models/Qwen/Qwen3.5-9B/config.json')
    model = json.loads(model_path.read_text())
    assert model['architectures'] == ['Qwen3_5ForConditionalGeneration']
    assert model['text_config']['num_hidden_layers'] == 32
    report = {'passed_configuration_only': True, 'time': time.time(), 'commands': commands,
              'differences': differences, 'source_head': subprocess.check_output(
                  ['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
              'model_config_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'variants_sha256': hashlib.sha256(variants_path.read_bytes()).hexdigest(),
              'model_architecture': model['architectures'],
              'remaining': ['Actual 9B GPU execution and memory', 'Warm equal-work inference timing',
                            'Nonzero LoRA synchronization and base restoration',
                            'Sampled-token probability semantics and official TIS',
                            'Official update correctness and actual learning evidence']}
    (OUT/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed_configuration_only': True, 'differences': differences}))


if __name__ == '__main__':
    main()
