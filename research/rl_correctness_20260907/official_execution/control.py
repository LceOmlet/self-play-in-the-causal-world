"""Prepare/launch one official DAPO update in a fresh audit-only directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
RECIPE = Path('/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1')
PYTHON = '/home/chen/.venvs/dolens-dapo-official/bin/python'
TRAIN = '/home/chen/runs/official-dapo-integration-20260906/data-v1/integration-train.parquet'
VAL = '/home/chen/runs/environment-validation-20260907/heldout-250.parquet'
MODEL = '/home/chen/models/Qwen/Qwen3.5-9B'


def environment(run):
    env = os.environ.copy()
    env.update(VERL_ROOT=str(VERL), VERL_RECIPE_ROOT=str(RECIPE),
               CPT_WORLD_TRAIN_DATA=TRAIN, CPT_WORLD_VAL_DATA=VAL,
               CPT_WORLD_MODEL=MODEL, CPT_WORLD_PROJECT=str(PROJECT),
               CPT_WORLD_EXPECTED_SOURCE=str(PROJECT/'src/cpt_world'),
               CPT_WORLD_RUN_DIR=str(run), CPT_WORLD_VERL_AUDIT_DIR=str(run/'environment'),
               CPT_DAPO_OBSERVATION_DIR=str(run/'observation'),
               PYTHONPATH=':'.join(map(str, (PROJECT/'src', PROJECT, RECIPE, VERL, HERE/'observer'))),
               TOKENIZERS_PARALLELISM='false', VLLM_ALLOW_RUNTIME_LORA_UPDATING='true')
    assert not env.get('CPT_WORLD_INITIAL_ADAPTER')
    assert not env.get('RESUME_FROM_CHECKPOINT')
    return env


def preflight():
    run = HERE/'preflight-mask-v1-02'
    run.mkdir(exist_ok=False)
    env = environment(run)
    commands = [
        [PYTHON, str(PROJECT/'scripts/verify_official_dapo.py'), '--output', str(run/'source.json')],
        [PYTHON, str(HERE.parent/'verify_official_token_tis.py')],
        [PYTHON, '-m', 'pytest', '-q', str(PROJECT/'tests/test_verl_environment.py'),
         str(PROJECT/'tests/test_official_dapo_action_mask.py'),
         str(PROJECT/'tests/test_official_dapo_provenance.py'),
         str(VERL/'tests/experimental/agent_loop/test_tool_termination_on_cpu.py')],
    ]
    with (run/'checks.log').open('w') as log:
        for command in commands:
            result = subprocess.run(command, env=env, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Preflight failed: {command}, exit {result.returncode}')
    errors = list((run/'observation').glob('observer-errors-*'))
    if errors:
        raise RuntimeError(f'Observation errors: {errors}')
    events = [json.loads(line) for file in (run/'observation').glob('*/events.jsonl')
              for line in file.read_text().splitlines()]
    for required in ('grpo', 'tis', 'policy_loss', 'tool_transition'):
        assert any(e['event'] == required for e in events), required
    result = {'passed': True, 'time': time.time(), 'observed_events': len(events),
              'kinds': sorted({e['event'] for e in events}),
              'observer_sha256': hashlib.sha256((HERE/'observer/execution_observer.py').read_bytes()).hexdigest()}
    (run/'acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def launch():
    data_acceptance = json.loads((HERE/'data-acceptance.json').read_text())
    assert data_acceptance['passed']
    assert data_acceptance['data_sha256'] == hashlib.sha256(Path(TRAIN).read_bytes()).hexdigest()
    accepted = json.loads((HERE/'preflight-mask-v1-02/acceptance.json').read_text())
    assert accepted['passed']
    assert accepted['observer_sha256'] == hashlib.sha256((HERE/'observer/execution_observer.py').read_bytes()).hexdigest()
    run = HERE/'run-02-mask-v1-audit-only'
    run.mkdir(exist_ok=False)
    env = environment(run)
    env_overrides = [f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}' for key in
                     ('CPT_DAPO_OBSERVATION_DIR', 'CPT_WORLD_VERL_AUDIT_DIR', 'CPT_WORLD_EXPECTED_SOURCE',
                      'VERL_ROOT', 'VERL_RECIPE_ROOT', 'PYTHONPATH')]
    command = ['bash', str(PROJECT/'scripts/run_official_dapo.sh'), 'data.gen_batch_size=1',
               'trainer.total_training_steps=1', 'trainer.resume_mode=disable',
               'actor_rollout_ref.model.lora_adapter_path=null',
               'trainer.experiment_name=correctness-20260907-one-update', *env_overrides]
    metadata = {'purpose': 'Isolated actual-path correctness audit; not capability evaluation or production training',
                'audit_only_checkpoint': True, 'base_model': MODEL, 'initial_adapter': None,
                'command': command, 'start_time': time.time(),
                'data': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (TRAIN, VAL)},
                'observer_sha256': accepted['observer_sha256']}
    (run/'launch.json').write_text(json.dumps(metadata, indent=2))
    log = (run/'train.log').open('w')
    supervisor = subprocess.Popen([PYTHON, str(HERE/'control.py'), 'supervise'], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (run/'supervisor.json').write_text(json.dumps({'pid': supervisor.pid, 'time': time.time()}))
    print(json.dumps({'run': str(run), 'supervisor_pid': supervisor.pid}))


def supervise():
    run = HERE/'run-02-mask-v1-audit-only'
    metadata = json.loads((run/'launch.json').read_text())
    process = subprocess.Popen(metadata['command'], env=environment(run))
    (run/'process.json').write_text(json.dumps({'pid': process.pid, 'time': time.time()}))
    code = process.wait()
    (run/'exit.json').write_text(json.dumps({'exit_code': code, 'time': time.time()}, indent=2))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'launch', 'supervise'])
    action = parser.parse_args().action
    if action == 'preflight':
        preflight()
    elif action == 'launch':
        launch()
    else:
        sys.exit(supervise())
