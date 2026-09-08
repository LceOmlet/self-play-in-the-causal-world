"""Submit the verified official DAPO recipe; no training algorithm lives here."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Archived fixed-500 submission entry point; all command-line actions are disabled. '
            'See research/long_run_10000_20260908/control.py for continuous '
            'restoration helpers; the training entry point is scripts/run_official_dapo.sh. '
            'Importable environment helpers remain available to the accepted continuation.'
        )
    )
    parser.parse_known_args()
    parser.error(
        'Fixed-500 training submission is retired. See '
        'research/long_run_10000_20260908/control.py for continuous restoration. '
        'Training uses scripts/run_official_dapo.sh and the accepted prepared stream. '
        'This archived controller cannot prepare, launch, supervise, or stop a run.'
    )

PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
RECIPE = Path('/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1')
PYTHON = '/home/chen/.venvs/dolens-dapo-official/bin/python'
ROOT = Path('/home/chen/runs/training-submission-20260907')
RUN = ROOT/'run-01'
WALL_SECONDS = 86400


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2)+'\n', encoding='utf-8')


def environment():
    env = os.environ.copy()
    for key in ('CPT_WORLD_INITIAL_ADAPTER', 'RESUME_FROM_CHECKPOINT'):
        assert not env.get(key), f'Fresh original base required: {key}'
    env.pop('CPT_DAPO_OBSERVATION_DIR', None)
    env.update(
        VERL_ROOT=str(VERL), VERL_RECIPE_ROOT=str(RECIPE), DAPO_PYTHON=PYTHON,
        CPT_WORLD_TRAIN_DATA=str(ROOT/'data/train.parquet'),
        CPT_WORLD_VAL_DATA=str(ROOT/'data/validation.parquet'),
        CPT_WORLD_PROJECT=str(PROJECT), CPT_WORLD_MODEL='/home/chen/models/Qwen/Qwen3.5-9B',
        CPT_WORLD_RUN_DIR=str(RUN), CPT_WORLD_VERL_AUDIT_DIR=str(RUN/'environment'),
        CPT_WORLD_EXPECTED_SOURCE=str(PROJECT/'src/cpt_world'),
        PYTHONPATH=':'.join(map(str, (PROJECT/'src', PROJECT, RECIPE, VERL))),
        PATH=str(Path(PYTHON).parent)+':'+env.get('PATH', ''),
        TOKENIZERS_PARALLELISM='false', VLLM_ALLOW_RUNTIME_LORA_UPDATING='true',
        RAY_TMPDIR='/tmp/cpt-train-0907-01',
    )
    return env


def stop_pilot():
    import psutil
    pilot = Path('/home/chen/runs/base-signal-diagnostic-20260907/run-01')
    saved = json.loads((pilot/'process.json').read_text())
    process = psutil.Process(saved['pid'])
    assert abs(process.create_time()-saved['create_time']) < .01
    assert 'dapo.main_dapo' in process.cmdline()
    children = process.children(recursive=True)
    write(pilot/'user-directed-stop.json', {
        'time': time.time(), 'reason': 'User requested a few trajectories then actual training; stop expanded pilot.',
        'process': saved, 'children': [{'pid': p.pid, 'create_time': p.create_time()} for p in children],
        'completed_evaluation': False,
    })
    os.killpg(process.pid, signal.SIGTERM)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([process, *children], timeout=15)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    print('Stopped only the identified pilot process tree; partial evidence preserved.')


def preflight():
    accepted = json.loads((ROOT/'data/acceptance.json').read_text())
    assert accepted['passed']
    for name, digest in accepted['sha256'].items():
        assert sha(ROOT/'data'/name) == digest
    RUN.mkdir(exist_ok=False)
    env = environment()
    env['CPT_WORLD_VERL_AUDIT_DIR'] = str(ROOT/'preflight-environment')
    commands = [
        [PYTHON, str(PROJECT/'scripts/verify_official_dapo.py'), '--output', str(ROOT/'preflight-source.json')],
        [PYTHON, '-m', 'pytest', '-q', str(PROJECT/'tests/test_verl_environment.py'),
         str(PROJECT/'tests/test_official_dapo_action_mask.py'),
         str(PROJECT/'tests/test_official_dapo_provenance.py'),
         str(PROJECT/'tests/test_rewards.py'), str(PROJECT/'tests/test_task_scoring.py'),
         str(PROJECT/'tests/test_trl_environment.py'),
         str(VERL/'tests/experimental/agent_loop/test_tool_termination_on_cpu.py')],
    ]
    with (ROOT/'preflight.log').open('x') as output:
        for command in commands:
            subprocess.run(command, env=env, cwd=PROJECT, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
    files = [p for folder in ('src', 'configs/verl', 'scripts') for p in (PROJECT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    write(ROOT/'preflight-acceptance.json', {
        'passed': True, 'time': time.time(), 'commands': commands,
        'project_files': {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)},
        'data_acceptance_sha256': sha(ROOT/'data/acceptance.json'),
    })
    print('Training preflight passed', flush=True)


def launch():
    accepted = json.loads((ROOT/'preflight-acceptance.json').read_text())
    assert accepted['passed']
    assert sha(ROOT/'data/acceptance.json') == accepted['data_acceptance_sha256']
    for file, digest in accepted['project_files'].items():
        assert sha(PROJECT/file) == digest, file
    assert not (RUN/'launch.json').exists(), 'Never relaunch an existing invocation'
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    env = environment()
    overrides = [f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}' for key in (
        'CPT_WORLD_VERL_AUDIT_DIR', 'CPT_WORLD_EXPECTED_SOURCE', 'VERL_ROOT', 'VERL_RECIPE_ROOT',
        'PYTHONPATH', 'PATH')]
    command = ['bash', str(PROJECT/'scripts/run_official_dapo.sh'),
               'trainer.total_training_steps=500', 'trainer.total_epochs=20',
               'trainer.val_before_train=false', 'trainer.val_only=false',
               'trainer.save_freq=5', 'trainer.test_freq=25',
               f'trainer.validation_data_dir={RUN}/validation',
               'trainer.experiment_name=verified-learning-20260907', 'data.shuffle=true',
               'actor_rollout_ref.rollout.val_kwargs.do_sample=false',
               'actor_rollout_ref.rollout.val_kwargs.temperature=0.0', *overrides]
    write(RUN/'launch.json', {
        'command': command, 'start_time': time.time(), 'wall_seconds_limit': WALL_SECONDS,
        'source_head': subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip(),
        'controller_sha256': sha(__file__), 'original_base': env['CPT_WORLD_MODEL'], 'initial_adapter': None,
        'training': {'retained_prompts_per_update': 1, 'trajectories_per_prompt': 4, 'steps_limit': 500},
        'authorization': 'User requested actual training once task value, computability and RL correctness were checked.',
        'interpretation': 'Training horizon and 24h resource window are not evidence that learning must succeed.',
        'initial_validation': 'Expanded pilot stopped per user direction; no claim of full pre/post evaluation.',
    })
    with (RUN/'train.log').open('x') as output:
        p = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), 'supervise'], env=env,
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    write(RUN/'supervisor.json', {'pid': p.pid, 'time': time.time()})
    print(json.dumps({'run': str(RUN), 'supervisor_pid': p.pid}))


def supervise():
    import psutil
    metadata = json.loads((RUN/'launch.json').read_text())
    p = subprocess.Popen(metadata['command'], env=environment(), start_new_session=True)
    write(RUN/'process.json', {'pid': p.pid, 'create_time': psutil.Process(p.pid).create_time()})
    expired = False
    try:
        code = p.wait(timeout=max(1, metadata['start_time']+WALL_SECONDS-time.time()))
    except subprocess.TimeoutExpired:
        expired = True
        children = psutil.Process(p.pid).children(recursive=True)
        os.killpg(p.pid, signal.SIGTERM)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=15)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            code = p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            code = p.wait()
    write(RUN/'exit.json', {'exit_code': code, 'wall_limit_reached': expired, 'time': time.time()})
    return code
