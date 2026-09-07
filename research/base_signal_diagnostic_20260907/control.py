"""Measure original-base task behavior with the official DAPO validation path.

No policy update, new trainer, or reward modification is implemented here.
The first 25 existing diagnostic worlds, four trajectories each, form a pilot
for observable failure modes and evaluation design, not a learning verdict.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
VERL = Path('/home/chen/vendor/dapo-official-20260906/verl-tool-termination-v1')
RECIPE = Path('/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1')
PYTHON = '/home/chen/.venvs/dolens-dapo-official/bin/python'
MODEL = '/home/chen/models/Qwen/Qwen3.5-9B'
ROOT = Path('/home/chen/runs/base-signal-diagnostic-20260907')
RUN = ROOT/'run-01'
HELDOUT = Path('/home/chen/runs/environment-validation-20260907/heldout-250.parquet')
TRAIN = Path('/home/chen/runs/official-dapo-integration-20260906/data-v1/integration-train.parquet')
WALL_SECONDS = 5400


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')


def environment():
    env = os.environ.copy()
    for key in ('CPT_WORLD_INITIAL_ADAPTER', 'RESUME_FROM_CHECKPOINT'):
        assert not env.get(key), f'Original base required: {key}'
    env.pop('CPT_DAPO_OBSERVATION_DIR', None)
    env.update(
        VERL_ROOT=str(VERL), VERL_RECIPE_ROOT=str(RECIPE), DAPO_PYTHON=PYTHON,
        CPT_WORLD_TRAIN_DATA=str(TRAIN), CPT_WORLD_VAL_DATA=str(RUN/'validation.parquet'),
        CPT_WORLD_MODEL=MODEL, CPT_WORLD_PROJECT=str(PROJECT),
        CPT_WORLD_EXPECTED_SOURCE=str(PROJECT/'src/cpt_world'),
        CPT_WORLD_RUN_DIR=str(RUN), CPT_WORLD_VERL_AUDIT_DIR=str(RUN/'environment'),
        PYTHONPATH=':'.join(map(str, (PROJECT/'src', PROJECT, RECIPE, VERL))),
        PATH=str(Path(PYTHON).parent)+':'+env.get('PATH', ''),
        TOKENIZERS_PARALLELISM='false', VLLM_ALLOW_RUNTIME_LORA_UPDATING='true',
        RAY_TMPDIR='/tmp/cpt-base-0907-01',
    )
    return env


def prepare():
    import pickle
    from collections import Counter
    from fractions import Fraction

    import pyarrow.parquet as pq

    sys.path.insert(0, str(PROJECT/'src'))
    from cpt_world import CPTWorldEnvironment, compute_query_truth
    from cpt_world.query_truth import nearest_backdoor_adjustment_set
    from cpt_world.world_space import _best_intervention_observational_relation

    def clean(value):
        if isinstance(value, Fraction):
            return float(value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    RUN.mkdir(parents=True, exist_ok=False)
    (RUN/'truth').mkdir()
    table = pq.read_table(HELDOUT).slice(0, 25)
    pq.write_table(table, RUN/'validation.parquet')
    records = []
    for i, record in enumerate(table.to_pylist()):
        row = json.loads(record['extra_info']['tools_kwargs']['act']['create_kwargs']['row_json'])
        env = CPTWorldEnvironment()
        env.reset(**row)
        ep = env.episode
        world, seed = ep.world, ep.seed
        family = row['query_type']
        truth = compute_query_truth(world, seed, counterfactual_endpoint_time_limit_seconds=5)
        cached = json.loads(row['terminal_truth_json']) if row['terminal_truth_json'] else None
        labels = seed['visible_schema']['variable_labels']
        if family == 'individual_counterfactual_probability':
            assert cached is not None
            tol = float(cached['endpoint_error'])+float(truth['endpoint_error'])+1e-8
            assert abs(float(cached['lower'])-float(truth['lower'])) <= tol
            assert abs(float(cached['upper'])-float(truth['upper'])) <= tol
            answer = {'type': 'answer', 'lower': max(0, min(1, float(cached['lower']))),
                      'upper': max(0, min(1, float(cached['upper'])))}
        elif family == 'ate':
            answer = {'type': 'answer', 'effect': {f'state_{j}': float(x)
                      for j, x in enumerate(truth['effect'])}}
        elif family == 'best_intervention':
            answer = {'type': 'answer', 'value': f'state_{truth["value"]}'}
        elif family == 'backadj_minimal_sets':
            inverse = {label: node for node, label in labels.items()}
            x, y = (inverse[seed['query'][k]] for k in ('treatment', 'outcome'))
            _, nearest = nearest_backdoor_adjustment_set(world, x, y, frozenset())
            answer = {'type': 'answer', 'adjustment_set': [labels[n] for n in nearest]}
        elif family == 'mediator_set':
            answer = {'type': 'answer', 'mediators': [labels[n] for n in truth['mediators']],
                      'order': [[labels[a], labels[b]] for a, b in truth['order']]}
        else:
            raise AssertionError(family)
        (RUN/'truth'/f'{i:03}.pkl').write_bytes(pickle.dumps((row, world, seed, truth)))
        env.act(answer)
        assert ep.completed and abs(env.get_reward()-1) < 1e-10, (i, ep.terminal_score)
        entry = {'index': i, 'query_type': family, 'sample_index': row['sample_index'],
                 'seed_id': seed['seed_id'], 'tape_key': row['tape_key'],
                 'truth': clean(truth), 'cached': cached, 'oracle_score': clean(ep.terminal_score)}
        if family == 'best_intervention':
            inverse = {label: world.variables.index(node) for node, label in labels.items()}
            query = seed['query']
            entry['discordant'], entry['observational_causal_gap'] = _best_intervention_observational_relation(world, {
                'decision_target': inverse[query['decision_target']],
                'outcome': inverse[query['outcome']],
                'outcome_state': int(query['outcome_state'].removeprefix('state_')),
                'objective': query['objective'],
            })
        records.append(entry)
        print(json.dumps({'index': i, 'query_type': family, 'oracle_reward': env.get_reward()}), flush=True)
    assert set(Counter(r['query_type'] for r in records).values()) == {5}
    assert sum(bool(r.get('discordant')) for r in records) == 4
    files = [p for folder in ('src', 'configs/verl', 'scripts') for p in (PROJECT/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    write(RUN/'data-acceptance.json', {
        'passed': True, 'records': records, 'validation_sha256': sha(RUN/'validation.parquet'),
        'source_dataset_sha256': sha(HELDOUT),
        'project_files': {str(p.relative_to(PROJECT)): sha(p) for p in sorted(files)},
        'selection': 'First 25 frozen rows in existing order, chosen before model results; 5 per family, decision layers 1:4.',
        'purpose': 'Original-base signal and failure diagnostic; not overall accuracy or learnability certification.',
    })


def preflight():
    assert json.loads((RUN/'data-acceptance.json').read_text())['passed']
    commands = [
        [PYTHON, str(PROJECT/'scripts/verify_official_dapo.py'), '--output', str(RUN/'preflight-source.json')],
        [PYTHON, '-m', 'pytest', '-q', str(PROJECT/'tests/test_verl_environment.py'),
         str(PROJECT/'tests/test_official_dapo_action_mask.py'),
         str(PROJECT/'tests/test_official_dapo_provenance.py'),
         str(VERL/'tests/experimental/agent_loop/test_tool_termination_on_cpu.py')],
    ]
    check_env = environment()
    check_env['CPT_WORLD_VERL_AUDIT_DIR'] = str(RUN/'preflight-environment')
    with (RUN/'preflight.log').open('w') as log:
        for command in commands:
            subprocess.run(command, env=check_env, cwd=PROJECT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    write(RUN/'preflight-acceptance.json', {'passed': True, 'time': time.time(), 'commands': commands})
    print('Preflight passed', flush=True)


def launch():
    assert json.loads((RUN/'preflight-acceptance.json').read_text())['passed']
    accepted = json.loads((RUN/'data-acceptance.json').read_text())
    assert sha(RUN/'validation.parquet') == accepted['validation_sha256']
    for file, value in accepted['project_files'].items():
        assert sha(PROJECT/file) == value, file
    assert not (RUN/'launch.json').exists(), 'Never relaunch an existing run'
    env = environment()
    overrides = [f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}' for key in (
        'CPT_WORLD_VERL_AUDIT_DIR', 'CPT_WORLD_EXPECTED_SOURCE', 'VERL_ROOT', 'VERL_RECIPE_ROOT',
        'PYTHONPATH', 'PATH')]
    command = ['bash', str(PROJECT/'scripts/run_official_dapo.sh'),
               'trainer.val_before_train=true', 'trainer.val_only=true',
               f'trainer.validation_data_dir={RUN}/validation',
               'trainer.experiment_name=base-signal-diagnostic-20260907',
               'data.val_batch_size=1', 'actor_rollout_ref.rollout.val_kwargs.n=4', *overrides]
    write(RUN/'launch.json', {
        'command': command, 'start_time': time.time(), 'wall_seconds_limit': WALL_SECONDS,
        'base_model': MODEL, 'initial_adapter': None, 'whole_process_observer': False,
        'policy_updates': 0, 'validation_worlds': 25, 'trajectories_per_world': 4,
        'source_head': subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip(),
        'controller_sha256': sha(__file__),
    })
    with (RUN/'train.log').open('x') as log:
        process = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), 'supervise'], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(RUN/'supervisor.json', {'pid': process.pid, 'time': time.time()})
    print(json.dumps({'run': str(RUN), 'supervisor_pid': process.pid}))


def supervise():
    import psutil
    metadata = json.loads((RUN/'launch.json').read_text())
    process = subprocess.Popen(metadata['command'], env=environment(), start_new_session=True)
    write(RUN/'process.json', {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time()})
    expired = False
    try:
        code = process.wait(timeout=max(1, metadata['start_time']+WALL_SECONDS-time.time()))
    except subprocess.TimeoutExpired:
        expired = True
        owned = psutil.Process(process.pid).children(recursive=True)
        os.killpg(process.pid, signal.SIGTERM)
        for child in owned:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=15)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
    write(RUN/'exit.json', {'exit_code': code, 'time': time.time(), 'wall_limit_reached': expired})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'preflight', 'launch', 'supervise'])
    action = parser.parse_args().action
    sys.exit(globals()[action]() or 0)
