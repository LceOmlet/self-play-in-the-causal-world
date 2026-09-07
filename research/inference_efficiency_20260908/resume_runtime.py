"""Preserve an early native checkpoint, then resume the official DAPO entry point.

This controls processes/configuration only. It contains no rollout, loss, gradient
or optimizer implementation. The original 500-update verification horizon is
retained; this is not the separately requested final 10,000-update training.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
ROOT = Path('/home/chen/runs/inference-efficiency-20260908')
ORIGINAL = Path('/home/chen/runs/training-submission-20260907/run-01')
PRESERVED = ROOT/'preserved/global_step_5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def original_controller():
    source = PROJECT/'research/training_submission_20260907/control.py'
    spec = importlib.util.spec_from_file_location('submission_controller', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_production():
    accepted = json.loads((ORIGINAL.parent/'preflight-acceptance.json').read_text())
    assert accepted['passed']
    for file, digest in accepted['project_files'].items():
        assert sha(PROJECT/file) == digest, file


def environment(run):
    env = original_controller().environment()
    env.update(CPT_WORLD_RUN_DIR=str(run), CPT_WORLD_VERL_AUDIT_DIR=str(run/'environment'),
               RAY_TMPDIR='/tmp/cpt-'+run.name)
    return env


def prepare(run, variant):
    check_production()
    from omegaconf import OmegaConf

    launch = json.loads((ORIGINAL/'launch.json').read_text())
    run.mkdir(parents=True, exist_ok=False)
    env = environment(run)
    replaced = ('trainer.validation_data_dir=', 'trainer.experiment_name=',
                'trainer.save_freq=', '++ray_kwargs.ray_init.runtime_env.env_vars.')
    overrides = [arg for arg in launch['command'][2:] if not arg.startswith(replaced)]
    overrides += ['trainer.resume_mode=resume_path', f'trainer.resume_from_path={PRESERVED}',
                  f'trainer.validation_data_dir={run}/validation', 'trainer.save_freq=1',
                  f'trainer.experiment_name=verified-resume-{variant}',
                  'actor_rollout_ref.rollout.disable_log_stats=false']
    if variant == 'compiled':
        overrides += ['actor_rollout_ref.rollout.enforce_eager=false',
                      'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4]']
    overrides += [f'++ray_kwargs.ray_init.runtime_env.env_vars.{key}={env[key]}' for key in (
        'CPT_WORLD_VERL_AUDIT_DIR', 'CPT_WORLD_EXPECTED_SOURCE', 'VERL_ROOT', 'VERL_RECIPE_ROOT',
        'PYTHONPATH', 'PATH')]
    controller = original_controller()
    inspection = [sys.executable, '-m', 'dapo.main_dapo',
                  f'hydra.searchpath=[file://{controller.VERL}/verl/trainer/config,'
                  f'file://{PROJECT}/configs/verl]', '+profiles@_global_=cpt_world_dapo',
                  *overrides, '--cfg', 'job', '--resolve']
    cpu = env | {'CUDA_VISIBLE_DEVICES': '', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run(inspection, env=cpu, cwd=PROJECT, capture_output=True, text=True)
    (run/'resolved.yaml').write_text(result.stdout)
    (run/'configuration.stderr').write_text(result.stderr)
    result.check_returncode()
    config = OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)
    base = OmegaConf.to_container(
        OmegaConf.load(ROOT/'configuration-v2/current.yaml'), resolve=True)
    spec = importlib.util.spec_from_file_location('configuration_diff',
             PROJECT/'research/inference_efficiency_20260908/validate_candidates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old, new = module.flatten(base), module.flatten(config)
    # The earlier inspection omitted explicit Ray environment overrides; use
    # their actual submitted values instead of treating them as new settings.
    for arg in launch['command'][2:]:
        if arg.startswith('++ray_kwargs.ray_init.runtime_env.env_vars.'):
            key, value = arg[2:].split('=', 1)
            old[key] = value
    diff = {key: {'before': old.get(key), 'after': new.get(key)}
            for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)}
    allowed = {'trainer.resume_mode', 'trainer.resume_from_path',
               'trainer.default_local_dir', 'trainer.rollout_data_dir',
               'trainer.validation_data_dir', 'trainer.experiment_name', 'trainer.save_freq',
               'actor_rollout_ref.rollout.disable_log_stats',
               'actor_rollout_ref.rollout.enforce_eager',
               'actor_rollout_ref.rollout.cudagraph_capture_sizes',
               'actor_rollout_ref.rollout.trace.experiment_name',
               'ray_kwargs.ray_init.runtime_env.env_vars.CPT_WORLD_VERL_AUDIT_DIR'}
    assert set(diff) <= allowed, diff
    assert config['trainer']['total_training_steps'] == 500
    assert config['trainer']['total_epochs'] == 20
    assert config['actor_rollout_ref']['model']['lora_adapter_path'] is None
    assert config['trainer']['resume_mode'] == 'resume_path'
    sys.path.insert(0, str(controller.VERL))
    from verl.utils.config import validate_config
    validate_config(OmegaConf.create(config), use_reference_policy=False, use_critic=False)
    command = ['bash', str(PROJECT/'scripts/run_official_dapo.sh'), *overrides]
    write(run/'prepared.json', {'time': time.time(), 'variant': variant, 'command': command,
          'resolved_sha256': sha(run/'resolved.yaml'), 'differences': diff,
          'checkpoint': str(PRESERVED), 'controller_sha256': sha(__file__),
          'wall_deadline': launch['start_time'] + launch['wall_seconds_limit'],
          'scope': 'Resume the existing verification, using a new run directory. '
          'Save every update for recovery verification; do not reset optimizer/data.'})
    print(json.dumps({'prepared': str(run), 'differences': diff}), flush=True)


def identified_original():
    import psutil
    saved = json.loads((ORIGINAL/'process.json').read_text())
    process = psutil.Process(saved['pid'])
    assert abs(process.create_time()-saved['create_time']) < .01
    assert 'dapo.main_dapo' in process.cmdline()
    return process


def terminate_tree(process):
    import psutil
    assert os.getpgid(process.pid) == process.pid, 'Expected a separately owned process group'
    children = process.children(recursive=True)
    identities = [(p.pid, p.create_time()) for p in [process, *children]]
    os.killpg(process.pid, signal.SIGTERM)
    for pid, created in identities:
        try:
            child = psutil.Process(pid)
            if abs(child.create_time()-created) < .01:
                child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([process, *children], timeout=15)
    for child in alive:
        try:
            if (child.pid, child.create_time()) in identities:
                child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)
    return [{'pid': pid, 'create_time': created} for pid, created in identities]


def preserve_and_pause():
    check_production()
    process = identified_original()
    tracker = ORIGINAL/'checkpoints/latest_checkpointed_iteration.txt'
    assert tracker.is_file() and int(tracker.read_text().strip()) == 5, 'Step 5 not published'
    source = ORIGINAL/'checkpoints/global_step_5'
    assert not PRESERVED.exists()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from torch.distributed.tensor import DTensor
    from torchdata.stateful_dataloader import StatefulDataLoader

    controller = original_controller()
    sys.path.insert(0, str(controller.VERL))
    from omegaconf import OmegaConf
    from verl.trainer.ppo.utils import create_rl_sampler

    def local(value):
        return value.to_local() if isinstance(value, DTensor) else value

    model = torch.load(source/'actor/model_world_size_1_rank_0.pt',
                       map_location='cpu', weights_only=False)
    optimizer = torch.load(source/'actor/optim_world_size_1_rank_0.pt',
                           map_location='cpu', weights_only=False)
    extra = torch.load(source/'actor/extra_state_world_size_1_rank_0.pt',
                       map_location='cpu', weights_only=False)
    data = torch.load(source/'data.pt', map_location='cpu', weights_only=False)
    assert len(model) == 496 and all('lora_' in key for key in model)
    assert all(local(t).dtype == torch.bfloat16 and torch.isfinite(local(t)).all()
               for t in model.values())
    states = [state for state in optimizer['state'].values() if state]
    assert len(states) == 496
    assert all(float(local(state['step'])) == 5 for state in states)
    assert all(torch.isfinite(local(state[key])).all()
               for state in states for key in ('exp_avg', 'exp_avg_sq'))
    assert extra['lr_scheduler']['last_epoch'] == 5
    generated = data['_num_yielded']
    assert 5 <= generated <= 50 < 500 and not data['_iterator_finished']
    original = OmegaConf.load(ROOT/'configuration-v2/current.yaml')
    dataset = list(range(500))

    def loader():
        return StatefulDataLoader(dataset, batch_size=1, num_workers=0, drop_last=True,
                                  sampler=create_rl_sampler(original.data, dataset))

    continuous = iter(loader())
    for _ in range(generated):
        next(continuous)
    expected = [int(next(continuous).item()) for _ in range(8)]
    resumed = loader()
    resumed.load_state_dict(data)
    iterator = iter(resumed)
    actual = [int(next(iterator).item()) for _ in range(8)]
    assert actual == expected
    assert not torch.cuda.is_initialized()
    files = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source, PRESERVED)
    for name, digest in files.items():
        assert sha(PRESERVED/name) == digest and sha(source/name) == digest
    write(ROOT/'preserved/acceptance.json', {'time': time.time(), 'original_run': str(ORIGINAL),
          'checkpoint': str(PRESERVED), 'updates': 5, 'generated_batches': generated,
          'model_tensors_finite': 496, 'optimizer_states_at_step_5': 496,
          'lr_scheduler_last_epoch': 5, 'expected_next_rows': expected,
          'restored_next_rows': actual, 'files': files,
          'scope': 'Published native checkpoint and CPU data-stream audit, '
          'not bitwise observation of a GPU restore.'})
    # Revalidate PID identity after the CPU work; never stop by name/port alone.
    process = identified_original()
    identities = terminate_tree(process)
    write(ROOT/'training-paused-for-benchmark.json', {'time': time.time(),
          'reason': 'Native update-5 checkpoint preserved for official resume; '
          'release the sole GPU for equal-work inference verification.',
          'processes': identities,
          'checkpoint_acceptance_sha256': sha(ROOT/'preserved/acceptance.json')})
    print(json.dumps({'paused': True, 'checkpoint': str(PRESERVED),
                      'generated_batches': generated, 'next_rows': actual}), flush=True)


def launch(run):
    check_production()
    plan = json.loads((run/'prepared.json').read_text())
    accepted = json.loads((ROOT/'preserved/acceptance.json').read_text())
    assert accepted['updates'] == 5 and plan['checkpoint'] == str(PRESERVED)
    assert plan['controller_sha256'] == sha(__file__)
    for name, digest in accepted['files'].items():
        assert sha(PRESERVED/name) == digest
    assert time.time() < plan['wall_deadline'] and not (run/'supervisor.json').exists()
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    with (run/'train.log').open('x') as output:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                              'supervise', '--run', str(run)], env=environment(run),
                             stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    write(run/'supervisor.json', {'pid': p.pid, 'time': time.time()})
    print(json.dumps({'launched': str(run), 'supervisor': p.pid}))


def supervise(run):
    import psutil
    plan = json.loads((run/'prepared.json').read_text())
    process = subprocess.Popen(plan['command'], env=environment(run), start_new_session=True)
    write(run/'process.json', {'pid': process.pid,
                              'create_time': psutil.Process(process.pid).create_time()})
    expired = False
    try:
        code = process.wait(timeout=max(1, plan['wall_deadline']-time.time()))
    except subprocess.TimeoutExpired:
        expired = True
        terminate_tree(psutil.Process(process.pid))
        code = process.wait(timeout=5)
    write(run/'exit.json', {'time': time.time(), 'exit_code': code, 'wall_limit_reached': expired})
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preserve_and_pause', 'launch', 'supervise'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--variant', choices=['eager', 'compiled'])
    args = parser.parse_args()
    if args.action == 'preserve_and_pause':
        preserve_and_pause()
    elif args.action == 'prepare':
        assert args.run and args.variant
        prepare(args.run, args.variant)
    else:
        assert args.run
        sys.exit(globals()[args.action](args.run) or 0)
