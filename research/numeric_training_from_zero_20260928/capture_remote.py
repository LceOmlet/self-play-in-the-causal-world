"""Read-only capture of the fresh numeric-tools training branch (no model load)."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    RUN = ROOT/'run-01'
    UNIT = args.unit
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'checks') and out != ROOT/'checks'
    out.mkdir(parents=True, exist_ok=False)
    status = json.loads((RUN/'status.json').read_text())
    checked = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for name in ['status.json','process.json','prepared.json','configuration.json','fresh-start.json',
                 'trainer-wiring.json','first-parameter-change.json','initial-update-observations.jsonl']:
        if (RUN/name).exists():
            shutil.copy2(RUN/name, out/name)
    # Use the initial atomic status as the snapshot's completed-step ceiling.
    (out/'status.json').write_text(json.dumps(status,indent=2))
    metrics = []
    if (RUN/'metrics.jsonl').exists():
        raw = (RUN/'metrics.jsonl').read_bytes()
        for line in raw.split(b'\n')[:-1]:
            row = json.loads(line)
            if row.get('step',0) <= status['completed_optimizer_steps']:
                metrics.append(row)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in metrics))
    if (ROOT/'telemetry/resources.jsonl').exists():
        raw = (ROOT/'telemetry/resources.jsonl').read_bytes()
        (out/'resources.jsonl').write_bytes(b'\n'.join(raw.split(b'\n')[:-1])+b'\n')
    spec = importlib.util.spec_from_file_location('budget_metrics', Path(__file__).with_name('budget_metrics.py'))
    budget_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget_module)
    try:
        budgets = budget_module.collect(RUN/'checkpoints/completions', status['completed_optimizer_steps'])
    except Exception as error:
        budgets = {'records': [], 'sources': [], 'error': str(error)}
    (out/'budgets.json').write_text(json.dumps(budgets, indent=2))
    completed = status.get('last_complete_checkpoint_step',0)
    if completed:
        checkpoint = RUN/'checkpoints'/f'checkpoint-{completed}'
        shutil.copy2(checkpoint/'trainer_state.json', out/'saved-trainer-state.json')
        (out/'checkpoint-files.json').write_text(json.dumps({p.name:p.stat().st_size
                                                   for p in checkpoint.iterdir() if p.is_file()},indent=2))
    with (ROOT/'train.log').open('rb') as stream:
        stream.seek(max(0,os.fstat(stream.fileno()).st_size-131072))
        (out/'train-tail.log').write_bytes(stream.read())
    process = json.loads((RUN/'process.json').read_text())
    pid = process['pid']
    proc = Path(f'/proc/{pid}')
    actual = {'pid':pid,'exists':proc.exists()}
    if proc.exists():
        actual.update(cwd=os.readlink(proc/'cwd'), exe=os.readlink(proc/'exe'),
                      argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0'))
        ticks = int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
        boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
        actual['created_unix'] = boot + ticks/os.sysconf('SC_CLK_TCK')
        actual['source_matches'] = (actual['cwd'] == str(ROOT) and
                                   str(ROOT/'source/train_from_zero.py') in actual['argv'])
    systemd = subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState',
               '-p','MainPID','-p','ExecMainStartTimestamp','-p','Result'],capture_output=True,text=True)
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
                          '--format=csv,noheader'],capture_output=True,text=True)
    source = json.loads((RUN/'prepared.json').read_text())['source_sha256']
    source_ok = {name:Path(name).is_file() and sha(Path(name))==expected for name,expected in source.items()}
    capture = {'checked_utc':checked,'run':str(RUN),'process':actual,'systemd':systemd.stdout,
        'monitor_source_sha256':{name:sha(Path(__file__).with_name(name)) for name in ['capture_remote.py','budget_metrics.py','resource_sampler.py']},
        'gpu':gpu.stdout,'gpu_read_returncode':gpu.returncode,'source_hash_matches':source_ok,
        'completed_optimizer_steps':status['completed_optimizer_steps'],
        'saved_step':completed,'logged_steps':len([r for r in metrics if 'loss' in r]),
        'numeric_sessions_created':len(list((RUN/'sandbox-sessions').glob('answer-*'))),
        'numeric_session_count_note':'Includes any restarted numerical session; not distinct worlds or call count.'}
    (out/'capture.json').write_text(json.dumps(capture,indent=2))
    manifest = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive = out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.name)
    print(json.dumps({'archive':str(archive),'sha256':sha(archive),'capture':capture},indent=2))

if __name__=='__main__': main()
