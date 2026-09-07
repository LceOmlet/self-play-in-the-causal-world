"""Archive prepared task evidence and a clearly labelled training-start snapshot."""

import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path('/home/chen/runs/training-submission-20260907')
PILOT = Path('/home/chen/runs/base-signal-diagnostic-20260907')
PROJECT = Path('/home/chen/projects/self-play-in-the-causal-world-rewardv10-a92ab8e-20260831')
TARGET = PROJECT/'research/training_submission_20260907'
BASE_TARGET = PROJECT/'research/base_signal_diagnostic_20260907'


def archive(path, root, files):
    assert not path.exists()
    with tarfile.open(path, 'w:gz') as tar:
        for file in sorted(set(files)):
            if file.is_file():
                tar.add(file, arcname=str(file.relative_to(root)), recursive=False)
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    assert json.loads((ROOT/'data/acceptance.json').read_text())['passed']
    assert json.loads((ROOT/'preflight-acceptance.json').read_text())['passed']
    assert (PILOT/'run-01/exit.json').exists()
    evidence = {}
    data_files = list((ROOT/'data').rglob('*')) + list((ROOT/'cf-diagnostics').rglob('*'))
    data_files += [ROOT/'prepare.py', ROOT/'prepare.log', ROOT/'prepare-process.json',
                   ROOT/'cf-attempts.jsonl', ROOT/'data-gate-summary.json']
    evidence['certified-data.tar.gz'] = archive(TARGET/'certified-data.tar.gz', ROOT, data_files)
    initial_files = [ROOT/name for name in (
        'control.py', 'analyze_case.py', 'trajectory-case.json', 'preflight.log',
        'preflight-source.json', 'preflight-acceptance.json')]
    initial_files += [ROOT/'run-01'/name for name in (
        'launch.json', 'process.json', 'supervisor.json', 'official-source-verification.json')]
    # Copy a stable prefix, explicitly not the eventual complete training log.
    (ROOT/'training-start.log').write_bytes((ROOT/'run-01/train.log').read_bytes())
    initial_files += [ROOT/'training-start.log']
    evidence['training-start.tar.gz'] = archive(TARGET/'training-start.tar.gz', ROOT, initial_files)
    pilot_files = [PILOT/'control.py', PILOT/'analyze_events.py', PILOT/'stopped-analysis.json']
    pilot_files += [p for p in (PILOT/'run-01').rglob('*') if p.is_file()
                    and not any(part in ('checkpoints', 'observation', '__pycache__') for part in p.parts)]
    pilot = archive(BASE_TARGET/'stopped-evidence.tar.gz', PILOT, pilot_files)
    (BASE_TARGET/'stopped-evidence.sha256.json').write_text(json.dumps(pilot, indent=2)+'\n')
    report = {'snapshot_time': time.time(), 'scope': 'Certified task data, stopped pilot, and training-start evidence; training is ongoing.',
              'files': evidence, 'pilot_archive': pilot}
    (TARGET/'evidence-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
