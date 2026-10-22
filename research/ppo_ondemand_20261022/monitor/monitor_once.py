"""One SSH capture, verified import and local plot refresh; no training restart."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
HOST = 'chen@e1b4acb6c4c142d5a00d326cec07fb11.hn.takin.cc'
ROOT = '/home/chen/runs/ppo-ondemand-from-base-20261022'
PYTHON = '/home/chen/runs/official-grpo-rollout-repair-20260928/venv/bin/python'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']


def main(args):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt = {'attempted_utc': datetime.now(timezone.utc).isoformat(), 'result': 'failed'}
    previous = None
    if (HERE / 'latest/capture.json').exists():
        previous = json.loads((HERE / 'latest/capture.json').read_text(encoding='utf-8'))
    try:
        target = args.remote_root + '/monitor/checks/' + stamp
        command = shlex.join([PYTHON, '-I', args.remote_root + '/monitor/capture_remote.py',
            '--root', args.remote_root, '--output', target])
        response = subprocess.run(['ssh', '-p', str(args.port), *OPTIONS, args.host, command],
            capture_output=True, text=True, encoding='utf-8', check=True, timeout=300)
        receipt = json.loads(response.stdout)
        assert receipt['archive'] == target + '.tar.gz'
        (HERE / 'archives').mkdir(exist_ok=True)
        archive = HERE / 'archives' / (stamp + '.tar.gz')
        subprocess.run(['scp', '-P', str(args.port), *OPTIONS, args.host + ':' + receipt['archive'], str(archive)],
                       capture_output=True, text=True, encoding='utf-8', check=True, timeout=300)
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
        out = HERE / 'checks' / stamp
        out.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                assert member.isfile() and member.name in ('capture.json', 'series.jsonl')
            source.extractall(out, filter='data')
        subprocess.run([sys.executable, str(HERE / 'plot_report.py'), '--capture-dir', str(out),
                        '--history', str(HERE / 'history.jsonl')], check=True, timeout=180)
        capture = json.loads((out / 'capture.json').read_text(encoding='utf-8'))
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        delta = None
        if previous and previous['source_receipt_sha256'] == capture['source_receipt_sha256']:
            delta = capture['captured_global_step'] - previous.get('captured_global_step', previous['completed_steps'])
        summary.update(updates_since_previous_check=delta,
            previous_checked_utc=previous['checked_utc'] if previous else None)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        latest = HERE / 'latest'
        latest.mkdir(exist_ok=True)
        for path in out.iterdir():
            shutil.copyfile(path, latest / path.name)
        attempt.update(result='ok', capture=capture, capture_dir=str(out), archive_sha256=receipt['sha256'], summary=summary)
        print(json.dumps(summary))
    except (OSError, ValueError, AssertionError, subprocess.SubprocessError) as error:
        attempt['error'] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            attempt['stderr'] = error.stderr
        if previous:
            attempt['last_trusted_utc'] = previous['checked_utc']
        raise
    finally:
        (HERE / 'check-attempts').mkdir(exist_ok=True)
        value = json.dumps(attempt, indent=2, ensure_ascii=False) + '\n'
        (HERE / 'check-attempts' / (stamp + '.json')).write_text(value, encoding='utf-8')
        (HERE / 'latest-check.json').write_text(value, encoding='utf-8')
        with (HERE / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(attempt, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=HOST)
    parser.add_argument('--port', type=int, default=10179)
    parser.add_argument('--remote-root', default=ROOT)
    main(parser.parse_args())
