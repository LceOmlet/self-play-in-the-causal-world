"""SSH read-only capture, SHA256 verification, safe import and five local plots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True, help='SSH user@host')
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--remote-python', required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
    remote_out = args.remote_root+'/checks/'+stamp
    remote = shlex.join([args.remote_python, '-I', args.remote_root+'/capture_remote.py', '--root', args.remote_root, '--unit', args.unit, '--output', remote_out])
    check = {'attempted_utc': datetime.now(timezone.utc).isoformat(), 'result': 'failed'}
    try:
        result = subprocess.run(['ssh', '-p', str(args.port), *options, args.host, remote], capture_output=True, text=True, timeout=240, check=True)
        receipt = json.loads(result.stdout)
        if receipt['archive'] != remote_out+'.tar.gz':
            raise ValueError('Unexpected archive path')
        archive = here/(stamp+'.tar.gz')
        subprocess.run(['scp', '-P', str(args.port), *options, args.host+':'+receipt['archive'], str(archive)], capture_output=True, text=True, timeout=240, check=True)
        subprocess.run([sys.executable, str(here/'import_capture.py'), '--archive', str(archive), '--sha256', receipt['sha256']], check=True, timeout=240)
        check.update(result='ok', capture_dir=str(here/'checks'/stamp), sha256=receipt['sha256'], trusted_utc=receipt['capture']['checked_utc'])
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        check['error'] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            check['stderr'] = error.stderr
        if (here/'latest/summary.json').exists():
            check['last_trusted_utc'] = json.loads((here/'latest/summary.json').read_text())['capture']['checked_utc']
        raise
    finally:
        (here/'check-attempts').mkdir(exist_ok=True)
        text = json.dumps(check, indent=2)
        (here/'check-attempts'/f'{stamp}.json').write_text(text)
        (here/'latest-check.json').write_text(text)
    print(json.dumps(check))


if __name__ == '__main__':
    main()
