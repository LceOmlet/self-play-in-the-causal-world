"""Extract only matching NVML and nvidia-smi from official Ubuntu packages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_pid():
    return subprocess.check_output(['systemctl', '--user', 'show',
        'cpt-numerics-ppo-ondemand-from-base-20261022.service', '-p', 'MainPID', '--value'], text=True).strip()


def extract(package, suffix, destination):
    metadata = subprocess.check_output(['dpkg-deb', '-f', str(package), 'Version', 'Architecture'], text=True)
    assert '595.84-0ubuntu0.24.04.1' in metadata and 'amd64' in metadata
    process = subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(package)], stdout=subprocess.PIPE)
    found = False
    with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
        for member in archive:
            if member.name.endswith('/' + suffix) and member.isfile():
                assert not found
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_suffix(destination.suffix + '.tmp')
                with archive.extractfile(member) as source, temp.open('wb') as target:
                    shutil.copyfileobj(source, target)
                assert temp.read_bytes()[:4] == b'\x7fELF'
                if destination.exists():
                    assert sha(destination) == sha(temp), 'Refuse to replace a different existing runtime'
                temp.replace(destination)
                destination.chmod(0o755)
                found = True
    process.stdout.close()
    assert process.wait() == 0 and found


def main(root):
    folder = root / 'nvml-repair-20261022'
    runtime = folder / 'runtime'
    before = training_pid()
    kernel = Path('/sys/module/nvidia/version').read_text().strip()
    assert kernel == '595.84'
    compute = folder / 'libnvidia-compute-595_595.84.deb'
    utils = folder / 'nvidia-utils-595_595.84.deb'
    extract(compute, 'libnvidia-ml.so.595.84', runtime / 'lib/libnvidia-ml.so.595.84')
    extract(utils, 'nvidia-smi', runtime / 'bin/nvidia-smi')
    for name in ('libnvidia-ml.so.1', 'libnvidia-ml.so'):
        link = runtime / 'lib' / name
        if link.is_symlink():
            assert os.readlink(link) == 'libnvidia-ml.so.595.84'
        else:
            assert not link.exists()
            link.symlink_to('libnvidia-ml.so.595.84')
    wrapper = root / 'monitor/nvidia-smi'
    wrapper.write_text('#!/bin/sh\n'
        'if [ "$(cat /sys/module/nvidia/version 2>/dev/null)" = 595.84 ]; then\n'
        '  nvml_runtime=' + shlex.quote(str(runtime)) + '\n'
        '  LD_LIBRARY_PATH="$nvml_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" '
        'exec "$nvml_runtime/bin/nvidia-smi" "$@"\n'
        'fi\nexec /usr/bin/nvidia-smi "$@"\n', encoding='utf-8')
    wrapper.chmod(0o755)
    result = subprocess.check_output([str(wrapper), '--query-gpu=uuid,driver_version,utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader'], text=True)
    after = training_pid()
    assert before == after and after != '0', 'Training PID changed during the repair'
    receipt = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'kernel_driver_version': kernel,
        'scope': 'Task-local user-space NVML and nvidia-smi; system driver/library links unchanged',
        'training_pid_before': before, 'training_pid_after': after, 'training_restarted': False,
        'sources': ['https://launchpad.net/ubuntu/+archive/primary/+files/libnvidia-compute-595_595.84-0ubuntu0.24.04.1_amd64.deb',
                    'https://launchpad.net/ubuntu/+archive/primary/+files/nvidia-utils-595_595.84-0ubuntu0.24.04.1_amd64.deb'],
        'sha256': {str(p.relative_to(root)): sha(p) for p in (compute, utils,
            runtime / 'lib/libnvidia-ml.so.595.84', runtime / 'bin/nvidia-smi', wrapper)}, 'query_output': result.strip()}
    (folder / 'repair-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
