"""Low-frequency, read-only Linux/systemd/NVIDIA telemetry. Never loads a model."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
    return result.stdout.strip()


def sample(unit, root):
    now = time.time()
    row = {'utc': datetime.fromtimestamp(now, timezone.utc).isoformat(), 'unix': now,
           'monotonic': time.monotonic(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    errors = []
    try:
        fields = dict(line.split('=', 1) for line in command([
            'systemctl', '--user', 'show', unit, '-p', 'ControlGroup', '-p', 'MainPID', '-p', 'ActiveState']).splitlines())
        row.update(main_pid=int(fields['MainPID']), service_active=fields['ActiveState'])
        group = fields['ControlGroup']
        if not group.startswith('/') or group == '/' or '..' in Path(group).parts:
            raise ValueError('Missing or invalid training cgroup')
        cg = Path('/sys/fs/cgroup') / group.lstrip('/')
        row['cgroup'] = group
        row['cgroup_inode'] = cg.stat().st_ino
        row['cpu_usage_usec'] = int(dict(line.split() for line in (cg/'cpu.stat').read_text().splitlines())['usage_usec'])
        for filename, key in [('memory.current', 'ram_bytes'), ('memory.peak', 'ram_peak_bytes'), ('pids.current', 'tasks')]:
            row[key] = int((cg/filename).read_text())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        errors.append('cgroup: ' + str(error))
    try:
        output = command(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu', '--format=csv,noheader,nounits'])
        row['gpus'] = []
        for values in csv.reader(output.splitlines()):
            index, uuid, name, *numbers = (v.strip() for v in values)
            gpu = {'index': int(index), 'uuid': uuid, 'name': name}
            for key, value in zip(['util_percent', 'used_mib', 'total_mib', 'power_w', 'temperature_c'], numbers):
                try:
                    gpu[key] = float(value)
                except ValueError:
                    gpu[key] = None
            row['gpus'].append(gpu)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        errors.append('gpu: ' + str(error))
    row['disk_free_bytes'] = shutil.disk_usage(root).free
    mem = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
    row['host_ram_used_bytes'] = mem['MemTotal'] - mem['MemAvailable']
    row['host_ram_total_bytes'] = mem['MemTotal']
    row['errors'] = errors
    return row


def cpu_cores(previous, current):
    keys = ('boot_id', 'main_pid', 'cgroup', 'cgroup_inode')
    if not previous or any(previous.get(k) != current.get(k) for k in keys):
        return None
    if 'cpu_usage_usec' not in previous or 'cpu_usage_usec' not in current:
        return None
    elapsed = current['monotonic'] - previous['monotonic']
    delta = current['cpu_usage_usec'] - previous['cpu_usage_usec']
    return delta / 1e6 / elapsed if elapsed > 0 and delta >= 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 30:
        parser.error('Use a low-frequency interval of at least 30 seconds')
    out = args.root/'telemetry'
    out.mkdir(exist_ok=True)
    previous = None
    while True:
        started = time.monotonic()
        row = sample(args.unit, args.root)
        row['cpu_cores'] = cpu_cores(previous, row)
        row['interval_seconds'] = row['monotonic'] - previous['monotonic'] if previous else None
        with (out/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        previous = row
        if args.once:
            return
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
