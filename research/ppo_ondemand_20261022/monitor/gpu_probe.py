"""Read real NVML metrics using a driver-matched, task-local userspace runtime."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def probe_gpu(root, samples=5, interval=1.):
    kernel = Path('/sys/module/nvidia/version').read_text().strip()
    runtime = root / 'nvml-repair-20261022' / 'runtime'
    binary = Path('/usr/bin/nvidia-smi')
    library = None
    env = os.environ.copy()
    if kernel == '595.84' and (runtime / 'lib/libnvidia-ml.so.1').exists():
        binary = runtime / 'bin/nvidia-smi'
        library = runtime / 'lib/libnvidia-ml.so.595.84'
        env['LD_LIBRARY_PATH'] = str(runtime / 'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    fields = ['uuid', 'driver_version', 'utilization.gpu', 'utilization.memory', 'memory.used',
              'memory.total', 'power.draw', 'temperature.gpu']
    values, errors = [], []
    for i in range(samples):
        try:
            result = subprocess.run([str(binary), '--id=0', '--query-gpu=' + ','.join(fields),
                '--format=csv,noheader,nounits'], env=env, capture_output=True, text=True, check=True, timeout=10)
            records = list(csv.reader(result.stdout.strip().splitlines()))
            assert len(records) == 1 and len(records[0]) == len(fields)
            raw = {key: value.strip() for key, value in zip(fields, records[0], strict=True)}
            record = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'unix': time.time(),
                      'uuid': raw['uuid'], 'driver_version': raw['driver_version']}
            for source, target, multiplier in [('utilization.gpu', 'gpu_utilization_percent', 1),
                ('utilization.memory', 'memory_bandwidth_utilization_percent', 1),
                ('memory.used', 'physical_memory_used_bytes', 2**20),
                ('memory.total', 'physical_memory_total_bytes', 2**20),
                ('power.draw', 'power_watts', 1), ('temperature.gpu', 'temperature_celsius', 1)]:
                try:
                    v = float(raw[source]) * multiplier
                    record[target] = v if math.isfinite(v) else None
                except ValueError:
                    record[target] = None
            values.append(record)
        except (OSError, AssertionError, subprocess.SubprocessError) as error:
            errors.append({'checked_utc': datetime.now(timezone.utc).isoformat(), 'error': str(error),
                           'stderr': getattr(error, 'stderr', None)})
        if i + 1 < samples:
            time.sleep(interval)
    util = [r['gpu_utilization_percent'] for r in values if r['gpu_utilization_percent'] is not None]
    latest = values[-1] if values else {}
    return {'available': bool(util), 'kernel_driver_version': kernel, 'nvidia_smi': str(binary),
        'nvml_library': str(library) if library else 'system default', 'samples': values, 'errors': errors,
        'requested_samples': samples, 'valid_utilization_samples': len(util), 'sample_interval_seconds': interval,
        'utilization_percent_mean': statistics.mean(util) if util else None,
        'utilization_percent_min': min(util) if util else None, 'utilization_percent_max': max(util) if util else None,
        'physical_memory_used_bytes': latest.get('physical_memory_used_bytes'),
        'physical_memory_total_bytes': latest.get('physical_memory_total_bytes'),
        'power_watts': latest.get('power_watts'), 'temperature_celsius': latest.get('temperature_celsius'),
        'scope': 'Actual NVML samples during this check; utilization mean is over these samples, not an hourly average'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--interval', type=float, default=1.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    assert 1 <= args.samples <= 120 and 0 <= args.interval <= 60
    result = probe_gpu(args.root, args.samples, args.interval)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(0 if result['available'] else 1)
