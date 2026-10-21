"""Resume verified archive parts over bounded parallel SSH byte-range streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main(root, host, port, remote, workers):
    index = json.loads((root / 'archive-restore-index.json').read_text())
    scratch = root / '.download-chunks'
    scratch.mkdir(exist_ok=True)
    jobs, plan = [], []
    for part in index['parts_in_order']:
        dest = root / part['name']
        offset = dest.stat().st_size if dest.exists() else 0
        assert offset <= part['bytes']
        chunks = []
        for start in range(offset, part['bytes'], 8 * 1024 * 1024):
            count = min(8 * 1024 * 1024, part['bytes'] - start)
            chunk = scratch / (part['name'] + '.offset' + str(start))
            jobs.append((chunk, part['name'], start, count))
            chunks.append(chunk)
        plan.append((dest, part, chunks))

    def get(job):
        chunk, name, start, count = job
        code = f"import sys;f=open({(remote+'/'+name)!r},'rb');f.seek({start});sys.stdout.buffer.write(f.read({count}))"
        command = '/home/chen/runs/official-grpo-rollout-repair-20260928/venv/bin/python -c ' + shlex.quote(code)
        for attempt in range(3):
            with chunk.open('wb') as out:
                result = subprocess.run(['ssh', '-p', str(port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', host, command], stdout=out)
            if result.returncode == 0 and chunk.stat().st_size == count:
                print('Downloaded', name, start, count, flush=True)
                return
        raise RuntimeError('Byte-range download failed: ' + str(chunk))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(get, jobs))
    for dest, part, chunks in plan:
        tmp = scratch / (dest.name + '.assembled')
        with tmp.open('wb') as out:
            if dest.exists():
                with dest.open('rb') as source:
                    shutil.copyfileobj(source, out)
            for chunk in chunks:
                with chunk.open('rb') as source:
                    shutil.copyfileobj(source, out)
        h = hashlib.sha256()
        with tmp.open('rb') as source:
            for b in iter(lambda: source.read(1024 * 1024), b''):
                h.update(b)
        assert tmp.stat().st_size == part['bytes'] and h.hexdigest() == part['sha256']
        tmp.replace(dest)
        for chunk in chunks:
            chunk.unlink()
        print('Verified part', dest.name, flush=True)
    scratch.rmdir()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--remote', required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    main(a.root, a.host, a.port, a.remote, a.workers)
