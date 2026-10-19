"""Dependency-free integrity verification; does not print hidden task contents."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')


def main():
    manifest = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    tasks = manifest['tasks']
    assert len(tasks) == 500 and len({t['task_id'] for t in tasks}) == 500
    assert hashlib.sha256(canonical(tasks)).hexdigest() == manifest['tasks_merkle_input_sha256']
    for task in tasks:
        blob = (ROOT/'data'/task['file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == task['compressed_sha256']
        assert hashlib.sha256(canonical(json.loads(gzip.decompress(blob)))).hexdigest() == task['sha256']
    for name, digest in manifest['owner']['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    evaluation = json.loads((ROOT/'data/eval100.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(canonical(evaluation)).hexdigest() == manifest['eval100_sha256']
    assert len(evaluation['task_ids']) == len(set(evaluation['task_ids'])) == 100
    print(json.dumps({'verified_tasks':500, 'evaluation_tasks':100,
        'dataset_sha256':manifest['tasks_merkle_input_sha256']}))


if __name__ == '__main__':
    main()
