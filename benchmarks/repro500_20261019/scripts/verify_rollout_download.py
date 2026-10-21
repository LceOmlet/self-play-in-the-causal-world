"""Verify every downloaded export, the concatenated stream, and every tar member."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Parts:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, n):
        chunks = []
        remaining = n
        while remaining:
            if self.current is None:
                try:
                    self.current = next(self.paths).open('rb')
                except StopIteration:
                    break
            chunk = self.current.read(remaining)
            if not chunk:
                self.current.close()
                self.current = None
            else:
                chunks.append(chunk)
                remaining -= len(chunk)
        return b''.join(chunks)


def verify(root):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    checksums = read(root / 'export-checksums.json')
    for name, expected in checksums.items():
        assert sha(root / name) == expected, name
    index = read(root / 'archive-restore-index.json')
    parts = [root / p['name'] for p in index['parts_in_order']]
    h, size = hashlib.sha256(), 0
    for p, expected in zip(parts, index['parts_in_order']):
        assert sha(p) == expected['sha256'] and p.stat().st_size == expected['bytes']
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
                size += len(chunk)
    assert h.hexdigest() == index['sha256'] and size == index['bytes']
    members = read(root / 'archive-members-sha256.json')
    actual, generated = {}, {'base': 0, 'checkpoint2350': 0}
    with tarfile.open(fileobj=Parts(parts), mode='r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name not in actual
            blob = tar.extractfile(member).read()
            actual[member.name] = hashlib.sha256(blob).hexdigest()
            if member.name.endswith('/events.jsonl'):
                model = member.name.split('/')[0]
                for line in io.BytesIO(blob):
                    e = json.loads(line)
                    if e['kind'] == 'generation_response':
                        generated[model] += sum(len(o['token_ids']) for o in e['outputs'])
    assert actual == members
    result = {'passed': True, 'downloaded_export_files_verified': len(checksums), 'archive_bytes': size,
              'archive_sha256': h.hexdigest(), 'archive_members_verified': len(actual),
              'full_generation_response_tokens': generated,
              'token_note': 'Full sampled response tokens are counted from archived events; rollout_model_tokens count only tokens retained by the unchanged training completion-budget logic. All response outputs remain archived.'}
    (root / 'local-verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root)
