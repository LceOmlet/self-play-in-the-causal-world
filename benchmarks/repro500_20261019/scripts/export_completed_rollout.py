"""Verify completed v4 evidence and export lossless Git-sized archive parts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def union_seconds(spans):
    total = 0
    end = float('-inf')
    for a, b in sorted(spans):
        assert b >= a
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def export(root):
    out = root / 'results'
    read = lambda p: json.loads(p.read_text())
    assert read(root / 'pipeline-status.json')['stage'] == 'complete'
    assert read(out / 'verification.json')['passed']
    for name, expected in read(out / 'all-result-checksums.json').items():
        assert sha(out / name) == expected, name
    members = read(out / 'archive-members-sha256.json')
    seen = set()
    with tarfile.open(out / 'raw-rollouts.tar.gz', 'r:gz') as tar:
        for entry in tar:
            assert entry.isfile() and entry.name not in seen
            seen.add(entry.name)
            assert hashlib.sha256(tar.extractfile(entry).read()).hexdigest() == members[entry.name]
    assert seen == set(members)
    speed = {'measurement': 'Wall clock from runner-source.json mtime to final progress.json mtime; event spans use recorded unix timestamps. Generation/tool busy time is interval union across cases, while request-seconds sum overlaps. No GPU occupancy inference.', 'models': {}}
    for model in ('base', 'checkpoint2350'):
        folder = root / model
        assert read(folder / 'progress.json')['completed'] == 100
        assert read(folder / 'runner-source.json')['run_training_rollout.py_sha256'] == '4146f53f964b8f9aede8954faf5aebb59a7fa8084a547f8394ee96c112517073'
        gs, ts, totals = [], [], {'generation_requests': 0, 'tool_requests': 0, 'rollout_model_tokens': 0, 'rollout_tool_feedback_tokens': 0, 'rollout_completion_tokens': 0}
        for case in (folder / 'cases').iterdir():
            row = read(case / 'private-result.json')
            for key in ('rollout_model_tokens', 'rollout_tool_feedback_tokens', 'rollout_completion_tokens'):
                totals[key] += row[key]
            pending_g, pending_t = {}, []
            for line in (case / 'events.jsonl').open():
                e = json.loads(line)
                if e['kind'] == 'generation_request':
                    pending_g[e['turn']] = e['unix']
                    totals['generation_requests'] += 1
                elif e['kind'] == 'generation_response':
                    gs.append((pending_g.pop(e['turn']), e['unix']))
                elif e['kind'] == 'tool_request':
                    pending_t.append(e['unix'])
                    totals['tool_requests'] += 1
                elif e['kind'] == 'tool_response':
                    ts.append((pending_t.pop(0), e['unix']))
            assert not pending_g and not pending_t
        start = (folder / 'runner-source.json').stat().st_mtime
        end = (folder / 'progress.json').stat().st_mtime
        totals.update(start_unix=start, end_unix=end, end_to_end_seconds=end-start,
                      generation_busy_union_seconds=union_seconds(gs),
                      generation_request_seconds_sum=sum(b-a for a,b in gs),
                      tool_busy_union_seconds=union_seconds(ts),
                      tool_request_seconds_sum=sum(b-a for a,b in ts),
                      model_tokens_per_end_to_end_second=totals['rollout_model_tokens']/(end-start))
        speed['models'][model] = totals
    speed['serial_two_model_wall_seconds'] = speed['models']['checkpoint2350']['end_unix'] - speed['models']['base']['start_unix']
    write(out / 'runtime-analysis.json', speed)
    parts, whole = [], hashlib.sha256()
    with (out / 'raw-rollouts.tar.gz').open('rb') as f:
        while True:
            b = f.read(80 * 1024 * 1024)
            if not b:
                break
            p = out / ('raw-rollouts.tar.gz.part%03d' % (len(parts)+1))
            p.write_bytes(b)
            whole.update(b)
            parts.append({'name': p.name, 'bytes': len(b), 'sha256': sha(p)})
    assert whole.hexdigest() == sha(out / 'raw-rollouts.tar.gz')
    # Re-read the parts, rather than trusting the input buffers used to create them.
    restored = hashlib.sha256()
    for part in parts:
        with (out / part['name']).open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                restored.update(b)
    assert restored.hexdigest() == whole.hexdigest()
    write(out / 'archive-restore-index.json', {'archive': 'raw-rollouts.tar.gz', 'bytes': (out / 'raw-rollouts.tar.gz').stat().st_size, 'sha256': whole.hexdigest(), 'parts_in_order': parts, 'restore': 'Concatenate the binary bytes of parts_in_order, then verify the full SHA256 before opening the gzip tar. The complete original tar remains on the remote host.'})
    metadata = out / 'execution-metadata'
    metadata.mkdir(exist_ok=True)
    for name in ('pipeline-status.json', 'preflight.json', 'base.log', 'checkpoint2350.log', 'export.log', 'synthetic-tests.log', 'official-rule-tests.log'):
        shutil.copyfile(root / name, metadata / name)
    for model in ('base', 'checkpoint2350'):
        for name in ('progress.json', 'runner-source.json', 'engine-execution.json', 'run-summary.json'):
            shutil.copyfile(root / model / name, metadata / (model + '-' + name))
    write(out / 'archive-verification.json', {'passed': True, 'members_verified': len(seen), 'all_original_result_checksums_verified': True, 'archive_sha256': whole.hexdigest(), 'all_parts_reread_and_reassembled_sha256_verified': True, 'runner_source_sha256': '4146f53f964b8f9aede8954faf5aebb59a7fa8084a547f8394ee96c112517073'})
    write(out / 'export-checksums.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('raw-rollouts.tar.gz', 'export-checksums.json')})
    print(json.dumps({'archive_verified': len(seen), 'parts': parts, 'runtime': speed}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    export(parser.parse_args().root)
