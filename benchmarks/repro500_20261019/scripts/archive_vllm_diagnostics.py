"""Archive completed measurements, source pins and withdrawn-gate evidence."""
import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_training_rollout import SUITE, read, write, digest

ROOT = Path('/home/chen/runs/repro-benchmark500-20261019')
NAMES = ('vllm-speed-tolerance-20261021','vllm-decode-graphs-tolerance-20261021',
    'vllm-sync-decode-graphs-tolerance-20261021','vllm-fixed-schedule-graphs-20261021',
    'vllm-speed-tolerance-20261021-api-preflight-failure',
    'vllm-decode-graphs-tolerance-20261021-withdrawn-project-gate',
    'training-rollout-v4-20261021-withdrawn-project-gate',
    'training-rollout-v4-20261021-gate-wait-attempt1',
    'training-rollout-v4-20261021-official-async-rejected',
    'training-rollout-v4-20261021-sync-control-rejected')


def main():
    assert all(read(ROOT/name/'status.json')['stage'] == 'complete' for name in NAMES[:2])
    assert read(ROOT/NAMES[3]/'status.json')['stage'] == 'complete'
    assert read(ROOT/NAMES[3]/'official-report.json')['passed']
    files = []
    for name in NAMES:
        for path in sorted((ROOT/name).rglob('*')):
            assert not path.is_symlink(),str(path)
            if path.is_file():
                files.append((path,path.relative_to(ROOT).as_posix()))
    for name in ('run_training_rollout.py','scripts/run_graph_rollout_pipeline.py',
        'scripts/vllm_speed_tolerance.py','scripts/vllm_decode_graphs_tolerance.py',
        'scripts/official_vllm_validation.py','scripts/tests_official_vllm_validation.py',
        'scripts/vllm_sync_graphs_tolerance.py','scripts/vllm_matched_schedule.py',
        'scripts/profile_recorded_generation.py','scripts/summarize_vllm_speed.py',
        'scripts/archive_vllm_diagnostics.py'):
        files.append((SUITE/name,'source-current/'+name))
    files.append((ROOT/'training-rollout-v3-contract.json','source-current/training-contract.json'))
    hashes = {}
    archive_path = ROOT/'vllm-official-diagnostics-20261021.tar.gz'
    assert not archive_path.exists(), 'Do not overwrite completed evidence'
    with tarfile.open(archive_path,'w:gz') as archive:
        for path,name in files:
            assert name not in hashes
            data = path.read_bytes()
            hashes[name] = digest(data)
            info = tarfile.TarInfo(name)
            info.size,info.mtime,info.mode = len(data),0,0o644
            archive.addfile(info,io.BytesIO(data))
    with tarfile.open(archive_path,'r:gz') as archive:
        assert {m.name for m in archive.getmembers()} == set(hashes)
        for member in archive.getmembers():
            assert digest(archive.extractfile(member).read()) == hashes[member.name]
    write(ROOT/'vllm-official-diagnostics-20261021-members.json',hashes)
    write(ROOT/'vllm-official-diagnostics-20261021-verification.json',{
        'passed':True,'files':len(hashes),'archive_bytes':archive_path.stat().st_size,
        'archive_sha256':digest(archive_path.read_bytes()),
        'all_members_verified_by_reading_archive':True,
        'formal_rollouts_included':False,'training_or_weight_changes':False,
        'acceptance_policy':'official-only-vllm-v0.26.0'})


if __name__ == '__main__':
    main()
