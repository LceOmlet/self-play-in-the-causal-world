"""Verify the independent local Git copy and archived byte fingerprints."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1] / 'self-play-in-the-causal-world'
research = PROJECT / 'research/rl_correctness_20260907'
manifest = json.loads((research / 'archive-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((research / name).read_bytes()).hexdigest() == expected, name
upstream = json.loads((PROJECT / 'configs/verl/upstream_sources.json').read_text())
for patch in upstream['reviewed_runtime_patches']:
    assert hashlib.sha256((PROJECT / patch['path']).read_bytes()).hexdigest() == patch['sha256']
for command in (['git', '-C', str(PROJECT), 'fsck', '--full'],
                ['git', '-C', str(PROJECT), 'bundle', 'verify', str(ROOT / 'local-recovery.bundle')]):
    result = subprocess.run(command, capture_output=True, check=True)
assert not subprocess.check_output(['git', '-C', str(PROJECT), 'status', '--porcelain']).strip()
report = {'passed': True, 'local_project': str(PROJECT),
          'commit': subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD']).decode().strip(),
          'archive_fingerprints_verified': len(manifest),
          'upstream_patches_verified': len(upstream['reviewed_runtime_patches']),
          'git_fsck': 'passed', 'recovery_bundle': 'verified', 'worktree_clean': True}
(ROOT / 'local-repository-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
